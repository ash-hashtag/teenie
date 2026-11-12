// Runs the model off the main thread, so the page stays responsive while it thinks.
//
// in:  { type: 'load', width, url, check }               download (with progress) and open the model for a width
//      { type: 'run', id, width, bits, asked, questions } one pass: bits [999, 3, W], asked [Q, 2, W] in {-1, 1}
// out: { type: 'progress', loaded, total } | { type: 'ready', width, backend, device } | { type: 'result', id,
//      probabilities, seconds, backend } | { type: 'error', id?, message }
//
// Backend: WebGPU when the device has a hardware GPU adapter and the model passes a self-check on it, else WebAssembly.
// The self-check is a fixed small episode rebuilt here exactly as export_web.py builds it, compared with the
// probabilities the export recorded; onnxruntime-web's WebGPU kernels have returned wrong numbers without an error
// before (export_web.py pads the graph around the one we found), so a GPU that disagrees is never used. Software
// adapters (SwiftShader) are skipped: they are correct but 30 times slower than WebAssembly. The choice is logged to the
// console and shown on the page.

// onnxruntime-web from the CDN, pinned. Its WebGPU build also runs WebAssembly, so one runtime covers both.
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
const CONTEXT = 999;
const CHECK_QUESTIONS = 4;

type Backend = 'webgpu' | 'wasm';
type Tensor = { data: Float32Array | number[] };
type Session = { run(feeds: Record<string, unknown>): Promise<Record<string, Tensor>>; release(): Promise<void> };
type Ort = {
  env: { wasm: { wasmPaths: string; numThreads: number; proxy: boolean } };
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => Tensor;
  InferenceSession: { create(model: Uint8Array, options: object): Promise<Session> };
};
type Opened = { session: Session; backend: Backend; device: string };

let runtime: Promise<Ort> | null = null;
const sessions = new Map<number, Promise<Opened>>();

function ort(): Promise<Ort> {
  runtime ??= (import(/* @vite-ignore */ `${ORT}ort.webgpu.min.mjs`) as Promise<Ort>).then((module) => {
    module.env.wasm.wasmPaths = ORT;
    module.env.wasm.numThreads = 1; // a static site is not cross-origin isolated, so one thread
    module.env.wasm.proxy = false; // already off the main thread
    return module;
  });
  runtime.catch(() => (runtime = null));
  return runtime;
}

type Adapter = { isFallbackAdapter?: boolean; info?: { architecture?: string; vendor?: string; isFallbackAdapter?: boolean } };

/** The hardware GPU's name, or why there is none to use. */
async function hardwareGpu(): Promise<{ ok: boolean; device: string }> {
  const gpu = (self.navigator as unknown as { gpu?: { requestAdapter(o: object): Promise<Adapter | null> } }).gpu;
  if (!gpu) return { ok: false, device: 'no WebGPU in this browser' };
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { ok: false, device: 'no WebGPU adapter' };
    const name = [adapter.info?.vendor, adapter.info?.architecture].filter(Boolean).join(' ') || 'GPU';
    if (adapter.isFallbackAdapter || adapter.info?.isFallbackAdapter || /swiftshader/i.test(name))
      return { ok: false, device: `software WebGPU only (${name})` };
    return { ok: true, device: name };
  } catch (error) {
    return { ok: false, device: `WebGPU failed: ${(error as Error).message}` };
  }
}

async function download(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`could not download ${url.split('/').pop()} (HTTP ${response.status})`);
  const total = Number(response.headers.get('content-length') ?? 0);
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    postMessage({ type: 'progress', loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

// The self-check episode (export_web.check_episode): a, b below 256 from a multiplicative hash, c = a*b + a.
function checkFeeds(module: Ort, width: number): Record<string, Tensor> {
  const hashed = (i: number) => Math.imul(i + 1, 0x9e3779b1) >>> 0;
  const signed = (target: Float32Array, offset: number, value: number) => {
    for (let k = 0; k < width; k++) target[offset + k] = Math.floor(value / 2 ** k) % 2 ? 1 : -1;
  };
  const pair = (n: number) => [hashed(2 * n) & 255, hashed(2 * n + 1) & 255] as const;
  const bits = new Float32Array(CONTEXT * 3 * width);
  for (let n = 0; n < CONTEXT; n++) {
    const [a, b] = pair(n);
    signed(bits, n * 3 * width, a);
    signed(bits, (n * 3 + 1) * width, b);
    signed(bits, (n * 3 + 2) * width, (a * b + a) % 2 ** width);
  }
  const asked = new Float32Array(CHECK_QUESTIONS * 2 * width);
  for (let q = 0; q < CHECK_QUESTIONS; q++) {
    const [a, b] = pair(CONTEXT + q);
    signed(asked, q * 2 * width, a);
    signed(asked, (q * 2 + 1) * width, b);
  }
  return {
    bits: new module.Tensor('float32', bits, [1, CONTEXT, 3, width]),
    asked: new module.Tensor('float32', asked, [1, CHECK_QUESTIONS, 2, width]),
  };
}

/** A WebGPU session that passed the self-check, or why WebGPU is not used. */
async function onGpu(module: Ort, bytes: Uint8Array, width: number, check: number[] | undefined):
    Promise<{ session: Session | null; device: string }> {
  const gpu = await hardwareGpu();
  if (!gpu.ok) return { session: null, device: gpu.device };
  if (!check?.length) return { session: null, device: 'no self-check for this model' };
  let session: Session | null = null;
  let why = '';
  try {
    session = await module.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });
    const got = (await session.run(checkFeeds(module, width))).probabilities!.data;
    const off = check.reduce((worst, x, i) => Math.max(worst, Math.abs(got[i]! - x)), got.length === check.length ? 0 : 1);
    if (off < 0.01) return { session, device: gpu.device };
    why = `${gpu.device} failed the self-check (off by ${off.toFixed(3)})`;
  } catch (error) {
    why = `${gpu.device}: ${(error as Error).message}`;
  }
  await session?.release().catch(() => {});
  return { session: null, device: why };
}

function open(width: number, url: string, check?: number[]): Promise<Opened> {
  if (!sessions.has(width)) {
    const opening = (async (): Promise<Opened> => {
      const [module, bytes] = await Promise.all([ort(), download(url)]);
      const gpu = await onGpu(module, bytes, width, check);
      if (gpu.session) {
        console.info(`[teenie] ${width}-bit model on the GPU (WebGPU, ${gpu.device}); self-check passed`);
        return { session: gpu.session, backend: 'webgpu', device: gpu.device };
      }
      console.info(`[teenie] ${width}-bit model on the CPU (WebAssembly): ${gpu.device}`);
      return { session: await module.InferenceSession.create(bytes, { executionProviders: ['wasm'] }), backend: 'wasm',
               device: gpu.device };
    })();
    opening.catch(() => sessions.delete(width)); // a failed download can be retried
    sessions.set(width, opening);
  }
  return sessions.get(width)!;
}

self.onmessage = async (event: MessageEvent) => {
  const message = event.data;
  try {
    if (message.type === 'load') {
      const { backend, device } = await open(message.width, message.url, message.check);
      postMessage({ type: 'ready', width: message.width, backend, device });
    } else if (message.type === 'run') {
      const { id, width, bits, asked, questions, url } = message;
      const [module, { session: s, backend }] = await Promise.all([ort(), open(width, url, message.check)]);
      const started = performance.now();
      const output = await s.run({
        bits: new module.Tensor('float32', bits, [1, CONTEXT, 3, width]),
        asked: new module.Tensor('float32', asked, [1, questions, 2, width]),
      });
      const probabilities = output.probabilities!.data as Float32Array;
      console.info(`[teenie] ${questions} questions at ${width} bits on the ${backend === 'webgpu' ? 'GPU' : 'CPU'}: ${(performance.now() - started).toFixed(0)} ms`);
      (self as unknown as Worker).postMessage({ type: 'result', id, probabilities, backend, seconds: (performance.now() - started) / 1000 },
        [probabilities.buffer]);
    }
  } catch (error) {
    postMessage({ type: 'error', id: message.id, message: (error as Error).message });
  }
};
