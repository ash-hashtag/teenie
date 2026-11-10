// Runs the model off the main thread, so the page stays responsive while it thinks.
//
// in:  { type: 'load', width, url }                      download (with progress) and open the model for a width
//      { type: 'run', id, width, bits, asked, questions } one pass: bits [999, 3, W], asked [Q, 2, W] in {-1, 1}
// out: { type: 'progress', loaded, total } | { type: 'ready', width } | { type: 'result', id, probabilities, seconds }
//      | { type: 'error', id?, message }

const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const CONTEXT = 999;

type Tensor = { data: Float32Array };
type Session = { run(feeds: Record<string, unknown>): Promise<Record<string, Tensor>> };
type Ort = {
  env: { wasm: { wasmPaths: string; numThreads: number; proxy: boolean } };
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => Tensor;
  InferenceSession: { create(model: Uint8Array, options: object): Promise<Session> };
};

let runtime: Promise<Ort> | null = null;
const sessions = new Map<number, Promise<Session>>();

function ort(): Promise<Ort> {
  // WebAssembly only: onnxruntime-web's WebGPU backend returns wrong answers on this graph without an error.
  runtime ??= (import(/* @vite-ignore */ `${ORT}ort.min.mjs`) as Promise<Ort>).then((module) => {
    module.env.wasm.wasmPaths = ORT;
    module.env.wasm.numThreads = 1; // a static site is not cross-origin isolated, so one thread
    module.env.wasm.proxy = false; // already off the main thread
    return module;
  });
  return runtime;
}

async function download(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`could not download the model (HTTP ${response.status})`);
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

function session(width: number, url: string): Promise<Session> {
  if (!sessions.has(width)) {
    const opening = (async () => {
      const [module, bytes] = await Promise.all([ort(), download(url)]);
      return module.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
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
      await session(message.width, message.url);
      postMessage({ type: 'ready', width: message.width });
    } else if (message.type === 'run') {
      const { id, width, bits, asked, questions, url } = message;
      const [module, s] = await Promise.all([ort(), session(width, url)]);
      const started = performance.now();
      const output = await s.run({
        bits: new module.Tensor('float32', bits, [1, CONTEXT, 3, width]),
        asked: new module.Tensor('float32', asked, [1, questions, 2, width]),
      });
      const probabilities = output.probabilities!.data;
      (self as unknown as Worker).postMessage({ type: 'result', id, probabilities, seconds: (performance.now() - started) / 1000 },
        [probabilities.buffer]);
    }
  } catch (error) {
    postMessage({ type: 'error', id: message.id, message: (error as Error).message });
  }
};
