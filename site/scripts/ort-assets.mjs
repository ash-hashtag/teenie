// Ships onnxruntime-web's WebAssembly binary from this site instead of a CDN. The binary the WebGPU build uses is
// about 29 MB, over Cloudflare's 25 MiB limit per static file, so it goes out gzipped (about 7 MB) with the runtime
// version in its name, and the worker (src/lib/infer.worker.ts) unpacks it with DecompressionStream.
import { createRequire } from 'node:module';
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { gzipSync } from 'node:zlib';

const require = createRequire(import.meta.url);
const root = dirname(require.resolve('onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm')).replace(/\/dist$/, '');
const { version } = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
const folder = new URL('../public/ort/', import.meta.url).pathname;
const name = `ort-${version}-asyncify.wasm.gz`;
mkdirSync(folder, { recursive: true });
for (const old of readdirSync(folder)) if (old !== name) rmSync(join(folder, old));
if (!existsSync(join(folder, name))) {
  const wasm = readFileSync(join(root, 'dist', 'ort-wasm-simd-threaded.asyncify.wasm'));
  writeFileSync(join(folder, name), gzipSync(wasm, { level: 9 }));
}
writeFileSync(join(folder, '..', '..', 'src', 'data', 'ort.json'), JSON.stringify({ version, wasm: `/ort/${name}` }) + '\n');
console.log(`onnxruntime-web ${version}: public/ort/${name}`);
