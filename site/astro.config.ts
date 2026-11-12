import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

// TEENIE_STATIC=1 (`pnpm build:static`, used by Cloudflare) leaves out the relay page, which needs the local Python
// relay (`uv run teenie-serve`). The public Try it page runs the model in the browser and needs no server.
const relay = process.env.TEENIE_STATIC !== '1';
// The public address, used for canonical links, social previews and the sitemap. SITE_URL overrides it.
const site = process.env.SITE_URL || 'https://teenie.lupyd.com';

// onnxruntime-web's bundle points at its 29 MB WebAssembly binary with new URL(..., import.meta.url), so Vite would copy
// it into the build, over Cloudflare's 25 MiB limit per file. The worker always hands the runtime that binary itself,
// gzipped from /ort/ (scripts/ort-assets.mjs), so the reference is never fetched: point it at the page instead.
const ortBinaryOutOfBuild = () => ({
  name: 'teenie-ort-binary-out-of-build',
  enforce: 'pre' as const,
  transform(code: string, id: string) {
    if (!id.includes('onnxruntime-web')) return null;
    const out = code.replace(/new URL\(("ort-wasm-simd-threaded[a-z.]*\.wasm"),\s*import\.meta\.url\)/g, 'new URL($1, self.location.href)');
    return out === code ? null : { code: out, map: null };
  },
});

export default defineConfig({
  // Static docs: `pnpm build` writes plain HTML/JS/JSON into dist/.
  output: 'static',
  // Keep whitespace as written: compression drops the line break before a tag that starts a line, which glues
  // words together ("give<strong>every bit</strong>").
  compressHTML: false,
  site,
  trailingSlash: 'always',
  devToolbar: { enabled: false },
  vite: {
    define: { __TEENIE_RELAY__: JSON.stringify(relay) },
    plugins: [ortBinaryOutOfBuild()],
    worker: { format: 'es', plugins: () => [ortBinaryOutOfBuild()] },
  },
  integrations: [
    // sitemap-index.xml for search engines; the relay page needs the local relay and is never public.
    sitemap({ filter: (page) => !page.includes('/relay/') }),
    {
      name: 'teenie-relay-pages',
      hooks: {
        'astro:config:setup': ({ injectRoute }) => {
          if (relay) injectRoute({ pattern: '/relay', entrypoint: './src/relay/try.astro' });
        },
      },
    },
  ],
});
