import { defineConfig } from 'astro/config';

// TEENIE_STATIC=1 (`pnpm build:static`, used by Cloudflare) leaves out the Try it page,
// which needs the local relay (`uv run teenie-serve`).
const relay = process.env.TEENIE_STATIC !== '1';
// The public address (e.g. https://teenie.example.com), used for canonical links, social previews and the sitemap.
// Set SITE_URL in the build environment; without it those absolute links are left out.
const site = process.env.SITE_URL || undefined;

export default defineConfig({
  // Static docs: `pnpm build` writes plain HTML/JS/JSON into dist/.
  output: 'static',
  site,
  trailingSlash: 'always',
  devToolbar: { enabled: false },
  vite: { define: { __TEENIE_RELAY__: JSON.stringify(relay) } },
  integrations: [
    {
      name: 'teenie-relay-pages',
      hooks: {
        'astro:config:setup': ({ injectRoute }) => {
          if (relay) injectRoute({ pattern: '/try', entrypoint: './src/relay/try.astro' });
        },
      },
    },
  ],
});
