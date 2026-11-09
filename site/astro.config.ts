import { defineConfig } from 'astro/config';

// TEENIE_STATIC=1 (`pnpm build:static`, used by Cloudflare) leaves out the Try it page,
// which needs the local relay (`uv run teenie-serve`).
const relay = process.env.TEENIE_STATIC !== '1';
// The public address, used for canonical links, social previews and the sitemap. SITE_URL overrides it.
const site = process.env.SITE_URL || 'https://teenie.lupyd.com';

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
