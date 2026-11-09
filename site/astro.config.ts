import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

// TEENIE_STATIC=1 (`pnpm build:static`, used by Cloudflare) leaves out the Try it page,
// which needs the local relay (`uv run teenie-serve`).
const relay = process.env.TEENIE_STATIC !== '1';
// The public address, used for canonical links, social previews and the sitemap. SITE_URL overrides it.
const site = process.env.SITE_URL || 'https://teenie.lupyd.com';

export default defineConfig({
  // Static docs: `pnpm build` writes plain HTML/JS/JSON into dist/.
  output: 'static',
  // Keep whitespace as written: compression drops the line break before a tag that starts a line, which glues
  // words together ("give<strong>every bit</strong>").
  compressHTML: false,
  site,
  trailingSlash: 'always',
  devToolbar: { enabled: false },
  vite: { define: { __TEENIE_RELAY__: JSON.stringify(relay) } },
  integrations: [
    // sitemap-index.xml for search engines; the Try it page needs the local relay and is never public.
    sitemap({ filter: (page) => !page.includes('/try/') }),
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
