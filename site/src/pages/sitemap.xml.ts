// sitemap.xml for search engines: every public page. Needs SITE_URL at build time for absolute addresses.
import type { APIRoute } from 'astro';

// The Try it page needs the local Python relay and is never part of the public site.
const PAGES = ['/', '/model/', '/results/', '/changelog/', '/first-model/', '/learning/', '/benchmarks/'];

export const GET: APIRoute = ({ site }) => {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  // Without SITE_URL there are no absolute addresses to list, so the sitemap stays empty.
  const urls = !site ? '' : PAGES.map((page) => `  <url><loc>${new URL(`${base}${page}`, site!).href}</loc></url>`).join('\n');
  return new Response(
    `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`,
    { headers: { 'Content-Type': 'application/xml' } },
  );
};
