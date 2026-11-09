// robots.txt: everything may be indexed; points to the sitemap when the public address is known.
import type { APIRoute } from 'astro';

export const GET: APIRoute = ({ site }) => {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const sitemap = site ? `\nSitemap: ${new URL(`${base}/sitemap.xml`, site).href}\n` : '';
  return new Response(`User-agent: *\nAllow: /\n${sitemap}`, { headers: { 'Content-Type': 'text/plain' } });
};
