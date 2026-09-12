// Live-site discoverability report, for the monthly scheduled workflow.
// check-seo.mjs gates the commit; this checks what Pages is actually serving —
// a deploy that silently dropped a page, a stale cache, or a crawl-policy
// accident are all invisible to a build-time check.
// Exits 1 (with a report on stdout) if anything below is wrong.
import fs from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd();
const site = JSON.parse(await fs.readFile(path.join(root, 'data/site.json'), 'utf8'));
const SITE = site.siteUrl;
const problems = [];
const UA = { 'user-agent': 'Mozilla/5.0 (compatible; raulv.dev discoverability checker)' };

async function get(url) {
  for (const method of ['GET', 'HEAD']) {
    try {
      const res = await fetch(url, { method, redirect: 'follow', signal: AbortSignal.timeout(20000), headers: UA });
      return { status: res.status, body: method === 'GET' ? await res.text() : '', url: res.url };
    } catch { /* try next method */ }
  }
  return { status: 0, body: '', url };
}

// --- robots.txt: crawl policy still what we think it is ---
const robots = await get(`${SITE}/robots.txt`);
if (robots.status !== 200) problems.push(`robots.txt returned ${robots.status}`);
else {
  if (/^\s*Disallow:\s*\/\s*$/mi.test(robots.body)) problems.push('robots.txt now disallows the entire site');
  if (!robots.body.includes(`${SITE}/sitemap.xml`)) problems.push('robots.txt no longer declares the sitemap');
}

// --- live sitemap vs the committed one ---
const liveSitemap = await get(`${SITE}/sitemap.xml`);
if (liveSitemap.status !== 200) {
  problems.push(`sitemap.xml returned ${liveSitemap.status}`);
} else {
  const liveUrls = [...liveSitemap.body.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1]);
  const localXml = await fs.readFile(path.join(root, 'sitemap.xml'), 'utf8');
  const localUrls = [...localXml.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1]);
  for (const url of localUrls) {
    if (!liveUrls.includes(url)) problems.push(`Sitemap on main lists ${url} but the live sitemap does not — deploy may be stale`);
  }
  for (const url of liveUrls) {
    if (!localUrls.includes(url)) problems.push(`Live sitemap lists ${url}, which main no longer builds`);
  }

  // --- every live sitemap URL serves, and serves what main built ---
  const ogImages = new Set();
  for (const url of liveUrls) {
    const res = await get(url);
    if (res.status !== 200) {
      problems.push(`Sitemap URL is not reachable: ${url} (${res.status})`);
      continue;
    }
    const liveTitle = res.body.match(/<title>([^<]*)<\/title>/)?.[1];
    const liveCanonical = res.body.match(/<link rel="canonical" href="([^"]*)"/)?.[1];
    const rel = url === `${SITE}/` ? 'index.html' : url.slice(SITE.length + 1);
    const local = await fs.readFile(path.join(root, rel), 'utf8').catch(() => null);
    if (local) {
      const localTitle = local.match(/<title>([^<]*)<\/title>/)?.[1];
      if (liveTitle !== localTitle) {
        problems.push(`Stale content at ${url}: live title "${liveTitle}" but main builds "${localTitle}"`);
      }
    }
    if (liveCanonical && liveCanonical !== url) {
      problems.push(`${url} is in the sitemap but canonicalises to ${liveCanonical}`);
    }
    if (/\bnoindex\b/.test(res.body.match(/<meta name="robots" content="([^"]*)"/)?.[1] || '')) {
      problems.push(`${url} is in the sitemap but serves a noindex robots meta`);
    }
    const og = res.body.match(/<meta property="og:image" content="([^"]*)"/)?.[1];
    if (og) ogImages.add(og);
  }

  // --- social cards resolve (a broken og:image kills every shared link) ---
  for (const img of ogImages) {
    const res = await get(img);
    if (res.status !== 200) problems.push(`og:image is not reachable: ${img} (${res.status})`);
  }
}

// --- feed ---
const rss = await get(`${SITE}/rss.xml`);
if (rss.status !== 200) problems.push(`rss.xml returned ${rss.status}`);
else {
  const links = [...rss.body.matchAll(/<link>([^<]+)<\/link>/g)].map((m) => m[1]);
  if (!links.length) problems.push('rss.xml contains no <link> entries');
  for (const link of links) {
    const res = await get(link);
    if (res.status !== 200) problems.push(`RSS link is not reachable: ${link} (${res.status})`);
  }
}

if (problems.length) {
  console.log(`DISCOVERABILITY REPORT (${problems.length} items):\n${problems.map((p) => `- ${p}`).join('\n')}`);
  console.log('\nNot checkable from CI: impressions, clicks, index coverage, and manual actions');
  console.log('live only in Google Search Console, which needs OAuth. Review those by hand.');
  process.exit(1);
}
console.log('discoverability check passed: live sitemap, crawl policy, social cards and feed all healthy');
