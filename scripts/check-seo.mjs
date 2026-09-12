// Discoverability gate: every indexable page must carry the metadata a crawler
// needs, and the sitemap must agree with what the build actually emitted.
// Pure fs — no browser, no network — so it tests the commit under test rather
// than whatever is currently deployed.
import fs from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd();
const site = JSON.parse(await fs.readFile(path.join(root, 'data/site.json'), 'utf8'));
const SITE = site.siteUrl;
const manifest = JSON.parse(await fs.readFile(path.join(root, 'data/image-manifest.json'), 'utf8'));

// Hard ceilings catch template runaway (a data field leaking into the whole
// tag); the SEO-optimal bands below are advisory, because tightening those is a
// copy decision, not a build regression.
const TITLE_MAX = 120;
const DESC_MAX = 320;
// Advisory bands flag only real truncation loss, not every page a few chars
// over the ideal. A gate nobody reads is worse than no gate.
const TITLE_BAND = [15, 80];
const DESC_BAND = [50, 200];
// Social cards below 200x200 are dropped outright by several platforms; below
// 1200x630 they render as a small thumbnail instead of a large card.
const OG_MIN = [200, 200];
const OG_IDEAL = [1200, 630];

const SKIP = new Set(['node_modules', '.git', '.cache', 'ieee-sps-uf-site']);
async function htmlPages(dir = root, prefix = '') {
  const out = [];
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    if (SKIP.has(entry.name) || entry.name.startsWith('.')) continue;
    if (entry.isDirectory()) out.push(...await htmlPages(path.join(dir, entry.name), `${prefix}${entry.name}/`));
    else if (entry.name.endsWith('.html')) out.push(`${prefix}${entry.name}`);
  }
  return out;
}

const attr = (html, re) => html.match(re)?.[1];
const meta = (html, name) => attr(html, new RegExp(`<meta name="${name}" content="([^"]*)"`));
const prop = (html, name) => attr(html, new RegExp(`<meta property="${name}" content="([^"]*)"`));
const canonicalFor = (p) => (p === 'index.html' ? `${SITE}/` : `${SITE}/${p}`);
const pathForUrl = (url) => {
  if (!url.startsWith(`${SITE}/`)) return null;
  const rel = url.slice(SITE.length + 1).split(/[?#]/)[0];
  return rel === '' ? 'index.html' : rel;
};

const failures = [];
const advisories = [];
const fail = (p, msg) => failures.push(`${p}: ${msg}`);

const pages = (await htmlPages()).sort();
const sitemapXml = await fs.readFile(path.join(root, 'sitemap.xml'), 'utf8');
const sitemapUrls = [...sitemapXml.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1]);
const sitemapPaths = new Set(sitemapUrls.map(pathForUrl));

const titles = new Map();
const descriptions = new Map();
const assetCache = new Map();
async function exists(rel) {
  if (!assetCache.has(rel)) {
    assetCache.set(rel, await fs.access(path.join(root, rel)).then(() => true, () => false));
  }
  return assetCache.get(rel);
}

for (const p of pages) {
  const html = await fs.readFile(path.join(root, p), 'utf8');
  const robots = meta(html, 'robots') || '';
  // A page that asks not to be indexed is exempt from the indexability rules —
  // 404.html and the deliberate "moved" stubs are the only such pages.
  const indexable = !/\bnoindex\b/.test(robots);
  const title = (attr(html, /<title>([^<]*)<\/title>/) || '').trim();
  const description = (meta(html, 'description') || '').trim();
  const canonical = attr(html, /<link rel="canonical" href="([^"]*)"/);

  // --- title ---
  if (!title) fail(p, 'missing or empty <title>');
  else {
    if (title.length > TITLE_MAX) fail(p, `title is ${title.length} chars, over the ${TITLE_MAX}-char ceiling`);
    if (titles.has(title)) fail(p, `duplicate <title> shared with ${titles.get(title)}`);
    else titles.set(title, p);
    if (indexable && (title.length < TITLE_BAND[0] || title.length > TITLE_BAND[1])) {
      advisories.push(`${p}: title is ${title.length} chars (outside the ${TITLE_BAND.join('-')} band search results show in full)`);
    }
  }

  // --- description ---
  if (indexable && !description) fail(p, 'missing or empty meta description');
  if (description) {
    if (description.length > DESC_MAX) fail(p, `meta description is ${description.length} chars, over the ${DESC_MAX}-char ceiling`);
    if (descriptions.has(description)) fail(p, `duplicate meta description shared with ${descriptions.get(description)}`);
    else descriptions.set(description, p);
    if (indexable && (description.length < DESC_BAND[0] || description.length > DESC_BAND[1])) {
      advisories.push(`${p}: meta description is ${description.length} chars (outside the ${DESC_BAND.join('-')} band search results show in full)`);
    }
  }

  // --- canonical ---
  if (indexable) {
    if (!canonical) fail(p, 'no <link rel="canonical">');
    else if (canonical !== canonicalFor(p)) fail(p, `canonical points at ${canonical}, not its own URL ${canonicalFor(p)}`);
  }

  // --- exactly one h1 ---
  const h1s = [...html.matchAll(/<h1[\s>]/g)].length;
  if (h1s !== 1) fail(p, `has ${h1s} <h1> elements, expected exactly 1`);

  // --- social card ---
  if (indexable) {
    const image = prop(html, 'og:image');
    if (!image) fail(p, 'no og:image');
    else if (!image.startsWith(`${SITE}/`)) fail(p, `og:image "${image}" is not an absolute ${SITE} URL`);
    else if (!(await exists(pathForUrl(image)))) fail(p, `og:image ${image} does not exist in the build`);
    else {
      const dim = manifest[pathForUrl(image)];
      if (!dim) fail(p, `og:image ${image} has no entry in data/image-manifest.json (run npm run images)`);
      else if (dim.width < OG_MIN[0] || dim.height < OG_MIN[1]) {
        fail(p, `og:image is ${dim.width}x${dim.height}, under the ${OG_MIN.join('x')} minimum several platforms require`);
      } else if (dim.width < OG_IDEAL[0] || dim.height < OG_IDEAL[1]) {
        advisories.push(`${p}: og:image is ${dim.width}x${dim.height} (under ${OG_IDEAL.join('x')}, so it renders as a thumbnail rather than a large card)`);
      }
    }
    for (const tag of ['og:title', 'og:description', 'og:url']) {
      if (!prop(html, tag)) fail(p, `no ${tag}`);
    }
  }

  // --- sitemap agreement, both directions ---
  if (indexable && !sitemapPaths.has(p)) fail(p, 'is indexable but missing from sitemap.xml (orphaned from discovery)');
  if (!indexable && sitemapPaths.has(p)) fail(p, 'is noindex but listed in sitemap.xml');

  // --- structured data ---
  const blocks = [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)].map((m) => m[1]);
  if (indexable && !blocks.length) fail(p, 'no JSON-LD block');
  for (const [i, raw] of blocks.entries()) {
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      fail(p, `JSON-LD block ${i + 1} does not parse: ${err.message}`);
      continue;
    }
    if (!parsed['@context']) fail(p, `JSON-LD block ${i + 1} has no @context`);
    const defined = new Set();
    const referenced = new Set();
    (function walk(node) {
      if (Array.isArray(node)) return node.forEach(walk);
      if (!node || typeof node !== 'object') return;
      if (node['@id']) {
        if (Object.keys(node).length === 1) referenced.add(node['@id']);
        else defined.add(node['@id']);
      }
      for (const value of Object.values(node)) walk(value);
    })(parsed);
    // Required properties are checked on the page's primary entities only --
    // the top-level @graph members. Nodes nested inside an ItemList are
    // deliberately thin summaries (headline + url) and must not be held to the
    // same bar as the standalone article node.
    const entities = [].concat(parsed['@graph'] || parsed).filter((n) => n && typeof n === 'object' && n['@type']);
    for (const id of defined) {
      if (!id.startsWith(`${SITE}/`)) fail(p, `JSON-LD @id "${id}" is not an absolute ${SITE} URL`);
    }
    for (const ref of referenced) {
      if (!defined.has(ref)) fail(p, `JSON-LD references @id "${ref}", which nothing in the page defines`);
    }
    // Required properties for the types this site actually emits. A template
    // edit that drops a field is the realistic failure here, not hand-authored
    // bad schema — all 36 pages come from one renderer.
    const REQUIRED = {
      Person: ['name', 'url'],
      WebPage: ['url', 'name'],
      ProfilePage: ['url', 'name'],
      ContactPage: ['url', 'name'],
      CollectionPage: ['url', 'name'],
      BlogPosting: ['headline', 'datePublished', 'author'],
      BreadcrumbList: ['itemListElement'],
      ItemList: ['itemListElement'],
      FAQPage: ['mainEntity']
    };
    for (const node of entities) {
      for (const type of [].concat(node['@type'])) {
        for (const key of REQUIRED[type] || []) {
          if (node[key] === undefined) fail(p, `JSON-LD ${type} node is missing required "${key}"`);
        }
      }
      // The page-level node must agree with the canonical, or the two halves of
      // the page disagree about its identity.
      if ([].concat(node['@type']).some((t) => t.endsWith('Page')) && node.url && canonical && node.url !== canonical) {
        fail(p, `JSON-LD ${node['@type']} url ${node.url} disagrees with canonical ${canonical}`);
      }
    }
  }
}

// --- sitemap -> build ---
for (const url of sitemapUrls) {
  const rel = pathForUrl(url);
  if (!rel) fail('sitemap.xml', `lists ${url}, which is not under ${SITE}/`);
  else if (!(await exists(rel))) fail('sitemap.xml', `lists ${url}, which the build does not emit`);
}
if (new Set(sitemapUrls).size !== sitemapUrls.length) fail('sitemap.xml', 'contains duplicate <loc> entries');

// --- robots.txt: a silent accident here is the most expensive one on the site ---
const robotsTxt = await fs.readFile(path.join(root, 'robots.txt'), 'utf8');
if (/^\s*Disallow:\s*\/\s*$/mi.test(robotsTxt)) fail('robots.txt', 'disallows the whole site');
const declared = robotsTxt.match(/^\s*Sitemap:\s*(\S+)/mi)?.[1];
if (declared !== `${SITE}/sitemap.xml`) fail('robots.txt', `declares sitemap "${declared}", expected ${SITE}/sitemap.xml`);

// --- llms.txt is hand-maintained and nothing else checks it ---
const llms = await fs.readFile(path.join(root, 'llms.txt'), 'utf8');
for (const [, url] of llms.matchAll(new RegExp(`\\]\\((${SITE}[^)]*)\\)`, 'g'))) {
  const rel = pathForUrl(url);
  if (!(await exists(rel))) fail('llms.txt', `links ${url}, which the build does not emit`);
  // Only pages are expected in the sitemap; llms.txt also links robots/sitemap.
  else if (rel.endsWith('.html') && !sitemapPaths.has(rel)) fail('llms.txt', `links ${url}, which is not in sitemap.xml`);
}

if (advisories.length) {
  console.log(`${advisories.length} advisories (not failures) — set SEO_VERBOSE=1 to list them`);
  if (process.env.SEO_VERBOSE) console.log(advisories.map((a) => `- ${a}`).join('\n'));
  console.log('');
}
if (failures.length) {
  console.error(`SEO FAILURES (${failures.length}):\n${failures.join('\n')}`);
  process.exit(1);
}
console.log(`seo check passed: ${pages.length} pages, ${sitemapUrls.length} sitemap URLs, ${titles.size} unique titles`);
