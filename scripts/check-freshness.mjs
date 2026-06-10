// Content freshness + external link rot, for the monthly scheduled workflow.
// Exits 1 (with a report on stdout) if the newest writing entry is older than
// 120 days or any external link no longer resolves.
import fs from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd();
const MAX_AGE_DAYS = 120;
const problems = [];

// freshness of writing
const dir = path.join(root, 'content/writing');
let newest = null;
for (const file of await fs.readdir(dir)) {
  if (!file.endsWith('.md')) continue;
  const text = await fs.readFile(path.join(dir, file), 'utf8');
  const date = text.match(/^date:\s*(\d{4}-\d{2}-\d{2})/m)?.[1];
  if (date && (!newest || date > newest)) newest = date;
}
if (newest) {
  const ageDays = Math.floor((Date.now() - new Date(`${newest}T00:00:00Z`)) / 86400000);
  if (ageDays > MAX_AGE_DAYS) {
    problems.push(`Newest writing entry is ${ageDays} days old (${newest}). Consider publishing a new note or reframing the journal copy.`);
  }
}

// external link rot across generated pages
const externalLinks = new Set();
const htmlFiles = (await fs.readdir(root)).filter((f) => f.endsWith('.html'));
for (const sub of ['projects', 'notes']) {
  htmlFiles.push(...(await fs.readdir(path.join(root, sub))).map((f) => `${sub}/${f}`));
}
for (const file of htmlFiles.filter((f) => f.endsWith('.html'))) {
  const html = await fs.readFile(path.join(root, file), 'utf8');
  for (const match of html.matchAll(/href="(https?:\/\/[^"]+)"/g)) {
    externalLinks.add(match[1].replaceAll('&amp;', '&'));
  }
}

for (const url of [...externalLinks].sort()) {
  if (url.startsWith('https://www.raulv.dev')) continue;
  let ok = false;
  for (const method of ['HEAD', 'GET']) {
    try {
      const res = await fetch(url, {
        method,
        redirect: 'follow',
        signal: AbortSignal.timeout(15000),
        headers: { 'user-agent': 'Mozilla/5.0 (compatible; raulv.dev link checker)' }
      });
      // 403/405/429 usually mean a bot wall, not a dead link
      if (res.status < 400 || [403, 405, 429].includes(res.status)) { ok = true; break; }
    } catch { /* try next method */ }
  }
  if (!ok) problems.push(`External link may be dead: ${url}`);
}

if (problems.length) {
  console.log(`FRESHNESS REPORT (${problems.length} items):\n${problems.map((p) => `- ${p}`).join('\n')}`);
  process.exit(1);
}
console.log(`freshness check passed: writing current as of ${newest}, ${externalLinks.size} external links alive`);
