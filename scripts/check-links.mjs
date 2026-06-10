// Crawls every generated page and verifies all internal links and images resolve.
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { startServer } from './lib/server.mjs';

const root = process.cwd();

async function htmlPages() {
  const pages = (await fs.readdir(root)).filter((f) => f.endsWith('.html'));
  for (const dir of ['projects', 'notes']) {
    pages.push(...(await fs.readdir(path.join(root, dir))).map((f) => `${dir}/${f}`));
  }
  return pages.filter((f) => f.endsWith('.html'));
}

const { server, port } = await startServer(root);
const base = `http://localhost:${port}`;
const browser = await chromium.launch();
const page = await browser.newPage();
const broken = [];
const checked = new Map();

for (const p of await htmlPages()) {
  await page.goto(`${base}/${p}`, { waitUntil: 'domcontentloaded' });
  const refs = await page.evaluate(() =>
    [...document.querySelectorAll('a[href], img[src]')].map((el) => ({
      url: el.href || el.currentSrc || el.src,
      text: (el.textContent || el.alt || '').trim().slice(0, 60)
    }))
  );
  for (const ref of refs) {
    if (!ref.url.startsWith(base)) continue;
    const clean = ref.url.split('#')[0];
    if (!checked.has(clean)) {
      const res = await fetch(clean, { method: 'HEAD' }).catch(() => null);
      checked.set(clean, res ? res.status : 0);
    }
    const status = checked.get(clean);
    if (status >= 400 || status === 0) {
      broken.push(`${p}: "${ref.text}" -> ${clean.replace(base, '')} (${status})`);
    }
  }
}

await browser.close();
server.close();

const unique = [...new Set(broken)];
if (unique.length) {
  console.error(`BROKEN INTERNAL LINKS (${unique.length}):\n${unique.join('\n')}`);
  process.exit(1);
}
console.log(`link check passed: ${checked.size} unique internal URLs`);
