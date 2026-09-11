// Runs axe-core against every generated page and fails on serious/critical violations.
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { startServer } from './lib/server.mjs';

const root = process.cwd();

// Recursive, matching check-links.mjs. A top-level-only readdir scanned 14 pages
// and skipped the 26 under projects/ and notes/, so anything rendered only on a
// project or note page was never audited at all.
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
const pages = (await htmlPages()).sort();
const { server, port } = await startServer(root);
const base = `http://localhost:${port}`;
const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
const failures = [];

for (const p of pages) {
  await page.goto(`${base}/${p}`, { waitUntil: 'networkidle' });
  const results = await new AxeBuilder({ page }).analyze();
  const bad = results.violations.filter((v) => ['serious', 'critical'].includes(v.impact));
  for (const v of bad) {
    failures.push(`${p}: [${v.impact}] ${v.id} — ${v.help} (${v.nodes.length} nodes, e.g. ${v.nodes[0]?.target?.join(' ')})`);
  }
}

await browser.close();
server.close();

if (failures.length) {
  console.error(`A11Y VIOLATIONS (${failures.length}):\n${failures.join('\n')}`);
  process.exit(1);
}
console.log(`a11y check passed: ${pages.length} pages, no serious/critical violations`);
