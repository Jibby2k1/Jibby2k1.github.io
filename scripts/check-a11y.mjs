// Runs axe-core against every top-level page and fails on serious/critical violations.
import fs from 'node:fs/promises';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { startServer } from './lib/server.mjs';

const root = process.cwd();
const pages = (await fs.readdir(root)).filter((f) => f.endsWith('.html'));
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
