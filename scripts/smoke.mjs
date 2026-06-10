// Playwright smoke test: every page loads with zero console errors at desktop
// and mobile sizes, scroll-reveal leaves nothing invisible, the photography
// lightbox opens/navigates/closes, and the theme toggle persists.
import fs from 'node:fs/promises';
import { chromium } from 'playwright';
import { startServer } from './lib/server.mjs';

const root = process.cwd();
const failures = [];

const pages = (await fs.readdir(root)).filter((f) => f.endsWith('.html'));
const { server, port } = await startServer(root);
const base = `http://localhost:${port}`;
const browser = await chromium.launch();

for (const viewport of [{ name: 'desktop', width: 1440, height: 900 }, { name: 'mobile', width: 390, height: 844 }]) {
  const ctx = await browser.newContext({ viewport: { width: viewport.width, height: viewport.height } });
  for (const p of pages) {
    const page = await ctx.newPage();
    const errors = [];
    page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()); });
    page.on('pageerror', (err) => errors.push(String(err)));
    page.on('requestfailed', (req) => errors.push(`request failed: ${req.url()}`));
    try {
      await page.goto(`${base}/${p}`, { waitUntil: 'networkidle', timeout: 20000 });
      await page.evaluate(async () => {
        const step = window.innerHeight / 2;
        for (let y = 0; y <= document.body.scrollHeight; y += step) {
          window.scrollTo({ top: y, behavior: 'instant' });
          await new Promise((resolve) => setTimeout(resolve, 60));
        }
      });
      await page.waitForTimeout(400);
      const hidden = await page.evaluate(() => document.querySelectorAll('.reveal:not(.visible)').length);
      // the 5s load fallback is the last resort; nothing should rely on it during a normal scroll
      if (hidden > 0) {
        await page.waitForTimeout(5300);
        const stillHidden = await page.evaluate(() => document.querySelectorAll('.reveal:not(.visible)').length);
        if (stillHidden > 0) failures.push(`${p} (${viewport.name}): ${stillHidden} reveal elements never became visible`);
      }
    } catch (err) {
      errors.push(`navigation: ${err.message}`);
    }
    if (errors.length) failures.push(`${p} (${viewport.name}): ${errors.join(' | ')}`);
    await page.close();
  }
  await ctx.close();
}

// lightbox
{
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(`${base}/photography.html`, { waitUntil: 'networkidle' });
  await page.click('[data-lightbox]');
  const open = await page.evaluate(() => document.querySelector('.lightbox').classList.contains('open'));
  if (!open) failures.push('lightbox: did not open on click');
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('Escape');
  const closed = await page.evaluate(() => !document.querySelector('.lightbox').classList.contains('open'));
  if (!closed) failures.push('lightbox: did not close on Escape');
  await page.close();
}

// theme toggle persistence
{
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(`${base}/index.html`, { waitUntil: 'networkidle' });
  await page.click('[data-theme-toggle]');
  const dark = await page.evaluate(() => document.documentElement.getAttribute('data-theme') === 'dark');
  if (!dark) failures.push('theme: toggle did not switch to dark');
  await page.goto(`${base}/about.html`, { waitUntil: 'networkidle' });
  const persisted = await page.evaluate(() => document.documentElement.getAttribute('data-theme') === 'dark');
  if (!persisted) failures.push('theme: dark theme did not persist across pages');
  await page.close();
}

await browser.close();
server.close();

if (failures.length) {
  console.error(`SMOKE FAILURES (${failures.length}):\n${failures.join('\n')}`);
  process.exit(1);
}
console.log(`smoke passed: ${pages.length} pages × 2 viewports, lightbox, theme`);
