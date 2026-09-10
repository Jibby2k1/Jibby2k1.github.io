// Playwright smoke test: every page loads with zero console errors at desktop
// and mobile sizes, nothing overflows the phone viewport sideways,
// scroll-reveal leaves nothing invisible, the photography lightbox
// opens/navigates/closes at both sizes, and the theme toggle persists.
import fs from 'node:fs/promises';
import { chromium } from 'playwright';
import { startServer } from './lib/server.mjs';

const root = process.cwd();
const failures = [];

const pages = (await fs.readdir(root)).filter((f) => f.endsWith('.html'));
const { server, port } = await startServer(root);
const base = `http://localhost:${port}`;
const browser = await chromium.launch();

// hasTouch matters: without it the mobile context still reports a fine
// pointer and real hover, so the (pointer: coarse) tap-target rules and the
// (hover: hover) guard never evaluate the way they do on an actual phone.
// 880 is not decoration: it is the two-column band where the diagram
// container queries actually fire, and nothing else in this loop covers it.
for (const viewport of [{ name: 'desktop', width: 1440, height: 900, touch: false }, { name: 'tablet', width: 880, height: 1000, touch: true }, { name: 'mobile', width: 390, height: 844, touch: true }]) {
  const ctx = await browser.newContext({ viewport: { width: viewport.width, height: viewport.height }, hasTouch: viewport.touch, isMobile: viewport.touch });
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
      // body{overflow-x:hidden} silently clips sideways overflow, so it looks
      // fine in a screenshot but strands content off-screen on a real phone.
      // Deliberate horizontal scrollers (phone diagram wells) are excluded.
      if (viewport.touch) {
        const overflow = await page.evaluate(() => {
          const doc = document.documentElement;
          if (doc.scrollWidth > window.innerWidth) return `document scrollWidth ${doc.scrollWidth} > viewport ${window.innerWidth}`;
          const scroller = (el) => {
            const o = getComputedStyle(el).overflowX;
            return o === 'auto' || o === 'scroll' || o === 'hidden';
          };
          for (const el of document.querySelectorAll('main *')) {
            const r = el.getBoundingClientRect();
            if (r.width === 0 && r.height === 0) continue;
            if (r.right > window.innerWidth + 1 || r.left < -1) {
              let p = el.parentElement, contained = false;
              while (p) { if (scroller(p)) { contained = true; break; } p = p.parentElement; }
              if (!contained) return `${el.tagName.toLowerCase()}.${el.className.toString().split(' ')[0]} spills to ${Math.round(r.right)}px`;
            }
          }
          return null;
        });
        if (overflow) failures.push(`${p} (${viewport.name}): horizontal overflow — ${overflow}`);
      }
      // Interactive controls must be finger-sized on a coarse pointer.
      if (viewport.touch) {
        const small = await page.evaluate(() => {
          if (!window.matchMedia('(pointer: coarse)').matches) return ['context is not reporting a coarse pointer'];
          const sel = '.btn, .filter-btn, .icon-link, .profile-icon, .burger, .navlinks a, .lightbox-btn';
          return [...document.querySelectorAll(sel)]
            .filter((el) => el.offsetParent !== null || getComputedStyle(el).position === 'fixed')
            .map((el) => ({ el, r: el.getBoundingClientRect() }))
            .filter(({ r }) => r.width > 0 && r.height > 0 && r.height < 40)
            .map(({ el, r }) => `${el.className.toString().split(' ')[0] || el.tagName.toLowerCase()} ${Math.round(r.width)}x${Math.round(r.height)}`)
            .slice(0, 4);
        });
        if (small.length) failures.push(`${p} (${viewport.name}): tap targets under 40px — ${small.join(', ')}`);
      }
      // Any well that scrolls horizontally must be keyboard reachable, or its
      // right-hand side is unreachable without a mouse or a finger.
      const unreachable = await page.evaluate(() => {
        return [...document.querySelectorAll('.media, .hero-media, .figure-svg')]
          .filter((el) => el.scrollWidth > el.clientWidth + 1 && el.getAttribute('tabindex') !== '0')
          .map((el) => el.className.toString().split(' ')[0])
          .slice(0, 3);
      });
      if (unreachable.length) failures.push(`${p} (${viewport.name}): scrollable diagram wells not keyboard reachable — ${unreachable.join(', ')}`);
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

// lightbox — at both sizes, since its sizing and controls are viewport-specific
for (const vp of [{ name: 'desktop', width: 1440, height: 900, touch: false }, { name: 'mobile', width: 390, height: 844, touch: true }]) {
  const page = await browser.newPage({ viewport: { width: vp.width, height: vp.height }, hasTouch: vp.touch, isMobile: vp.touch });
  await page.goto(`${base}/photography.html`, { waitUntil: 'networkidle' });
  await page.click('[data-lightbox]');
  const open = await page.evaluate(() => document.querySelector('.lightbox').classList.contains('open'));
  if (!open) failures.push(`lightbox (${vp.name}): did not open on click`);
  // the image and its controls must actually fit the viewport it opened in
  const fits = await page.evaluate(() => {
    const img = document.querySelector('.lightbox-img');
    const r = img.getBoundingClientRect();
    const bad = [...document.querySelectorAll('.lightbox-btn')]
      .filter((b) => { const q = b.getBoundingClientRect(); return q.right > window.innerWidth + 1 || q.bottom > window.innerHeight + 1 || q.left < -1; })
      .map((b) => b.getAttribute('aria-label'));
    return { w: Math.round(r.width), h: Math.round(r.height), vw: window.innerWidth, vh: window.innerHeight, bad };
  });
  if (fits.w > fits.vw + 1 || fits.h > fits.vh + 1) failures.push(`lightbox (${vp.name}): image ${fits.w}x${fits.h} exceeds viewport ${fits.vw}x${fits.vh}`);
  if (fits.bad.length) failures.push(`lightbox (${vp.name}): controls off-screen — ${fits.bad.join(', ')}`);
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('Escape');
  const closed = await page.evaluate(() => !document.querySelector('.lightbox').classList.contains('open'));
  if (!closed) failures.push(`lightbox (${vp.name}): did not close on Escape`);
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
console.log(`smoke passed: ${pages.length} pages × 3 viewports (tablet/mobile with touch), overflow, tap targets, keyboard-reachable scrollers, lightbox × 2 viewports, theme`);
