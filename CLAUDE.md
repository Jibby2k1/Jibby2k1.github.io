# raulv.dev — personal site

Static site for www.raulv.dev, served by GitHub Pages from the repo root (`main` = production). All HTML is **generated** — never edit `*.html` files directly.

## How pages are produced

- `scripts/build.mjs` renders every page from `data/*.json` + `content/writing/*.md` + the templates in `scripts/pages.mjs` (about, cv, contact). It also generates `sitemap.xml` and `rss.xml`.
- To change a page: edit the template in `scripts/build.mjs`/`scripts/pages.mjs` or the data/content files, then run `npm run build` and commit the regenerated output **with** the source change. CI fails if committed output doesn't match a fresh build.
- `sitemap.xml` lastmod is deterministic (last git commit date, or `SITE_LASTMOD` env).

## Images

- Run `npm run images` (`scripts/optimize-images.mjs`) after adding any image: it converts photography to 1600w + 640w WebP under `assets/img/photography/web/`, recompresses anything over 300 KB, and regenerates `data/image-manifest.json` (used by the build to emit `width`/`height` attributes — required, missing entries warn at build time).
- Budgets (CI-enforced): no asset over 500 KB; photography total under 8 MB.
- `npm run brand` regenerates favicon + OG card (`scripts/make-brand-assets.mjs`).

## Commands

- `npm run build` — regenerate the site
- `npm run dev` — build + serve on :8080
- `npm run check` — budgets, internal links, Playwright smoke (console errors / scroll-reveal / lightbox / theme), axe a11y scan; same as CI
- `npm run check:freshness` — stale-writing + external link rot (monthly workflow opens an issue from this)

## Conventions

- Light/dark themes via CSS custom properties in `:root` / `[data-theme="dark"]` at the top of `assets/css/styles.css`. Use the existing tokens (`--surface`, `--raised`, `--menu-bg`, …) — no hardcoded colors in rules.
- `.reveal` scroll animations are scoped to `.js` (no-JS users see everything) and have a 5s post-load fallback; don't remove either safety net.
- `publications.html` and `project.html` are intentional noindex "moved" stubs.
- The `ieee-sps-uf-site/` subsite and `CNAME` are managed separately — leave them alone.
