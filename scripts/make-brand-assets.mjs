// Regenerates the favicon set and the Open Graph card from the site's design
// tokens and the "streamlines and events" motif. Run with `npm run brand`.
//
// The OG card is rasterised with sharp/librsvg, which finds fonts through
// fontconfig. The site's webfonts are woff2 (not readable by fontconfig), so
// this script fetches the matching TTFs once into .cache/fonts/ (gitignored)
// and points fontconfig at them before sharp is loaded.
import fs from 'node:fs/promises';
import path from 'node:path';
import { execSync } from 'node:child_process';

const root = process.cwd();
const cache = path.join(root, '.cache/fonts');
const FONTS = {
  'Fraunces.ttf': 'https://github.com/google/fonts/raw/main/ofl/fraunces/Fraunces%5BSOFT%2CWONK%2Copsz%2Cwght%5D.ttf',
  'SourceSerif4.ttf': 'https://github.com/google/fonts/raw/main/ofl/sourceserif4/SourceSerif4%5Bopsz%2Cwght%5D.ttf',
  'IBMPlexMono-Regular.ttf': 'https://github.com/google/fonts/raw/main/ofl/ibmplexmono/IBMPlexMono-Regular.ttf'
};

await fs.mkdir(cache, { recursive: true });
for (const [file, url] of Object.entries(FONTS)) {
  const target = path.join(cache, file);
  try {
    await fs.access(target);
  } catch {
    console.log(`fetching ${file}`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status}`);
    await fs.writeFile(target, Buffer.from(await res.arrayBuffer()));
  }
}
const fontsConf = path.join(cache, 'fonts.conf');
await fs.writeFile(fontsConf, `<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>${cache}</dir>
  <cachedir>${path.join(cache, 'fc-cache')}</cachedir>
  <config></config>
</fontconfig>
`);
process.env.FONTCONFIG_FILE = fontsConf;
try { execSync(`fc-cache -f "${cache}"`, { stdio: 'ignore', env: { ...process.env, FONTCONFIG_FILE: fontsConf } }); } catch { /* optional */ }

const { default: sharp } = await import('sharp');

// Light tokens (mirror assets/css/styles.css)
const PAPER = '#f3f0e9';
const PAPER_3 = '#fcfbf8';
const INK = '#1b1816';
const INK_SOFT = '#433d38';
const FAINT = '#69615a';
const LINE_STRONG = '#8b8272';
const ACCENT = '#1f7a72';
const ACCENT_INK = '#1a5f59';
const CNEL = '#8a3b2c';
// Dark tokens
const D_PAPER = '#171412';
const D_ACCENT = '#6fbfb2';
const D_CNEL = '#e4a58c';

// Same seeded generator as build.mjs so brand assets share the motif.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function motif({ width, height, lines, dots, seed, lineColor, dotColor, lineOpacity, dotOpacity }) {
  const rand = mulberry32(seed);
  const paths = [];
  for (let i = 0; i < lines; i += 1) {
    const baseY = height * ((i + 0.6) / (lines + 0.2));
    const amp = height * (0.05 + rand() * 0.08);
    const freq = 1.2 + rand() * 1.6;
    const phase = rand() * Math.PI * 2;
    const drift = (rand() - 0.5) * height * 0.18;
    const pts = [];
    for (let s = 0; s <= 24; s += 1) {
      const t = s / 24;
      const x = t * width;
      const y = baseY + drift * t + amp * Math.sin(t * Math.PI * freq + phase) * (0.35 + 0.65 * Math.sin(t * Math.PI));
      pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    paths.push(`<polyline points="${pts.join(' ')}" fill="none" stroke="${lineColor}" stroke-width="1.6" opacity="${lineOpacity}"/>`);
  }
  const circles = [];
  for (let i = 0; i < dots; i += 1) {
    const x = width * (0.06 + rand() * 0.88);
    const y = height * (0.1 + rand() * 0.8);
    const r = 2.2 + rand() * 2.2;
    circles.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${dotColor}" opacity="${dotOpacity}"/>`);
  }
  return paths.join('') + circles.join('');
}

// Favicon: one streamline crossing the square, one event dot. Outlined paths
// only (no live text), with an internal dark-scheme block for browsers that
// honour it in SVG favicons.
const faviconSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <style>
    .bg{fill:${PAPER}} .edge{stroke:${LINE_STRONG}} .flow{stroke:${ACCENT}} .flow2{stroke:${ACCENT}} .dot{fill:${CNEL}}
    @media (prefers-color-scheme: dark){ .bg{fill:${D_PAPER}} .edge{stroke:#776e63} .flow{stroke:${D_ACCENT}} .flow2{stroke:${D_ACCENT}} .dot{fill:${D_CNEL}} }
  </style>
  <rect class="bg" width="64" height="64"/>
  <rect class="edge" x="2.5" y="2.5" width="59" height="59" fill="none" stroke-width="1.5"/>
  <path class="flow" d="M6 40 C 18 26, 26 50, 38 34 S 54 20, 60 28" fill="none" stroke-width="3.2" stroke-linecap="round"/>
  <path class="flow2" d="M6 50 C 20 40, 30 58, 42 46 S 54 36, 60 40" fill="none" stroke-width="2" stroke-linecap="round" opacity="0.55"/>
  <circle class="dot" cx="43" cy="19" r="4.6"/>
</svg>
`;

await fs.writeFile(path.join(root, 'assets/img/favicon.svg'), faviconSvg);
const faviconLight = faviconSvg.replace(/@media[^}]*\}\s*\}/s, '');
await sharp(Buffer.from(faviconLight)).resize(48, 48).png().toFile(path.join(root, 'assets/img/favicon-48.png'));
await sharp(Buffer.from(faviconLight)).resize(180, 180).png().toFile(path.join(root, 'assets/img/apple-touch-icon.png'));

// OG card: paper ground, motif behind the right half, name and statement in
// Fraunces, a mono credit line, and the matted portrait.
const ogSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630">
  <rect width="1200" height="630" fill="${PAPER}"/>
  <g>${motif({ width: 1200, height: 630, lines: 9, dots: 10, seed: 20260907, lineColor: ACCENT, dotColor: CNEL, lineOpacity: 0.28, dotOpacity: 0.6 })}</g>
  <rect x="0" y="0" width="760" height="630" fill="${PAPER}" opacity="0.82"/>
  <text x="84" y="128" font-family="IBM Plex Mono" font-size="22" letter-spacing="3.5" fill="${ACCENT_INK}">RAUL VALLE · UNIVERSITY OF FLORIDA</text>
  <text x="84" y="236" font-family="Fraunces" font-size="78" font-weight="500" fill="${INK}" letter-spacing="-1">Machine learning</text>
  <text x="84" y="322" font-family="Fraunces" font-size="78" font-weight="500" fill="${INK}" letter-spacing="-1">for <tspan font-style="italic" fill="${ACCENT_INK}">physical</tspan> and</text>
  <text x="84" y="408" font-family="Fraunces" font-size="78" font-weight="500" fill="${INK}" letter-spacing="-1"><tspan font-style="italic" fill="${ACCENT_INK}">neural</tspan> systems.</text>
  <text x="84" y="486" font-family="Source Serif 4" font-size="27" fill="${INK_SOFT}">Ph.D. student in Electrical and Computer Engineering.</text>
  <text x="84" y="524" font-family="Source Serif 4" font-size="27" fill="${INK_SOFT}">CFD surrogates, neural imaging, research software.</text>
  <text x="84" y="572" font-family="IBM Plex Mono" font-size="20" letter-spacing="1.5" fill="${FAINT}">www.raulv.dev</text>
</svg>
`;

const portrait = await sharp(path.join(root, 'assets/img/me/Raul_me.webp'))
  .resize(324, 405, { fit: 'cover', position: 'attention' })
  .toBuffer();
const mat = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="348" height="429">
  <rect width="348" height="429" fill="${PAPER_3}" stroke="${LINE_STRONG}" stroke-width="2"/>
</svg>`);
await sharp(Buffer.from(ogSvg))
  .composite([
    { input: mat, left: 792, top: 100 },
    { input: portrait, left: 804, top: 112 }
  ])
  .png({ compressionLevel: 9 })
  .toFile(path.join(root, 'assets/img/og-card.png'));

console.log('brand assets written: favicon.svg, favicon-48.png, apple-touch-icon.png, og-card.png');
