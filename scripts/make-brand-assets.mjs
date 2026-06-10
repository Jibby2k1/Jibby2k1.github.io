import fs from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';

const root = process.cwd();
const PAPER = '#efe7d9';
const PAPER_LIGHT = '#fbf7ef';
const INK = '#1f1a16';
const LINE = '#9d8f76';
const ACCENT = '#5f6a51';

const faviconSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" fill="${PAPER}"/>
  <rect x="3" y="3" width="58" height="58" fill="none" stroke="${LINE}" stroke-width="2"/>
  <text x="32" y="44" text-anchor="middle" font-family="Iowan Old Style, Palatino Linotype, Georgia, serif" font-size="34" font-weight="700" fill="${INK}">RV</text>
</svg>
`;

const ogSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630">
  <rect width="1200" height="630" fill="${PAPER_LIGHT}"/>
  <rect width="1200" height="630" fill="${PAPER}" opacity="0.55"/>
  <rect x="28" y="28" width="1144" height="574" fill="none" stroke="${LINE}" stroke-width="2"/>
  <text x="84" y="150" font-family="Consolas, Liberation Mono, monospace" font-size="26" letter-spacing="6" fill="${ACCENT}">WWW.RAULV.DEV</text>
  <text x="84" y="280" font-family="Palatino Linotype, Book Antiqua, Georgia, serif" font-size="92" font-weight="700" fill="${INK}">Raul Valle</text>
  <text x="84" y="362" font-family="Palatino Linotype, Book Antiqua, Georgia, serif" font-size="40" fill="${INK}" opacity="0.85">Machine Learning and</text>
  <text x="84" y="414" font-family="Palatino Linotype, Book Antiqua, Georgia, serif" font-size="40" fill="${INK}" opacity="0.85">Neuroengineering.</text>
  <text x="84" y="492" font-family="Consolas, Liberation Mono, monospace" font-size="22" letter-spacing="2" fill="${INK}" opacity="0.7">Ph.D. student · UF ECE</text>
  <text x="84" y="530" font-family="Consolas, Liberation Mono, monospace" font-size="22" letter-spacing="2" fill="${INK}" opacity="0.7">Gainesville, Florida</text>
  <rect x="84" y="556" width="220" height="6" fill="${ACCENT}"/>
</svg>
`;

await fs.writeFile(path.join(root, 'assets/img/favicon.svg'), faviconSvg);
await sharp(Buffer.from(faviconSvg)).resize(48, 48).png().toFile(path.join(root, 'assets/img/favicon-48.png'));
await sharp(Buffer.from(faviconSvg)).resize(180, 180).png().toFile(path.join(root, 'assets/img/apple-touch-icon.png'));

const portrait = await sharp(path.join(root, 'assets/img/me/Raul_me.jpeg'))
  .resize(340, 446, { fit: 'cover', position: 'attention' })
  .toBuffer();
const frame = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="356" height="462">
  <rect width="356" height="462" fill="none" stroke="${LINE}" stroke-width="4"/>
</svg>`);
await sharp(Buffer.from(ogSvg))
  .composite([
    { input: portrait, left: 776, top: 92 },
    { input: frame, left: 768, top: 84 }
  ])
  .png({ compressionLevel: 9 })
  .toFile(path.join(root, 'assets/img/og-card.png'));

console.log('brand assets written: favicon.svg, favicon-48.png, apple-touch-icon.png, og-card.png');
