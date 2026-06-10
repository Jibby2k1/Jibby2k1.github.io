import fs from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';

const root = process.cwd();
const SIZE_LIMIT = 150 * 1024;
const FULL_WIDTH = 1600;
const THUMB_WIDTH = 640;

const replacements = new Map();

async function walk(dir) {
  const out = [];
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...await walk(full));
    else out.push(full);
  }
  return out;
}

function rel(file) {
  return path.relative(root, file);
}

async function toWebp(srcFile, destFile, width) {
  const image = sharp(srcFile).rotate();
  const meta = await image.metadata();
  const resize = Math.max(meta.width, meta.height) > width
    ? (meta.width >= meta.height ? { width } : { height: width })
    : null;
  await fs.mkdir(path.dirname(destFile), { recursive: true });
  let pipeline = sharp(srcFile).rotate();
  if (resize) pipeline = pipeline.resize(resize);
  await pipeline.webp({ quality: 82 }).toFile(destFile);
}

async function optimizePhotography() {
  const dir = path.join(root, 'assets/img/photography');
  const files = (await fs.readdir(dir)).filter((f) => /\.jpe?g$/i.test(f));
  for (const file of files) {
    const base = path.parse(file).name;
    const src = path.join(dir, file);
    const full = path.join(dir, 'web', `${base}-1600.webp`);
    const thumb = path.join(dir, 'web', `${base}-640.webp`);
    await toWebp(src, full, FULL_WIDTH);
    await toWebp(src, thumb, THUMB_WIDTH);
    replacements.set(rel(src), rel(full));
    await fs.unlink(src);
    console.log(`photography: ${file} -> web/${base}-{1600,640}.webp`);
  }
}

async function optimizeLargeRasters() {
  const files = await walk(path.join(root, 'assets/img'));
  for (const file of files) {
    if (!/\.(png|jpe?g)$/i.test(file)) continue;
    if (file.includes(`${path.sep}photography${path.sep}`)) continue;
    // og-card/favicons stay PNG for scraper compatibility
    if (/og-card\.png$|favicon|apple-touch/i.test(file)) continue;
    const stat = await fs.stat(file);
    if (stat.size <= SIZE_LIMIT) continue;
    const dest = file.replace(/\.(png|jpe?g)$/i, '.webp');
    await toWebp(file, dest, FULL_WIDTH);
    replacements.set(rel(file), rel(dest));
    await fs.unlink(file);
    const after = (await fs.stat(dest)).size;
    console.log(`recompressed: ${rel(file)} (${(stat.size / 1024 / 1024).toFixed(1)}MB) -> ${rel(dest)} (${(after / 1024).toFixed(0)}KB)`);
  }
}

async function rewriteReferences() {
  const targets = [
    ...(await fs.readdir(path.join(root, 'data'))).map((f) => `data/${f}`),
    ...(await fs.readdir(path.join(root, 'content/writing'))).map((f) => `content/writing/${f}`),
    'scripts/build.mjs',
    'scripts/pages.mjs'
  ];
  for (const target of targets) {
    const file = path.join(root, target);
    let text = await fs.readFile(file, 'utf8');
    let changed = false;
    for (const [from, to] of replacements) {
      if (text.includes(from)) {
        text = text.replaceAll(from, to);
        changed = true;
      }
    }
    if (changed) {
      await fs.writeFile(file, text);
      console.log(`rewrote references in ${target}`);
    }
  }
}

async function writeManifest() {
  const manifest = {};
  for (const file of await walk(path.join(root, 'assets/img'))) {
    if (!/\.(png|jpe?g|webp|gif|avif)$/i.test(file)) continue;
    const meta = await sharp(file).metadata();
    manifest[rel(file)] = { width: meta.width, height: meta.height };
  }
  const sorted = Object.fromEntries(Object.entries(manifest).sort(([a], [b]) => a.localeCompare(b)));
  await fs.writeFile(path.join(root, 'data/image-manifest.json'), `${JSON.stringify(sorted, null, 2)}\n`);
  console.log(`manifest: ${Object.keys(sorted).length} images`);
}

async function makeHeroPortrait() {
  // dedicated small portrait for the hero/about pages (displayed ~340px wide)
  const src = (await fs.readdir(path.join(root, 'assets/img/me'))).find((f) => /^Raul_me\.(jpe?g|webp)$/i.test(f));
  if (!src) return;
  await toWebp(path.join(root, 'assets/img/me', src), path.join(root, 'assets/img/me/portrait-680.webp'), 680);
  console.log('portrait: assets/img/me/portrait-680.webp');
}

await optimizePhotography();
await optimizeLargeRasters();
await makeHeroPortrait();
await rewriteReferences();
await writeManifest();
