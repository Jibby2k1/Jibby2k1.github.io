// Asset weight budgets: no single asset file over 500 KB, photography under 8 MB total.
import fs from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd();
const SINGLE_LIMIT = 500 * 1024;
const PHOTOGRAPHY_LIMIT = 8 * 1024 * 1024;

async function walk(dir) {
  const out = [];
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...await walk(full));
    else out.push(full);
  }
  return out;
}

const failures = [];
let photographyTotal = 0;

for (const file of await walk(path.join(root, 'assets'))) {
  const { size } = await fs.stat(file);
  const rel = path.relative(root, file);
  if (size > SINGLE_LIMIT) failures.push(`${rel}: ${(size / 1024).toFixed(0)} KB exceeds ${SINGLE_LIMIT / 1024} KB limit`);
  if (rel.startsWith('assets/img/photography/')) photographyTotal += size;
}

if (photographyTotal > PHOTOGRAPHY_LIMIT) {
  failures.push(`photography total ${(photographyTotal / 1024 / 1024).toFixed(1)} MB exceeds ${PHOTOGRAPHY_LIMIT / 1024 / 1024} MB limit`);
}

if (failures.length) {
  console.error(`BUDGET FAILURES (${failures.length}):\n${failures.join('\n')}`);
  process.exit(1);
}
console.log(`budgets passed: photography total ${(photographyTotal / 1024 / 1024).toFixed(1)} MB`);
