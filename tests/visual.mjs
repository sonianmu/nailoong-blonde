// Visual acceptance (PROMPT §10.1) on the real GPU:  node tests/visual.mjs [label]
// Renders beauty / ID mask / clean / wall-only, projects the 3D landmarks, then tools/qa_visual.py computes
// silhouette IoU, landmark errors, colour deltaE, tile periodicity, layout errors and the triptych.
import { launch, open } from './browser.mjs';
import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

const label = process.argv[2] ?? 'final';
const TMP = 'docs/screenshots/.qa';
mkdirSync(TMP, { recursive: true });
const browser = await launch();
const { page, errors } = await open(browser, { size: 1000, dpr: 2 });
const gpu = await page.evaluate(() => window.__blonde.gpuInfo());
const canvas = page.locator('.photo');          // the photograph rectangle (the canvas covers the whole cover)
const shot = async (mode, file) => {
  await page.evaluate((m) => window.__blonde.setMode(m), mode);
  await page.waitForTimeout(350);
  await canvas.screenshot({ path: `${TMP}/${file}` });
};
await page.mouse.move(5, 5);
await page.waitForTimeout(600);
await shot('beauty', 'beauty.png');
await shot('id', 'id.png');
await shot('clean', 'clean.png');
await shot('wall', 'wall.png');
await page.evaluate(() => window.__blonde.setMode('beauty'));
await page.waitForTimeout(300);
await page.screenshot({ path: `${TMP}/page.png` });
await page.screenshot({ path: 'docs/screenshots/01-rest-cover.png' });
const lm = await page.evaluate(() => {
  const b = window.__blonde, out = {};
  for (const [k, p] of Object.entries(b.meta.landmarks3d)) { const q = b.project(p); out[k] = [q.u, q.v]; }
  return out;
});
writeFileSync(`${TMP}/landmarks.json`, JSON.stringify(lm));

// joint close-ups at 2x DPR (PROMPT §3.4)
const closeups = { 'head-body': [0.80, 0.50, 0.22], armpit: [0.76, 0.62, 0.16], elbow: [0.78, 0.82, 0.18], wrist: [0.55, 0.57, 0.14], 'finger-roots': [0.40, 0.48, 0.12],
  'finger-tips': [0.32, 0.44, 0.10], thumb: [0.53, 0.46, 0.10], 'eye-right': [0.48, 0.40, 0.11], 'eye-left': [0.27, 0.41, 0.10], 'belly-edge': [0.36, 0.68, 0.18] };
const box = await canvas.boundingBox();
for (const [name, [u, v, r]] of Object.entries(closeups)) {
  const cx = box.x + u * box.width, cy = box.y + v * box.height, half = r * box.width;
  const clip = { x: Math.max(box.x, cx - half), y: Math.max(box.y, cy - half), width: 0, height: 0 };
  clip.width = Math.min(box.x + box.width, cx + half) - clip.x; clip.height = Math.min(box.y + box.height, cy + half) - clip.y;
  await page.screenshot({ path: `docs/screenshots/closeup-${name}.png`, clip });
}
const perf = await page.evaluate(() => { const s = window.__blonde.stats(); return { fps: s.fps, cpuFrameMs: s.frameMs, backend: s.backend, solver: s.mode }; });
await browser.close();

const py = existsSync('.venv/bin/python') ? '.venv/bin/python' : 'python3';
const out = execFileSync(py, ['tools/qa_visual.py', TMP, `${TMP}/visual.json`, `docs/screenshots/triptych-${label}.png`], { encoding: 'utf8' });
console.log(out);
const visual = JSON.parse(readFileSync(`${TMP}/visual.json`, 'utf8'));
const model = existsSync('docs/model-report.json') ? JSON.parse(readFileSync('docs/model-report.json', 'utf8')) : null;
const report = existsSync('docs/qa-report.json') ? JSON.parse(readFileSync('docs/qa-report.json', 'utf8')) : {};
report.visual = { label, gpu, perfAtRest: perf, errors, ...visual };
report.model = model;
writeFileSync('docs/qa-report.json', JSON.stringify(report, null, 1));
const swift = /swiftshader|llvmpipe/i.test(JSON.stringify(gpu));
console.log('GPU:', JSON.stringify(gpu), swift ? '  !! SOFTWARE RENDERER' : '');
if (errors.length) console.log('PAGE ERRORS:', errors.slice(0, 5).join('\n'));
const pass = Object.values(visual.pass).every(Boolean) && !errors.length;
console.log(pass ? 'VISUAL PASS' : 'VISUAL FAIL', JSON.stringify(visual.pass));
process.exit(pass ? 0 : 1);
