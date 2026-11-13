// quick look:  node tests/shot.mjs [out.png] [query] [dpr]
import { launch, open } from './browser.mjs';
const out = process.argv[2] ?? 'docs/screenshots/_shot.png';
const b = await launch();
const { page, errors } = await open(b, { query: process.argv[3] ?? '', dpr: +(process.argv[4] ?? 1) });
await page.waitForTimeout(1200);
console.log(JSON.stringify(await page.evaluate(() => window.__blonde.gpuInfo())));
console.log(JSON.stringify(await page.evaluate(() => { const s = window.__blonde.stats(); return { fps: s.fps, frameMs: s.frameMs, step: s.avgStepMs, mode: s.mode, backend: s.backend }; })));
await page.screenshot({ path: out });
if (errors.length) console.log('ERRORS:\n' + errors.slice(0, 12).join('\n'));
await b.close();
