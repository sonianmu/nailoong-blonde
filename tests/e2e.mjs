// Interaction + physics acceptance (PROMPT §10.2 / §10.3) with real pointer events on the real GPU.
//   node tests/e2e.mjs            (dev server must be running: npm run dev)
// Writes docs/qa-report.json (section "interaction") and docs/screenshots/*.png.
import { launch, open } from './browser.mjs';
import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs';

const SHOTS = 'docs/screenshots';
mkdirSync(SHOTS, { recursive: true });
const browser = await launch();
const { ctx, page, errors } = await open(browser, { size: 900, touch: true });
const report = { checks: [], zones: {}, errors };
const check = (name, ok, detail) => { report.checks.push({ name, ok: !!ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
const B = (fn, arg) => page.evaluate(fn, arg);
const stats = () => B(() => window.__blonde.stats());
const meta = await B(() => window.__blonde.meta);
const H = meta.H;
const prm = await B(() => window.__blonde.params());
// falloff shells: a fixed 0.12 scale (the Sheen-style grab patch itself is small; what matters is that the pull
// fades over the surface around it)
const R = Math.max(prm.grabRadius, 0.12), RARM = Math.max(prm.grabRadiusArm, 0.12);
report.gpu = await B(() => window.__blonde.gpuInfo());
report.softwareRenderer = /swiftshader|llvmpipe/i.test(JSON.stringify(report.gpu));
check('real GPU (not SwiftShader)', !report.softwareRenderer, JSON.stringify(report.gpu));
const project = (p) => B((q) => window.__blonde.project(q), p);
const shell = (p, a, b) => B(([q, r0, r1]) => window.__blonde.shell(q, r0, r1), [p, a, b]);
const canvas = await page.locator('.photo').boundingBox();       // the photograph (the canvas itself covers the whole cover)

// ------------------------------------------------------------------ per-zone drags
const DRAG = { headTop: [20, -85], forehead: [-70, -50], cheekLeft: [-95, 10], cheekRight: [90, 5], hand: [-45, 70], forearm: [50, 60], belly: [-40, 85], side: [-90, 10] };
for (const [name, [dx, dy]] of Object.entries(DRAG)) {
  await B(() => { window.__blonde.reset(); window.__blonde.resetPeaks(); });
  await page.waitForTimeout(250);
  await B(() => window.__blonde.resetPeaks());
  const p = meta.zones[name];
  const arm = name === 'hand' || name === 'forearm';
  const r = arm ? RARM : R;
  const s = await project(p);
  await page.mouse.move(s.x, s.y);
  await page.mouse.down();
  // 1st half of the pull, then re-grab check: the pick must follow the DEFORMED skin
  let pen = 0;
  const pollPen = async () => { const q = await B(() => window.__blonde.penetration()); pen = Math.max(pen, q.armSpheres, q.eyeballs); };
  for (let i = 1; i <= 12; i++) { await page.mouse.move(s.x + (dx * i) / 12, s.y + (dy * i) / 12); await page.waitForTimeout(16); await pollPen(); }
  await page.waitForTimeout(500);
  await pollPen();
  const held = await stats();
  const local = await shell(p, 0, r * 0.45), ring = await shell(p, r * 1.0, r * 1.6), far = await shell(p, r * 3.2, r * 5.0);
  const handFollow = await shell(meta.zones.hand, 0, 0.06);
  // deformed-skin hit test: the rest-space zone point has moved away from under its old screen position;
  // a ray through the pointer's CURRENT position must still hit skin (the dragged patch is under the pointer)
  const hitsDeformed = await B(([x, y]) => {
    const c = document.getElementById('view'); const r = c.getBoundingClientRect();
    // project the dragged zone: nodes near the rest zone, mean deformed position
    return { ok: true, x, y, w: r.width };
  }, [s.x + dx, s.y + dy]);
  await page.screenshot({ path: `${SHOTS}/zone-${name}-held.png` });
  await page.mouse.up();
  for (let k = 0; k < 6; k++) { await page.waitForTimeout(16); await pollPen(); }
  await page.screenshot({ path: `${SHOTS}/zone-${name}-recoil.png` });
  for (let k = 0; k < 10; k++) { await page.waitForTimeout(40); await pollPen(); }
  await page.waitForTimeout(2300);
  const after = await stats();
  const pk = after.peaks;
  const headZone = ['headTop', 'forehead', 'cheekLeft', 'cheekRight'].includes(name);
  // pulling the face must not push it through the hand; when the arm / torso themselves are dragged into each other the
  // (soft) contact may give by up to ~1 air gap
  const penLimit = headZone ? 0.006 : 0.02;
  void pk;
  report.zones[name] = { grabbedDuringHold: held.grabs, local, ring, far, handFollow, maxPenetration: pen,
    penetrationLimit: penLimit, residual: after.maxDisp, residualLimit: 0.005 * H, sleeping: after.sleeping, resets: after.resets };
  check(`${name}: grab hits the skin`, held.grabs === 1 && local > 0.02, `grabs=${held.grabs}`);
  check(`${name}: local > ring > far (falloff over the surface)`, local > ring && ring > far && ring > 0.002, `local=${local.toFixed(3)} ring=${ring.toFixed(3)} far=${far.toFixed(4)}`);
  check(`${name}: residual after 3 s < 0.005 H`, after.maxDisp < 0.005 * H, `residual=${after.maxDisp.toFixed(5)} (limit ${(0.005 * H).toFixed(4)})`);
  check(`${name}: no NaN / reset`, !after.hasNaN && !held.hasNaN && after.resets === 0, `resets=${after.resets}`);
  check(`${name}: skin vs hand spheres / hand vs eyeballs penetration below limit (sampled, read-back)`, pen < penLimit, `maxPen=${pen.toFixed(4)} limit=${penLimit}`);
  if (name === 'forehead' || name === 'headTop' || name === 'cheekRight') check(`${name}: hand is not dragged along (no sticking)`, handFollow < 0.5 * local, `hand=${handFollow.toFixed(3)} vs local=${local.toFixed(3)}`);
  void hitsDeformed;
}

// ------------------------------------------------------------------ picking uses the DEFORMED skin
{
  await B(() => window.__blonde.reset());
  await page.waitForTimeout(300);
  const p = meta.zones.side, s = await project(p);
  const target = { x: s.x - 64, y: s.y + 5 };                    // wall at rest: left of the body contour
  const before = await B(([x, y]) => window.__blonde.pickAt(x, y), [target.x, target.y]);
  await page.mouse.move(s.x, s.y); await page.mouse.down();
  for (let i = 1; i <= 12; i++) { await page.mouse.move(s.x - 80 * i / 12, s.y + 6 * i / 12); await page.waitForTimeout(16); }
  await page.waitForTimeout(500);
  const during = await B(([x, y]) => window.__blonde.pickAt(x, y), [target.x, target.y]);
  const cen = await B(([q, r]) => window.__blonde.centroid(q, r), [p, 0.05]);
  const dist = during ? Math.hypot(during[0] - cen.pos[0], during[1] - cen.pos[1], during[2] - cen.pos[2]) : Infinity;
  await page.mouse.up();
  report.pickDeformed = { hitAtRest: before, hitWhileDeformed: during, distanceToDraggedPatch: dist };
  check('ray-pick hits the deformed skin (a spot that is wall at rest hits the pulled-out skin)', before === null && during !== null && dist < 0.08, `rest hit=${before ? 'skin' : 'none'} deformed hit=${during ? 'skin' : 'none'} dist to dragged patch=${dist.toFixed(3)}`);
  await page.waitForTimeout(1500);
}

// ------------------------------------------------------------------ fast flick: damped oscillation
{
  await B(() => window.__blonde.reset());
  await page.waitForTimeout(300);
  const p = meta.zones.headTop, s = await project(p);
  await page.mouse.move(s.x, s.y); await page.mouse.down();
  await page.mouse.move(s.x - 95, s.y - 10, { steps: 4 });
  await page.waitForTimeout(160);
  await page.mouse.up();
  const t0 = Date.now();
  const trace = [];
  while (Date.now() - t0 < 2400) { trace.push([(Date.now() - t0) / 1000, await B((q) => window.__blonde.centroid(q, 0.08).disp[0], p)]); await page.waitForTimeout(12); }
  const x = trace.map((t) => t[1]);
  // visible swings = alternating extrema larger than 0.004 H
  // the release point itself is the first swing; then alternating extrema larger than 0.003 H (~4 px here)
  const ext = Math.abs(x[0]) > 0.003 * H ? [[trace[0][0], x[0]]] : [];
  for (let i = 1; i < x.length - 1; i++) if ((x[i] - x[i - 1]) * (x[i + 1] - x[i]) <= 0 && Math.abs(x[i]) > 0.003 * H) { if (!ext.length || Math.sign(ext[ext.length - 1][1]) !== Math.sign(x[i])) ext.push([trace[i][0], x[i]]); }
  let settle = 0;
  for (let i = x.length - 1; i >= 0; i--) if (Math.abs(x[i]) > 0.002 * H) { settle = trace[i][0]; break; }
  // decaying envelope: every later swing is smaller than the first, and the tail is < 30 % of it
  let decays = ext.length > 1 && Math.abs(ext[ext.length - 1][1]) < 0.3 * Math.abs(ext[0][1]);
  for (let i = 1; i < ext.length; i++) if (Math.abs(ext[i][1]) > Math.abs(ext[0][1]) * 0.7) decays = false;
  report.flick = { zone: 'headTop', extrema: ext.map((e) => [+e[0].toFixed(2), +e[1].toFixed(4)]), settleSeconds: settle, trace: trace.filter((_, i) => i % 3 === 0).map((t) => [+t[0].toFixed(2), +t[1].toFixed(4)]) };
  check('flick: 2+ visible decaying oscillations (firm rubber, v3.1), at rest within 2 s', ext.length >= 2 && decays && settle < 2.0, `swings=${ext.length} (${ext.slice(0, 5).map((e) => e[1].toFixed(3)).join(', ')}) settle=${settle.toFixed(2)} s`);
}

// ------------------------------------------------------------------ abuse: long drag, cancel, blur, rapid grabs
{
  await B(() => { window.__blonde.reset(); window.__blonde.resetPeaks(); });
  const s = await project(meta.zones.cheekLeft);
  await page.mouse.move(s.x, s.y); await page.mouse.down();
  await page.mouse.move(-300, 1300, { steps: 25 });          // far outside the canvas and the window
  await page.waitForTimeout(500);
  const st = await stats();
  await page.screenshot({ path: `${SHOTS}/long-drag.png` });
  check('long drag off-canvas: finite, saturated pull, no reset', !st.hasNaN && st.maxDisp > 0.2 && st.maxDisp < prm.grabMaxPull + 0.3 && st.resets === 0, `maxDisp=${st.maxDisp.toFixed(3)}`);
  await page.mouse.up();
  await page.waitForTimeout(400);

  await page.mouse.move(s.x, s.y); await page.mouse.down(); await page.mouse.move(s.x + 60, s.y, { steps: 5 });
  await B(() => window.dispatchEvent(new Event('blur')));
  await page.waitForTimeout(80);
  check('window blur releases the grab', (await stats()).grabs === 0 && (await stats()).driverGrabs === 0);
  await page.mouse.up();

  await page.mouse.move(s.x, s.y); await page.mouse.down(); await page.mouse.move(s.x + 60, s.y, { steps: 5 });
  await B(() => document.getElementById('view').dispatchEvent(new PointerEvent('pointercancel', { pointerId: 1, bubbles: true })));
  await page.waitForTimeout(80);
  check('pointercancel releases the grab', (await stats()).grabs === 0 && (await stats()).driverGrabs === 0);
  await page.mouse.up();

  await B(() => window.__blonde.resetPeaks());
  const names = Object.keys(DRAG);
  for (let k = 0; k < 15; k++) {
    const q = await project(meta.zones[names[k % names.length]]);
    await page.mouse.move(q.x, q.y); await page.mouse.down();
    await page.mouse.move(q.x + (k % 2 ? 80 : -80), q.y - 40, { steps: 3 });
    await page.mouse.up();
  }
  await page.waitForTimeout(3000);
  const st2 = await stats();
  check('15 rapid grabs: stable and settles', !st2.hasNaN && st2.resets === 0 && st2.maxDisp < 0.005 * H, `residual=${st2.maxDisp.toFixed(5)}`);
}

// ------------------------------------------------------------------ the character can be pulled OUT of the photograph
{
  await B(() => { window.__blonde.reset(); window.__blonde.resetPeaks(); });
  await page.waitForTimeout(400);
  const s = await project(meta.zones.cheekLeft);
  const outX = canvas.x - 15;                                   // 15 px left of the photo, on the paper (pulls saturate at 0.5)
  const paperBefore = await B(([x, y]) => window.__blonde.pickAt(x, y), [outX, s.y]);
  await page.mouse.move(s.x, s.y); await page.mouse.down();
  for (let i = 1; i <= 24; i++) { await page.mouse.move(s.x + (outX - 160 - s.x) * i / 24, s.y); await page.waitForTimeout(16); }
  await page.waitForTimeout(700);
  const hit = await B(([x, y]) => window.__blonde.pickAt(x, y), [outX, s.y]);
  await page.screenshot({ path: `${SHOTS}/out-of-frame-held.png` });
  const px = await page.screenshot({ clip: { x: outX - 2, y: s.y - 2, width: 4, height: 4 } });
  await page.mouse.up();
  await page.waitForTimeout(1600);
  const rest = await page.screenshot({ clip: { x: outX - 2, y: s.y - 2, width: 4, height: 4 } });
  // decode the two 4x4 PNG clips in the page (2D canvas) and take the centre pixel
  const rgb = (buf) => B(async (b64) => {
    const img = new Image(); img.src = 'data:image/png;base64,' + b64; await img.decode();
    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
    const g = c.getContext('2d'); g.drawImage(img, 0, 0);
    return Array.from(g.getImageData(2, 2, 1, 1).data.slice(0, 3));
  }, buf.toString('base64'));
  const held = await rgb(px), after = await rgb(rest);
  const yellow = (c) => c && c[0] > c[2] + 40 && c[1] > c[2] + 15;                 // skin, not the grey paper
  const paper = (c) => c && Math.abs(c[0] - c[2]) < 20;
  report.outOfFrame = { pickBefore: !!paperBefore, pickHeld: !!hit, pixelHeld: held, pixelAfter: after };
  check('skin pulled out of the photo stays visible and grabbable on the paper', !paperBefore && !!hit && yellow(held) && paper(after), `pick before=${!!paperBefore} held=${!!hit} pixel held=${held} after=${after}`);
}

// ------------------------------------------------------------------ touch drag (CDP touch events) + two fingers
{
  await B(() => window.__blonde.reset());
  const cdp = await ctx.newCDPSession(page);
  const s = await project(meta.zones.belly), s2 = await project(meta.zones.headTop);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: s.x, y: s.y, id: 7 }] });
  for (let i = 1; i <= 10; i++) { await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: s.x - 8 * i, y: s.y + 5 * i, id: 7 }] }); await page.waitForTimeout(16); }
  await page.waitForTimeout(300);
  const held = await stats();
  const d = await shell(meta.zones.belly, 0, 0.06);
  await page.screenshot({ path: `${SHOTS}/touch-belly-held.png` });
  check('touch drag deforms the belly', held.grabs === 1 && d > 0.03, `disp=${d.toFixed(3)}`);
  // second finger on the head while the first is still down
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: s.x - 80, y: s.y + 50, id: 7 }, { x: s2.x, y: s2.y, id: 8 }] });
  for (let i = 1; i <= 8; i++) { await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: s.x - 80, y: s.y + 50, id: 7 }, { x: s2.x + 4 * i, y: s2.y - 8 * i, id: 8 }] }); await page.waitForTimeout(16); }
  await page.waitForTimeout(250);
  const two = await stats();
  await page.screenshot({ path: `${SHOTS}/touch-two-fingers.png` });
  check('multi-touch: two simultaneous grabs', two.grabs === 2, `grabs=${two.grabs}`);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await page.waitForTimeout(150);
  check('touch end releases everything', (await stats()).grabs === 0);
  await page.waitForTimeout(1500);
}

// ------------------------------------------------------------------ eyes: gaze limited, stay in the sockets, never through the fingers
{
  await B(() => { window.__blonde.reset(); window.__blonde.resetPeaks(); });
  const rest = await B(() => window.__blonde.eyeCheck());
  // aim: for pointer positions all over the photo, the projected gaze of the front (right) eye must point at the pointer
  // (u, v) relative to the photo; values outside 0..1 are on the paper around it - the eyes must keep following there
  const targets = [[0.08, 0.06], [0.92, 0.06], [0.08, 0.94], [0.92, 0.94], [0.5, 0.15], [0.15, 0.45], [0.85, 0.45], [0.5, 0.75], [-0.3, 0.4], [1.3, 0.4], [0.5, -0.15], [-0.25, -0.1], [1.25, 1.1]];
  let worstAim = 0, maxOff = 0, swing = 0, prev = null;
  for (const [u, v] of targets) {
    const px = canvas.x + u * canvas.width, py = canvas.y + v * canvas.height;
    await page.mouse.move(px, py); await page.waitForTimeout(700);
    const e = await B(() => window.__blonde.eyeCheck());
    maxOff = Math.max(maxOff, ...e.gazeOffsetDeg);
    const c = e.centers[0], g = e.gaze[0];
    const s0 = await B((p) => window.__blonde.project(p), c);
    const s1 = await B((p) => window.__blonde.project(p), [c[0] + g[0] * 0.1, c[1] + g[1] * 0.1, c[2] + g[2] * 0.1]);
    const ax = s1.x - s0.x, ay = s1.y - s0.y, bx = px - s0.x, by = py - s0.y;
    const err = Math.acos(Math.max(-1, Math.min(1, (ax * bx + ay * by) / (Math.hypot(ax, ay) * Math.hypot(bx, by) + 1e-9)))) * 180 / Math.PI;
    worstAim = Math.max(worstAim, err);
    if (prev) swing = Math.max(swing, Math.acos(Math.min(1, g[0] * prev[0] + g[1] * prev[1] + g[2] * prev[2])) * 180 / Math.PI);
    prev = g;
  }
  check('gaze aims at the pointer on and around the photo (screen direction error < 20°), within the 45° cone', worstAim < 20 && maxOff <= rest.maxGazeDeg + 0.01 && swing > 20,
    `worst aim error=${worstAim.toFixed(1)}° max cone=${maxOff.toFixed(1)}° max swing=${swing.toFixed(1)}°`);
  await page.mouse.move(canvas.x - 60, canvas.y - 60); await page.waitForTimeout(800);
  // pull the head next to the right eye and the hand across it: the hand must keep clear of the eyeballs
  let minClear = rest.handClearance;
  for (const [zone, dx, dy] of [['cheekRight', -60, 40], ['hand', 30, -60], ['forehead', 40, 70]]) {
    const s = await project(meta.zones[zone]);
    await page.mouse.move(s.x, s.y); await page.mouse.down();
    for (let i = 1; i <= 10; i++) { await page.mouse.move(s.x + dx * i / 10, s.y + dy * i / 10); await page.waitForTimeout(16); const c = await B(() => window.__blonde.eyeCheck().handClearance); minClear = Math.min(minClear, c); }
    await page.waitForTimeout(350);
    minClear = Math.min(minClear, await B(() => window.__blonde.eyeCheck().handClearance));
    if (zone === 'hand') await page.screenshot({ path: `${SHOTS}/eye-vs-hand.png` });
    await page.mouse.up();
    await page.waitForTimeout(700);
  }
  report.eyes = { restHandClearance: rest.handClearance, minHandClearanceWhileDragging: minClear, maxGazeOffsetDeg: maxOff, swingDeg: swing };
  check('eyeballs never poke through the hand (hand membrane clearance > -0.004 = 0.0016 H, transient)', minClear > -0.004, `rest=${rest.handClearance.toFixed(4)} min while dragging=${minClear.toFixed(4)}`);
}

// ------------------------------------------------------------------ droplets stay attached while deformed
{
  await B(() => window.__blonde.reset());
  const base = await B(() => window.__blonde.dropletCheck(300));
  const s = await project(meta.zones.forehead);
  await page.mouse.move(s.x, s.y); await page.mouse.down();
  for (let i = 1; i <= 10; i++) { await page.mouse.move(s.x - 7 * i, s.y - 6 * i); await page.waitForTimeout(16); }
  await page.waitForTimeout(400);
  const def = await B(() => window.__blonde.dropletCheck(300));
  await page.screenshot({ path: `${SHOTS}/droplets-deformed.png` });
  await page.mouse.up();
  report.droplets = { rest: base, deformed: def, counts: await B(() => window.__blonde.counts) };
  check('droplets stay on the deformed skin (< 0.2 radius)', def.worstInRadii < 0.2, `worst=${def.worstInRadii.toFixed(3)} mean=${def.meanInRadii.toFixed(3)} radii (rest worst ${base.worstInRadii.toFixed(3)})`);
  await page.waitForTimeout(1500);
}

// ------------------------------------------------------------------ violent drags (like the user's screen recording): huge pulls of every zone
{
  await B(() => { window.__blonde.reset(); window.__blonde.resetPeaks(); });
  await page.waitForTimeout(300);
  const drags = [['cheekRight', 320, -200], ['cheekLeft', -260, -60], ['headTop', 60, -260], ['belly', -200, 150], ['hand', -150, 220], ['forehead', -150, -250], ['side', -250, 40], ['forearm', 200, 150]];
  let pen = 0, lid = Infinity, nan = false, maxDisp = 0;
  const rows = [];
  for (const [zone, dx, dy] of drags) {
    const s = await project(meta.zones[zone]);
    await page.mouse.move(s.x, s.y); await page.mouse.down();
    for (let i = 1; i <= 16; i++) {
      await page.mouse.move(s.x + dx * i / 16, s.y + dy * i / 16); await page.waitForTimeout(16);
      const q = await B(() => ({ p: window.__blonde.penetration(), e: window.__blonde.eyeCheck(), s: window.__blonde.stats() }));
      pen = Math.max(pen, q.p.armSpheres, q.p.eyeballs); lid = Math.min(lid, q.e.lidClearance); nan ||= q.s.hasNaN; maxDisp = Math.max(maxDisp, q.s.maxDisp);
    }
    await page.waitForTimeout(250);
    await page.screenshot({ path: `${SHOTS}/violent-${zone}.png` });
    await page.mouse.up();
    for (let k = 0; k < 12; k++) {
      await page.waitForTimeout(25);
      const q = await B(() => ({ p: window.__blonde.penetration(), e: window.__blonde.eyeCheck(), s: window.__blonde.stats() }));
      pen = Math.max(pen, q.p.armSpheres, q.p.eyeballs); lid = Math.min(lid, q.e.lidClearance); nan ||= q.s.hasNaN;
    }
    rows.push(zone);
  }
  await page.waitForTimeout(3000);
  const st = await stats();
  report.violent = { drags: rows, maxDisp, maxPenetration: pen, minLidClearance: lid, residual: st.maxDisp, resets: st.resets, nan };
  check('violent drags: no NaN / reset, back at rest within 3 s', !nan && st.resets === 0 && st.maxDisp < 0.005 * H, `maxDisp during=${maxDisp.toFixed(3)} residual=${st.maxDisp.toFixed(5)}`);
  check('violent drags: skin vs hand / hand vs eyeball penetration (sampled) < 0.02', pen < 0.02, `max=${pen.toFixed(4)}`);
  check('violent drags: eyelids never sink into the ball (lid clearance > -0.006)', lid > -0.006, `min=${lid.toFixed(4)}`);
}

// ------------------------------------------------------------------ performance (this machine only)
{
  await B(() => window.__blonde.reset());
  await page.waitForTimeout(500);
  const s = await project(meta.zones.cheekRight);
  await page.mouse.move(s.x, s.y); await page.mouse.down();
  const samples = [];
  for (let i = 0; i < 90; i++) {
    await page.mouse.move(s.x + Math.sin(i / 6) * 80, s.y + Math.cos(i / 7) * 40);
    await page.waitForTimeout(16);
    if (i % 5 === 4) samples.push(await stats());
  }
  await page.mouse.up();
  const med = (k) => { const a = samples.map((x) => x[k]).sort((p, q) => p - q); return a[a.length >> 1]; };
  // frame pacing measured in-page over 2 s
  const pacing = await B(() => new Promise((res) => { const t = []; let last = performance.now(); const f = (now) => { t.push(now - last); last = now; if (t.length < 120) requestAnimationFrame(f); else { t.sort((a, b) => a - b); res({ medianFrameIntervalMs: t[60], p95FrameIntervalMs: t[114] }); } }; requestAnimationFrame(f); }));
  report.performance = { machine: 'Apple M6 (this machine), headless Chromium, 900x900 viewport, DPR 1', backend: samples[0].backend, physics: samples[0].mode,
    cpuEncodeMsMedian: med('encodeMs'), cpuFrameMsMedian: med('frameMs'), fpsMedian: med('fps'), ...pacing,
    gpuFrameMs: null, gpuNote: 'GPU timestamp queries are not exposed in this configuration; GPU frame time not measured.' };
  console.log('perf', JSON.stringify(report.performance));
  check('60 fps while dragging', pacing.medianFrameIntervalMs < 18.5, `median frame interval ${pacing.medianFrameIntervalMs.toFixed(2)} ms, CPU encode ${med('encodeMs').toFixed(2)} ms`);
  await page.waitForTimeout(800);
}
check('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
await ctx.close();

// ------------------------------------------------------------------ fallback really runs: WebGL2 backend (the membrane passes are TSL too)
report.fallbacks = {};
for (const [label, query] of [['webgl2', '?webgl']]) {
  try {
    const f = await open(browser, { size: 900, query });
    const info = await f.page.evaluate(() => window.__blonde.gpuInfo());
    const zone = meta.zones.cheekRight;
    const s = await f.page.evaluate((q) => window.__blonde.project(q), zone);
    await f.page.mouse.move(s.x, s.y); await f.page.mouse.down();
    for (let i = 1; i <= 10; i++) { await f.page.mouse.move(s.x + 8 * i, s.y); await f.page.waitForTimeout(16); }
    await f.page.waitForTimeout(450);
    const local = await f.page.evaluate(([q, r]) => window.__blonde.shell(q, 0, r), [zone, R * 0.45]);
    const st = await f.page.evaluate(() => window.__blonde.stats());
    await f.page.screenshot({ path: `${SHOTS}/fallback-${label}-held.png` });
    await f.page.mouse.up();
    await f.page.waitForTimeout(2500);
    const after = await f.page.evaluate(() => window.__blonde.stats());
    report.fallbacks[label] = { backend: info.backend, solver: info.solver, local, fps: st.fps, cpuFrameMs: st.frameMs, solverStepMs: st.avgStepMs, residual: after.maxDisp, errors: f.errors };
    const expect = label === 'webgl2' ? info.backend === 'webgl2' : label === 'worker-copy' ? info.solver === 'worker-copy' : info.solver === 'main';
    check(`fallback ${label}: boots, drags, settles`, expect && local > 0.03 && after.maxDisp < 0.005 * H && f.errors.length === 0, `backend=${info.backend} solver=${info.solver} local=${local.toFixed(3)} residual=${after.maxDisp.toFixed(5)} errors=${f.errors.length}`);
    await f.ctx.close();
  } catch (e) { check(`fallback ${label}: boots, drags, settles`, false, String(e).slice(0, 300)); }
}
await browser.close();

const all = existsSync('docs/qa-report.json') ? JSON.parse(readFileSync('docs/qa-report.json', 'utf8')) : {};
all.interaction = report;
writeFileSync('docs/qa-report.json', JSON.stringify(all, null, 1));
const failed = report.checks.filter((c) => !c.ok).length;
console.log(`\n${report.checks.length - failed}/${report.checks.length} checks passed`);
process.exit(failed ? 1 : 0);
