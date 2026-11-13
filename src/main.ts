/**
 * blonde (Nailoong edition) — boot + frame loop + QA hooks.
 * URL switches:  ?debug  (panel)   ?webgl  (force the WebGL2 backend)
 */
import './styles.css';
import { Color, Vector2, Vector3 } from 'three/webgpu';
import { loadAssets } from './assets/load.ts';
import { Membrane } from './sim/membrane.ts';
import { boundCPU, ringNormalsCPU } from './render/surface.ts';
import { createSkin, skinUniforms } from './render/skin.ts';
import { Eyes, MAX_GAZE_DEG } from './render/eyes.ts';
import { createDroplets } from './render/droplets.ts';
import { createWall } from './render/wall.ts';
import { createStage, postUniforms } from './render/stage.ts';
import { fitCamera } from './render/frame.ts';
import { GrabController } from './interaction/grab.ts';
import { buildCover, LAYOUT } from './ui/cover.ts';
import { DebugPanel } from './ui/debug.ts';

type Mode = 'beauty' | 'id' | 'wall' | 'nodrops' | 'clean';

async function boot() {
  const q = new URLSearchParams(location.search);
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  const coverInfo = await buildCover(document.getElementById('type') as unknown as SVGSVGElement);
  const assets = await loadAssets();
  const sj = assets.scene;
  const stage = await createStage(canvas, sj, q.has('webgl'));
  const { renderer, camera, scene, pipeline } = stage;

  const mem = new Membrane(renderer, assets.membrane);
  await mem.calibrate();
  const skin = createSkin(assets, mem);
  const eyes = new Eyes(sj.eyes, mem);
  const drops = createDroplets(assets, mem);
  const wall = createWall(sj, camera);
  scene.add(skin.mesh, ...eyes.meshes, drops.mesh, wall.mesh);
  const photoEl = document.querySelector('.photo') as HTMLElement;
  const grab = new GrabController(canvas, camera, mem, assets.membrane.tris, photoEl);
  const debug = new DebugPanel(document.getElementById('debug')!, scene, mem, grab, skin.material);
  if (q.has('debug')) debug.toggle(true);
  window.addEventListener('keydown', (e) => { if (e.key === 'd' || e.key === 'D') debug.toggle(); if (e.key === 'r') mem.reset(); });

  const resize = () => {
    const r = canvas.parentElement!.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(2, Math.round(r.width)), h = Math.max(2, Math.round(r.height));
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    fitCamera(camera, w, h, LAYOUT.photo);           // photo frustum inside a canvas that covers the whole cover
  };
  resize();
  window.addEventListener('resize', resize);

  // ------------------------------------------------------------------ frame loop
  let mode: Mode = 'beauty';
  let last = performance.now(), fps = 60, frameMs = 0, gpuMs: number | null = null, frames = 0;
  const beautyMat = skin.mesh.material;

  const applyMode = () => {
    const id = mode === 'id';
    (skin.mesh as any).material = id ? skin.idMaterial : beautyMat;
    eyes.meshes.forEach((m, i) => (m.material = id ? eyes.idMaterial[i] : eyes.materials[i]));
    skin.mesh.visible = mode !== 'wall';
    eyes.meshes.forEach((m) => (m.visible = mode !== 'wall'));
    drops.mesh.visible = mode === 'beauty';
    wall.mesh.visible = !id;
    scene.background = id ? new Color('#000000') : null;
    postUniforms.grain.value = mode === 'beauty' ? 1 : 0;
    postUniforms.vignette.value = mode === 'beauty' || mode === 'nodrops' ? 1 : 0;   // QA modes measure the scene, not the lens
  };

  const frame = (now: number) => {
    const t0 = performance.now();
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    mem.update(dt);
    eyes.update(mem.positions, grab.hasLook ? grab.look : null, dt);
    debug.update({ fps, frameMs, gpuMs, backend: stage.backendName });
    if (mode === 'id') {
      const tm = renderer.toneMapping;
      renderer.toneMapping = stage.NoToneMapping;
      renderer.render(scene, camera);
      renderer.toneMapping = tm;
    } else pipeline.render();
    const cpu = performance.now() - t0;
    frameMs += (cpu - frameMs) * 0.1;
    if (dt > 0) fps += (1 / dt - fps) * 0.05;
    frames++;
  };
  renderer.setAnimationLoop(frame);

  // ------------------------------------------------------------------ QA / test hooks (membrane read-back, 1-2 frames old)
  const rest = mem.rest, pinned = assets.membrane.pinned, nNodes = mem.n;
  /** client position of a world point, and (u, v) relative to the PHOTO rectangle */
  const project = (p: number[]) => {
    const v = new Vector3(p[0], p[1], p[2]).project(camera);
    const r = canvas.getBoundingClientRect(), ph = photoEl.getBoundingClientRect();
    const x = r.left + (v.x * 0.5 + 0.5) * r.width, y = r.top + (-v.y * 0.5 + 0.5) * r.height;
    return { x, y, u: (x - ph.left) / ph.width, v: (y - ph.top) / ph.height };
  };
  /** mean displacement of the membrane vertices whose REST position lies in the shell [r0, r1) around p */
  const shell = (p: number[], r0: number, r1: number) => {
    const pos = mem.positions;
    let s = 0, c = 0;
    for (let i = 0; i < nNodes; i++) {
      const d = Math.hypot(rest[i * 3] - p[0], rest[i * 3 + 1] - p[1], rest[i * 3 + 2] - p[2]);
      if (d < r0 || d >= r1) continue;
      s += Math.hypot(pos[i * 3] - rest[i * 3], pos[i * 3 + 1] - rest[i * 3 + 1], pos[i * 3 + 2] - rest[i * 3 + 2]); c++;
    }
    return c ? s / c : 0;
  };
  const maxDisp = () => ({ maxDisp: mem.stats.maxDisp, hasNaN: mem.stats.hasNaN });
  const group = assets.membrane.group;
  /** face/chest membrane vertices inside an arm collision sphere, and hand vertices inside an eyeball (CPU, read-back) */
  const penetration = () => {
    const P = mem.positions, pk = assets.pack;
    const ids = pk.proxyIds as Float32Array, off = pk.proxyOffset as Float32Array, rad = pk.proxyRadius as Float32Array, typ = pk.proxyType as Float32Array;
    const nP = rad.length, C = new Float32Array(nP * 3);
    let ek = 0;
    for (let p = 0; p < nP; p++) {
      if (typ[p] > 0.5) { const c = eyes.centerFrom(P, ek++).c; C[p * 3] = c.x; C[p * 3 + 1] = c.y; C[p * 3 + 2] = c.z; continue; }
      let c = 0; const a = [0, 0, 0];
      for (let k = 0; k < 24; k++) { const id = ids[p * 24 + k]; if (id < 0) continue; a[0] += P[id * 3]; a[1] += P[id * 3 + 1]; a[2] += P[id * 3 + 2]; c++; }
      for (let r = 0; r < 3; r++) C[p * 3 + r] = a[r] / c + off[p * 3 + r];
    }
    let worst = 0, eyeWorst = 0, wv = -1, wp = -1;
    const cb = assets.membrane.collideBody, ce = assets.membrane.collideEye;
    for (let i = 0; i < nNodes; i++) {
      if (!cb[i] && !ce[i]) continue;
      for (let p = 0; p < nP; p++) {
        if ((typ[p] > 0.5 && !ce[i]) || (typ[p] < 0.5 && !cb[i])) continue;
        const pen = rad[p] - Math.hypot(P[i * 3] - C[p * 3], P[i * 3 + 1] - C[p * 3 + 1], P[i * 3 + 2] - C[p * 3 + 2]);
        if (typ[p] > 0.5) eyeWorst = Math.max(eyeWorst, pen); else if (pen > worst) { worst = pen; wv = i; wp = p; }
      }
    }
    return { armSpheres: worst, eyeballs: eyeWorst, worstVertex: wv, worstProxy: wp };
  };
  /** droplet attachment: distance of each sampled drop's base to the deformed skin around it, in drop radii */
  const dropletCheck = (samples = 300) => {
    const P = new Float32Array(mem.positions), pk = assets.pack;
    const Nr = ringNormalsCPU(P, assets.membrane.adj, 12);
    const sp = assets.skin.attributes.position.array as Float32Array, idx = assets.skin.index!.array;
    const nd = pk.dropSeed.length, nv = sp.length / 3;
    const picks: { i: number; v: number }[] = [];
    for (let s = 0; s < samples; s++) {
      const i = Math.floor((s + 0.5) * nd / samples);
      const x = pk.dropPos[i * 3], y = pk.dropPos[i * 3 + 1], z = pk.dropPos[i * 3 + 2];
      let best = Infinity, bv = 0;
      for (let v = 0; v < nv; v++) { const dx = sp[v * 3] - x, dy = sp[v * 3 + 1] - y, dz = sp[v * 3 + 2] - z; const dd = dx * dx + dy * dy + dz * dz; if (dd < best) { best = dd; bv = v; } }
      picks.push({ i, v: bv });
    }
    const fan = new Map<number, number[]>();
    for (const pq of picks) fan.set(pq.v, []);
    for (let t = 0; t < idx.length; t += 3) for (let k = 0; k < 3; k++) fan.get(idx[t + k])?.push(t);
    // widen every fan to the 2-ring: under stretch the nearest-at-rest vertex's own fan can miss the drop
    const ring1 = new Map<number, Set<number>>();
    for (const [v, ts] of fan) { const s = new Set<number>(); for (const t of ts) for (let k = 0; k < 3; k++) s.add(idx[t + k]); ring1.set(v, s); }
    const need = new Set<number>(); for (const s of ring1.values()) for (const v of s) need.add(v);
    const fan2 = new Map<number, number[]>(); for (const v of need) fan2.set(v, []);
    for (let t = 0; t < idx.length; t += 3) for (let k = 0; k < 3; k++) fan2.get(idx[t + k])?.push(t);
    for (const [v, s] of ring1) { const all = new Set<number>(); for (const u of s) for (const t of fan2.get(u)!) all.add(t); fan.set(v, [...all]); }
    let worst = 0, sum = 0, count = 0;
    const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0], d = [0, 0, 0];
    for (const pq of picks) {
      const i = pq.i;
      boundCPU(P, Nr, pk.dropTri, pk.dropBary, pk.dropResid, i, d);
      let dist = Infinity;
      for (const t of fan.get(pq.v)!) {
        boundCPU(P, Nr, pk.skinTri, pk.skinBary, pk.skinResid, idx[t], a); boundCPU(P, Nr, pk.skinTri, pk.skinBary, pk.skinResid, idx[t + 1], b); boundCPU(P, Nr, pk.skinTri, pk.skinBary, pk.skinResid, idx[t + 2], c);
        const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const nl = Math.hypot(n[0], n[1], n[2]) || 1;
        dist = Math.min(dist, Math.abs(((d[0] - a[0]) * n[0] + (d[1] - a[1]) * n[1] + (d[2] - a[2]) * n[2]) / nl));
      }
      const rel = dist / pk.dropScale[i * 3];
      worst = Math.max(worst, rel); sum += rel; count++;
    }
    return { worstInRadii: worst, meanInRadii: sum / count, samples: count };
  };
  /** eyes: gaze limits, and how far each ball centre is from its rest place relative to the lid ring */
  const eyeCheck = () => {
    const P = mem.positions;
    let clear = Infinity, lid = Infinity;
    const sw = assets.membrane.socketW, se = assets.membrane.socketEye;
    sj.eyes.forEach((ej, e) => {
      const { c, scale } = eyes.centerFrom(P, e);
      for (let i = 0; i < nNodes; i++) {
        const d = Math.hypot(P[i * 3] - c.x, P[i * 3 + 1] - c.y, P[i * 3 + 2] - c.z);
        if (group[i] === 1) { if (d - ej.radius < clear) clear = d - ej.radius; }
        else if (sw[i] > 0.5 && Math.round(se[i]) === e) { if (d - ej.radius * scale < lid) lid = d - ej.radius * scale; }
      }
    });
    return { handClearance: clear, lidClearance: lid, gazeOffsetDeg: [...eyes.gazeOffsetDeg], maxGazeDeg: MAX_GAZE_DEG, gaze: eyes.gaze.map((g) => g.toArray()), centers: eyes.centers.map((c) => c.toArray()) };
  };
  const pickAt = (x: number, y: number) => {
    const r = canvas.getBoundingClientRect();
    const hit = grab.pick(new Vector2(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1));
    return hit ? hit.toArray() : null;
  };
  const centroid = (p: number[], radius: number) => {
    const pos = mem.positions; let c = 0; const o = [0, 0, 0], d = [0, 0, 0];
    for (let i = 0; i < nNodes; i++) {
      if (Math.hypot(rest[i * 3] - p[0], rest[i * 3 + 1] - p[1], rest[i * 3 + 2] - p[2]) >= radius) continue;
      for (let k = 0; k < 3; k++) { o[k] += pos[i * 3 + k]; d[k] += pos[i * 3 + k] - rest[i * 3 + k]; } c++;
    }
    return { pos: o.map((v) => v / c), disp: d.map((v) => v / c), count: c };
  };
  void pinned;
  const gpuInfo = async () => {
    const out: Record<string, unknown> = { backend: stage.backendName, physics: 'gpu-membrane', rowCalibration: mem.calibration };
    try {
      const adapter = await (navigator as any).gpu?.requestAdapter();
      out.webgpuAdapter = adapter ? { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture, device: adapter.info?.device, description: adapter.info?.description, isFallbackAdapter: adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter } : null;
    } catch { out.webgpuAdapter = null; }
    const gl = document.createElement('canvas').getContext('webgl2');
    const ext = gl?.getExtension('WEBGL_debug_renderer_info');
    out.webglRenderer = ext ? gl!.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'unknown';
    return out;
  };

  (window as any).__blonde = {
    ready: true, meta: sj, layout: LAYOUT, coverInfo, scene, skin, eyes, counts: { ...sj.counts, drops: drops.count },
    project, shell, maxDisp, pickAt, centroid, dropletCheck, eyeCheck, penetration, gpuInfo,
    stats: () => ({ ...mem.stats, ...maxDisp(), peaks: { ...mem.peaks }, grabs: grab.count, driverGrabs: mem.activeGrabs.size, frameMs, fps, gpuMs, frames, mode: 'gpu-membrane', backend: stage.backendName }),
    resetPeaks: () => mem.resetPeaks(),
    reset: () => mem.reset(),
    setMode: (m: Mode) => { mode = m; applyMode(); },
    setDebug: (on: boolean) => debug.toggle(on),
    uniforms: { skin: skinUniforms, post: postUniforms },
    setParams: (p: Record<string, number>) => { Object.assign(mem.params, p); mem.wake(); },
    params: () => ({ ...mem.params }),
    membrane: mem,
    wallInfo: { pitch: wall.pitch, offset: wall.offset },
  };
}

boot().catch((err) => {
  console.error(err);
  const d = document.createElement('pre');
  d.style.cssText = 'position:fixed;left:8px;bottom:8px;color:#a00;font:12px monospace;white-space:pre-wrap;max-width:90vw';
  d.textContent = 'Failed to start: ' + (err?.message ?? err);
  document.body.append(d);
  (window as any).__blonde = { ready: false, error: String(err?.stack ?? err) };
});
