/**
 * Sheen-style GPU membrane — the physics of the character (v3).
 *
 * Learned from sciecode/sheen (src/modules/fbo.js + src/glsl/*.frag.js) and applied to the whole Nailoong surface:
 *   - state lives in float textures, one fullscreen pass per operation, positions never leave the GPU for rendering;
 *   - integrate  : Verlet with a shape-recovery acceleration towards the original surface (Sheen: 345 (org - pos)) and
 *                  0.98 velocity retention per sub-step;
 *   - constraints: Jacobi relaxation of the 1-ring edge lengths (Sheen: 0.5 (d - L) dir, averaged over the ring);
 *   - mouse      : the grabbed patch is moved to org + (pointer - grabbed point) before the constraint passes, the
 *                  rest of the surface follows through the springs; 6 sub-steps x 5 iterations per frame (Sheen's numbers);
 *   - normals    : recomputed from the deformed 1-ring every frame (Sheen normals.frag).
 * Added for this scene: per-region shape recovery / stiffness (the hand keeps its shape, the cheeks stretch), collision
 * spheres riding inside the arm (face / chest skin is pushed out of them, so the face cannot be pulled through the
 * hand) and the two eyeballs (hand skin is pushed out of them), up to 4 simultaneous grabs (multi-touch), pins below
 * the frame, sleeping when at rest, and an asynchronous read-back for picking and the QA hooks.
 */
import { DataTexture, FloatType, NearestFilter, NodeMaterial, QuadMesh, RGBAFormat, RenderTarget, Vector3, Vector4, type WebGPURenderer } from 'three/webgpu';
import { Fn, If, Loop, dot, float, int, ivec2, max, min, mix, normalize, screenCoordinate, smoothstep, textureLoad, uniform, uniformArray, vec3, vec4, cross, length } from 'three/tsl';

export const SIM_W = 256;
export const MAX_GRABS = 4;
const RING = 12;
const PROXY_K = 24;

export interface MembraneData {
  rest: Float32Array; adj: Float32Array; dist: Float32Array; restore: Float32Array; stiff: Float32Array; pinned: Float32Array;
  group: Uint8Array; collideBody: Float32Array; collideEye: Float32Array; tris: Uint32Array;
  proxyIds: Float32Array; proxyOffset: Float32Array; proxyRadius: Float32Array; proxyType: Float32Array; proxyRest: Float32Array;
  socketW: Float32Array; socketEye: Float32Array;
  /** rigid groups (eye sockets, hand): ring + 3 anchors each; socketEye = group index of each member vertex */
  eyes: { ring: number[]; anchors: number[] }[];
}

export interface MembraneParams {
  substeps: number; iterations: number; retain: number; restoreScale: number; stiffScale: number; socketHold: number;
  grabRadius: number; grabRadiusArm: number; grabCore: number; grabMaxPull: number; grabMaxPullArm: number; grabSpread: number; grabRadiusMax: number;
}
export const DEFAULT_MEMBRANE: MembraneParams = {
  // v3.1: firmer than Sheen's sphere (user: "too soft, like a water balloon, moves too much"): 3x shape recovery,
  // pulls saturate at 0.5 (body) / 0.22 (arm), the grab patch no longer widens with the pull, a little more damping
  substeps: 6, iterations: 5, retain: 0.965, restoreScale: 3.0, stiffScale: 2.0, socketHold: 0.9,
  grabRadius: 0.07, grabRadiusArm: 0.16, grabCore: 0.35, grabMaxPull: 0.5, grabMaxPullArm: 0.22, grabSpread: 0, grabRadiusMax: 0.16,
};

/** centroid + orthonormal frame of a lid ring (CPU; eyes.ts and the GPU eye-frame pass use the same construction) */
export function ringFrameCPU(P: Float32Array, ring: number[], anchors: number[]) {
  const c = [0, 0, 0];
  for (const i of ring) for (let r = 0; r < 3; r++) c[r] += P[i * 3 + r] / ring.length;
  const g = (i: number) => [P[i * 3], P[i * 3 + 1], P[i * 3 + 2]];
  const [up, sp, sm] = anchors.map(g);
  const nz = (a: number[]) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return a.map((x) => x / l); };
  const cr = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const X = nz([sp[0] - sm[0], sp[1] - sm[1], sp[2] - sm[2]]);
  const Z = nz(cr(X, [up[0] - c[0], up[1] - c[1], up[2] - c[2]]));
  const Y = cr(Z, X);
  return { c, X, Y, Z, spacing: Math.hypot(sp[0] - sm[0], sp[1] - sm[1], sp[2] - sm[2]) };
}

const tex = (data: Float32Array, w: number, h: number) => {
  const t = new DataTexture(data, w, h, RGBAFormat, FloatType);
  t.minFilter = t.magFilter = NearestFilter;
  t.generateMipmaps = false;
  t.needsUpdate = true;
  return t;
};
const rt = (w: number, h: number) => new RenderTarget(w, h, { type: FloatType, format: RGBAFormat, depthBuffer: false, minFilter: NearestFilter, magFilter: NearestFilter, generateMipmaps: false });

interface GrabState { slot: number; restPoint: Vector3; target: Vector3; smoothed: Vector3; arm: boolean }

export class Membrane {
  readonly n: number;
  readonly W = SIM_W;
  readonly H: number;
  readonly rest: Float32Array;
  readonly params: MembraneParams = { ...DEFAULT_MEMBRANE };
  readonly data: MembraneData;
  /** latest CPU copy of the positions (asynchronous read-back, 1-2 frames old) */
  readonly positions: Float32Array;
  /** fixed render targets read by every deformed material */
  readonly posOut: RenderTarget;
  readonly nrmOut: RenderTarget;
  readonly activeGrabs = new Map<number, GrabState>();
  stats = { steps: 0, maxDisp: 0, hasNaN: false, sleeping: false, readbacks: 0, encodeMs: 0, resets: 0 };
  peaks = { maxDisp: 0, maxPenetration: 0 };

  private renderer: WebGPURenderer;
  private pos: RenderTarget[];
  private proxyRT: RenderTarget;
  private orgTex: DataTexture; private matTex: DataTexture; private adjTex: DataTexture[]; private distTex: DataTexture[];
  private pIdsTex: DataTexture; private pInfoTex: DataTexture; private pRestTex: DataTexture;
  private sockTex: DataTexture;                 // (local x, y, z in the eye's rest frame, +w eye 0 / -w eye 1)
  private eyeRT: RenderTarget;                  // 4 x nEyes: centroid, X, Y, Z of every lid ring
  private nProxy: number;
  private prev = 0; private cur = 1; private spare = 2;
  private quads = new Map<string, QuadMesh>();
  private u = {
    dt: uniform(1 / 360), retain: uniform(0.98), restoreScale: uniform(1), stiffScale: uniform(2), socketHold: uniform(0.6),
    grabOrg: uniformArray(Array.from({ length: MAX_GRABS }, () => new Vector4()), 'vec4'),       // xyz rest point, w radius
    grabTgt: uniformArray(Array.from({ length: MAX_GRABS }, () => new Vector4()), 'vec4'),       // xyz target, w active
    grabMeta: uniformArray(Array.from({ length: MAX_GRABS }, () => new Vector4()), 'vec4'),      // x group (0 body 1 arm), y core
    grabOn: uniform(0),
  };
  private reading = false;
  private quietFrames = 0;
  private readonly zero: Float32Array;

  constructor(renderer: WebGPURenderer, data: MembraneData) {
    this.renderer = renderer;
    this.data = data;
    const n = (this.n = data.rest.length / 3);
    const W = this.W, H = (this.H = Math.ceil(n / W));
    this.rest = data.rest;
    this.positions = new Float32Array(data.rest);
    this.zero = new Float32Array(W * H * 4);
    const T = W * H * 4;
    const org = new Float32Array(T), mat = new Float32Array(T);
    for (let i = 0; i < n; i++) {
      org.set([data.rest[i * 3], data.rest[i * 3 + 1], data.rest[i * 3 + 2], data.restore[i]], i * 4);
      mat.set([data.stiff[i], data.pinned[i], data.group[i], data.collideBody[i] + 2 * data.collideEye[i]], i * 4);
    }
    this.orgTex = tex(org, W, H);
    this.matTex = tex(mat, W, H);
    this.adjTex = []; this.distTex = [];
    for (let k = 0; k < RING / 4; k++) {
      const a = new Float32Array(T).fill(-1), d = new Float32Array(T);
      for (let i = 0; i < n; i++) for (let c = 0; c < 4; c++) { a[i * 4 + c] = data.adj[i * RING + k * 4 + c]; d[i * 4 + c] = data.dist[i * RING + k * 4 + c]; }
      this.adjTex.push(tex(a, W, H)); this.distTex.push(tex(d, W, H));
    }
    const P = (this.nProxy = data.proxyRadius.length);
    const pid = new Float32Array((PROXY_K / 4) * P * 4).fill(-1), pinfo = new Float32Array(P * 4), prest = new Float32Array(P * 4);
    const rigidF = data.eyes.map((e) => ringFrameCPU(data.rest, e.ring, e.anchors));
    let eyeK = 0;
    for (let p = 0; p < P; p++) {
      if (data.proxyType[p] > 0.5) {            // eyeball: store the rest centre in its eye's rest frame, flag = -(eye + 1)
        const f = rigidF[eyeK], c = [data.proxyRest[p * 3], data.proxyRest[p * 3 + 1], data.proxyRest[p * 3 + 2]];
        const d = [c[0] - f.c[0], c[1] - f.c[1], c[2] - f.c[2]], dt = (a: number[]) => d[0] * a[0] + d[1] * a[1] + d[2] * a[2];
        data.proxyOffset.set([dt(f.X), dt(f.Y), dt(f.Z)], p * 3);
        data.proxyIds.fill(-(eyeK + 1), p * PROXY_K, (p + 1) * PROXY_K);
        eyeK++;
      }
      for (let k = 0; k < PROXY_K; k++) pid[(p * (PROXY_K / 4)) * 4 + k] = data.proxyIds[p * PROXY_K + k];
      prest.set([data.proxyRest[p * 3], data.proxyRest[p * 3 + 1], data.proxyRest[p * 3 + 2], 0], p * 4);
      pinfo.set([data.proxyOffset[p * 3], data.proxyOffset[p * 3 + 1], data.proxyOffset[p * 3 + 2], data.proxyRadius[p] * (data.proxyType[p] > 0.5 ? -1 : 1)], p * 4);
    }
    this.pIdsTex = tex(pid, PROXY_K / 4, P);
    // socket shape matching: every socket vertex's position in its eye's REST ring frame
    const sock = new Float32Array(T).fill(-1);
    for (let i = 0; i < T; i += 4) sock[i] = sock[i + 1] = sock[i + 2] = 0;
    const frames = data.eyes.map((e) => ringFrameCPU(data.rest, e.ring, e.anchors));
    for (let i = 0; i < n; i++) {
      const w = data.socketW[i];
      if (!(w > 0)) continue;
      const e = Math.round(data.socketEye[i]), f = frames[e];
      const d = [data.rest[i * 3] - f.c[0], data.rest[i * 3 + 1] - f.c[1], data.rest[i * 3 + 2] - f.c[2]];
      const dot = (a: number[]) => d[0] * a[0] + d[1] * a[1] + d[2] * a[2];
      sock.set([dot(f.X), dot(f.Y), dot(f.Z), e * 2 + w], i * 4);   // group index * 2 + weight (0 < w <= 1)
    }
    this.sockTex = tex(sock, W, H);
    this.eyeRT = rt(4, Math.max(2, data.eyes.length));
    this.pInfoTex = tex(pinfo, 1, P);
    this.pRestTex = tex(prest, 1, P);
    this.pos = [rt(W, H), rt(W, H), rt(W, H)];
    this.posOut = rt(W, H);
    this.nrmOut = rt(W, H);
    this.proxyRT = rt(P, 1);
    this.reset();
  }

  // ------------------------------------------------------------------ TSL helpers
  /** texel row/col of the fragment being written (WebGL's screenCoordinate is flipped by three) */
  private fragIndex(_h: number) {
    const x = int(screenCoordinate.x);
    const y = int(screenCoordinate.y);
    return { x, y, i: y.mul(this.W).add(x) };
  }

  /**
   * Row conventions. In TSL, screenCoordinate.y and textureLoad() on RENDER TARGETS agree on both backends (three flips
   * both under WebGL), so the vertex index i = floor(y) * W + x is used for every pass. DataTextures are not flipped
   * (indexed by i directly, also consistent), but readRenderTargetPixelsAsync returns GL rows under WebGL: measure
   * that once and flip the read-back rows if needed.
   */
  async calibrate() {
    const probe = rt(this.W, this.H);
    const q = new QuadMesh(Object.assign(new NodeMaterial(), { fragmentNode: vec4(screenCoordinate.y.floor(), 0, 0, 1), depthTest: false, depthWrite: false }));
    const prev = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(probe); q.render(this.renderer); this.renderer.setRenderTarget(prev);
    const a = (await this.renderer.readRenderTargetPixelsAsync(probe, 0, 0, 1, 2)) as Float32Array;
    probe.dispose();
    this.readFlip = a[4] - a[0] < 0;
    this.calibration = { readbackRow0: a[0], readbackRow1: a[4], readFlip: this.readFlip ? 1 : 0 };
    this.reset();
  }
  private readFlip = false;
  calibration: Record<string, number> = {};
  private at(i: any) { return ivec2(i.mod(this.W), i.div(this.W)); }
  private load(t: any, i: any) { return textureLoad(t, this.at(i)); }

  private quad(key: string, build: () => any) {
    let q = this.quads.get(key);
    if (!q) {
      const m = new NodeMaterial();
      m.fragmentNode = build();
      m.depthTest = false; m.depthWrite = false;
      q = new QuadMesh(m);
      this.quads.set(key, q);
    }
    return q;
  }

  private run(key: string, target: RenderTarget, build: () => any) {
    const q = this.quad(key, build);
    this.renderer.setRenderTarget(target);
    q.render(this.renderer);
  }

  // ------------------------------------------------------------------ passes
  private integratePass(prevT: any, curT: any) {
    const { orgTex, matTex, u } = this;
    return Fn(() => {
      const { i } = this.fragIndex(this.H);
      const o = this.load(orgTex, i), m = this.load(matTex, i);
      const x = this.load(curT, i).xyz, xp = this.load(prevT, i).xyz;
      const k = o.w.mul(u.restoreScale);
      const vel = x.sub(xp).div(u.dt).add(o.xyz.sub(x).mul(k).mul(u.dt));
      const nx = x.add(vel.mul(u.retain).mul(u.dt));
      return vec4(mix(nx, o.xyz, m.y), 1);
    })();
  }

  private proxyPass(curT: any) {
    const { pIdsTex, pInfoTex, eyeRT } = this;
    return Fn(() => {
      const p = int(screenCoordinate.x);
      const first = textureLoad(pIdsTex, ivec2(int(0), p)).x;
      const acc = vec3(0).toVar(), cnt = float(0).toVar();
      for (let k = 0; k < PROXY_K / 4; k++) {
        const ids = textureLoad(pIdsTex, ivec2(int(k), p));
        for (const c of ['x', 'y', 'z', 'w'] as const) {
          const id = ids[c];
          If(id.greaterThanEqual(0), () => { acc.addAssign(this.load(curT, int(id)).xyz); cnt.addAssign(1); });
        }
      }
      const info = textureLoad(pInfoTex, ivec2(int(0), p));
      const armC = acc.div(max(cnt, 1)).add(info.xyz);
      // eyeballs: the rigid frame of the lid ring, exactly like the rendered ball (eyes.ts)
      const e = int(first.negate().sub(1));
      const ec = textureLoad(eyeRT.texture, ivec2(int(0), e)).xyz, X = textureLoad(eyeRT.texture, ivec2(int(1), e)).xyz;
      const Y = textureLoad(eyeRT.texture, ivec2(int(2), e)).xyz, Z = textureLoad(eyeRT.texture, ivec2(int(3), e)).xyz;
      const eyeC = ec.add(X.mul(info.x)).add(Y.mul(info.y)).add(Z.mul(info.z));
      return vec4(first.lessThan(-0.5).select(eyeC, armC), info.w);
    })();
  }

  /** texel row of a fragment in the small eye target (calibrated separately: its height differs) */
  private eyeRow(y: any) { return int(y); }

  /** per eye: centroid and orthonormal frame (X, Y, Z) of its lid ring - same construction as eyes.ts */
  private eyeFramePass(curT: any) {
    const eyes = this.data.eyes;
    return Fn(() => {
      const col = int(screenCoordinate.x), row = this.eyeRow(screenCoordinate.y.floor());
      const out = vec4(0).toVar();
      eyes.forEach((e, k) => {
        If(row.equal(k), () => {
          let acc: any = vec3(0);
          for (const id of e.ring) acc = acc.add(this.load(curT, int(id)).xyz);
          const c = acc.div(e.ring.length);
          const up = this.load(curT, int(e.anchors[0])).xyz, sp = this.load(curT, int(e.anchors[1])).xyz, sm = this.load(curT, int(e.anchors[2])).xyz;
          const X = normalize(sp.sub(sm)), Z = normalize(cross(X, up.sub(c))), Y = cross(Z, X);
          If(col.equal(0), () => out.assign(vec4(c, 1)));
          If(col.equal(1), () => out.assign(vec4(X, 0)));
          If(col.equal(2), () => out.assign(vec4(Y, 0)));
          If(col.equal(3), () => out.assign(vec4(Z, 0)));
        });
      });
      return out;
    })();
  }

  private constraintPass(curT: any, withGrab: boolean, collideOnly = false) {
    const { orgTex, matTex, adjTex, distTex, proxyRT, u, sockTex, eyeRT, pRestTex } = this;
    const nP = this.nProxy;
    return Fn(() => {
      const { i } = this.fragIndex(this.H);
      const o = this.load(orgTex, i), m = this.load(matTex, i);
      const p0 = this.load(curT, i).xyz.toVar();
      // ---- mouse (Sheen mouse.frag): the patch around the grabbed point goes to org + (pointer - grabbed point)
      if (withGrab) {
        If(u.grabOn.greaterThan(0.5), () => {
          for (let g = 0; g < MAX_GRABS; g++) {
            const go: any = u.grabOrg.element(g), gt: any = u.grabTgt.element(g), gm: any = u.grabMeta.element(g);
            const sameGroup = m.z.equal(gm.x).or(m.z.equal(2));
            If(gt.w.greaterThan(0.5).and(sameGroup), () => {
              const d = length(o.xyz.sub(go.xyz)).div(go.w);
              const q = max(d.sub(gm.y), 0).div(float(1).sub(gm.y));
              const w = float(1).sub(smoothstep(0.0, 1.0, q));
              If(w.greaterThan(0), () => { p0.assign(mix(p0, o.xyz.add(gt.xyz.sub(go.xyz)), w)); });
            });
          }
        });
      }
      // ---- springs (Sheen constraints.frag): Jacobi average over the 1-ring
      if (!collideOnly) {
      const acc = vec3(0).toVar(), cnt = float(0).toVar();
      for (let k = 0; k < RING / 4; k++) {
        const ids = this.load(adjTex[k], i), ds = this.load(distTex[k], i);
        for (const c of ['x', 'y', 'z', 'w'] as const) {
          const id = ids[c], L = ds[c];
          If(id.greaterThanEqual(0), () => {
            const p1 = this.load(curT, int(id)).xyz;
            const e = p1.sub(p0), d = max(length(e), 1e-6);
            acc.addAssign(e.mul(d.sub(L).div(d)).mul(0.5));
            cnt.addAssign(1);
          });
        }
      }
      p0.addAssign(acc.div(max(cnt, 1)).mul(min(m.x.mul(u.stiffScale), 1.0)));
      // ---- rigid groups: eye sockets keep their shape around the balls, the hand moves as a limb (local shape matching)
      const sk = this.load(sockTex, i);
      If(sk.w.greaterThan(0), () => {
        const ey = int(sk.w.div(2).floor());
        const wgt = sk.w.sub(float(ey).mul(2));
        const c = textureLoad(eyeRT.texture, ivec2(int(0), ey)).xyz, X = textureLoad(eyeRT.texture, ivec2(int(1), ey)).xyz;
        const Y = textureLoad(eyeRT.texture, ivec2(int(2), ey)).xyz, Z = textureLoad(eyeRT.texture, ivec2(int(3), ey)).xyz;
        const target = c.add(X.mul(sk.x)).add(Y.mul(sk.y)).add(Z.mul(sk.z));
        p0.assign(mix(p0, target, wgt.mul(u.socketHold)));
      });
      }
      // ---- collisions: face / chest skin vs arm spheres (radius > 0), hand skin vs eyeballs (radius < 0)
      const flags = m.w;
      If(flags.greaterThan(0.5), () => {
        const body = flags.equal(1).or(flags.equal(3)), eye = flags.greaterThanEqual(2);
        Loop({ start: int(0), end: int(nP), type: 'int', condition: '<' }, ({ i: p }: { i: any }) => {
          const s = textureLoad(proxyRT.texture, ivec2(p, int(0)));
          const isEye = s.w.lessThan(0), r = s.w.abs();
          If(isEye.and(eye).or(isEye.not().and(body)), () => {
            const e = p0.sub(s.xyz), d = length(e);
            If(d.lessThan(r).and(d.greaterThan(1e-6)), () => {
              // push out on the side this skin belongs to at rest: a fast recoil that carried it through a finger
              // is sent back behind the finger instead of out in front of it
              const side = normalize(o.xyz.sub(textureLoad(pRestTex, ivec2(int(0), p)).xyz));
              const dir = e.div(d);
              p0.assign(s.xyz.add(dot(dir, side).lessThan(0).select(side, dir).mul(r)));
            });
          });
        });
      });
      return vec4(mix(p0, o.xyz, m.y), 1);
    })();
  }

  private normalPass() {
    const { adjTex } = this;
    const posT = this.posOut.texture;
    return Fn(() => {
      const { i } = this.fragIndex(this.H);
      const p0 = this.load(posT, i).xyz;
      const ids: any[] = [];
      for (let k = 0; k < RING / 4; k++) { const t = this.load(adjTex[k], i); ids.push(t.x, t.y, t.z, t.w); }
      const nrm = vec3(0).toVar();
      const first = this.load(posT, int(ids[0])).xyz.sub(p0);
      const prev = first.toVar();
      for (let k = 1; k <= RING; k++) {
        if (k < RING) {
          If(ids[k].greaterThanEqual(0), () => {
            const e = this.load(posT, int(ids[k])).xyz.sub(p0);
            nrm.addAssign(cross(prev, e)); prev.assign(e);
          });
        }
      }
      nrm.addAssign(cross(prev, first));                  // close the fan
      return vec4(normalize(nrm), 1);
    })();
  }

  private copyPass(srcT: any) {
    return Fn(() => { const { i } = this.fragIndex(this.H); return vec4(this.load(srcT, i).xyz, 1); })();
  }

  // ------------------------------------------------------------------ public
  reset() {
    const rest = this.orgTex;
    const copy = (src: any, dst: RenderTarget, key: string) => this.run(key, dst, () => Fn(() => { const { i } = this.fragIndex(this.H); return vec4(this.load(src, i).xyz, 1); })());
    const prevRT = this.renderer.getRenderTarget();
    for (let k = 0; k < 3; k++) copy(rest, this.pos[k], 'reset' + k);
    copy(rest, this.posOut, 'resetOut');
    this.run('normals', this.nrmOut, () => this.normalPass());
    this.renderer.setRenderTarget(prevRT);
    this.positions.set(this.rest);
    this.activeGrabs.clear();
    this.syncGrabs();
    this.stats.sleeping = false;
    this.quietFrames = 0;
  }

  /** advance one frame (Sheen: 6 sub-steps, 5 constraint iterations each, mouse before every iteration but the last two) */
  update(frameDt: number) {
    const t0 = performance.now();
    if (this.stats.sleeping && this.activeGrabs.size === 0) { this.stats.encodeMs = 0; return false; }
    // smoothed, speed-limited grab targets (no teleporting skin)
    for (const g of this.activeGrabs.values()) g.smoothed.lerp(g.target, 0.5);
    this.syncGrabs();
    const p = this.params, u = this.u;
    // one block of sub-steps per frame, frame time clamped to 1/30 s (no catch-up loop: a slow GPU must not get slower)
    const S = p.substeps, I = p.iterations;
    u.dt.value = Math.min(frameDt, 1 / 30) / S;
    u.retain.value = p.retain; u.restoreScale.value = p.restoreScale; u.stiffScale.value = p.stiffScale; u.socketHold.value = p.socketHold;
    const prevRT = this.renderer.getRenderTarget();
    const grabbing = this.activeGrabs.size > 0;
    for (let s = 0; s < S; s++) {
      const [a, b, c] = [this.prev, this.cur, this.spare];
      this.run(`int${a}${b}`, this.pos[c], () => this.integratePass(this.pos[a].texture, this.pos[b].texture));
      this.prev = b; this.cur = c; this.spare = a;
      this.run(`eyes${this.cur}`, this.eyeRT, () => this.eyeFramePass(this.pos[this.cur].texture));
      this.run(`proxy${this.cur}`, this.proxyRT, () => this.proxyPass(this.pos[this.cur].texture));
      for (let it = 0; it < I; it++) {
        this.run(`eyes${this.cur}`, this.eyeRT, () => this.eyeFramePass(this.pos[this.cur].texture));   // fresh lid frame every iteration
        const withGrab = grabbing && (s < S - 2 || it < I - 2);
        const [x, y] = [this.cur, this.spare];
        this.run(`con${x}${y}${withGrab ? 'g' : ''}`, this.pos[y], () => this.constraintPass(this.pos[x].texture, withGrab));
        this.cur = y; this.spare = x;
      }
      // a collision-only sweep closes the sub-step (proxies refreshed first): contacts get the last word
      this.run(`eyes${this.cur}`, this.eyeRT, () => this.eyeFramePass(this.pos[this.cur].texture));
      this.run(`proxy${this.cur}`, this.proxyRT, () => this.proxyPass(this.pos[this.cur].texture));
      { const [x, y] = [this.cur, this.spare];
        this.run(`col${x}${y}`, this.pos[y], () => this.constraintPass(this.pos[x].texture, false, true));
        this.cur = y; this.spare = x; }
    }
    this.run(`copy${this.cur}`, this.posOut, () => this.copyPass(this.pos[this.cur].texture));
    this.run('normals', this.nrmOut, () => this.normalPass());
    this.renderer.setRenderTarget(prevRT);
    this.stats.steps++;
    this.readBack();
    this.stats.encodeMs = performance.now() - t0;
    return true;
  }

  private readBack() {
    if (this.reading) return;
    this.reading = true;
    this.renderer.readRenderTargetPixelsAsync(this.posOut, 0, 0, this.W, this.H).then((buf: any) => {
      this.reading = false;
      const a = buf as Float32Array, n = this.n, out = this.positions, r = this.rest, W = this.W, H = this.H, flip = this.readFlip;
      let md = 0, nan = false;
      for (let i = 0; i < n; i++) {
        const row = Math.floor(i / W), j = ((flip ? H - 1 - row : row) * W + (i % W)) * 4;
        const x = a[j], y = a[j + 1], z = a[j + 2];
        if (!(x === x && y === y && z === z)) { nan = true; continue; }
        out[i * 3] = x; out[i * 3 + 1] = y; out[i * 3 + 2] = z;
        const d = Math.hypot(x - r[i * 3], y - r[i * 3 + 1], z - r[i * 3 + 2]);
        if (d > md) md = d;
      }
      this.stats.readbacks++;
      this.stats.maxDisp = md; this.stats.hasNaN = nan;
      if (md > this.peaks.maxDisp) this.peaks.maxDisp = md;
      if (nan) { this.stats.resets++; this.reset(); return; }
      // sleep: nothing held and back at rest for half a second -> snap exactly to rest and stop simulating
      if (this.activeGrabs.size === 0 && md < 6e-4) { if (++this.quietFrames > 30 && !this.stats.sleeping) { this.reset(); this.stats.sleeping = true; } }
      else this.quietFrames = 0;
    }).catch(() => { this.reading = false; });
  }

  // ------------------------------------------------------------------ grabs
  private syncGrabs() {
    const u = this.u, p = this.params;
    for (let s = 0; s < MAX_GRABS; s++) (u.grabTgt.array[s] as Vector4).w = 0;
    for (const g of this.activeGrabs.values()) {
      const pull = g.smoothed.distanceTo(g.restPoint);
      const r0 = g.arm ? p.grabRadiusArm : p.grabRadius;
      const R = g.arm ? r0 : Math.min(Math.max(r0, p.grabSpread * pull), Math.max(r0, p.grabRadiusMax));
      (u.grabOrg.array[g.slot] as Vector4).set(g.restPoint.x, g.restPoint.y, g.restPoint.z, R);
      (u.grabTgt.array[g.slot] as Vector4).set(g.smoothed.x, g.smoothed.y, g.smoothed.z, 1);
      (u.grabMeta.array[g.slot] as Vector4).set(g.arm ? 1 : 0, p.grabCore, 0, 0);
    }
    u.grabOn.value = this.activeGrabs.size ? 1 : 0;
  }

  /** rest-space point + group of a hit on the current surface: triangle `t`, barycentric (w1, w2) */
  restPointOf(t: number, w1: number, w2: number) {
    const f = this.data.tris, r = this.rest;
    const a = f[t * 3], b = f[t * 3 + 1], c = f[t * 3 + 2], w0 = 1 - w1 - w2;
    const p = new Vector3(r[a * 3] * w0 + r[b * 3] * w1 + r[c * 3] * w2, r[a * 3 + 1] * w0 + r[b * 3 + 1] * w1 + r[c * 3 + 1] * w2, r[a * 3 + 2] * w0 + r[b * 3 + 2] * w1 + r[c * 3 + 2] * w2);
    const g = this.data.group, arm = (g[a] === 1 ? 1 : 0) + (g[b] === 1 ? 1 : 0) + (g[c] === 1 ? 1 : 0) >= 2;
    return { p, arm };
  }

  beginGrab(id: number, t: number, w1: number, w2: number, hitWorld: ArrayLike<number>) {
    if (this.activeGrabs.size >= MAX_GRABS) return false;
    const used = new Set([...this.activeGrabs.values()].map((g) => g.slot));
    let slot = 0; while (used.has(slot)) slot++;
    const { p, arm } = this.restPointOf(t, w1, w2);
    // the pointer target is expressed relative to where the patch is NOW: start the target at the current hit point
    const now = new Vector3(hitWorld[0], hitWorld[1], hitWorld[2]);
    this.activeGrabs.set(id, { slot, restPoint: p, target: now.clone(), smoothed: now.clone(), arm });
    this.stats.sleeping = false; this.quietFrames = 0;
    return true;
  }

  moveGrab(id: number, target: ArrayLike<number>) {
    const g = this.activeGrabs.get(id);
    if (!g) return;
    const d = new Vector3(target[0], target[1], target[2]).sub(g.restPoint);
    const L = d.length(), m = g.arm ? this.params.grabMaxPullArm : this.params.grabMaxPull;
    if (L > 1e-9) d.multiplyScalar((m * Math.tanh(L / m)) / L);              // linear near the grab, saturating far away
    g.target.copy(g.restPoint).add(d);
  }

  endGrab(id: number) { this.activeGrabs.delete(id); this.syncGrabs(); }
  endAllGrabs() { this.activeGrabs.clear(); this.syncGrabs(); }
  wake() { this.stats.sleeping = false; this.quietFrames = 0; }
  resetPeaks() { this.peaks.maxDisp = 0; this.peaks.maxPenetration = 0; }

  /** TSL accessors for the deformed materials */
  posAt(id: any) { return textureLoad(this.posOut.texture, this.at(id)).xyz; }
  nrmAt(id: any) { return textureLoad(this.nrmOut.texture, this.at(id)).xyz; }
  orgAt(id: any) { return textureLoad(this.orgTex, this.at(id)).xyz; }
}
