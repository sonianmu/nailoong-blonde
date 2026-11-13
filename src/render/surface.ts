/**
 * Render binding to the simulated membrane (tools/model/membrane.py):
 *   x = PhongTess(bary, x_i, n_i) + T r_t + B r_b + N r_n      n = T c_t + B c_b + N c_n
 * where (T, B, N) is the frame of the bound point: N = interpolated membrane normal, T = edge 0->1 projected.
 * Exact at rest; the Phong term keeps a coarse, stretched membrane round between its vertices, the residual
 * carries the fine detail. Same code for the skin (per vertex) and the droplets (per instance).
 */
import { cross, dot, float, int, normalize, vec3 } from 'three/tsl';
import type { Membrane } from '../sim/membrane.ts';

export const PHONG_ALPHA = 0.75;

export function bound(mem: Membrane, ids: any, bary: any, resid: any, ncoef: any) {
  const i0 = int(ids.x), i1 = int(ids.y), i2 = int(ids.z);
  const w0 = bary.x, w1 = bary.y, w2 = float(1).sub(bary.x).sub(bary.y);
  const x0 = mem.posAt(i0), x1 = mem.posAt(i1), x2 = mem.posAt(i2);
  const n0 = mem.nrmAt(i0), n1 = mem.nrmAt(i1), n2 = mem.nrmAt(i2);
  const flat = x0.mul(w0).add(x1.mul(w1)).add(x2.mul(w2));
  const proj = (x: any, n: any) => flat.sub(n.mul(dot(flat.sub(x), n)));
  const curved = flat.mul(1 - PHONG_ALPHA).add(proj(x0, n0).mul(w0).add(proj(x1, n1).mul(w1)).add(proj(x2, n2).mul(w2)).mul(PHONG_ALPHA));
  const N = normalize(n0.mul(w0).add(n1.mul(w1)).add(n2.mul(w2)));
  const e = x1.sub(x0);
  const T = normalize(e.sub(N.mul(dot(e, N))));
  const B = cross(N, T);
  const pos = curved.add(T.mul(resid.x)).add(B.mul(resid.y)).add(N.mul(resid.z));
  const nrm = normalize(T.mul(ncoef.x).add(B.mul(ncoef.y)).add(N.mul(ncoef.z)));
  /** rest-space point (flat interpolation of the original membrane) and how far the patch has moved */
  const restFlat = mem.orgAt(i0).mul(w0).add(mem.orgAt(i1).mul(w1)).add(mem.orgAt(i2).mul(w2));
  return { pos, nrm, T, B, N, restFlat, moved: flat.sub(restFlat).length() };
}

/** CPU mirror (QA): positions (n*3) and normals (n*3) of the membrane */
export function boundCPU(P: Float32Array, Nr: Float32Array, ids: ArrayLike<number>, bary: ArrayLike<number>, resid: ArrayLike<number>, k: number, out: number[]) {
  const a = ids[k * 3], b = ids[k * 3 + 1], c = ids[k * 3 + 2];
  const w = [bary[k * 2], bary[k * 2 + 1], 1 - bary[k * 2] - bary[k * 2 + 1]], v = [a, b, c];
  const flat = [0, 0, 0], curved = [0, 0, 0], nn = [0, 0, 0];
  for (let j = 0; j < 3; j++) for (let r = 0; r < 3; r++) flat[r] += P[v[j] * 3 + r] * w[j];
  for (let j = 0; j < 3; j++) {
    const x = v[j] * 3;
    const d = (flat[0] - P[x]) * Nr[x] + (flat[1] - P[x + 1]) * Nr[x + 1] + (flat[2] - P[x + 2]) * Nr[x + 2];
    for (let r = 0; r < 3; r++) { curved[r] += w[j] * (flat[r] - Nr[x + r] * d); nn[r] += Nr[x + r] * w[j]; }
  }
  for (let r = 0; r < 3; r++) curved[r] = flat[r] * (1 - PHONG_ALPHA) + curved[r] * PHONG_ALPHA;
  const nl = Math.hypot(nn[0], nn[1], nn[2]); for (let r = 0; r < 3; r++) nn[r] /= nl;
  const e = [P[b * 3] - P[a * 3], P[b * 3 + 1] - P[a * 3 + 1], P[b * 3 + 2] - P[a * 3 + 2]];
  const ed = e[0] * nn[0] + e[1] * nn[1] + e[2] * nn[2];
  const T = [e[0] - nn[0] * ed, e[1] - nn[1] * ed, e[2] - nn[2] * ed]; const tl = Math.hypot(T[0], T[1], T[2]); for (let r = 0; r < 3; r++) T[r] /= tl;
  const B = [nn[1] * T[2] - nn[2] * T[1], nn[2] * T[0] - nn[0] * T[2], nn[0] * T[1] - nn[1] * T[0]];
  for (let r = 0; r < 3; r++) out[r] = curved[r] + T[r] * resid[k * 3] + B[r] * resid[k * 3 + 1] + nn[r] * resid[k * 3 + 2];
  return out;
}

/** CPU vertex normals of the membrane from its ordered 1-rings (same as the GPU normal pass) */
export function ringNormalsCPU(P: Float32Array, adj: Float32Array, ring: number, out = new Float32Array(P.length)) {
  const n = P.length / 3;
  for (let i = 0; i < n; i++) {
    let nx = 0, ny = 0, nz = 0;
    const ids: number[] = [];
    for (let k = 0; k < ring; k++) { const j = adj[i * ring + k]; if (j >= 0) ids.push(j); }
    for (let k = 0; k < ids.length; k++) {
      const a = ids[k] * 3, b = ids[(k + 1) % ids.length] * 3, o = i * 3;
      const ex = P[a] - P[o], ey = P[a + 1] - P[o + 1], ez = P[a + 2] - P[o + 2];
      const fx = P[b] - P[o], fy = P[b + 1] - P[o + 1], fz = P[b + 2] - P[o + 2];
      nx += ey * fz - ez * fy; ny += ez * fx - ex * fz; nz += ex * fy - ey * fx;
    }
    const l = Math.hypot(nx, ny, nz) || 1;
    out[i * 3] = nx / l; out[i * 3 + 1] = ny / l; out[i * 3 + 2] = nz / l;
  }
  return out;
}

void vec3;
