/**
 * Water: 3600 beads + 44 running streaks as ONE InstancedMesh of real dome geometry.
 * Every instance is bound to the simulated membrane exactly like the skin under it (surface.ts): its base point and
 * its frame (normal + down-slope tangent, stored in the triangle frame) follow the membrane every frame.
 * so drops stay glued to the deforming skin and turn with it. Shading is a cheap custom lens model
 * (Fresnel reflection, sharp key highlight, dark near edge, bright refracted crescent) - no transmission pass.
 */
import { InstancedBufferAttribute, InstancedMesh, MeshBasicNodeMaterial, SphereGeometry } from 'three/webgpu';
import { attribute, bool, cameraPosition, clamp, cross, dot, float, instancedBufferAttribute, max, mix, normalize, pow, reflect, smoothstep, varying, vec3, vec4 } from 'three/tsl';
import type { Assets } from '../assets/load.ts';
import type { Membrane } from '../sim/membrane.ts';
import { bound } from './surface.ts';
import { photoMask } from './frame.ts';
import { skinUniforms } from './skin.ts';

export function createDroplets(assets: Assets, mem: Membrane) {
  const p = assets.pack;
  const count = p.dropSeed.length;
  // unit dome: z = up (normal), flattened later by the per-instance scale
  const geo = new SphereGeometry(1, 12, 6, 0, Math.PI * 2, 0, Math.PI / 2);
  geo.rotateX(Math.PI / 2);
  const inst = (arr: Float32Array, size: number, type: string): any => instancedBufferAttribute(new InstancedBufferAttribute(arr, size), type as any);
  // WebGPU allows 8 vertex buffers: geometry (2) + 5 packed per-instance buffers
  const A = new Float32Array(count * 4), Bv = new Float32Array(count * 4), C = new Float32Array(count * 4), D = new Float32Array(count * 4), E = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    A.set([p.dropNrm[i * 3], p.dropNrm[i * 3 + 1], p.dropNrm[i * 3 + 2], p.dropSeed[i]], i * 4);          // normal in the triangle frame, seed
    Bv.set([p.dropTan[i * 3], p.dropTan[i * 3 + 1], p.dropTan[i * 3 + 2], p.dropScale[i * 3]], i * 4);     // tangent in the frame, width
    C.set([p.dropScale[i * 3 + 1], p.dropScale[i * 3 + 2], p.dropBary[i * 2], p.dropBary[i * 2 + 1]], i * 4);
    D.set([p.dropTri[i * 3], p.dropTri[i * 3 + 1], p.dropTri[i * 3 + 2], p.dropResid[i * 3]], i * 4);
    E.set([p.dropResid[i * 3 + 1], p.dropResid[i * 3 + 2], 0, 0], i * 4);
  }
  const aA = inst(A, 4, 'vec4'), aB = inst(Bv, 4, 'vec4'), aC = inst(C, 4, 'vec4'), aD = inst(D, 4, 'vec4'), aE = inst(E, 4, 'vec4');
  const iSeed = aA.w, iS = vec3(aB.w, aC.x, aC.y.mul(1.45)) as any;   // beads stand proud of the skin
  const bd = bound(mem, aD.xyz, aC.zw, vec3(aD.w, aE.x, aE.y), aA.xyz);
  const center: any = bd.pos;
  const N: any = bd.nrm;
  const ft: any = normalize(bd.T.mul(aB.x).add(bd.B.mul(aB.y)).add(bd.N.mul(aB.z)));
  const T: any = normalize(ft.sub(N.mul(dot(N, ft))));
  const B: any = cross(N, T);
  const lp: any = attribute('position', 'vec3'), ln: any = attribute('normal', 'vec3');
  // sink the dome slightly so its rim never floats above the curved skin
  const world = center.add(B.mul(lp.x.mul(iS.x))).add(T.mul(lp.y.mul(iS.y))).add(N.mul(lp.z.mul(iS.z).sub(iS.z.mul(0.12))));
  const nrm = normalize(B.mul(ln.x.div(iS.x)).add(T.mul(ln.y.div(iS.y))).add(N.mul(ln.z.div(iS.z))));
  const vNrm: any = varying(nrm), vBase: any = varying(N), vPos: any = varying(world), vLocal: any = varying(lp), vSeed: any = varying(iSeed);

  const mat = new MeshBasicNodeMaterial();
  mat.positionNode = world;
  const n: any = normalize(vNrm), nb: any = normalize(vBase);
  const V: any = normalize(cameraPosition.sub(vPos));
  const L: any = normalize(skinUniforms.keyDir);
  const ndv = clamp(dot(n, V), 0.0, 1.0);
  const fres = float(0.02).add(float(0.98).mul(pow(float(1).sub(ndv), 5.0)));
  const R: any = reflect(V.negate(), n);
  const spec = pow(max(dot(R, L), 0.0), 70.0).mul(2.6).add(pow(max(dot(R, L), 0.0), 12.0).mul(0.22));
  // lens: the side of the dome facing away from the light gathers it (bright crescent), the near side goes dark
  const lit = clamp(dot(nb, L).mul(0.75).add(0.3), 0.0, 1.0);                    // is the skin under the drop lit at all?
  const side = dot(normalize(n.sub(nb.mul(dot(n, nb))).add(1e-5)), normalize(L.sub(nb.mul(dot(L, nb))).add(1e-5)));
  const rimness = float(1).sub(clamp(dot(n, nb), 0.0, 1.0));
  const crescent = smoothstep(-0.1, 0.9, side.negate()).mul(smoothstep(0.05, 0.55, rimness));
  const darkEdge = smoothstep(0.0, 0.9, side).mul(smoothstep(0.12, 0.7, rimness));
  const env = mix(vec3(0.20, 0.17, 0.14), vec3(0.95, 0.85, 0.72), smoothstep(-0.2, 0.9, dot(R, L)));
  const warm = vec3(1.0, 0.82, 0.5);
  const col = warm.mul(crescent.mul(0.9)).mul(lit).add(env.mul(fres).mul(lit.mul(0.8).add(0.2))).add(vec3(1.0, 0.96, 0.9).mul(spec).mul(lit));
  const alpha = clamp(crescent.mul(0.5).mul(lit).add(darkEdge.mul(0.55)).add(fres.mul(0.6)).add(spec).add(0.10), 0.0, 1.0);
  mat.colorNode = vec4(col.div(max(alpha, 0.05)).mul(float(0.9).add(vSeed.mul(0.2))), 1.0);
  mat.opacityNode = alpha.mul(smoothstep(-0.02, 0.1, vLocal.z));
  mat.maskNode = photoMask(world, bd.moved);
  mat.maskShadowNode = bool(true);
  mat.transparent = true;
  mat.depthWrite = false;

  const mesh = new InstancedMesh(geo, mat, count);
  mesh.frustumCulled = false;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  mesh.renderOrder = 5;
  return { mesh, count };
}
