/**
 * Skin: the fine render mesh (100 k vertices) riding on the simulated membrane (surface.ts):
 *   position = Phong-tessellated membrane triangle + baked residual, normal = membrane frame x baked detail normal.
 * Wrinkles and stretch folds are therefore real: they come from the membrane's own buckling (Sheen-style), shaded with
 * the membrane normals recomputed every frame. A stretched wall gets a little lighter / glossier (thinner rubber).
 */
import { type BufferGeometry, BufferAttribute, Color, Mesh, MeshBasicNodeMaterial, MeshPhysicalNodeMaterial } from 'three/webgpu';
import {
  attribute, bool, clamp, color, dot, float, max, mix, mx_noise_float, normalize, select, smoothstep,
  transformNormalToView, uniform, varying, vec3,
} from 'three/tsl';
import type { Assets } from '../assets/load.ts';
import type { Membrane } from '../sim/membrane.ts';
import { bound } from './surface.ts';
import { photoMask } from './frame.ts';

export const skinUniforms = {
  thinning: uniform(1.0),
  debugStrain: uniform(0.0),
  keyDir: uniform(vec3(-0.5, 0.6, 0.62)),
  keyStrength: uniform(1.0),
  sss: uniform(1.0),
};

export function createSkin(assets: Assets, mem: Membrane) {
  const geo: BufferGeometry = assets.skin;
  const P = assets.pack;
  geo.setAttribute('simTri', new BufferAttribute(P.skinTri as Float32Array, 3));
  geo.setAttribute('simBary', new BufferAttribute(P.skinBary as Float32Array, 2));
  geo.setAttribute('simResid', new BufferAttribute(P.skinResid as Float32Array, 3));
  geo.setAttribute('simNrm', new BufferAttribute(P.skinNrm as Float32Array, 3));
  geo.deleteAttribute('normal');
  geo.computeBoundingSphere();
  geo.boundingSphere!.radius *= 3;

  const restP = attribute('position', 'vec3');
  const label = attribute('color', 'vec4');                  // r belly, g hand, b eyelid zone, a mouth line
  const b = bound(mem, attribute('simTri', 'vec3'), attribute('simBary', 'vec2'), attribute('simResid', 'vec3'), attribute('simNrm', 'vec3'));
  const position = b.pos;
  const vP: any = varying(restP), vL: any = varying(label), vNrm: any = varying(b.nrm);
  // membrane stretch: how far the patch moved relative to its neighbourhood is not available per vertex, so use the
  // displacement as a proxy for "pulled thin" (a pulled lobe is stretched along its whole length)
  const vMoved: any = varying(b.moved);
  const thin = smoothstep(0.03, 0.35, vMoved).mul(skinUniforms.thinning);

  const pore = vec3(mx_noise_float(vP.mul(260.0)), mx_noise_float(vP.mul(260.0).add(17.3)), mx_noise_float(vP.mul(260.0).add(41.9))).mul(float(0.05).mul(float(1).sub(thin.mul(0.8))));
  const nWorld = normalize(normalize(vNrm).add(pore));

  const skinCol = color(new Color('#d6a640')), bellyCol = color(new Color('#cbb393')), handCol = color(new Color('#a07c4e'));
  const mottled = mx_noise_float(vP.mul(4.0)).mul(0.05).add(mx_noise_float(vP.mul(31.0)).mul(0.03)).add(mx_noise_float(vP.mul(420.0)).mul(0.09).mul(float(1).sub(thin.mul(0.7))));
  const base = mix(mix(skinCol, bellyCol, smoothstep(0.35, 0.65, vL.x)), handCol, smoothstep(0.08, 0.92, vL.y));
  const lip = smoothstep(0.15, 0.85, vL.w);
  const albedo0 = base.mul(float(1).add(mottled)).mul(float(1).sub(vL.z.mul(0.07)));
  const albedo1 = mix(albedo0, albedo0.mul(vec3(0.16, 0.10, 0.06)), lip);
  const albedo = mix(albedo1, albedo1.mul(vec3(1.10, 1.07, 1.0)).add(vec3(0.03, 0.025, 0.0)), thin.mul(0.7));
  const strainView = mix(vec3(0.08, 0.1, 0.5), vec3(1.0, 0.25, 0.05), thin);

  const mat = new MeshPhysicalNodeMaterial();
  mat.positionNode = position;
  mat.normalNode = transformNormalToView(nWorld);
  mat.colorNode = mix(albedo, strainView, skinUniforms.debugStrain);
  mat.roughnessNode = clamp(float(0.66).add(mx_noise_float(vP.mul(70.0)).mul(0.07)).add(vL.y.mul(0.16)).add(vL.x.mul(0.06)).sub(thin.mul(0.18)).add(lip.mul(0.1)), 0.2, 1.0);
  mat.metalness = 0;
  mat.clearcoat = 0.09;
  mat.clearcoatRoughness = 0.32;
  mat.specularIntensity = 0.38;
  const ndl = dot(nWorld, normalize(skinUniforms.keyDir));
  const wrap = clamp(ndl.add(0.42).div(1.42), 0.0, 1.0);
  const bleed = wrap.sub(clamp(ndl, 0.0, 1.0)).add(thin.mul(0.12).mul(clamp(ndl.add(0.2), 0.0, 1.0)));
  mat.emissiveNode = albedo.mul(vec3(1.0, 0.36, 0.12)).mul(bleed).mul(skinUniforms.keyStrength).mul(skinUniforms.sss).mul(0.5).mul(float(1).sub(vL.y.mul(0.6)));
  mat.maskNode = photoMask(position, b.moved);
  mat.maskShadowNode = bool(true);

  const mesh = new Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.frustumCulled = false;

  const idMat = new MeshBasicNodeMaterial();
  idMat.positionNode = position;
  idMat.colorNode = select(vL.y.greaterThan(0.5), vec3(0, 1, 0), select(vL.x.greaterThan(0.5), vec3(0, 0, 1), vec3(1, 0, 0)));
  idMat.maskNode = photoMask(position, b.moved);
  idMat.toneMapped = false;
  void max;
  return { mesh, material: mat, idMaterial: idMat };
}
