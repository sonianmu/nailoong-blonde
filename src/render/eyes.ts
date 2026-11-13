/**
 * Eyeballs that ride on their eyelids.
 * Each ball is attached to a ring of lid vertices of the simulated membrane (tools/model/membrane.py eye_rings):
 *   centre  = ring centroid + R (c_rest - centroid_rest) s,   R = frame of three ring anchors (up, side+, side-)
 *   size    unchanged: the socket around it is held rigid by the membrane's socket shape matching
 * The ball is rigid (always a ball) but moves, turns and grows a little with its lids, so it cannot leave the socket;
 * the lids are pushed off the ball by the membrane's eyeball collision proxy.
 * Gaze: iris / pupil are drawn around a rest-space gaze axis. Over the photo the ball turns to look straight at the pointer
 * (clamped to MAX_GAZE_DEG around the socket normal); elsewhere it returns to the cover gaze.
 */
import { Color, Matrix4, Mesh, MeshBasicNodeMaterial, MeshPhysicalNodeMaterial, SphereGeometry, Vector3 } from 'three/webgpu';
import {
  attribute, bool, clamp, color, cross, dot, float, int, mix, normalize, pow, smoothstep, transformNormalToView, uniform, varying, vec3,
} from 'three/tsl';
import type { EyeJson } from '../assets/load.ts';
import type { Membrane } from '../sim/membrane.ts';
import { photoMask } from './frame.ts';

/** gaze cone around the eye's forward axis (the socket normal); the cover gaze (looking down) lies inside it */
export const MAX_GAZE_DEG = 45;
/** how fast the eyes catch up with the pointer (1/s) */
const GAZE_RATE = 16;

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const nrm = (a: V3): V3 => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const crs = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** ring frame from positions (CPU): centroid, X, Y, Z, spacing */
export function ringFrame(P: Float32Array, ring: number[], anchors: number[]) {
  const c: V3 = [0, 0, 0];
  for (const i of ring) { c[0] += P[i * 3]; c[1] += P[i * 3 + 1]; c[2] += P[i * 3 + 2]; }
  c[0] /= ring.length; c[1] /= ring.length; c[2] /= ring.length;
  const g = (i: number): V3 => [P[i * 3], P[i * 3 + 1], P[i * 3 + 2]];
  const [up, sp, sm] = anchors.map(g);
  const X = nrm(sub(sp, sm));
  const Z = nrm(crs(X, sub(up, c)));
  const Y = crs(Z, X);
  const d = sub(sp, sm);
  return { c, X, Y, Z, spacing: Math.hypot(d[0], d[1], d[2]) };
}

export class Eyes {
  readonly meshes: Mesh[] = [];
  readonly idMaterial: MeshBasicNodeMaterial[] = [];
  readonly materials: MeshPhysicalNodeMaterial[] = [];
  readonly gaze: Vector3[] = [];
  readonly gazeOffsetDeg: number[] = [];
  readonly centers: Vector3[] = [];
  private restFrame: Matrix4[] = [];
  private restRing: ReturnType<typeof ringFrame>[] = [];
  private offset: Vector3[] = [];
  private gazeRest: any[] = [];
  private eyes: EyeJson[];

  constructor(eyes: EyeJson[], mem: Membrane) {
    this.eyes = eyes;
    eyes.forEach((e) => {
      const rr = ringFrame(mem.rest, e.ring, e.anchors);
      this.restRing.push(rr);
      // ---- current ring frame on the GPU (ids are constants: the loop unrolls)
      const centroid = (() => { let acc: any = vec3(0); for (const i of e.ring) acc = acc.add(mem.posAt(int(i))); return acc.div(e.ring.length); })();
      const up = mem.posAt(int(e.anchors[0])), sp = mem.posAt(int(e.anchors[1])), sm = mem.posAt(int(e.anchors[2]));
      const X = normalize(sp.sub(sm));
      const Z = normalize(cross(X, up.sub(centroid)));
      const Y = cross(Z, X);
      const s = float(1);
      const restP: any = attribute('position', 'vec3'), restN: any = attribute('normal', 'vec3');
      const C = vec3(...rr.c), X0 = vec3(...rr.X), Y0 = vec3(...rr.Y), Z0 = vec3(...rr.Z);
      const d = restP.sub(C);
      const local = vec3(dot(d, X0), dot(d, Y0), dot(d, Z0));
      const pos = centroid.add(X.mul(local.x).add(Y.mul(local.y)).add(Z.mul(local.z)).mul(s));
      const nl = vec3(dot(restN, X0), dot(restN, Y0), dot(restN, Z0));
      const nWorld = normalize(X.mul(nl.x).add(Y.mul(nl.y)).add(Z.mul(nl.z)));

      const geo = new SphereGeometry(e.radius, 72, 48);
      geo.translate(e.center[0], e.center[1], e.center[2]);    // rest space
      const gz = uniform(new Vector3(...e.gaze));
      this.gazeRest.push(gz);
      const ctr = vec3(e.center[0], e.center[1], e.center[2]);
      const dir: any = varying(normalize(restP.sub(ctr)));
      const cz = dot(dir, normalize(gz));
      // figure A: pale matte sage ball, darker towards the limbus, black pupil
      const sage = color(new Color('#93a27c')), deep = color(new Color('#5d6b48'));
      const pupil = smoothstep(0.80, 0.83, cz);
      const iris = mix(deep, sage, smoothstep(-0.6, 0.35, cz)).mul(mix(float(1), float(0.82), smoothstep(0.62, 0.79, cz)));
      const moved = centroid.sub(C).length();
      const mat = new MeshPhysicalNodeMaterial();
      mat.positionNode = pos;
      mat.normalNode = transformNormalToView(nWorld);
      mat.colorNode = mix(iris, vec3(0.012, 0.01, 0.008), pupil);
      mat.roughnessNode = mix(float(0.42), float(0.2), pupil);
      mat.clearcoat = 0.6;
      mat.clearcoatRoughness = 0.2;
      mat.specularIntensity = 0.6;
      mat.maskNode = photoMask(pos, moved);
      mat.maskShadowNode = bool(true);
      const idm = new MeshBasicNodeMaterial();
      idm.positionNode = pos;
      idm.colorNode = vec3(1, 1, 0);
      idm.maskNode = photoMask(pos, moved);
      idm.toneMapped = false;
      this.idMaterial.push(idm);
      this.materials.push(mat);
      const m = new Mesh(geo, mat);
      m.castShadow = false;
      m.receiveShadow = true;
      m.frustumCulled = false;
      this.meshes.push(m);

      const fr = this.frame(new Vector3(...rr.X), new Vector3(...rr.Y), new Vector3(...rr.Z));
      this.restFrame.push(fr);
      this.offset.push(new Vector3(...e.gaze).applyMatrix4(fr.clone().invert()).normalize());
      this.gaze.push(new Vector3(...e.gaze));
      this.gazeOffsetDeg.push(0);
      this.centers.push(new Vector3(...e.center));
    });
  }

  private frame(X: Vector3, Y: Vector3, Z: Vector3) { return new Matrix4().makeBasis(X, Y, Z); }

  /** centre of eye i from (read-back) membrane positions */
  centerFrom(P: Float32Array, i: number) {
    const e = this.eyes[i], r0 = this.restRing[i], r = ringFrame(P, e.ring, e.anchors);
    const s = 1;
    const d = sub(e.center as V3, r0.c);
    const l = [d[0] * r0.X[0] + d[1] * r0.X[1] + d[2] * r0.X[2], d[0] * r0.Y[0] + d[1] * r0.Y[1] + d[2] * r0.Y[2], d[0] * r0.Z[0] + d[1] * r0.Z[1] + d[2] * r0.Z[2]];
    const c = new Vector3(...r.c);
    c.addScaledVector(new Vector3(...r.X), l[0] * s).addScaledVector(new Vector3(...r.Y), l[1] * s).addScaledVector(new Vector3(...r.Z), l[2] * s);
    return { c, frame: this.frame(new Vector3(...r.X), new Vector3(...r.Y), new Vector3(...r.Z)), scale: s };
  }

  /** look: world-space point the pointer indicates (or null); P: read-back membrane positions */
  update(P: Float32Array, look: Vector3 | null, dt: number) {
    const maxAng = (MAX_GAZE_DEG * Math.PI) / 180;
    this.eyes.forEach((e, i) => {
      const { c, frame } = this.centerFrom(P, i);
      const restInv = this.restFrame[i].clone().invert();
      const restGaze = new Vector3(...e.gaze).applyMatrix4(restInv).normalize();
      const fwd = new Vector3(...e.normal).applyMatrix4(restInv).normalize();
      let want = restGaze.clone();
      if (look) {
        // look straight at the pointer: direction from the (current) ball centre, in the lid-ring frame
        want = look.clone().sub(c).normalize().applyMatrix4(frame.clone().invert()).normalize();
        const ang = fwd.angleTo(want);
        if (ang > maxAng) want = fwd.clone().applyAxisAngle(new Vector3().crossVectors(fwd, want).normalize(), maxAng);
      }
      // rotate the current gaze towards the target (constant-rate slerp, no lerp shortcut through the ball)
      const k = 1 - Math.exp(-dt * GAZE_RATE);
      const ang = this.offset[i].angleTo(want);
      if (ang > 1e-6) {
        const axis = new Vector3().crossVectors(this.offset[i], want);
        if (axis.lengthSq() < 1e-12) this.offset[i].copy(want);
        else this.offset[i].applyAxisAngle(axis.normalize(), ang * k).normalize();
      }
      this.gazeOffsetDeg[i] = (this.offset[i].angleTo(fwd) * 180) / Math.PI;
      this.gaze[i].copy(this.offset[i]).applyMatrix4(frame).normalize();
      this.gazeRest[i].value.copy(this.offset[i]).applyMatrix4(this.restFrame[i]).normalize();
      this.centers[i].copy(c);
    });
  }
}
