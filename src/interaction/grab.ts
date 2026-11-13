/**
 * Pointer -> membrane grab (Sheen's mouse.js):
 *  1. ray-cast the CURRENT deformed membrane (read-back positions, visible part only),
 *  2. build a camera-facing drag plane through the hit (Sheen: plane.normal = camera direction),
 *  3. pointer motion projected on that plane is the target of the grabbed patch.
 * One grab per pointerId (multi-touch). Released on pointerup / pointercancel / lostpointercapture /
 * window blur / visibilitychange.
 */
import { Plane, Raycaster, Vector2, Vector3, type PerspectiveCamera } from 'three/webgpu';
import type { Membrane } from '../sim/membrane.ts';
import { PULL_FADE } from '../render/frame.ts';

/** depth (world z) of the point the eyes look at: the eyes sit at z 0.19-0.31, the camera at 3.0 */
const LOOK_PLANE_Z = 0.8;

export class GrabController {
  private ray = new Raycaster();
  private ndc = new Vector2();
  private planes = new Map<number, Plane>();
  readonly look = new Vector3();
  hasLook = false;
  /** world points for the debug overlay: pointerId -> [hit, target] */
  readonly debug = new Map<number, [Vector3, Vector3]>();
  private dom: HTMLElement; private camera: PerspectiveCamera; private driver: Membrane; private tris: Uint32Array;
  private photo: HTMLElement;
  private rest: Float32Array;
  private restNdc = new Vector3();

  constructor(dom: HTMLElement, camera: PerspectiveCamera, driver: Membrane, surfaceTris: Uint32Array, photo: HTMLElement) {
    this.dom = dom; this.camera = camera; this.driver = driver; this.tris = surfaceTris; this.photo = photo;
    this.rest = driver.rest;
    dom.addEventListener('pointerdown', this.onDown);
    dom.addEventListener('pointermove', this.onMove);
    dom.addEventListener('pointerup', this.onUp);
    dom.addEventListener('pointercancel', this.onUp);
    dom.addEventListener('lostpointercapture', this.onUp);
    dom.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('pointermove', this.onHover, { passive: true });
    window.addEventListener('blur', this.releaseAll);
    document.documentElement.addEventListener('pointerleave', () => { this.hasLook = false; });
    document.addEventListener('visibilitychange', () => { if (document.hidden) this.releaseAll(); });
  }

  private toNdc(e: { clientX: number; clientY: number }) {
    const r = this.dom.getBoundingClientRect();
    return this.ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  }

  /** is a canvas-NDC point inside the photo rectangle? */
  private inPhoto(x: number, y: number) {
    const r = this.dom.getBoundingClientRect(), ph = this.photo.getBoundingClientRect();
    const cx = r.left + (x * 0.5 + 0.5) * r.width, cy = r.top + (-y * 0.5 + 0.5) * r.height;
    return cx >= ph.left && cx <= ph.right && cy >= ph.top && cy <= ph.bottom;
  }

  /**
   * nearest VISIBLE intersection of the pointer ray with the deformed surface (Moller-Trumbore over the cage skin).
   * Same rule as the renderer (frame.ts): inside the photo everything is visible; outside it only skin whose rest
   * position belongs to the photo.
   */
  pick(ndc: Vector2): Vector3 | null { return this.pickHit(ndc)?.point ?? null; }

  pickHit(ndc: Vector2): { point: Vector3; tri: number; u: number; v: number } | null {
    this.ray.setFromCamera(ndc, this.camera);
    const o = this.ray.ray.origin, d = this.ray.ray.direction;
    const p = this.driver.positions, t = this.tris, X = this.rest;
    const inside = this.inPhoto(ndc.x, ndc.y);
    let best = Infinity, bt = -1, bu = 0, bv = 0;
    for (let i = 0; i < t.length; i += 3) {
      const a = t[i] * 3, b = t[i + 1] * 3, c = t[i + 2] * 3;
      const e1x = p[b] - p[a], e1y = p[b + 1] - p[a + 1], e1z = p[b + 2] - p[a + 2];
      const e2x = p[c] - p[a], e2y = p[c + 1] - p[a + 1], e2z = p[c + 2] - p[a + 2];
      const px = d.y * e2z - d.z * e2y, py = d.z * e2x - d.x * e2z, pz = d.x * e2y - d.y * e2x;
      const det = e1x * px + e1y * py + e1z * pz;
      if (det < 1e-12) continue;                               // back-facing or parallel
      const tx = o.x - p[a], ty = o.y - p[a + 1], tz = o.z - p[a + 2];
      const u = (tx * px + ty * py + tz * pz) / det;
      if (u < 0 || u > 1) continue;
      const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
      const v = (d.x * qx + d.y * qy + d.z * qz) / det;
      if (v < 0 || u + v > 1) continue;
      const dist = (e2x * qx + e2y * qy + e2z * qz) / det;
      if (!(dist > 0 && dist < best)) continue;
      if (!inside) {                                           // outside the photo: only skin that belongs to it
        const w0 = 1 - u - v;
        const rx = X[a] * w0 + X[b] * u + X[c] * v, ry = X[a + 1] * w0 + X[b + 1] * u + X[c + 1] * v, rz = X[a + 2] * w0 + X[b + 2] * u + X[c + 2] * v;
        const moved = Math.hypot(o.x + d.x * dist - rx, o.y + d.y * dist - ry, o.z + d.z * dist - rz);
        if (moved <= 0.5 * (PULL_FADE[0] + PULL_FADE[1])) continue;   // same rule as the renderer (frame.ts)
      }
      best = dist; bt = i / 3; bu = u; bv = v;
    }
    return bt >= 0 ? { point: o.clone().addScaledVector(d, best), tri: bt, u: bu, v: bv } : null;
  }

  private onDown = (e: PointerEvent) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    const h = this.pickHit(this.toNdc(e));
    if (!h) return;
    const hit = h.point;
    if (!this.driver.beginGrab(e.pointerId, h.tri, h.u, h.v, [hit.x, hit.y, hit.z])) return;
    const n = new Vector3();
    this.camera.getWorldDirection(n).negate();
    this.planes.set(e.pointerId, new Plane().setFromNormalAndCoplanarPoint(n, hit));
    this.debug.set(e.pointerId, [hit.clone(), hit.clone()]);
    try { this.dom.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
    this.dom.classList.add('grabbing');
    e.preventDefault();
  };

  private onMove = (e: PointerEvent) => {
    const plane = this.planes.get(e.pointerId);
    if (!plane) return;
    this.ray.setFromCamera(this.toNdc(e), this.camera);
    const q = new Vector3();
    if (this.ray.ray.intersectPlane(plane, q)) {
      this.driver.moveGrab(e.pointerId, [q.x, q.y, q.z]);
      this.debug.get(e.pointerId)?.[1].copy(q);
    }
  };

  private onUp = (e: PointerEvent) => {
    if (!this.planes.delete(e.pointerId)) return;
    this.driver.endGrab(e.pointerId);
    this.debug.delete(e.pointerId);
    try { this.dom.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    if (!this.planes.size) this.dom.classList.remove('grabbing');
  };

  releaseAll = () => {
    this.planes.clear();
    this.debug.clear();
    this.driver.endAllGrabs();
    this.dom.classList.remove('grabbing');
  };

  get count() { return this.planes.size; }

  private onHover = (e: PointerEvent) => {
    if (!this.planes.size) this.dom.classList.toggle('over', this.pick(this.toNdc(e)) !== null);
    // the eyes follow the pointer anywhere on the page (the gaze cone in eyes.ts limits how far they turn);
    // they return to the cover gaze only when the pointer leaves the window
    this.ray.setFromCamera(this.toNdc(e), this.camera);
    const plane = new Plane(new Vector3(0, 0, 1), -LOOK_PLANE_Z);   // just in front of the face: the eyes aim at the pointer itself
    if (this.ray.ray.intersectPlane(plane, this.look)) this.hasLook = true;
  };
}
