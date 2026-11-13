/**
 * The photograph is a window, but the character can be pulled out of it.
 * The canvas covers the whole cover; the camera keeps the photo's frustum for the photo rectangle
 * (PerspectiveCamera.setViewOffset) and simply sees more around it. A fragment is drawn when
 *   - it lies inside the photo rectangle (the photograph itself), or
 *   - the skin under it has been pulled away from its rest place: visibility fades in with the displacement
 *     (dithered between PULL_FADE[0] and PULL_FADE[1]), so a lobe dragged across the border keeps a soft, organic
 *     outline on the paper - no straight cut along the frame - while everything at rest stays cropped.
 */
import { Matrix4, Vector4, type PerspectiveCamera } from 'three/webgpu';
import { float, fract, screenCoordinate, smoothstep, uniform, varying, vec4 } from 'three/tsl';

export const PULL_FADE = [0.025, 0.04];

export const frameUniforms = {
  viewProj: uniform(new Matrix4()),
  /** photo rectangle in canvas NDC: (x0, y0, x1, y1) */
  rect: uniform(new Vector4(-1, -1, 1, 1)),
  /** photo rectangle in screenUV (y down): (u0, v0, u1, v1) */
  rectUV: uniform(new Vector4(0, 0, 1, 1)),
};

/** canvas covers the cover square; photo = rectangle in cover fractions */
export function fitCamera(camera: PerspectiveCamera, cw: number, ch: number, photo: { x0: number; x1: number; y0: number; y1: number }) {
  const pw = (photo.x1 - photo.x0) * cw, ph = (photo.y1 - photo.y0) * ch;
  camera.setViewOffset(pw, ph, -photo.x0 * cw, -photo.y0 * ch, cw, ch);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  frameUniforms.viewProj.value.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  frameUniforms.rect.value.set(photo.x0 * 2 - 1, 1 - photo.y1 * 2, photo.x1 * 2 - 1, 1 - photo.y0 * 2);
  frameUniforms.rectUV.value.set(photo.x0, photo.y0, photo.x1, photo.y1);
}

const inside = (c: any) => {
  const r: any = frameUniforms.rect;
  const n = c.xy.div(c.w);
  return n.x.greaterThanEqual(r.x).and(n.x.lessThanEqual(r.z)).and(n.y.greaterThanEqual(r.y)).and(n.y.lessThanEqual(r.w)).and(c.w.greaterThan(0));
};

/** interleaved gradient noise (Jimenez): a stable per-pixel threshold for the dithered fade */
const ign = () => fract(float(52.9829189).mul(fract(screenCoordinate.x.mul(0.06711056).add(screenCoordinate.y.mul(0.00583715)))));

/** mask node: `current` = deformed position, `moved` = how far that skin is from its rest place (null: wall) */
export function photoMask(current: any, moved: any | null) {
  const vc: any = varying(frameUniforms.viewProj.mul(vec4(current, 1.0)));
  if (!moved) return inside(vc);
  const mv: any = varying(moved);
  return inside(vc).or(smoothstep(PULL_FADE[0], PULL_FADE[1], mv).greaterThan(ign()));
}
