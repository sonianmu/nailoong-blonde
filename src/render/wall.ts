/**
 * One tiled bathroom wall, fully procedural and continuous (a single plane much larger than the frame,
 * tile pattern evaluated per fragment in wall space - nothing can open up when the character deforms).
 * Measured from figure A: square tiles, pitch 0.194 x photo width, thin grout slightly DARKER than the
 * faces, lines rising ~3 deg towards the right (the wall is not perfectly parallel to the sensor).
 */
import { Mesh, MeshPhysicalNodeMaterial, PlaneGeometry, type PerspectiveCamera, Raycaster, Vector2, Plane, Vector3, Color } from 'three/webgpu';
import { abs, clamp, color, float, floor, fract, hash, max, mix, mx_noise_float, normalize, positionLocal, positionWorld, sign, smoothstep, transformNormalToView, uniform, vec2, vec3 } from 'three/tsl';
import { photoMask } from './frame.ts';
import type { SceneJson } from '../assets/load.ts';

export const WALL_YAW = -0.052;   // rad

export function createWall(scene: SceneJson, camera: PerspectiveCamera) {
  const z = scene.wall.z;
  const mesh = new Mesh(new PlaneGeometry(9, 9, 1, 1));
  mesh.position.set(0, 0, z);
  mesh.rotation.y = WALL_YAW;
  mesh.updateMatrixWorld(true);
  // wall-space position of a photo point
  const ray = new Raycaster();
  const plane = new Plane().setFromNormalAndCoplanarPoint(new Vector3(0, 0, 1).applyQuaternion(mesh.quaternion), mesh.position);
  const onWall = (u: number, v: number) => {
    ray.setFromCamera(new Vector2(u * 2 - 1, 1 - v * 2), camera);
    const p = ray.ray.intersectPlane(plane, new Vector3())!;
    return mesh.worldToLocal(p);
  };
  const aspect = scene.camera.aspect;
  // pitch: distance between two measured vertical grout lines on the wall
  const gx = scene.wall.groutXPhoto.filter((x) => x > 0.1 && x < 0.9);
  const a = onWall(gx[0], 0.08), b = onWall(gx[gx.length - 1], 0.08);
  const pitch = (b.x - a.x) / (gx.length - 1);
  const gy = scene.wall.groutYLeftPhotoWidths.filter((y) => y > 0.1)[0];
  const c = onWall(0.055, gy * aspect);
  const offset = uniform(new Vector2(a.x, c.y));
  const uPitch = uniform(pitch);

  const pl: any = positionLocal;
  const p = pl.xy.sub(offset).div(uPitch);
  const cell: any = floor(p);
  const f: any = fract(p);
  const edge = vec2(abs(f.x.sub(0.5)), abs(f.y.sub(0.5)));      // 0 centre .. 0.5 at the grout line
  const gw = float(0.007);                                       // grout half width (in tiles)
  const groove = (d: any) => smoothstep(float(0.5).sub(gw.mul(2.6)), float(0.5).sub(gw.mul(0.5)), d);
  const gxm = groove(edge.x), gym = groove(edge.y);
  const grout = max(gxm, gym);
  const rnd: any = hash(cell.x.mul(13.7).add(cell.y.mul(71.3)).add(5.0));
  const tone = float(1).add(rnd.sub(0.5).mul(0.05));            // per-tile variation <= 3 %
  const cloud = mx_noise_float(vec3(pl.xy.mul(2.3), 1.7) as any).mul(0.035).add(mx_noise_float(vec3(pl.xy.mul(11.0), 4.2) as any).mul(0.02));
  const tile = color(new Color('#98928c')).mul(tone).mul(float(1).add(cloud));
  const groutCol = color(new Color('#6c665f'));
  const mat = new MeshPhysicalNodeMaterial();
  mat.colorNode = mix(tile, groutCol, grout.mul(0.55));
  mat.roughnessNode = clamp(float(0.46).add(rnd.sub(0.5).mul(0.1)).add(grout.mul(0.4)).add(cloud.mul(1.5)), 0.2, 1.0);
  mat.metalness = 0;
  mat.specularIntensity = 0.35;
  // recessed grout: tilt the normal on the bevel
  const bev = (d: any, s: any) => smoothstep(float(0.5).sub(gw.mul(3.2)), float(0.5).sub(gw.mul(1.2)), d).mul(float(1).sub(smoothstep(float(0.5).sub(gw.mul(1.0)), float(0.5), d))).mul(s);
  const nx: any = bev(edge.x, sign(f.x.sub(0.5))), ny: any = bev(edge.y, sign(f.y.sub(0.5)));
  mat.normalNode = transformNormalToView(normalize(vec3(nx.mul(0.55), ny.mul(0.55), 1.0) as any));
  mat.maskNode = photoMask(positionWorld, null);     // the wall only exists inside the photograph
  mesh.material = mat;
  mesh.receiveShadow = true;
  return { mesh, pitch, offset: [a.x, c.y] };
}
