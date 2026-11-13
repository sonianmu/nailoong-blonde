/** Asset loading: scene.json (camera, landmarks, eyes), pack.bin (membrane, bindings, droplets), skin.glb (render mesh). */
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import type { BufferGeometry, Mesh } from 'three';
import type { MembraneData } from '../sim/membrane.ts';

export interface EyeJson { center: number[]; normal: number[]; up: number[]; radius: number; gaze: number[]; ring: number[]; anchors: number[] }
export interface SceneJson {
  camera: { position: number[]; target: number[]; fovY: number; aspect: number };
  H: number; gap: number; pinY: number;
  eyes: EyeJson[];
  landmarks3d: Record<string, number[]>;
  zones: Record<string, number[]>;
  counts: Record<string, number>;
  wall: { z: number; tilePitchPhoto: number; groutXPhoto: number[]; groutYLeftPhotoWidths: number[] };
}
type Typed = Float32Array | Uint32Array | Uint16Array | Uint8Array;
const TYPES: Record<string, { new(b: ArrayBuffer, o: number, l: number): Typed }> = { float32: Float32Array, uint32: Uint32Array, uint16: Uint16Array, uint8: Uint8Array };

export interface Assets { scene: SceneJson; pack: Record<string, any>; membrane: MembraneData; skin: BufferGeometry }

export async function loadAssets(base = './assets/'): Promise<Assets> {
  const [scene, index, bin, gltf] = await Promise.all([
    fetch(base + 'scene.json').then((r) => r.json() as Promise<SceneJson>),
    fetch(base + 'pack.json').then((r) => r.json() as Promise<Record<string, { dtype: string; shape: number[]; offset: number }>>),
    fetch(base + 'pack.bin').then((r) => r.arrayBuffer()),
    new GLTFLoader().loadAsync(base + 'skin.glb'),
  ]);
  const pack: Record<string, any> = {};
  for (const [name, e] of Object.entries(index)) pack[name] = new TYPES[e.dtype](bin, e.offset, e.shape.reduce((a, b) => a * b, 1));
  let skin: BufferGeometry | null = null;
  gltf.scene.traverse((o) => { if ((o as Mesh).isMesh && !skin) skin = (o as Mesh).geometry; });
  if (!skin) throw new Error('skin.glb contains no mesh');
  const g = skin as BufferGeometry;
  if (g.attributes.position.count * 3 !== pack.skinTri.length) throw new Error('skin.glb vertex count does not match the binding data');
  const membrane: MembraneData = {
    rest: pack.simRest, adj: pack.simAdj, dist: pack.simDist, restore: pack.simRestore, stiff: pack.simStiff, pinned: pack.simPinned,
    group: pack.simGroup, collideBody: pack.simCollideBody, collideEye: pack.simCollideEye, tris: pack.simTris,
    proxyIds: pack.proxyIds, proxyOffset: pack.proxyOffset, proxyRadius: pack.proxyRadius, proxyType: pack.proxyType, proxyRest: pack.proxyRest,
    socketW: pack.simSocketW, socketEye: pack.simSocketEye, eyes: (scene as any).rigid,
  };
  return { scene, pack, membrane, skin: g };
}
