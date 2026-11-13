/**
 * Renderer, camera, light rig and post chain.
 * WebGPURenderer (WebGPU backend, or its WebGL2 backend with ?webgl / when WebGPU is unavailable).
 * Light: figure A is lit by a warm soft key from the upper right-front (the right flank of the head and the
 * near shoulder are the brightest areas; the face side is in soft shade) plus a dim fill - low-key exposure.
 */
import {
  ACESFilmicToneMapping, AgXToneMapping, AmbientLight, Color, DirectionalLight, HemisphereLight, NoToneMapping, PCFSoftShadowMap, PMREMGenerator, PerspectiveCamera,
  RenderPipeline, Scene, SRGBColorSpace, Vector3, WebGPURenderer,
} from 'three/webgpu';
import { clamp, distance, max, float, hash, mix, pass, renderOutput, screenCoordinate, screenUV, smoothstep, time, uniform, vec2, vec3, vec4 } from 'three/tsl';
import { frameUniforms } from './frame.ts';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import type { SceneJson } from '../assets/load.ts';
import { skinUniforms } from './skin.ts';

export const postUniforms = { ao: uniform(1.0), grain: uniform(1.0), vignette: uniform(1.0) };

export async function createStage(canvas: HTMLCanvasElement, sj: SceneJson, forceWebGL: boolean) {
  const renderer = new WebGPURenderer({ canvas, antialias: true, forceWebGL, alpha: true });
  await renderer.init();
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.70;
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = PCFSoftShadowMap;

  const camera = new PerspectiveCamera(sj.camera.fovY, sj.camera.aspect, 0.4, 30);
  camera.position.fromArray(sj.camera.position);
  camera.lookAt(new Vector3().fromArray(sj.camera.target));
  camera.updateMatrixWorld(true);

  const scene = new Scene();
  scene.background = null;                       // transparent around the photo: the paper (DOM) shows through
  renderer.setClearColor(0x000000, 0);
  const pmrem = new PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.10;

  const keyDir = new Vector3(0.46, 0.88, -0.09).normalize();
  const key = new DirectionalLight(new Color('#fff0dc'), 2.7);
  key.position.copy(keyDir).multiplyScalar(5);
  key.target.position.set(0, -0.1, 0);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  const sc = key.shadow.camera;
  sc.left = -1.5; sc.right = 1.5; sc.top = 1.5; sc.bottom = -1.5; sc.near = 1; sc.far = 10;
  key.shadow.bias = -0.0006;
  key.shadow.normalBias = 0.012;
  key.shadow.radius = 7;
  scene.add(key, key.target);
  skinUniforms.keyDir.value.copy(keyDir);
  const fill = new DirectionalLight(new Color('#d8cdbf'), 0.74);
  fill.position.set(-1.6, -1.7, 2.6);
  scene.add(fill);
  const side = new DirectionalLight(new Color('#ffe9c8'), 0.55);     // low warm bounce from the right
  side.position.set(2.6, -0.3, 1.6);
  scene.add(side);
  const hemi = new HemisphereLight(new Color('#c2b3a4'), new Color('#4a3d33'), 0.24);
  scene.add(hemi);
  const amb = new AmbientLight(new Color('#6b5a4c'), 0.05);
  scene.add(amb);

  // ---- post: GTAO -> AgX -> warm grade -> vignette -> film grain
  const pipeline = new RenderPipeline(renderer);
  const scenePass = pass(scene, camera);
  const colorTex = scenePass.getTextureNode('output');
  const depthTex = scenePass.getTextureNode('depth');
  const aoPass = ao(depthTex, null as any, camera);
  aoPass.radius.value = 0.11;
  aoPass.thickness.value = 0.6;
  aoPass.distanceExponent.value = 1.4;
  aoPass.scale.value = 0.85;
  aoPass.samples.value = 16;
  aoPass.resolutionScale = 1;
  const aoVal = mix(float(1), aoPass.getTextureNode().r, postUniforms.ao);
  const alpha = colorTex.a;
  const lit = vec4(colorTex.rgb.div(max(alpha, 1e-4)).mul(aoVal), 1.0);   // un-premultiply the scene pass
  const out = renderOutput(lit);                                   // ACES filmic + sRGB
  const graded = out.rgb.mul(vec3(1.035, 1.0, 0.955));             // slight warm grade
  // lens vignette of the PHOTOGRAPH (photo-relative coordinates; clamped, so skin pulled out of the frame keeps
  // the value at the border)
  const r: any = frameUniforms.rectUV;
  const puv = clamp(screenUV.sub(r.xy).div(r.zw.sub(r.xy)), 0.0, 1.0);
  const vig = smoothstep(0.95, 0.22, distance(puv, vec2(0.56, 0.40)));
  const vigCol = graded.mul(mix(float(1), mix(float(0.55), float(1), vig), postUniforms.vignette));
  const g = hash(screenCoordinate.x.add(screenCoordinate.y.mul(1733.0)).add(time.mul(60.0).floor().mul(977.0))).sub(0.5);
  const final = vigCol.add(g.mul(0.028).mul(postUniforms.grain));
  pipeline.outputNode = vec4(final.mul(alpha), alpha);            // premultiplied for the transparent canvas
  pipeline.outputColorTransform = false;

  const backendName = (renderer.backend as any).isWebGPUBackend ? 'webgpu' : 'webgl2';
  return { renderer, camera, scene, pipeline, key, backendName, NoToneMapping, AgXToneMapping };
}
