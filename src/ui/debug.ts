/** Debug panel (hidden by default): press `d` or open with ?debug. */
import { Mesh, MeshBasicNodeMaterial, SphereGeometry, type Scene } from 'three/webgpu';
import type { Membrane } from '../sim/membrane.ts';
import { DEFAULT_MEMBRANE } from '../sim/membrane.ts';
import type { GrabController } from '../interaction/grab.ts';
import { skinUniforms } from '../render/skin.ts';

export class DebugPanel {
  visible = false;
  private markers: Mesh[] = [];
  private readout: HTMLPreElement;
  private showGrab = false;
  private root: HTMLElement; private scene: Scene; private mem: Membrane; private grab: GrabController; private skinMat: { wireframe: boolean };

  constructor(root: HTMLElement, scene: Scene, mem: Membrane, grab: GrabController, skinMat: { wireframe: boolean }) {
    this.root = root; this.scene = scene; this.mem = mem; this.grab = grab; this.skinMat = skinMat;
    const d = DEFAULT_MEMBRANE;
    root.innerHTML = `<h4>blonde · debug</h4>
      <label>wireframe <input type="checkbox" data-k="wire"></label>
      <label>grab points <input type="checkbox" data-k="grab"></label>
      <label>stretch view <input type="checkbox" data-k="strain"></label>
      <label>velocity kept / sub-step <input type="range" data-p="retain" min="0.9" max="0.999" step="0.001" value="${d.retain}"></label>
      <label>shape recovery × <input type="range" data-p="restoreScale" min="0.1" max="3" step="0.05" value="${d.restoreScale}"></label>
      <label>spring stiffness × <input type="range" data-p="stiffScale" min="0.2" max="2" step="0.05" value="${d.stiffScale}"></label>
      <label>iterations <input type="range" data-p="iterations" min="1" max="12" step="1" value="${d.iterations}"></label>
      <label>membrane thinning <input type="range" data-p="thin" min="0" max="2" step="0.05" value="1"></label>
      <pre></pre>`;
    this.readout = root.querySelector('pre')!;
    root.addEventListener('input', (ev) => {
      const t = ev.target as HTMLInputElement;
      if (t.dataset.k === 'wire') this.skinMat.wireframe = t.checked;
      if (t.dataset.k === 'grab') this.showGrab = t.checked;
      if (t.dataset.k === 'strain') skinUniforms.debugStrain.value = t.checked ? 1 : 0;
      const v = parseFloat(t.value), p = t.dataset.p;
      if (p === 'thin') skinUniforms.thinning.value = v;
      else if (p) { (mem.params as any)[p] = v; mem.wake(); }
    });
  }

  toggle(force?: boolean) {
    this.visible = force ?? !this.visible;
    this.root.hidden = !this.visible;
  }

  update(info: { fps: number; frameMs: number; gpuMs: number | null; backend: string }) {
    if (!this.visible) { this.markers.forEach((m) => (m.visible = false)); return; }
    const s = this.mem.stats;
    this.readout.textContent =
      `renderer   ${info.backend}\nphysics    GPU membrane, ${this.mem.n} vertices\nfps        ${info.fps.toFixed(0)}\ncpu frame  ${info.frameMs.toFixed(2)} ms\n` +
      `gpu frame  ${info.gpuMs == null ? 'n/a' : info.gpuMs.toFixed(2) + ' ms'}\nencode     ${s.encodeMs.toFixed(2)} ms${s.sleeping ? ' (asleep)' : ''}\n` +
      `grabs      ${this.grab.count}\nmax disp   ${s.maxDisp.toFixed(4)}\nresets     ${s.resets}`;
    let k = 0;
    if (this.showGrab) for (const [hit, target] of this.grab.debug.values()) for (const [pt, col] of [[hit, 0xff3355], [target, 0x33ddff]] as const) {
      if (!this.markers[k]) { const m = new Mesh(new SphereGeometry(0.012, 12, 8), new MeshBasicNodeMaterial({ depthTest: false })); m.renderOrder = 30; this.scene.add(m); this.markers[k] = m; }
      (this.markers[k].material as MeshBasicNodeMaterial).color.setHex(col);
      this.markers[k].position.copy(pt); this.markers[k].visible = true; k++;
    }
    for (; k < this.markers.length; k++) this.markers[k].visible = false;
  }
}
