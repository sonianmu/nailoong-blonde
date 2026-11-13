/**
 * Cover typography, drawn as live SVG text (no image slices):
 *  - `blonde`: Inter Tight Black Italic (OFL, self-hosted via @fontsource), fitted so that its INK box matches the
 *    box measured from figure A (x 0.3179-0.6917, y 0.0395-0.1301 of the cover).
 *  - Parental Advisory label rebuilt from rectangles + text (x 0.4434-0.5587, y 0.8958-0.9646).
 */
import '@fontsource/inter-tight/900-italic.css';
import '@fontsource/inter-tight/900.css';
import '@fontsource/inter-tight/800.css';

export const LAYOUT = {
  title: { x0: 0.3179, x1: 0.6917, y0: 0.0395, y1: 0.1301 },
  advisory: { x0: 0.4434, x1: 0.5587, y0: 0.8958, y1: 0.9646 },
  photo: { x0: 0.24966, x1: 0.75171, y0: 0.15123, y1: 0.87466 },
};
const NS = 'http://www.w3.org/2000/svg';
const el = (name: string, attrs: Record<string, string | number>, text?: string) => {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (text) e.textContent = text;
  return e;
};

/** ink box of `text` at font size 200px: [left, right, ascent, descent] relative to the baseline origin */
function ink(text: string, font: string) {
  const c = document.createElement('canvas').getContext('2d')!;
  c.font = font;
  const m = c.measureText(text);
  return { left: -m.actualBoundingBoxLeft, right: m.actualBoundingBoxRight, ascent: m.actualBoundingBoxAscent, descent: m.actualBoundingBoxDescent };
}

export async function buildCover(svg: SVGSVGElement) {
  await Promise.all([document.fonts.load('italic 900 200px "Inter Tight"', 'blonde'), document.fonts.load('900 200px "Inter Tight"', 'ADVISORY'), document.fonts.load('800 200px "Inter Tight"', 'PARENTAL')]);
  svg.replaceChildren();
  const U = 1000;
  // ---- title
  const t = LAYOUT.title;
  const m = ink('blonde', 'italic 900 200px "Inter Tight"');
  const fs = ((t.y1 - t.y0) * U) / ((m.ascent + m.descent) / 200);
  const k = fs / 200;
  const sx = ((t.x1 - t.x0) * U) / ((m.right - m.left) * k);
  const title = el('text', {
    x: 0, y: 0, 'font-family': '"Inter Tight"', 'font-weight': 900, 'font-style': 'italic', 'font-size': fs, fill: '#060606', class: 'title',
    transform: `translate(${t.x0 * U - m.left * k * sx} ${t.y0 * U + m.ascent * k}) scale(${sx} 1)`,
  }, 'blonde');
  svg.append(title);
  // ---- parental advisory
  const a = LAYOUT.advisory;
  const x = a.x0 * U, y = a.y0 * U, w = (a.x1 - a.x0) * U, h = (a.y1 - a.y0) * U;
  const g = el('g', { class: 'advisory' });
  g.append(el('rect', { x, y, width: w, height: h, rx: 1.2, fill: '#0a0a0a' }));
  const pad = w * 0.035, bandH = h * 0.215;
  g.append(el('rect', { x: x + pad, y: y + pad, width: w - pad * 2, height: bandH, fill: '#f4f1ec' }));
  g.append(el('rect', { x: x + pad, y: y + h - pad - bandH, width: w - pad * 2, height: bandH, fill: '#f4f1ec' }));
  const line = (txt: string, cy: number, size: number, weight: number, fill: string, len: number) => {
    const mm = ink(txt, `${weight} 200px "Inter Tight"`);
    const s = size / ((mm.ascent + mm.descent) / 200);
    g.append(el('text', {
      x: x + (w - len) / 2, y: cy + size / 2 - (mm.descent * s) / 200, 'font-family': '"Inter Tight"', 'font-weight': weight, 'font-size': s, fill,
      textLength: len, lengthAdjust: 'spacingAndGlyphs',
    }, txt));
  };
  line('PARENTAL', y + pad + bandH / 2, bandH * 0.62, 800, '#0a0a0a', w - pad * 4.2);
  line('ADVISORY', y + h / 2, h * 0.30, 900, '#f4f1ec', w - pad * 3.0);
  line('EXPLICIT CONTENT', y + h - pad - bandH / 2, bandH * 0.60, 800, '#0a0a0a', w - pad * 3.6);
  svg.append(g);
  return { titleFontSize: fs, titleScaleX: sx };
}
