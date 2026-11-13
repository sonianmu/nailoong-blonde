/**
 * Loading state shown in the photo window until the first frames are on screen.
 * The markup lives in index.html so it is visible before the JS bundle has arrived; this module only drives it.
 */
const root = () => document.getElementById('loader');

export const loader = {
  /** 0..1 */
  progress(p: number, label?: string) {
    const el = root();
    if (!el) return;
    const v = Math.max(0, Math.min(1, p));
    el.style.setProperty('--p', v.toFixed(4));
    const pct = el.querySelector('.ld-pct');
    if (pct) pct.textContent = `${Math.round(v * 100)}%`;
    if (label) { const l = el.querySelector('.ld-label'); if (l) l.textContent = label; }
  },
  done() {
    const el = root();
    if (!el) return;
    // automated runs (tests, demo recorder) screenshot right after `ready`: no fade, remove at once
    if (navigator.webdriver) { el.remove(); return; }
    el.classList.add('is-done');
    el.addEventListener('transitionend', () => el.remove(), { once: true });
    setTimeout(() => el.remove(), 1500);
  },
  fail(msg: string) {
    const el = root();
    if (!el) return;
    el.classList.add('is-error');
    const l = el.querySelector('.ld-label');
    if (l) l.textContent = msg;
    const pct = el.querySelector('.ld-pct');
    if (pct) pct.textContent = '';
  },
};

/** fetch with byte progress; `expected` is used when the response has no usable Content-Length (compressed transfer) */
export async function fetchBytes(url: string, expected: number, onBytes: (loaded: number, total: number) => void): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const len = Number(res.headers.get('content-length')) || 0;
  const total = len && !res.headers.get('content-encoding') ? len : expected;
  if (!res.body) { const b = await res.arrayBuffer(); onBytes(b.byteLength, b.byteLength); return b; }
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onBytes(loaded, Math.max(total, loaded));
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.byteLength; }
  onBytes(loaded, loaded);
  return out.buffer;
}
