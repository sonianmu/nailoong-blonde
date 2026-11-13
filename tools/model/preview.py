"""Fast numpy sphere-traced preview of the SDF model through the shared camera, compared with figure A.

  python tools/model/preview.py [out.png] [width]
Prints silhouette IoU, per-row contour errors and projected landmark errors. Used to fit the model
before the (slow) full mesh build; the authoritative metrics come from the browser render (tests/visual.mjs).
"""
import os
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera as cam
from sdf import Model, LM, ROOT


def trace(m, W, crop=None):
    Hh = int(round(W / cam.ASPECT))
    us, vs = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(Hh) + 0.5) / Hh)
    if crop:   # (u0, v0, u1, v1): W x W pixels over that window
        Hh = W
        us, vs = np.meshgrid(crop[0] + (np.arange(W) + 0.5) / W * (crop[2] - crop[0]), crop[1] + (np.arange(W) + 0.5) / W * (crop[3] - crop[1]))
    o, d = cam.ray(us.ravel(), vs.ravel())
    n = len(d)
    t = np.full(n, 1.7)
    alive = np.ones(n, bool)
    hit = np.zeros(n, bool)
    fn = lambda p: np.minimum(m.skin(p), m.eyeballs(p))
    for _ in range(140):
        idx = np.where(alive)[0]
        if not len(idx):
            break
        p = o + d[idx] * t[idx, None]
        dist = fn(p)
        h = dist < 4e-4
        hit[idx[h]] = True
        alive[idx[h]] = False
        t[idx] += np.maximum(dist, 0) * 0.7
        alive[idx[t[idx] > 5.2]] = False
    p = o + d * t[:, None]
    return hit.reshape(Hh, W), p, d, fn


def render(m, W=230, crop=None):
    hit, p, d, fn = trace(m, W, crop)
    Hh = hit.shape[0]
    img = np.zeros((Hh * W, 3))
    img[:] = (0.31, 0.27, 0.23)
    idx = np.where(hit.ravel())[0]
    q = p[idx]
    nrm = Model.grad(fn, q, 2e-4)
    L = np.array([-0.55, 0.65, 0.55])
    L /= np.linalg.norm(L)
    lam = np.clip(nrm @ L, 0, 1) * 0.75 + 0.25 + 0.15 * np.clip(nrm @ np.array([0.8, 0.1, 0.3]), 0, 1)
    lab = m.labels(q)
    col = np.tile(np.array([[0.80, 0.56, 0.16]]), (len(q), 1))
    col = col * (1 - lab["belly"][:, None]) + np.array([0.62, 0.50, 0.38]) * lab["belly"][:, None]
    col = col * (1 - lab["hand"][:, None]) + np.array([0.33, 0.22, 0.09]) * lab["hand"][:, None]
    eye = m.eyeballs(q) < m.skin(q)
    col[eye] = (0.55, 0.62, 0.45)
    for e in m.eyes:   # pupil looking down-ish
        pd = (e["normal"] * 0.9 + np.array([0, -0.45, 0.1]))
        pd /= np.linalg.norm(pd)
        pc = e["center"] + pd * m.P["eye_r"]
        col[eye & (np.linalg.norm(q - pc, axis=1) < m.P["eye_r"] * 0.62)] = (0.05, 0.04, 0.03)
    img[idx] = col * lam[:, None]
    return hit, (np.clip(img, 0, 1).reshape(Hh, W, 3) * 255).astype(np.uint8)


def metrics(m, hit):
    Hh, W = hit.shape
    ref = np.asarray(Image.open(os.path.join(ROOT, "reference", "mask_A.png")).convert("L").resize((W, Hh), Image.BILINEAR)) > 127
    iou = (hit & ref).sum() / max(1, (hit | ref).sum())
    rows = {}
    worst = 0
    for k, (x0, x1) in LM["silhouette"]["rows"].items():
        xs = np.where(hit[int(float(k) * Hh)])[0]
        if len(xs):
            a, b = xs.min() / W, (xs.max() + 1) / W
            rows[k] = (round(a - x0, 3), round(b - x1, 3))
            worst = max(worst, abs(a - x0), abs(b - x1))
    ys = np.where(hit.any(axis=1))[0]
    top = ys.min() / Hh
    lm = LM["landmarks"]
    pr = lambda p: [float(x[0]) for x in cam.project(np.asarray(p)[None, :])[:2]]
    err = lambda a, b: float(np.hypot(a[0] - b[0], (a[1] - b[1]) / cam.ASPECT))
    out = {
        "iou": float(iou), "row_worst": float(worst), "top_err": float(top - lm["head_top"][1]) / cam.ASPECT,
        "eye_right": err(pr(m.eyes[0]["center"]), lm["eye_right"]["center"]),
        "eye_left": err(pr(m.eyes[1]["center"]), lm["eye_left"]["center"]),
        "palm": err(pr(m.palm_c), lm["hand"]["palm_center"]),
        "elbow": err(pr(m.elbow), lm["elbow"]["center"]),
        "wrist": err(pr(m.wrist), lm["hand"]["wrist"]),
    }
    return out, rows, ref


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "reference", "_preview.png")
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 230
    m = Model()
    hit, img = render(m, W)
    met, rows, ref = metrics(m, hit)
    Hh = hit.shape[0]
    photo = np.asarray(Image.open(os.path.join(ROOT, "reference", "photo_A.png")).convert("RGB").resize((W, Hh), Image.LANCZOS))
    diff = photo.astype(float) * 0.5 + img.astype(float) * 0.5
    diff[hit & ~ref] = diff[hit & ~ref] * 0.4 + np.array([255, 0, 0]) * 0.6      # model outside reference
    diff[~hit & ref] = diff[~hit & ref] * 0.4 + np.array([0, 90, 255]) * 0.6     # reference not covered
    Image.fromarray(np.concatenate([photo, img, diff.astype(np.uint8)], axis=1)).save(out)
    crop = (0.15, 0.30, 0.65, 0.30 + 0.5 * cam.ASPECT)
    _, zoom = render(m, 520, crop)
    ph = Image.open(os.path.join(ROOT, "reference", "photo_A.png")).convert("RGB")
    pw, phh = ph.size
    ref = ph.crop((int(crop[0] * pw), int(crop[1] * phh), int(crop[2] * pw), int(crop[3] * phh))).resize((520, 520), Image.LANCZOS)
    Image.fromarray(np.concatenate([np.asarray(ref), zoom], axis=1)).save(os.path.join(ROOT, "reference", "_pv_face.png"))
    print({k: round(v, 4) for k, v in met.items()})
    print("rows (left err, right err):", rows)
    print("eyes az/el:", [(round(e["az"], 1), round(e["el"], 1)) for e in m.eyes], "fit residual", round(m.eye_fit_residual, 5))
    print("palm", m.palm_c.round(3), "wrist", m.wrist.round(3), "elbow", m.elbow.round(3))


if __name__ == "__main__":
    main()
