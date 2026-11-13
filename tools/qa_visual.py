"""Visual acceptance metrics (PROMPT §10.1). Called by tests/visual.mjs:
   python tools/qa_visual.py <dir with id.png clean.png beauty.png wall.png page.png landmarks.json> <out.json> <triptych.png>
"""
import json
import os
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color as skc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LM = json.load(open(os.path.join(ROOT, "reference", "landmarks.json")))
ASPECT = LM["photo"]["aspect"]


def load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32)


def main(d, out_json, trip_path):
    res = {}
    ref_mask = np.asarray(Image.open(os.path.join(ROOT, "reference", "mask_A.png")).convert("L")) > 127
    H, W = ref_mask.shape
    photo = load(os.path.join(ROOT, "reference", "photo_A.png"))
    rs = lambda im: np.asarray(Image.fromarray(im.astype(np.uint8)).resize((W, H), Image.LANCZOS)).astype(np.float32)
    idm = rs(load(os.path.join(d, "id.png")))
    mask = idm.max(axis=2) > 100
    mask[:, :4] = False      # the canvas screenshot can include a sub-pixel sliver of paper at its left edge
    mask[:, -1:] = mask[:, -2:-1]
    # ---- 1 silhouette IoU
    iou = float((mask & ref_mask).sum() / (mask | ref_mask).sum())
    res["silhouette_iou"] = iou
    rows, worst = {}, 0.0
    for k, (x0, x1) in LM["silhouette"]["rows"].items():
        xs = np.where(mask[int(float(k) * H)])[0]
        if len(xs):
            e = (xs.min() / W - x0, (xs.max() + 1) / W - x1)
            rows[k] = [round(float(e[0]), 4), round(float(e[1]), 4)]
            worst = max(worst, abs(e[0]), abs(e[1]))
    res["row_edge_errors"] = rows
    res["row_edge_worst"] = float(worst)
    res["row_edge_mean_abs"] = float(np.mean([abs(v) for r in rows.values() for v in r]))
    # ---- 2 landmarks (projected 3D landmarks from the page + mask-derived ones)
    proj = json.load(open(os.path.join(d, "landmarks.json")))
    lm = LM["landmarks"]
    err = lambda a, b: float(np.hypot(a[0] - b[0], (a[1] - b[1]) / ASPECT))   # in photo widths
    ys = np.where(mask.any(axis=1))[0]
    top_v = ys.min() / H
    top_u = float(np.mean(np.where(mask[ys.min() + 2])[0])) / W
    belly = (idm[..., 2] > 100) & (idm[..., 0] < 100)
    belly = ndi.binary_opening(belly, iterations=2)
    by = np.where(belly.any(axis=1))[0]
    belly_top = [float(np.mean(np.where(belly[by.min() + 3])[0])) / W, by.min() / H] if len(by) else [0, 0]
    eyes_px = (idm[..., 0] > 100) & (idm[..., 1] > 100) & (idm[..., 2] < 100)
    lab, n = ndi.label(eyes_px)
    L = {
        "head_top": err([top_u, top_v], lm["head_top"]),
        "head_top_v_only": abs(top_v - lm["head_top"][1]) / ASPECT,
        "eye_right": err(proj["eyeRight"], lm["eye_right"]["center"]),
        "eye_left": err(proj["eyeLeft"], lm["eye_left"]["center"]),
        "palm_center": err(proj["palm"], lm["hand"]["palm_center"]),
        "wrist": err(proj["wrist"], lm["hand"]["wrist"]),
        "elbow": err(proj["elbow"], lm["elbow"]["center"]),
        "belly_top": err(belly_top, lm["belly_top"]),
        "belly_top_v_only": abs(belly_top[1] - lm["belly_top"][1]) / ASPECT,
    }
    res["landmark_errors_photo_widths"] = L
    keys = ["head_top_v_only", "eye_right", "eye_left", "palm_center", "elbow", "belly_top_v_only"]
    res["landmark_worst"] = float(max(L[k] for k in keys))
    res["landmark_keys_used"] = keys
    # ---- 5 colour: deltaE at the sample points (tone-mapped render without droplets/grain vs figure A)
    clean = rs(load(os.path.join(d, "clean.png")))
    cols = {}
    de_list = []
    for name, c in LM["colors"].items():
        u, v = c["pos"]
        x, y = int(u * W), int(v * H)
        a = photo[y - 9:y + 9, x - 9:x + 9].reshape(-1, 3)
        b = clean[y - 9:y + 9, x - 9:x + 9].reshape(-1, 3)
        # median: robust to droplets / highlights inside the reference patch
        ra, rb = np.median(a, axis=0), np.median(b, axis=0)
        la, lb = skc.rgb2lab((ra / 255)[None, None, :])[0, 0], skc.rgb2lab((rb / 255)[None, None, :])[0, 0]
        de76 = float(np.linalg.norm(la - lb))
        de00 = float(skc.deltaE_ciede2000(la[None, :], lb[None, :])[0])
        cols[name] = {"ref": "#%02X%02X%02X" % tuple(int(round(t)) for t in ra), "render": "#%02X%02X%02X" % tuple(int(round(t)) for t in rb),
                      "dE76": round(de76, 2), "dE2000": round(de00, 2)}
        de_list.append(de00)
    # wall samples
    for name, (u, v) in {"wall_top_left": (0.10, 0.07), "wall_top_mid": (0.50, 0.06), "wall_left_mid": (0.07, 0.30), "wall_right_top": (0.90, 0.10)}.items():
        x, y = int(u * W), int(v * H)
        ra = np.median(photo[y - 9:y + 9, x - 9:x + 9].reshape(-1, 3), axis=0)
        rb = np.median(clean[y - 9:y + 9, x - 9:x + 9].reshape(-1, 3), axis=0)
        la, lb = skc.rgb2lab((ra / 255)[None, None, :])[0, 0], skc.rgb2lab((rb / 255)[None, None, :])[0, 0]
        de00 = float(skc.deltaE_ciede2000(la[None, :], lb[None, :])[0])
        cols[name] = {"ref": "#%02X%02X%02X" % tuple(int(round(t)) for t in ra), "render": "#%02X%02X%02X" % tuple(int(round(t)) for t in rb),
                      "dE76": round(float(np.linalg.norm(la - lb)), 2), "dE2000": round(de00, 2)}
        de_list.append(de00)
    res["colors"] = cols
    res["color_dE2000_max"] = float(max(de_list))
    res["color_dE2000_mean"] = float(np.mean(de_list))
    # ---- 6 tiles (character hidden)
    wall = rs(load(os.path.join(d, "wall.png")))
    lum = wall @ np.array([0.2126, 0.7152, 0.0722], np.float32)

    def lines(profile, mind=80):
        p = ndi.gaussian_filter1d(profile, 1.2) - ndi.gaussian_filter1d(profile, 25)
        thr = -max(1.0, 0.35 * abs(p.min()))
        return [i for i in range(3, len(p) - 3) if p[i] < thr and p[i] == p[max(0, i - mind):i + mind + 1].min()]

    vx = lines(lum[int(H * 0.45):int(H * 0.55)].mean(axis=0))
    hy = lines(lum[:, int(W * 0.45):int(W * 0.55)].mean(axis=1))
    dx, dy = np.diff(vx) / W, np.diff(hy) / W
    centers = []
    for j in range(len(hy) - 1):
        for i in range(len(vx) - 1):
            cx, cy = (vx[i] + vx[i + 1]) // 2, (hy[j] + hy[j + 1]) // 2
            centers.append((j, i, float(lum[cy - 6:cy + 6, cx - 6:cx + 6].mean())))
    grid = {}
    for j, i, v in centers:
        grid[(j, i)] = v
    jumps = [abs(grid[(j, i)] - grid[(j, i + 1)]) / max(grid[(j, i)], 1) for (j, i) in grid if (j, i + 1) in grid] + \
            [abs(grid[(j, i)] - grid[(j + 1, i)]) / max(grid[(j, i)], 1) for (j, i) in grid if (j + 1, i) in grid]
    res["tiles"] = {"vertical_lines": [round(x / W, 4) for x in vx], "horizontal_lines_in_widths": [round(y / W, 4) for y in hy],
                    "pitch_x": float(dx.mean()) if len(dx) else None, "pitch_y": float(dy.mean()) if len(dy) else None,
                    "pitch_x_rel_std": float(dx.std() / dx.mean()) if len(dx) else None, "pitch_y_rel_std": float(dy.std() / dy.mean()) if len(dy) else None,
                    "reference_pitch": LM["tiles"].get("pitch_x"), "tile_centre_luminance_max_neighbour_jump": float(max(jumps)) if jumps else None,
                    "tiles_sampled": len(centers)}
    # ---- layout (whole page)
    page = load(os.path.join(d, "page.png"))
    PH, PW, _ = page.shape
    pl = page @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    ph = LM["photo"]
    ink = pl < 70
    lay = {}
    for name, key, (ya, yb) in (("title", "title", (0.0, ph["y"][0] - 0.008)), ("advisory", "advisory", (ph["y"][1] + 0.008, 1.0))):
        reg = ink[int(ya * PH):int(yb * PH)]
        yy, xx = np.where(reg)
        box = [xx.min() / PW, (xx.max() + 1) / PW, (yy.min() + int(ya * PH)) / PH, (yy.max() + 1 + int(ya * PH)) / PH]
        refb = LM[key]["x"] + LM[key]["y"]
        lay[name] = {"box": [round(float(b), 4) for b in box], "ref": [round(float(b), 4) for b in refb], "max_err": float(max(abs(a - b) for a, b in zip(box, refb)))}
    # photo window: columns/rows where the page is not paper
    dark = pl < 195
    col = dark[int(PH * 0.3):int(PH * 0.8)].mean(axis=0) > 0.9
    row = dark[:, int(PW * 0.4):int(PW * 0.6)].mean(axis=1) > 0.9
    xs, ysr = np.where(col)[0], np.where(row)[0]
    # longest run of rows
    runs = np.split(ysr, np.where(np.diff(ysr) > 1)[0] + 1)
    ysr = max(runs, key=len)
    box = [xs.min() / PW, (xs.max() + 1) / PW, ysr.min() / PH, (ysr.max() + 1) / PH]
    refb = ph["x"] + ph["y"]
    lay["photo"] = {"box": [round(float(b), 4) for b in box], "ref": [round(float(b), 4) for b in refb], "max_err": float(max(abs(a - b) for a, b in zip(box, refb)))}
    res["layout"] = lay
    res["layout_worst"] = float(max(v["max_err"] for v in lay.values()))
    # ---- triptych: A | render | 50 % overlay + silhouette difference heat map
    beauty = rs(load(os.path.join(d, "beauty.png")))
    over = photo * 0.5 + beauty * 0.5
    over[mask & ~ref_mask] = over[mask & ~ref_mask] * 0.35 + np.array([255, 40, 40]) * 0.65     # render outside reference
    over[~mask & ref_mask] = over[~mask & ref_mask] * 0.35 + np.array([40, 110, 255]) * 0.65    # reference not covered
    Image.fromarray(np.concatenate([photo, beauty, over], axis=1).astype(np.uint8)).save(trip_path)
    res["pass"] = {"iou>=0.85": iou >= 0.85, "landmarks<=0.03": res["landmark_worst"] <= 0.03, "dE<=10": res["color_dE2000_max"] <= 10,
                   "layout<=0.01": res["layout_worst"] <= 0.01}
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("silhouette_iou", "row_edge_worst", "landmark_worst", "color_dE2000_max", "color_dE2000_mean", "layout_worst", "pass")}))
    print("landmarks", json.dumps({k: round(v, 4) for k, v in L.items()}))
    print("colors", json.dumps({k: (v["ref"], v["render"], v["dE2000"]) for k, v in cols.items()}))
    print("tiles", json.dumps(res["tiles"]))
    print("layout", json.dumps(lay))


if __name__ == "__main__":
    main(*sys.argv[1:4])
