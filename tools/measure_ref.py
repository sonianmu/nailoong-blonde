"""Re-measure figure A (reference/A_blonde_cover.png) and write reference/landmarks.json
plus reference/mask_A.png (character silhouette inside the photo window).

Everything downstream (model fitting, camera, layout CSS, QA thresholds) reads landmarks.json.
Usage: python tools/measure_ref.py
"""
import json
import os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "reference", "A_blonde_cover.png")


def runs(mask1d):
    """[start, end) runs of True in a 1-D bool array."""
    d = np.diff(np.concatenate([[0], mask1d.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def main():
    img = np.asarray(Image.open(SRC).convert("RGB")).astype(np.float32)
    H, W, _ = img.shape
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    out = {"image": {"file": "reference/A_blonde_cover.png", "width": W, "height": H}}

    # ---- paper colour: median of a clean margin strip
    paper = np.median(img[int(H * 0.3):int(H * 0.7), int(W * 0.03):int(W * 0.2)].reshape(-1, 3), axis=0)
    out["paper"] = {"rgb": [int(round(c)) for c in paper], "hex": "#%02X%02X%02X" % tuple(int(round(c)) for c in paper)}

    # ---- photo window: large dark rectangle. Use column/row dark-fraction profiles.
    dark = lum < 195
    col = dark[int(H * 0.3):int(H * 0.8)].mean(axis=0)
    row = dark[:, int(W * 0.4):int(W * 0.6)].mean(axis=1)
    cx = max(runs(col > 0.9), key=lambda r: r[1] - r[0])
    cy = max(runs(row > 0.9), key=lambda r: r[1] - r[0])
    px0, px1, py0, py1 = int(cx[0]), int(cx[1]), int(cy[0]), int(cy[1])
    out["photo"] = {"x": [px0 / W, px1 / W], "y": [py0 / H, py1 / H], "px": [px0, py0, px1, py1],
                    "aspect": (px1 - px0) / (py1 - py0)}

    # ---- title + advisory: black ink bounding boxes above / below the photo
    ink = lum < 70

    def bbox(region, y0, y1):
        region = region.copy()
        region[:, :int(W * 0.08)] = False  # the source PNG has a dark 1-2 px screenshot border
        region[:, int(W * 0.92):] = False
        ys, xs = np.where(region)
        return {"x": [float(xs.min()) / W, float(xs.max() + 1) / W],
                "y": [float(ys.min() + y0) / H, float(ys.max() + 1 + y0) / H]}

    out["title"] = bbox(ink[int(H * 0.02):py0 - 8], int(H * 0.02), py0)
    out["advisory"] = bbox(ink[py1 + 8:int(H * 0.985)], py1 + 8, H)

    # ---- inside the photo
    ph = img[py0:py1, px0:px1]
    PH, PW, _ = ph.shape
    mx = ph.max(axis=2)
    mn = ph.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    r, g, b = ph[..., 0], ph[..., 1], ph[..., 2]
    # wall: desaturated warm grey (sat ~0.2-0.3). character: saturated yellow / tan / olive-brown.
    # 'warmth' r-b grows with brightness, so normalise with saturation.
    char = (sat > 0.37) & ((r - b) > 26)
    # eyes: pale sage / dark pupils live inside the head -> filled by hole filling
    char = ndi.binary_opening(char, iterations=2)
    char = ndi.binary_closing(char, iterations=6)
    char = ndi.binary_fill_holes(char)
    lab, n = ndi.label(char)
    if n > 1:
        sizes = ndi.sum(char, lab, range(1, n + 1))
        char = lab == (1 + int(np.argmax(sizes)))
    # the left eye pokes out of the head contour: sage-green region adjacent to the silhouette
    eye = (g > r * 0.97) & (g > b * 1.12) & (mx > 60) & (sat > 0.12)
    eye = ndi.binary_opening(eye, iterations=2)
    elab, en = ndi.label(eye)
    grown = ndi.binary_dilation(char, iterations=6)
    for i in range(1, en + 1):
        comp = elab == i
        if comp.sum() > 150 and (comp & grown).any():
            char |= comp
    char = ndi.binary_closing(char, iterations=8)
    char = ndi.binary_fill_holes(char)
    Image.fromarray((char * 255).astype(np.uint8)).save(os.path.join(ROOT, "reference", "mask_A.png"))
    Image.fromarray(ph.astype(np.uint8)).save(os.path.join(ROOT, "reference", "photo_A.png"))

    rows = {}
    for yf in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        xs = np.where(char[int(yf * PH)])[0]
        if len(xs):
            rows["%.2f" % yf] = [round(float(xs.min()) / PW, 4), round(float(xs.max() + 1) / PW, 4)]
    ys, xs = np.where(char)
    top_y = int(ys.min())
    top_x = float(np.mean(np.where(char[top_y + 2])[0]))
    out["silhouette"] = {"rows": rows, "top": [top_x / PW, top_y / PH], "area_fraction": float(char.mean()),
                         "mask": "reference/mask_A.png"}

    # ---- eyes (sage iris region) ---------------------------------------------------------
    iris = (g > r * 1.0) & (g > b * 1.15) & (mx > 70)
    iris = ndi.binary_opening(iris, iterations=2)
    ilab, inn = ndi.label(iris)
    eyes = []
    for i in range(1, inn + 1):
        comp = ilab == i
        if comp.sum() < 400:
            continue
        ys_, xs_ = np.where(comp)
        eyes.append({"area": int(comp.sum()), "bbox": [float(xs_.min()) / PW, float(ys_.min()) / PH, float(xs_.max()) / PW, float(ys_.max()) / PH],
                     "centroid": [float(xs_.mean()) / PW, float(ys_.mean()) / PH]})
    eyes.sort(key=lambda e: -e["area"])
    out["eye_regions"] = eyes[:4]

    # ---- hand (olive brown: dark, strongly saturated, low green)
    hand = (sat > 0.5) & (sat < 0.76) & (mx < 112) & (mx > 40) & (r > g * 1.25)
    hand[:, int(PW * 0.60):] = False
    hand[:, :int(PW * 0.25)] = False
    hand[int(PH * 0.60):] = False
    hand[:int(PH * 0.38)] = False
    hand = ndi.binary_opening(hand, iterations=3)
    hlab, hn = ndi.label(hand)
    if hn:
        sizes = ndi.sum(hand, hlab, range(1, hn + 1))
        hand = hlab == (1 + int(np.argmax(sizes)))
        ys_, xs_ = np.where(hand)
        out["hand_region"] = {"bbox": [float(xs_.min()) / PW, float(ys_.min()) / PH, float(xs_.max()) / PW, float(ys_.max()) / PH],
                              "centroid": [float(xs_.mean()) / PW, float(ys_.mean()) / PH], "area_fraction": float(hand.mean())}
        Image.fromarray((hand * 255).astype(np.uint8)).save(os.path.join(ROOT, "reference", "mask_A_hand.png"))

    # ---- belly patch (tan: mid saturation, inside the character, lower half)
    belly = char & (sat > 0.33) & (sat < 0.56) & (mx > 85)
    belly[:int(PH * 0.55)] = False
    belly = ndi.binary_opening(belly, iterations=4)
    blab, bn = ndi.label(belly)
    if bn:
        sizes = ndi.sum(belly, blab, range(1, bn + 1))
        belly = blab == (1 + int(np.argmax(sizes)))
        belly = ndi.binary_closing(belly, iterations=6)
        ys_, xs_ = np.where(belly)
        brow = {}
        for yf in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            q = np.where(belly[int(yf * PH)])[0]
            if len(q):
                brow["%.2f" % yf] = [round(float(q.min()) / PW, 4), round(float(q.max() + 1) / PW, 4)]
        out["belly_region"] = {"bbox": [float(xs_.min()) / PW, float(ys_.min()) / PH, float(xs_.max()) / PW, float(ys_.max()) / PH],
                               "top": [float(np.mean(np.where(belly[ys_.min() + 3])[0])) / PW, float(ys_.min()) / PH], "rows": brow}
        Image.fromarray((belly * 255).astype(np.uint8)).save(os.path.join(ROOT, "reference", "mask_A_belly.png"))

    # ---- tiles: grout lines are thin and slightly DARKER than the tile faces (the prompt guessed lighter).
    wall = ~ndi.binary_dilation(char, iterations=10)
    plum = lum[py0:py1, px0:px1]

    def dark_lines(profile, mind=90):
        p = ndi.gaussian_filter1d(profile, 1.5) - ndi.gaussian_filter1d(profile, 30)
        return [i for i in range(4, len(p) - 4) if p[i] < -0.9 and p[i] == p[max(0, i - mind):i + mind + 1].min()]

    vx = dark_lines(plum[int(PH * 0.01):int(PH * 0.15)].mean(axis=0))
    hy = [i for i in dark_lines(plum[:int(PH * 0.72), int(PW * 0.01):int(PW * 0.10)].mean(axis=1))]
    hy_r = [i for i in dark_lines(plum[:int(PH * 0.40), int(PW * 0.90):int(PW * 0.99)].mean(axis=1))]
    tiles = {"grout": "darker than tile face, ~2-3 px wide of 736",
             "vertical_grout_x": [round(x / PW, 4) for x in vx],
             "horizontal_grout_y_left_in_widths": [round(y / PW, 4) for y in hy],
             "horizontal_grout_y_right_in_widths": [round(y / PW, 4) for y in hy_r]}
    if len(vx) > 1:
        tiles["pitch_x"] = float(np.median(np.diff(vx))) / PW
    if len(hy) > 1:
        tiles["pitch_y_in_widths"] = float(np.median(np.diff(hy))) / PW
    wallpix = ph[wall]
    tiles["wall_mean_rgb"] = [int(round(c)) for c in wallpix.mean(axis=0)]

    def patch(xf, yf, s=10):
        x, y = int(xf * PW), int(yf * PH)
        return [int(round(c)) for c in ph[y - s:y + s, x - s:x + s].reshape(-1, 3).mean(axis=0)]

    tiles["wall_top_left_rgb"] = patch(0.10, 0.07)
    tiles["wall_top_right_rgb"] = patch(0.90, 0.07)
    tiles["wall_mid_left_rgb"] = patch(0.08, 0.45)
    tiles["wall_right_rgb"] = patch(0.93, 0.33)
    out["tiles"] = tiles

    # ---- colour samples (photo-relative positions; sRGB after lighting)
    samples = {
        "skin_lit_forehead": (0.42, 0.28), "skin_bright_right_head": (0.62, 0.40), "skin_shoulder_right": (0.86, 0.62),
        "skin_shadow_left_shoulder": (0.16, 0.80), "skin_arm": (0.66, 0.68), "belly_centre": (0.36, 0.84),
        "belly_upper": (0.40, 0.72), "hand_palm": (0.40, 0.50), "hand_dark": (0.33, 0.55),
        "skin_head_top": (0.48, 0.21), "skin_lower_right": (0.90, 0.90),
    }
    out["colors"] = {k: {"pos": list(v), "rgb": patch(*v, s=6), "hex": "#%02X%02X%02X" % tuple(patch(*v, s=6))} for k, v in samples.items()}

    # ---- landmarks read off a gridded 3x enlargement (tools/measure_ref.py cannot segment these reliably;
    # values are photo-relative, tolerance +-0.015). Auto values above are used to cross-check.
    eye_auto = out["eye_regions"][0]["centroid"] if out["eye_regions"] else None
    out["landmarks"] = {
        "head_top": [round(out["silhouette"]["top"][0], 4), round(out["silhouette"]["top"][1], 4)],
        "eye_right": {"center": [0.482, 0.405], "radius": 0.068, "pupil": [0.487, 0.427], "pupil_radius": 0.038,
                      "auto_iris_centroid": eye_auto},
        "eye_left": {"center": [0.255, 0.409], "radius": 0.066, "note": "on the left head contour, ~35% visible"},
        "hand": {"palm_center": [0.440, 0.520], "wrist": [0.548, 0.575], "thumb_tip": [0.548, 0.428],
                 "index_tip": [0.346, 0.403], "middle_tip": [0.288, 0.438], "ring_tip": [0.279, 0.475]},
        "elbow": {"center": [0.76, 0.80], "outer_corner": [0.795, 0.871],
                  "note": "prompt guessed (0.75,0.75); the forearm's lower contour bottoms out at v~0.87"},
        "armpit_crease": [0.734, 0.612],
        "belly_top": [round(out["belly_region"]["top"][0], 4), round(out["belly_region"]["top"][1], 4)],
    }
    out["corrections_vs_prompt"] = [
        "paper is #E2DEDB (prompt: #E4E0DD)",
        "grout lines are darker than the tile faces, not lighter; pitch 0.194 x photo width",
        "wall is not exactly parallel: horizontal grout lines rise ~3 deg towards the right",
        "right eye centre (0.482,0.405) r=0.068 (prompt: (0.47,0.40) r=0.06)",
        "elbow centre ~(0.76,0.80), outer corner (0.795,0.871) (prompt: (0.75,0.75))",
        "back-of-hand centre re-read from a 3x crop: (0.440,0.520); wrist (0.548,0.575)",
        "hand spans u 0.27-0.56, v 0.39-0.60; three fingers visible + thumb, 4th finger hidden behind",
    ]
    with open(os.path.join(ROOT, "reference", "landmarks.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k in ["paper", "photo", "title", "advisory", "hand_region", "belly_region", "tiles", "landmarks"]:
        print(k, json.dumps(out[k]))


if __name__ == "__main__":
    main()
