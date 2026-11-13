"""Signed-distance description of the character (pose of figure A).

Two groups are kept strictly apart:
  * BODY  = head + muzzle + torso + back hump + belly bulge (+ eyelids, - eye openings, - mouth groove)
  * ARM   = near (character-left) upper arm + forearm + palm + 3 fingers + thumb (Nailoong has four digits)
The arm is fused to the body ONLY around the shoulder. Everywhere else the body is carved by the
arm inflated by GAP, so hand/face and forearm/chest keep a real, constant air gap (PROMPT §4).

All placement is driven by photo-space landmarks (reference/landmarks.json) through the shared
camera: points are found by shooting the landmark's camera ray at the body surface.
"""
import json
import os
import numpy as np
from scipy.optimize import least_squares

import camera as cam

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LM = json.load(open(os.path.join(ROOT, "reference", "landmarks.json")))

H_CHAR = 2.5  # full character height in world units (head width 0.21 H, as in figure B)

DEFAULT = dict(
    # ---- body ------------------------------------------------------------------------
    body_yaw=-27.0,                       # deg, character faces viewer-left
    head_yaw=-27.0,                       # deg, the head is turned further than the body
    head_pitch=12.0,                      # deg, head leans forward (looking down)
    eye_left_uv=None, eye_right_uv=None,  # optional overrides of the landmark eye centres
    head_c=(-0.022, 0.196, 0.10), head_r=(0.252, 0.258, 0.262),
    jaw_c=(-0.06, 0.02, 0.10), jaw_r=(0.21, 0.22, 0.21), k_jaw=0.10,   # lower face under the hand
    torso_c=(0.10, -0.78, -0.16), torso_r=(0.615, 0.78, 0.56),
    neck_a=(0.0, 0.17, 0.0), neck_ra=0.24, neck_b=(0.22, -0.22, -0.15), neck_rb=0.28,   # back of the neck -> near shoulder slope
    patch=dict(a=0.36, y0=0.0, b_up=0.62, b_dn=0.60, soft=0.03, x0=0.0),             # belly colour patch (body frame, rel. belly_c)
    belly_c=(-0.10, -0.86, 0.10), belly_r=(0.50, 0.50, 0.50),
    k_head=0.20, k_hump=0.22, k_belly=0.25,
    y_bottom=-1.02, k_bottom=0.10,
    # ---- eyes ------------------------------------------------------------------------
    eye_depth_left=None,                  # the far eye may sit deeper (only a sliver shows in figure A)
    eye_r=0.064, eye_depth=0.50,          # centre sits eye_depth*eye_r under the head surface
    rim_h=0.016, k_rim=0.045, socket_gap=0.0035, k_socket=0.012,
    # eyelids: a skin shell of thickness rim_h around the ball, opened by an elliptic cone around the eye axis
    # (tilted down by lid_tilt). The upper lid covers more of the ball than the lower one.
    lid_open_x=57.0, lid_open_up=47.0, lid_open_dn=57.0, lid_tilt=6.0, k_lid=0.009,
    # ---- mouth (figure B: a thin smile line ~2.2 eye radii under the eyes on a slightly raised muzzle) -----
    mouth_drop=0.43, mouth_half=0.31, mouth_smile=0.035, mouth_r=0.0062, k_mouth=0.006,
    muzzle_up=0.12, muzzle_r=(0.115, 0.075, 0.08), muzzle_in=0.058, k_muzzle=0.07,
    # ---- arm -------------------------------------------------------------------------
    shoulder=(0.56, -0.30, 0.02), r_shoulder=0.165,
    elbow_uv=(0.762, 0.782), r_elbow=0.118, elbow_lift=0.95,
    wrist_uv=(0.543, 0.565), r_wrist=0.082, wrist_dz=-0.035,
    k_elbow=0.07, k_wrist=0.07,
    fuse_r0=0.20, fuse_r1=0.40, k_shoulder=0.16,
    # ---- hand ------------------------------------------------------------------------
    palm_uv=(0.435, 0.492), palm_half=(0.098, 0.090, 0.040), palm_round=0.036, palm_lift=0.004,
    finger_r=(0.0300, 0.0305, 0.0295), finger_taper=0.88,
    finger_tips_uv=((0.352, 0.409), (0.296, 0.443), (0.285, 0.480)),
    finger_base_t=(0.052, 0.0, -0.052), finger_base_f=0.070,
    thumb_tip_uv=(0.545, 0.436), thumb_r=0.0315, thumb_base=(-0.035, 0.078),  # (along f, along t)
    finger_press=0.80,                    # finger centre sits press*r above the face (it dents it)
    k_finger=0.030, k_knuckle=0.012,
    # ---- separation ------------------------------------------------------------------
    gap=0.013, k_carve=0.022, k_wall=0.034,
)


# ----------------------------------------------------------------------------- primitives
def smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a, b, k):
    return -smin(-a, -b, k)


def sd_sphere(p, c, r):
    return np.linalg.norm(p - c, axis=1) - r


def sd_ellipsoid(p, c, rad, R=None):
    q = p - c
    if R is not None:
        q = q @ R
    k0 = np.linalg.norm(q / rad, axis=1)
    k1 = np.linalg.norm(q / (rad * rad), axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_capsule(p, a, b, ra, rb):
    pa = p - a
    ba = b - a
    t = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return np.linalg.norm(pa - t[:, None] * ba, axis=1) - (ra + (rb - ra) * t)


def sd_rbox(p, c, R, half, rnd):
    q = np.abs((p - c) @ R) - (half - rnd)
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0) - rnd


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class Model:
    def __init__(self, params=None):
        P = dict(DEFAULT)
        ov = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json")
        if os.path.exists(ov):          # fitted overrides (written while matching figure A)
            P.update(json.load(open(ov)))
        if params:
            P.update(params)
        self.P = P
        A = lambda k: np.array(P[k], dtype=np.float64)
        yaw = np.radians(P["body_yaw"])
        self.fwd = np.array([np.sin(yaw), 0.0, np.cos(yaw)])       # character forward
        self.left = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])     # character left (towards viewer right / camera)
        self.Rbody = np.stack([self.left, [0, 1, 0], self.fwd], axis=1)  # columns = local axes
        pt = np.radians(P["head_pitch"])
        Rx = np.array([[1, 0, 0], [0, np.cos(pt), -np.sin(pt)], [0, np.sin(pt), np.cos(pt)]])
        hy = np.radians(P["head_yaw"])
        Rh = np.stack([[np.cos(hy), 0.0, -np.sin(hy)], [0, 1, 0], [np.sin(hy), 0.0, np.cos(hy)]], axis=1)
        self.head_fwd = Rh[:, 2]
        self.Rhead = Rh @ Rx.T     # local up tilts towards the character's forward
        self.head_c, self.head_r = A("head_c"), A("head_r")
        self.torso_c, self.torso_r = A("torso_c"), A("torso_r")
        self.belly_c, self.belly_r = A("belly_c"), A("belly_r")
        self.eyes = []          # list of dict(center, normal, up)
        self.muzzle = None
        self.mouth = None
        self._setup_eyes()
        self._setup_mouth()
        self._setup_arm()
        self._mouth_on_carved_face()

    # ------------------------------------------------------------------ body
    def body_base(self, p):
        P = self.P
        d = self.head_only(p)
        d = smin(d, sd_ellipsoid(p, self.torso_c, self.torso_r, self.Rbody), P["k_head"])
        d = smin(d, sd_capsule(p, np.array(P["neck_a"]), np.array(P["neck_b"]), P["neck_ra"], P["neck_rb"]), P["k_hump"])
        d = smin(d, sd_ellipsoid(p, self.belly_c, self.belly_r), P["k_belly"])
        return smax(d, P["y_bottom"] - p[:, 1], P["k_bottom"])

    def head_only(self, p):
        d = sd_ellipsoid(p, self.head_c, self.head_r, self.Rhead)
        d = smin(d, sd_ellipsoid(p, np.array(self.P["jaw_c"]), np.array(self.P["jaw_r"]), self.Rhead), self.P["k_jaw"])
        if getattr(self, "muzzle", None) is not None:
            mz = self.muzzle
            d = smin(d, sd_ellipsoid(p, mz["c"], np.array(self.P["muzzle_r"]), mz["R"]), self.P["k_muzzle"])
        return d

    def mouth_groove(self, p):
        """distance to the mouth line (a tapered capsule chain lying on the muzzle surface)"""
        if getattr(self, "mouth", None) is None:
            return np.full(len(p), 1e9)
        pts, rr = self.mouth["pts"], self.mouth["r"]
        d = None
        for i in range(len(pts) - 1):
            c = sd_capsule(p, pts[i], pts[i + 1], rr[i], rr[i + 1])
            d = c if d is None else np.minimum(d, c)
        return d

    def body_rims(self, p):
        P = self.P
        d = self.body_base(p)
        for e in self.eyes:
            d = smin(d, sd_sphere(p, e["center"], P["eye_r"] + P["rim_h"]), P["k_rim"])
        return d

    def eye_hole(self, p, e):
        """volume removed for one eye: the ball (+ air gap) and the lid opening (elliptic cone through the lid shell)"""
        P = self.P
        q = p - e["center"]
        z, x, y = q @ e["axis"], q @ e["side"], q @ e["upl"]
        ay = np.where(y > 0, np.tan(np.radians(P["lid_open_up"])), np.tan(np.radians(P["lid_open_dn"])))
        ax = np.tan(np.radians(P["lid_open_x"]))
        cone = (np.sqrt((x / ax) ** 2 + (y / ay) ** 2) - z) * 0.62
        shell = np.linalg.norm(q, axis=1) - (P["eye_r"] + P["rim_h"] + 0.05)
        opening = smax(cone, shell, 0.01)
        return smin(sd_sphere(p, e["center"], P["eye_r"] + P["socket_gap"]), opening, P["k_lid"])

    def body(self, p):
        """body with eyelids and eye openings (still un-carved by the arm)."""
        P = self.P
        d = self.body_rims(p)
        for e in self.eyes:
            d = smax(d, -self.eye_hole(p, e), P["k_socket"])
        return d

    def eyeballs(self, p):
        d = np.full(len(p), 1e9)
        for e in self.eyes:
            d = np.minimum(d, sd_sphere(p, e["center"], self.P["eye_r"]))
        return d

    def collide_body(self, p):
        return np.minimum(self.body(p), self.eyeballs(p))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def grad(fn, p, eps=1e-4):
        g = np.zeros_like(p)
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            g[:, i] = fn(p + e) - fn(p - e)
        n = np.linalg.norm(g, axis=1, keepdims=True)
        return g / np.maximum(n, 1e-12)

    def on_surface(self, uv, fn, lift=0.0):
        """Shoot the camera ray of photo point uv at fn's zero set. Returns (point + lift*normal, normal)."""
        o, d = cam.ray(uv[0], uv[1])
        d = d[0]
        t = 1.6
        best = (1e9, t)
        for _ in range(400):
            q = (o + d * t)[None, :]
            dist = float(fn(q)[0])
            if dist < best[0]:
                best = (dist, t)
            if dist < 2e-5:
                break
            t += dist * 0.6
            if t > 5.5:
                break
        q = (o + d * best[1])[None, :]
        n = self.grad(fn, q)[0]
        if best[0] > 2e-5:          # the ray misses: fall back to the nearest surface point to the ray
            for _ in range(8):
                q = q - n * float(fn(q)[0])
                n = self.grad(fn, q)[0]
        return q[0] + n * lift, n

    @staticmethod
    def on_ray_at_z(uv, z):
        o, d = cam.ray(uv[0], uv[1])
        d = d[0]
        return o + d * ((z - o[2]) / d[2])

    # ------------------------------------------------------------------ eyes
    def _eye_center(self, az, el, depth=None):
        d = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
        lo, hi = 0.0, 0.6
        for _ in range(40):          # bisect the head surface along the radial direction
            mid = 0.5 * (lo + hi)
            if self.body_base((self.head_c + d * mid)[None, :])[0] < 0:
                lo = mid
            else:
                hi = mid
        return self.head_c + d * (lo - (depth if depth is not None else self.P["eye_depth"]) * self.P["eye_r"]), d

    def _setup_eyes(self):
        er = self.P["eye_right_uv"] or LM["landmarks"]["eye_right"]["center"]
        el = self.P["eye_left_uv"] or LM["landmarks"]["eye_left"]["center"]

        def resid(x):
            out = []
            for (az, e), target, dep in (((x[0], x[1]), er, None), ((x[2], x[3]), el, self.P["eye_depth_left"])):
                c, _ = self._eye_center(az, e, dep)
                u, v, _ = cam.project(c[None, :])
                out += [u[0] - target[0], (v[0] - target[1]) / cam.ASPECT]
            return out

        x0 = np.radians([2.0, -15.0, -60.0, -15.0])
        sol = least_squares(resid, x0, bounds=(np.radians([-40, -50, -120, -50]), np.radians([40, 30, -20, 30])))
        self.eye_fit_residual = float(np.abs(sol.fun).max())
        for az, e, dep in ((sol.x[0], sol.x[1], None), (sol.x[2], sol.x[3], self.P["eye_depth_left"])):
            c, n = self._eye_center(az, e, dep)
            up = unit(np.array([0, 1, 0]) - n * n[1])
            tl = np.radians(self.P["lid_tilt"])
            axis = unit(n * np.cos(tl) - up * np.sin(tl))
            side = unit(np.cross(up, axis))
            upl = np.cross(axis, side)
            self.eyes.append({"center": c, "normal": n, "up": up, "axis": axis, "side": side, "upl": upl,
                              "az": float(np.degrees(az)), "el": float(np.degrees(e))})

    # ------------------------------------------------------------------ mouth
    def _dir(self, az, el):
        az, el = np.radians(az), np.radians(el)
        return np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])

    def _surface_along(self, d, fn):
        lo, hi = 0.0, 0.6
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if fn((self.head_c + d * mid)[None, :])[0] < 0:
                lo = mid
            else:
                hi = mid
        return self.head_c + d * lo

    def _setup_mouth(self):
        """muzzle bump + smile line laid out in the FACE frame of the two eyes: the line runs parallel to the
        eye-to-eye axis, centred under the midpoint between the eyes, and is projected onto the face along the
        face normal (so it sits straight and symmetric however the head is turned)."""
        P = self.P
        c0, c1 = self.eyes[0]["center"], self.eyes[1]["center"]
        M = 0.5 * (c0 + c1)
        side = unit(c0 - c1)
        fwd = unit(self.eyes[0]["normal"] + self.eyes[1]["normal"])
        fwd = unit(fwd - side * (fwd @ side))
        down = unit(np.cross(fwd, side))
        if down[1] > 0:
            down = -down
        ew = np.linalg.norm(c0 - c1)                            # eye spacing (centre to centre)

        def project(q):                                          # march from in front of the face along -fwd
            t = 0.0
            o = q + fwd * 0.4
            for _ in range(200):
                d = float(self.head_only((o - fwd * t)[None, :])[0])
                if abs(d) < 1e-5:
                    break
                t += d * 0.8
            return o - fwd * t

        drop = P["mouth_drop"] * ew                              # mouth_drop / mouth_half are fractions of the eye spacing
        half = P["mouth_half"] * ew
        mz = project(M + down * (drop - P["muzzle_up"] * ew))
        self.muzzle = {"c": mz - fwd * P["muzzle_in"], "R": self.Rhead}
        pts, rr = [], []
        n = 11
        for i in range(n):
            t = -1 + 2 * i / (n - 1)
            q = M + side * (t * half) + down * (drop - P["mouth_smile"] * ew * t * t)
            pts.append(project(q))
            rr.append(P["mouth_r"] * (1.0 - 0.55 * t ** 4))
        self.mouth = {"pts": np.array(pts), "r": np.array(rr)}
        self._mouth_fwd, self._mouth_r = fwd, np.array(rr)

    def _mouth_on_carved_face(self):
        """move the mouth line onto the face surface as carved by the pressing hand (same face-frame projection)"""
        pts = self.mouth["pts"]
        self.mouth = None
        fwd = self._mouth_fwd
        face = lambda q: self.skin_parts(q)[1]          # carved body (without the arm itself)
        out = []
        for q in pts:
            o = q + fwd * 0.1
            t = 0.0
            for _ in range(300):
                d = float(face((o - fwd * t)[None, :])[0])
                if abs(d) < 1e-5:
                    break
                t += max(d, -0.01) * 0.6 if d > 0 else d * 0.6
            out.append(o - fwd * t)
        self.mouth = {"pts": np.array(out), "r": self._mouth_r}

    # ------------------------------------------------------------------ arm + hand
    def _setup_arm(self):
        P = self.P
        g = P["gap"]
        cb = self.collide_body
        # palm
        hit, n = self.on_surface(P["palm_uv"], cb)
        n = unit(n * 0.75 + unit(cam.CAM_POS - hit) * 0.25)   # face normal, slightly turned to the camera
        self.palm_n = n
        half = np.array(P["palm_half"])
        self.palm_c = hit + n * (half[2] + P["palm_lift"])
        # finger tips
        fr = P["finger_r"]
        tips = [self.on_surface(uv, cb, lift=fr[i] * P["finger_taper"] * P["finger_press"] + g)[0]
                for i, uv in enumerate(P["finger_tips_uv"])]
        f = np.mean(tips[:3], axis=0) - self.palm_c
        f = unit(f - n * (f @ n))
        thumb_tip = self.on_surface(P["thumb_tip_uv"], cb, lift=P["thumb_r"] * 0.9 * P["finger_press"] + g)[0]
        t = np.cross(n, f)
        if (thumb_tip - self.palm_c) @ t < 0:
            t = -t
        self.palm_f, self.palm_t = f, t
        self.palm_R = np.stack([f, t, n], axis=1)
        # finger chains: base on the palm's distal edge, two knuckles following the face, tip
        self.fingers = []
        for i, tip in enumerate(tips):
            base = self.palm_c + f * P["finger_base_f"] + t * P["finger_base_t"][i]
            ub, vb, _ = cam.project(base[None, :])
            pts = [base]
            for s in (0.40, 0.72):
                uv = (ub[0] + (P["finger_tips_uv"][i][0] - ub[0]) * s, vb[0] + (P["finger_tips_uv"][i][1] - vb[0]) * s)
                pts.append(self.on_surface(uv, cb, lift=fr[i] * P["finger_press"] + g + 0.004)[0])
            pts.append(tip)
            self.fingers.append({"pts": np.array(pts), "r0": fr[i], "r1": fr[i] * P["finger_taper"]})
        tb = self.palm_c + f * P["thumb_base"][0] + t * P["thumb_base"][1]
        ub, vb, _ = cam.project(tb[None, :])
        uvm = (0.5 * (ub[0] + P["thumb_tip_uv"][0]), 0.5 * (vb[0] + P["thumb_tip_uv"][1]))
        tm = self.on_surface(uvm, cb, lift=P["thumb_r"] * P["finger_press"] + g + 0.006)[0]
        self.thumb = {"pts": np.array([tb, tm, thumb_tip]), "r0": P["thumb_r"] * 1.05, "r1": P["thumb_r"] * 0.9}
        # digits never cut into an eyeball: push chain points (and segment mid-points' owners) radially off the balls
        for ch in self.fingers + [self.thumb]:
            pts = ch["pts"]
            for _ in range(6):
                moved = False
                for i in range(len(pts)):
                    rr = ch["r0"] + (ch["r1"] - ch["r0"]) * (i / (len(pts) - 1))
                    for e in self.eyes:
                        q = pts[i] - e["center"]
                        need = P["eye_r"] + g + rr + 0.004
                        dq = np.linalg.norm(q)
                        if dq < need:
                            pts[i] = e["center"] + q / dq * need
                            moved = True
                for i in range(len(pts) - 1):       # segment interior
                    for e in self.eyes:
                        a, b = pts[i], pts[i + 1]
                        t = np.clip((e["center"] - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
                        c = a + (b - a) * t
                        rr = ch["r0"] + (ch["r1"] - ch["r0"]) * ((i + t) / (len(pts) - 1))
                        q = c - e["center"]
                        need = P["eye_r"] + g + rr + 0.004
                        dq = np.linalg.norm(q)
                        if dq < need:
                            push = q / dq * (need - dq)
                            pts[i] = pts[i] + push * (1 - t) * 1.2
                            pts[i + 1] = pts[i + 1] + push * t * 1.2
                            moved = True
                if not moved:
                    break
        # arm joints
        self.wrist = self.on_ray_at_z(P["wrist_uv"], self.palm_c[2] + P["wrist_dz"])
        eh, en = self.on_surface(P["elbow_uv"], self.body_base)
        self.elbow = eh + unit(cam.CAM_POS - eh) * (P["r_elbow"] * P["elbow_lift"])
        self.shoulder = np.array(P["shoulder"], dtype=np.float64)
        self.fore_dir = unit(self.wrist - self.elbow)

    def _chain(self, p, ch, k):
        pts = ch["pts"]
        n = len(pts) - 1
        d = None
        for i in range(n):
            ra = ch["r0"] + (ch["r1"] - ch["r0"]) * (i / n)
            rb = ch["r0"] + (ch["r1"] - ch["r0"]) * ((i + 1) / n)
            c = sd_capsule(p, pts[i], pts[i + 1], ra, rb)
            d = c if d is None else smin(d, c, k)
        return d

    def hand(self, p):
        P = self.P
        d = sd_rbox(p, self.palm_c, self.palm_R, np.array(P["palm_half"]), P["palm_round"])
        for ch in self.fingers:
            d = smin(d, self._chain(p, ch, P["k_knuckle"]), P["k_finger"])
        d = smin(d, self._chain(p, self.thumb, P["k_knuckle"]), P["k_finger"] * 1.3)
        return d

    def arm_parts(self, p):
        P = self.P
        upper = sd_capsule(p, self.shoulder, self.elbow, P["r_shoulder"], P["r_elbow"])
        fore = sd_capsule(p, self.elbow, self.wrist, P["r_elbow"], P["r_wrist"])
        # the forearm continues into the palm so the wrist is one tapering volume
        wristlink = sd_capsule(p, self.wrist, self.palm_c - self.palm_f * 0.03, P["r_wrist"], P["r_wrist"] * 0.78)
        hand = self.hand(p)
        return upper, fore, wristlink, hand

    def arm(self, p):
        P = self.P
        upper, fore, link, hand = self.arm_parts(p)
        d = smin(upper, fore, P["k_elbow"])
        d = smin(d, link, P["k_wrist"])
        return smin(d, hand, P["k_wrist"])

    # ------------------------------------------------------------------ full skin
    def fuse_weight(self, p):
        r = np.linalg.norm(p - self.shoulder, axis=1)
        return 1.0 - smoothstep(self.P["fuse_r0"], self.P["fuse_r1"], r)

    def skin_parts(self, p):
        """returns (d_skin, d_body_carved, d_arm)"""
        P = self.P
        db = self.body_rims(p)
        da = self.arm(p)
        # volume removed from the body: eye sockets and the arm inflated by GAP. The two are blended
        # (smooth union) so that no paper-thin wall of skin can survive between a socket and a finger groove.
        removed = da - P["gap"]
        holes = [self.eye_hole(p, e) for e in self.eyes]
        for h in holes:
            removed = smin(removed, h, P["k_wall"])
        carved = smax(db, -removed, P["k_carve"])
        # eye openings only (used where the arm is fused): keep them crisp
        sock = db
        for h in holes:
            sock = smax(sock, -h, P["k_socket"])
        carved = np.maximum(carved, sock)
        # the mouth line is cut LAST, into the face as the hand left it (it lies under the fingers in the cover pose
        # and shows as soon as the hand is pulled away)
        if getattr(self, "mouth", None) is not None:
            g = self.mouth_groove(p)
            carved = smax(carved, -g, P["k_mouth"])
            sock = smax(sock, -g, P["k_mouth"])
        sep = np.minimum(carved, da)
        fused = smin(sock, da, P["k_shoulder"])
        w = self.fuse_weight(p)
        return sep + (fused - sep) * w, carved, da

    def skin(self, p):
        return self.skin_parts(p)[0]

    def skin_chunked(self, p, chunk=400000, level=0.0):
        out = np.empty(len(p), dtype=np.float32)
        for i in range(0, len(p), chunk):
            out[i:i + chunk] = self.skin(p[i:i + chunk].astype(np.float64)) - level
        return out

    # ------------------------------------------------------------------ labels (rest-space, per point)
    def labels(self, p):
        """Material / physics partition weights for surface (or interior) points."""
        P = self.P
        d, carved, da = self.skin_parts(p)
        w_fuse = self.fuse_weight(p)
        is_arm = (da < carved).astype(np.float64)
        upper, fore, link, hand = self.arm_parts(p)
        # olive hand colour: fades into yellow along the wrist / lower forearm
        s = (p - self.wrist) @ self.fore_dir
        hand_w = is_arm * smoothstep(-0.085, 0.075, s) * (1.0 - w_fuse)
        # belly patch: elliptic cylinder along the character's forward axis, front side only
        BP = P["patch"]
        q = (p - self.belly_c) @ self.Rbody
        qy = q[:, 1] - BP["y0"]
        e = np.sqrt(((q[:, 0] - BP["x0"]) / BP["a"]) ** 2 + (qy / np.where(qy > 0, BP["b_up"], BP["b_dn"])) ** 2)
        belly_w = (1.0 - is_arm) * (1.0 - smoothstep(1.0 - BP["soft"], 1.0 + BP["soft"], e)) * smoothstep(-0.05, 0.05, q[:, 2])
        # socket rim
        de = np.full(len(p), 1e9)
        for ey in self.eyes:
            de = np.minimum(de, np.linalg.norm(p - ey["center"], axis=1))
        socket_w = (1.0 - is_arm) * (1.0 - smoothstep(P["eye_r"] * 1.0, P["eye_r"] * 1.45, de))
        # physics regions
        d_head = self.head_only(p)
        head_w = (1.0 - is_arm) * (1.0 - smoothstep(-0.02, 0.10, d_head))
        face_w = head_w * smoothstep(-0.05, 0.12, (p - self.head_c) @ self.head_fwd)   # cheeks + forehead
        d_belly = sd_ellipsoid(p, self.belly_c, self.belly_r)
        bellyvol_w = (1.0 - is_arm) * (1.0 - smoothstep(-0.05, 0.12, d_belly))
        hand_solid = is_arm * (1.0 - smoothstep(0.0, 0.05, np.minimum(hand, link)))
        mouth_w = (1.0 - is_arm) * (1.0 - smoothstep(0.0, P["mouth_r"] * 1.4, self.mouth_groove(p)))
        return dict(is_arm=is_arm, hand=hand_w, belly=belly_w, socket=socket_w, head=head_w, face=face_w, mouth=mouth_w,
                    bellyvol=bellyvol_w, hand_solid=hand_solid, fuse=w_fuse, eye_dist=de)
