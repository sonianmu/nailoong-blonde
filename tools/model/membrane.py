"""Data for the Sheen-style GPU membrane (src/sim/membrane.ts).

  * ordered 1-rings (for Jacobi edge constraints and for vertex normals, like sheen/src/modules/pre.js)
  * per-vertex material: shape-recovery acceleration, constraint stiffness, group, collision flags, pin
  * render binding: every fine skin vertex / droplet sits on a membrane triangle:
        x = PhongTess(bary, x_i, n_i) + T r_t + B r_b + N r_n          (r = residual in the triangle frame)
    exact at rest; the curved (Phong) interpolation keeps a stretched membrane round, the residual keeps the
    fine detail (mouth line, eyelid edges, finger creases) that the coarser membrane cannot carry.
  * eyes: a ring of lid vertices per eye (centroid + 3 anchors for the frame)
  * collision proxies: spheres inside the hand / forearm / upper arm (ride on nearby arm vertices) and the
    two eyeballs (ride on the lid ring); face / chest vertices are pushed out of the arm spheres, hand vertices
    out of the eyeballs.
"""
import numpy as np
from scipy.spatial import cKDTree

PHONG_ALPHA = 0.75
MAX_RING = 12


def unit(a, axis=-1):
    return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), 1e-12)


def ordered_rings(nv, f):
    """for every vertex: its neighbours in fan order (counter-clockwise around the outward normal)"""
    nxt = [dict() for _ in range(nv)]
    for a, b, c in f:
        nxt[a][b] = c
        nxt[b][c] = a
        nxt[c][a] = b
    rings = []
    for i in range(nv):
        d = nxt[i]
        start = next(iter(d))
        ring = [start]
        cur = start
        while True:
            cur = d.get(cur)
            if cur is None or cur == start:
                break
            ring.append(cur)
            if len(ring) > 32:
                break
        rings.append(ring)
    return rings


def vertex_normals(v, rings):
    n = np.zeros_like(v)
    for i, r in enumerate(rings):
        p0 = v[i]
        acc = np.zeros(3)
        for k in range(len(r)):
            acc += np.cross(v[r[k]] - p0, v[r[(k + 1) % len(r)]] - p0)
        n[i] = acc
    return unit(n)


def phong(P, w, X, N):
    """Phong tessellation of the triangle (X: (n,3,3) corners, N: corner normals, w: (n,3) barycentrics)"""
    flat = (X * w[:, :, None]).sum(1)
    proj = np.zeros_like(flat)
    for k in range(3):
        d = flat - X[:, k]
        proj += w[:, k:k + 1] * (flat - N[:, k] * (d * N[:, k]).sum(1, keepdims=True))
    return (1 - PHONG_ALPHA) * flat + PHONG_ALPHA * proj


def tri_frame(X, N, w):
    """local frame of a bound point: N = interpolated vertex normal, T along edge 0->1 projected, B = N x T"""
    n = unit((N * w[:, :, None]).sum(1))
    e = X[:, 1] - X[:, 0]
    t = unit(e - n * (e * n).sum(1, keepdims=True))
    b = np.cross(n, t)
    return t, b, n


def closest_on_tris(p, a, b, c):
    from build import closest_on_tris as cot
    return cot(p, a, b, c)


def bind_points(p, pn, sv, sn, sf, s_group, p_group, K=16):
    """bind points (positions p, normals pn) to the nearest membrane triangle of an allowed group.
    group: 0 body, 1 arm, 2 either. Returns tri ids, barycentrics, residual (t,b,n), normal coeffs (t,b,n), distance"""
    tg = s_group[sf]                                     # (nf,3)
    tri_body = (tg != 1).all(1)
    tri_arm = (tg != 0).all(1)
    cen = sv[sf].mean(1)
    out_t = np.zeros(len(p), np.int64)
    out_w = np.zeros((len(p), 3))
    best = np.full(len(p), np.inf)
    for g in (0, 1, 2):
        sel = np.where(p_group == g)[0]
        if not len(sel):
            continue
        allowed = np.where(tri_body if g == 0 else tri_arm if g == 1 else np.ones(len(sf), bool))[0]
        tree = cKDTree(cen[allowed])
        _, idx = tree.query(p[sel], k=K)
        for j in range(K):
            t = allowed[idx[:, j]]
            tri = sf[t]
            w = closest_on_tris(p[sel], sv[tri[:, 0]], sv[tri[:, 1]], sv[tri[:, 2]])
            q = (sv[tri] * w[:, :, None]).sum(1)
            d = np.linalg.norm(p[sel] - q, axis=1)
            better = d < best[sel]
            ii = sel[better]
            best[ii] = d[better]
            out_t[ii] = t[better]
            out_w[ii] = w[better]
    X, N = sv[sf[out_t]], sn[sf[out_t]]
    base = phong(None, out_w, X, N)
    T, B, Nn = tri_frame(X, N, out_w)
    r = p - base
    resid = np.stack([(r * T).sum(1), (r * B).sum(1), (r * Nn).sum(1)], 1)
    ncoef = np.stack([(pn * T).sum(1), (pn * B).sum(1), (pn * Nn).sum(1)], 1)
    return out_t, out_w, resid, ncoef, best


def eye_rings(sv, s_group, eyes, r_eye, rim, n_ring=24):
    """lid ring of every eye: membrane vertices on the lid around the opening + 3 anchors (up, side+, side-)"""
    out = []
    for e in eyes:
        c = e["center"]
        q = sv - c
        d = np.linalg.norm(q, axis=1)
        z = q @ e["axis"]
        cand = np.where((s_group != 1) & (d > r_eye + 0.002) & (d < r_eye + rim + 0.03) & (z > -0.2 * r_eye))[0]
        # spread over the angle around the axis
        ang = np.arctan2(q[cand] @ e["upl"], q[cand] @ e["side"])
        order = np.argsort(ang)
        cand = cand[order][np.linspace(0, len(cand) - 1, min(n_ring, len(cand))).astype(int)]
        ring = list(dict.fromkeys(cand.tolist()))
        qa = sv[ring] - c
        up = ring[int(np.argmax(qa @ e["upl"]))]
        sp = ring[int(np.argmax(qa @ e["side"]))]
        sm = ring[int(np.argmin(qa @ e["side"]))]
        out.append({"ring": ring, "anchors": [up, sp, sm]})
    return out


def arm_proxies(m, sv, s_group):
    """spheres inside the arm (fingers, thumb, palm, wrist, forearm, upper arm); each rides on the K nearest arm
    membrane vertices: centre = mean(current) + (rest centre - mean(rest))"""
    P = m.P
    spheres = []

    def chain(pts, r0, r1, step=0.6):
        n = len(pts) - 1
        for i in range(n):
            a, b = pts[i], pts[i + 1]
            ra = r0 + (r1 - r0) * i / n
            rb = r0 + (r1 - r0) * (i + 1) / n
            L = np.linalg.norm(b - a)
            k = max(1, int(np.ceil(L / (step * min(ra, rb)))))
            for j in range(k + (1 if i == n - 1 else 0)):
                t = j / k
                spheres.append((a + (b - a) * t, ra + (rb - ra) * t))

    for ch in m.fingers + [m.thumb]:
        chain(ch["pts"], ch["r0"], ch["r1"])
    half = np.array(P["palm_half"])
    for u in np.linspace(-0.6, 0.6, 4):
        for w in np.linspace(-0.6, 0.6, 4):
            c = m.palm_c + m.palm_f * (u * half[0]) + m.palm_t * (w * half[1])
            spheres.append((c, half[2] * 0.95))
    chain([m.wrist, m.palm_c - m.palm_f * 0.03], P["r_wrist"], P["r_wrist"] * 0.78)
    chain([m.elbow, m.wrist], P["r_elbow"], P["r_wrist"], step=0.5)
    chain([m.shoulder + (m.elbow - m.shoulder) * 0.35, m.elbow], P["r_shoulder"] * 0.9, P["r_elbow"], step=0.5)
    centres = np.array([s[0] for s in spheres])
    radii = np.array([s[1] for s in spheres]) * 0.96
    arm_v = np.where(s_group == 1)[0]
    tree = cKDTree(sv[arm_v])
    binds = []
    for c, r in zip(centres, radii):
        idx = tree.query_ball_point(c, r * 1.6)
        if len(idx) < 4:
            _, idx = tree.query(c, k=8)
        idx = np.array(idx)
        if len(idx) > 24:
            dd = np.linalg.norm(sv[arm_v[idx]] - c, axis=1)
            idx = idx[np.argsort(dd)[:24]]
        binds.append(arm_v[idx].tolist())
    return centres, radii, binds
