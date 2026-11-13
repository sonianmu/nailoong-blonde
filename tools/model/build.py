"""Deterministic asset build:  python tools/model/build.py   (npm run model)

  SDF (tools/model/sdf.py)
    -> narrow-band sparse marching cubes (voxel 0.0018 H, only blocks that straddle the surface)
    -> Taubin smoothing -> region-adaptive isotropic explicit remeshing (pymeshlab)
    -> re-projection of every vertex onto the SDF zero set, normals from the SDF gradient
    -> topology / bounding box / edge length / curvature / hand-face gap checks
  -> simulation membrane (coarser remesh of the same surface, Sheen-style GPU cloth on a closed surface):
     ordered 1-rings, rest lengths, per-region shape recovery, collision proxies (arm spheres, eyeballs),
     render binding (Phong-tessellated membrane triangle + residual), eye lid rings, droplets

Outputs: public/assets/skin.glb, public/assets/pack.bin + pack.json, public/assets/scene.json,
         reference/mask_model.png, docs/model-report.json
"""
import json
import os
import sys
import time

import numpy as np
import pymeshlab
import trimesh
from scipy.spatial import cKDTree
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera as cam
from sdf import Model, LM, ROOT, H_CHAR

OUT = os.path.join(ROOT, "public", "assets")
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)

VOXEL = 0.0040            # 0.0016 H  (requirement: <= 0.004 H)
L_FINE, L_MID, L_BODY, L_HIDDEN = 0.0042, 0.0066, 0.0098, 0.020   # target edge lengths (world); H = 2.5
INSET = 0.009             # cage shell = SDF iso-surface at -INSET
CAGE_VOXEL = 0.006
SURF_EPS = 0.0012          # final cage surface sits this far inside the skin
CL_FINGER, CL_HAND, CL_MID, CL_BODY = 0.0135, 0.021, 0.046, 0.072
PIN_Y = -0.80
N_DROPS, N_STREAKS = 2200, 44
REPORT = {}
T0 = time.time()


def log(*a):
    print("[%6.1fs]" % (time.time() - T0), *a, flush=True)


# --------------------------------------------------------------------------- iso-surface extraction
def extract(m, level, h, pad=0.03):
    """Narrow-band marching cubes of {skin = level}. Only 8^3 blocks that straddle the surface are sampled finely."""
    lo = np.array([-0.75, -1.16, -0.86])
    hi = np.array([0.95, 0.56, 0.74])
    B = 8
    nb = np.ceil((hi - lo) / (h * B)).astype(int)
    n = nb * B + 1
    # coarse pass at block centres
    cx = [lo[i] + (np.arange(nb[i]) + 0.5) * h * B for i in range(3)]
    C = np.stack(np.meshgrid(*cx, indexing="ij"), axis=-1).reshape(-1, 3)
    dc = m.skin_chunked(C, level=level).reshape(nb)
    band = np.abs(dc) < h * B * 0.5 * np.sqrt(3) * 1.6 + pad * 0
    vol = np.empty(n, dtype=np.float32)
    # default: sign of the nearest block centre (never crosses zero outside the band)
    sign = np.where(dc < 0, -1.0, 1.0).astype(np.float32)
    vol[:-1, :-1, :-1] = np.repeat(np.repeat(np.repeat(sign, B, 0), B, 1), B, 2)
    vol[-1, :, :] = vol[-2, :, :]
    vol[:, -1, :] = vol[:, -2, :]
    vol[:, :, -1] = vol[:, :, -2]
    blocks = np.argwhere(band)
    off = np.stack(np.meshgrid(*[np.arange(B + 1)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    CH = 600
    for s in range(0, len(blocks), CH):
        bl = blocks[s:s + CH]
        idx = (bl[:, None, :] * B + off[None, :, :]).reshape(-1, 3)
        d = m.skin_chunked(lo + idx * h, level=level)
        vol[idx[:, 0], idx[:, 1], idx[:, 2]] = d
    assert (vol[0] > 0).all() and (vol[-1] > 0).all() and (vol[:, :, 0] > 0).all() and (vol[:, :, -1] > 0).all() and (vol[:, -1] > 0).all(), "surface touches the grid bounds"
    v, f, _, _ = measure.marching_cubes(vol, level=0.0, spacing=(h, h, h))
    return v + lo, f.astype(np.int64), int(band.sum()), tuple(int(x) for x in n)


def largest_component(v, f):
    """Drop numerical specks (a few 8-triangle bubbles where two SDF branches cross inside the solid)."""
    tm = trimesh.Trimesh(v, f, process=False)
    comps = trimesh.graph.connected_components(tm.face_adjacency, nodes=np.arange(len(f)))
    big = max(comps, key=len)
    dropped = int(len(f) - len(big))
    ff = f[np.sort(big)]
    used = np.unique(ff)
    remap = -np.ones(len(v), np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[ff], dropped, len(comps) - 1


def project_to_surface(m, v, level=0.0, iters=4):
    v = v.copy()
    for _ in range(iters):
        for s in range(0, len(v), 300000):
            p = v[s:s + 300000]
            d = m.skin(p) - level
            g = np.zeros_like(p)
            e = 1e-4
            for i in range(3):
                dp = np.zeros(3)
                dp[i] = e
                g[:, i] = (m.skin(p + dp) - m.skin(p - dp)) / (2 * e)
            gg = np.maximum((g * g).sum(1), 1e-6)
            step = np.clip(d / gg, -0.01, 0.01)
            v[s:s + 300000] = p - g * step[:, None]
    return v


def sdf_normals(m, v):
    out = np.zeros_like(v)
    for s in range(0, len(v), 300000):
        out[s:s + 300000] = Model.grad(m.skin, v[s:s + 300000], 1.5e-4)
    return out


# --------------------------------------------------------------------------- remeshing
def face_field(v, f, vals):
    return vals[f].mean(axis=1)


def remesh_pass(v, f, sel_vertex_mask, target, iters=5):
    q = sel_vertex_mask.astype(np.float64)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f, v_scalar_array=q))
    if sel_vertex_mask.all():
        ms.meshing_isotropic_explicit_remeshing(iterations=iters, targetlen=pymeshlab.PureValue(target), adaptive=False,
                                                featuredeg=180, checksurfdist=False, reprojectflag=True)
    else:
        ms.compute_selection_by_condition_per_face(condselect="(q0+q1+q2) > 1.5")
        ms.meshing_isotropic_explicit_remeshing(iterations=iters, targetlen=pymeshlab.PureValue(target), adaptive=False,
                                                featuredeg=180, checksurfdist=False, reprojectflag=True, selectedonly=True)
    cm = ms.current_mesh()
    return cm.vertex_matrix().copy(), cm.face_matrix().astype(np.int64).copy()


def zone_lengths(m, v, nrm):
    """Target edge length per point (PROMPT §3.3)."""
    P = m.P
    lab = m.labels(v)
    d, carved, da = m.skin_parts(v)
    db = m.body(v)
    L = np.full(len(v), L_BODY)
    arm = lab["is_arm"] > 0.5
    mid = arm | (lab["head"] > 0.25) | (np.linalg.norm(v - np.array(P["neck_a"]), axis=1) < 0.36)
    L[mid] = L_MID
    upper, fore, link, hand = m.arm_parts(v)
    fine = (arm & (np.minimum(hand, link) < 0.03))                           # hand, fingers, wrist
    fine |= lab["eye_dist"] < P["eye_r"] * 1.75                              # eyelids
    fine |= m.mouth_groove(v) < 0.03                                          # mouth line
    fine |= np.linalg.norm(v - m.elbow, axis=1) < P["r_elbow"] * 1.25        # elbow
    fine |= (~arm) & (da < 0.045)                                            # skin pressed by the arm / hand (contact folds)
    fine |= arm & (db < 0.045) & (lab["fuse"] < 0.999)                       # arm side of the same folds
    fine |= (lab["fuse"] > 0.02) & (lab["fuse"] < 0.98) & (np.abs(db - da) < 0.10)   # armpit / shoulder fusion seam
    L[fine] = L_FINE
    u, w, _ = cam.project(v)
    view = cam.CAM_POS - v
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    hidden = ((nrm * view).sum(1) < -0.28) | (v[:, 1] < -0.86) | (u > 1.16) | (w > 1.12)
    hidden &= ~fine
    L[hidden] = L_HIDDEN
    return L


def build_skin(m):
    log("extracting high-resolution iso-surface, voxel", VOXEL)
    v, f, nblocks, dims = extract(m, 0.0, VOXEL)
    v, f, dropped, nspeck = largest_component(v, f)
    eul = trimesh.Trimesh(v, f, process=False).euler_number
    assert eul == 2, "skin is not a genus-0 surface (euler %d)" % eul
    REPORT["highres"] = {"removed_numerical_specks": nspeck, "removed_speck_triangles": dropped, "euler": int(eul),"voxel": VOXEL, "voxel_over_H": VOXEL / H_CHAR, "grid": dims, "narrow_band_blocks": nblocks,
                         "triangles": int(len(f)), "vertices": int(len(v))}
    log("  high-res mesh: %d tris, %d band blocks of %s grid" % (len(f), nblocks, dims))
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f))
    ms.meshing_remove_duplicate_vertices()
    ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=6)
    cm = ms.current_mesh()
    v, f = cm.vertex_matrix().copy(), cm.face_matrix().astype(np.int64).copy()
    v = project_to_surface(m, v, iters=2)

    def lengths(v):
        return zone_lengths(m, v, sdf_normals(m, v))

    log("remesh pass A (all -> %.4f)" % L_BODY)
    v, f = remesh_pass(v, f, np.ones(len(v), bool), L_BODY, iters=6)
    v = project_to_surface(m, v, iters=2)
    L = lengths(v)
    log("remesh pass B (hidden -> %.4f)" % L_HIDDEN)
    v, f = remesh_pass(v, f, L >= L_HIDDEN - 1e-9, L_HIDDEN, iters=5)
    v = project_to_surface(m, v, iters=2)
    L = lengths(v)
    log("remesh pass C (head/arm -> %.4f)" % L_MID)
    v, f = remesh_pass(v, f, L <= L_MID + 1e-9, L_MID, iters=6)
    v = project_to_surface(m, v, iters=2)
    L = lengths(v)
    log("remesh pass D (hand/sockets/joints -> %.4f)" % L_FINE)
    v, f = remesh_pass(v, f, L <= L_FINE + 1e-9, L_FINE, iters=6)
    v = project_to_surface(m, v, iters=5)
    v, nfix = unfold(m, v, f)
    REPORT["unfold_vertices"] = nfix
    log("  render mesh: %d tris, %d verts (%d vertices relaxed out of folds)" % (len(f), len(v), nfix))
    return v, f


def unfold(m, v, f, limit=75.0, rounds=12):
    """Remeshing can leave a few folded triangles inside the deepest creases (forearm / chest groove). Relax the
    vertices around any edge sharper than `limit` towards their 1-ring centroid and re-project them onto the SDF."""
    v = v.copy()
    tm = trimesh.Trimesh(v, f, process=False)
    nb = tm.vertex_neighbors
    touched = set()
    for _ in range(rounds):
        tm = trimesh.Trimesh(v, f, process=False)
        fn = tm.face_normals
        adj = tm.face_adjacency
        ang = np.degrees(np.arccos(np.clip((fn[adj[:, 0]] * fn[adj[:, 1]]).sum(1), -1, 1)))
        bad = ang > limit
        if not bad.any():
            break
        ids = np.unique(f[adj[bad]].ravel())
        ring = set(ids.tolist())
        for i in ids:
            ring.update(nb[i])
        ids = np.array(sorted(ring))
        touched.update(ids.tolist())
        cen = np.array([v[nb[i]].mean(0) for i in ids])
        v[ids] = project_to_surface(m, cen, iters=3)
    return v, len(touched)


# --------------------------------------------------------------------------- simulation membrane (Sheen-style)
S_FINE, S_MID, S_BODY, S_HIDDEN = 0.0085, 0.0135, 0.021, 0.036


def sim_lengths(m, v, nrm):
    L = zone_lengths(m, v, nrm)
    out = np.full(len(v), S_BODY)
    out[L <= L_MID + 1e-9] = S_MID
    out[L <= L_FINE + 1e-9] = S_FINE
    out[L >= L_HIDDEN - 1e-9] = S_HIDDEN
    return out


def build_sim(m, v, f):
    """The surface that is SIMULATED (positions live in GPU textures, Sheen-style). Same SDF surface as the skin,
    remeshed coarser: Jacobi constraint relaxation must reach across the head within a frame."""
    log("simulation membrane: remeshing the skin")
    lens = lambda v: sim_lengths(m, v, sdf_normals(m, v))
    v, f = remesh_pass(v, f, np.ones(len(v), bool), S_BODY, iters=6)
    v = project_to_surface(m, v, iters=2)
    for target in (S_HIDDEN, S_MID, S_FINE):
        L = lens(v)
        v, f = remesh_pass(v, f, np.abs(L - target) < 1e-9, target, iters=6)
        v = project_to_surface(m, v, iters=3)
    v, _ = unfold(m, v, f, limit=55.0, rounds=30)
    topo = topology(v, f)
    log("  membrane: %d vertices, %d triangles, %s" % (len(v), len(f), topo))
    assert topo["boundary_edges"] == 0 and topo["non_manifold_edges"] == 0 and topo["components"] == 1 and topo["euler"] == 2
    return v, f


# --------------------------------------------------------------------------- mesh checks
def topology(v, f):
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    tm = trimesh.Trimesh(v, f, process=False)
    comps = trimesh.graph.connected_components(tm.face_adjacency, nodes=np.arange(len(f)))
    return {"boundary_edges": int((counts == 1).sum()), "non_manifold_edges": int((counts > 2).sum()),
            "components": int(len(comps)), "watertight": bool(tm.is_watertight), "euler": int(tm.euler_number),
            "bbox_min": v.min(0).round(4).tolist(), "bbox_max": v.max(0).round(4).tolist()}


def edge_stats(v, f, L):
    e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
    ln = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    Le = np.minimum(L[e[:, 0]], L[e[:, 1]])
    out = {}
    for name, lv, limit in (("fine_hand_sockets_joints", L_FINE, 0.003), ("head_arms", L_MID, 0.005), ("body", L_BODY, 0.008), ("hidden", L_HIDDEN, None)):
        sel = np.abs(Le - lv) < 1e-9
        if sel.any():
            out[name] = {"edges": int(sel.sum()), "mean": float(ln[sel].mean()), "mean_over_H": float(ln[sel].mean() / H_CHAR),
                         "p95_over_H": float(np.percentile(ln[sel], 95) / H_CHAR), "limit_over_H": limit,
                         "ok": bool(limit is None or ln[sel].mean() / H_CHAR <= limit)}
    return out


def curvature_report(m, v, f, L):
    """Discrete mean curvature (cotangent Laplacian) and its jump across edges, plus dihedral angles, in the joint zones."""
    tm = trimesh.Trimesh(v, f, process=False)
    # cotan mean curvature normal
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]

    def cot(p, q, r):  # angle at p
        u, w = q - p, r - p
        return (u * w).sum(1) / np.maximum(np.linalg.norm(np.cross(u, w), axis=1), 1e-12)

    Hn = np.zeros_like(v)
    area = np.zeros(len(v))
    fa = tm.area_faces
    for i, (p, q, r) in enumerate(((a, b, c), (b, c, a), (c, a, b))):
        ct = cot(p, q, r)           # angle at p, opposite edge q-r
        qi, ri = f[:, (i + 1) % 3], f[:, (i + 2) % 3]
        np.add.at(Hn, qi, (q - r) * ct[:, None] * 0.5)
        np.add.at(Hn, ri, (r - q) * ct[:, None] * 0.5)
        np.add.at(area, f[:, i], fa / 3)
    nrm = sdf_normals(m, v)
    Hm = (Hn * nrm).sum(1) / np.maximum(area, 1e-12) * 0.5
    e = tm.edges_unique
    jump = np.abs(Hm[e[:, 0]] - Hm[e[:, 1]])
    ln = tm.edges_unique_length
    joint = np.minimum(L[e[:, 0]], L[e[:, 1]]) <= L_FINE + 1e-9
    # dimensionless: curvature change per edge relative to the local curvature scale 1/edge
    rel = jump * ln
    dih = np.degrees(tm.face_adjacency_angles)
    fj = (L[tm.face_adjacency_edges[:, 0]] <= L_FINE + 1e-9)
    return {
        "joint_edges": int(joint.sum()),
        "mean_curvature_abs_p99_joint": float(np.percentile(np.abs(Hm[L <= L_FINE + 1e-9]), 99)),
        "curv_jump_rel_p99_joint": float(np.percentile(rel[joint], 99)), "curv_jump_rel_max_joint": float(rel[joint].max()),
        "curv_jump_rel_p99_all": float(np.percentile(rel, 99)),
        "dihedral_deg_p99_joint": float(np.percentile(dih[fj], 99)), "dihedral_deg_max_joint": float(dih[fj].max()),
        "dihedral_deg_p99_other": float(np.percentile(dih[~fj], 99)), "dihedral_deg_max_other": float(dih[~fj].max()),
        "note": "rel = |H_i - H_j| * edge length (dimensionless); dihedral = angle between adjacent face normals. "
                "Shading normals come from the SDF gradient, so facets only matter on silhouettes.",
    }, Hm


# --------------------------------------------------------------------------- cage
def closest_on_tris(p, a, b, c):
    """closest points on triangles (vectorised, Ericson). p,a,b,c: (N,3). returns (N,3) barycentric weights."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    denom = np.maximum(va + vb + vc, 1e-20)
    v, w = vb / denom, vc / denom
    out = np.stack([1 - v - w, v, w], axis=1)
    m_ = (d1 <= 0) & (d2 <= 0)
    out[m_] = [1, 0, 0]
    m_ = (d3 >= 0) & (d4 <= d3)
    out[m_] = [0, 1, 0]
    m_ = (d6 >= 0) & (d5 <= d6)
    out[m_] = [0, 0, 1]
    m_ = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.maximum(d1 - d3, 1e-20)
    out[m_] = np.stack([1 - t, t, np.zeros_like(t)], 1)[m_]
    m_ = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.maximum(d2 - d6, 1e-20)
    out[m_] = np.stack([1 - t, np.zeros_like(t), t], 1)[m_]
    m_ = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-20)
    out[m_] = np.stack([np.zeros_like(t), 1 - t, t], 1)[m_]
    return out



# --------------------------------------------------------------------------- droplets
def make_droplets(m, v, f, nrm, lab):
    rng = np.random.default_rng(20240817)
    tm = trimesh.Trimesh(v, f, process=False)
    fc = tm.triangles_center
    fn = nrm[f].mean(axis=1)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    view = cam.CAM_POS - fc
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    facing = np.clip((fn * view).sum(1) + 0.15, 0, 1)
    u, w, _ = cam.project(fc)
    fl = {k: face_field(v, f, val) for k, val in lab.items()}
    dens = np.full(len(f), 0.55)
    head_top = np.clip((fc[:, 1] - 0.15) / 0.25, 0, 1) * fl["head"]
    head_right = np.clip((fc[:, 0] - 0.02) / 0.2, 0, 1) * np.clip((fc[:, 1] + 0.25) / 0.3, 0, 1)
    dens += 1.6 * head_top + 1.5 * head_right
    dens += 0.9 * fl["is_arm"] * (1 - fl["hand"])
    dens += 0.5 * fl["belly"]
    dens *= 1 - 0.75 * fl["hand"]
    dens *= np.clip(1 - fl["socket"] * 3, 0, 1)
    dens *= facing * (u > -0.02) * (u < 1.03) * (w < 1.04)
    dens *= np.clip(0.35 + 0.65 * (fn[:, 1] * 0.5 + 0.5 + fn[:, 2] * 0.3), 0.15, 1)    # fewer on downward faces
    wts = dens * tm.area_faces
    cdf = np.cumsum(wts)
    cdf /= cdf[-1]

    def sample(n):
        fi = np.searchsorted(cdf, rng.random(n))
        r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
        bw = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], axis=1)
        p = (v[f[fi]] * bw[:, :, None]).sum(1)
        return p

    # beads: log-normal radii (diameter 0.002-0.01 of the photo width), rejection by spacing
    pts, rad = [], []
    cand = sample(N_DROPS * 5)
    r_c = np.clip(np.exp(rng.normal(np.log(0.0031), 0.38, len(cand))), 0.0014, 0.0050)
    for p, r in zip(cand, r_c):
        pts.append(p)
        rad.append(r)
    pts, rad = np.array(pts), np.array(rad)
    tree = cKDTree(pts)
    keep = np.ones(len(pts), bool)
    for i, j in tree.query_pairs(0.011):
        if keep[i] and keep[j] and np.linalg.norm(pts[i] - pts[j]) < (rad[i] + rad[j]) * 1.5:
            keep[j] = False
    pts, rad = pts[keep][:N_DROPS], rad[keep][:N_DROPS]
    # streaks: elongated, running down the surface; head right side, arm, belly
    sc = sample(N_STREAKS * 30)
    scol = m.labels(sc)
    pick = (scol["hand"] < 0.2) & (((sc[:, 0] > 0.05) & (sc[:, 1] > -0.25)) | (scol["belly"] > 0.6) | (scol["is_arm"] > 0.5))
    sc = sc[pick]
    ktree = cKDTree(sc)
    keep = np.ones(len(sc), bool)
    for i, j in ktree.query_pairs(0.05):
        if keep[i] and keep[j]:
            keep[j] = False
    sc = sc[keep][:N_STREAKS]
    P = np.concatenate([pts, sc])
    P = project_to_surface(m, P, iters=3)
    N = Model.grad(m.skin, P, 1.5e-4)
    down = np.array([0.0, -1.0, 0.0])
    T = down[None, :] - N * (N @ down)[:, None]
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-6)
    nb = len(pts)
    scale = np.zeros((len(P), 3))       # (width along bitangent, length along tangent, height along normal)
    asp = 1 + 0.25 * rng.random(nb) * np.clip(1 - np.abs(N[:nb, 1]), 0, 1)        # slight sag on steep skin
    scale[:nb] = np.stack([rad, rad * asp, rad * (0.62 + 0.2 * rng.random(nb))], axis=1)
    sl = 0.022 + 0.05 * rng.random(len(sc))
    sw = 0.0026 + 0.0016 * rng.random(len(sc))
    scale[nb:] = np.stack([sw, sl, sw * 0.55], axis=1)
    seed = rng.random(len(P))
    return P, N, T, scale, seed, nb


# --------------------------------------------------------------------------- export
class Pack:
    def __init__(self):
        self.buf = bytearray()
        self.index = {}

    def add(self, name, arr, dtype):
        a = np.ascontiguousarray(arr, dtype=dtype)
        while len(self.buf) % 8:
            self.buf.append(0)
        self.index[name] = {"dtype": np.dtype(dtype).name, "shape": list(a.shape), "offset": len(self.buf)}
        self.buf += a.tobytes()

    def write(self):
        open(os.path.join(OUT, "pack.bin"), "wb").write(bytes(self.buf))
        json.dump(self.index, open(os.path.join(OUT, "pack.json"), "w"), indent=1)


def rasterize_mask(v, f, eyes, r_eye, W=736):
    """Silhouette of the model through the shared camera (used for the offline IoU + reference/mask_model.png)."""
    from PIL import Image, ImageDraw
    Hh = int(round(W / cam.ASPECT))
    u, w, _ = cam.project(v)
    im = Image.new("L", (W, Hh), 0)
    dr = ImageDraw.Draw(im)
    pts = np.stack([u * W, w * Hh], axis=1)
    tri = pts[f]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    for t in tri[np.abs(area) > 1e-9]:
        dr.polygon([tuple(t[0]), tuple(t[1]), tuple(t[2])], fill=255)
    for e in eyes:
        uu, ww, z = cam.project(e["center"][None, :])
        rr = r_eye / (z[0] * cam.TAN_HALF_V * cam.ASPECT) * 0.5 * W
        dr.ellipse([uu[0] * W - rr, ww[0] * Hh - rr, uu[0] * W + rr, ww[0] * Hh + rr], fill=255)
    return np.asarray(im) > 127


def main():
    from PIL import Image
    m = Model()
    P = m.P
    v, f = build_skin(m)
    nrm = sdf_normals(m, v)
    L = zone_lengths(m, v, nrm)
    topo = topology(v, f)
    log("skin topology", topo)
    assert topo["boundary_edges"] == 0 and topo["non_manifold_edges"] == 0 and topo["components"] == 1, topo
    bb_ok = (np.array(topo["bbox_min"]) > [-0.70, -1.10, -0.82]).all() and (np.array(topo["bbox_max"]) < [0.90, 0.50, 0.70]).all()
    assert bb_ok, "bounding box outside the expected range: %s" % topo
    lab = m.labels(v)
    u, w, _ = cam.project(v)
    view = cam.CAM_POS - v
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    vis_v = ((nrm * view).sum(1) > -0.28) & (u > -0.05) & (u < 1.05) & (w < 1.05)
    vis_f = vis_v[f].any(axis=1)
    es = edge_stats(v, f, L)
    curv, Hm = curvature_report(m, v, f, L)
    # hand/face clearance measured on the final mesh: arm-group vertices vs body-group surface
    d, carved, da = m.skin_parts(v)
    arm_v = (lab["is_arm"] > 0.5) & (lab["fuse"] < 0.02)
    body_v = (lab["is_arm"] < 0.5) & (lab["fuse"] < 0.02)
    tb = cKDTree(v[body_v])
    gap_d, _ = tb.query(v[arm_v])
    eye_clear = float((m.arm(np.concatenate([e["center"][None, :] for e in m.eyes])) - P["eye_r"]).min())
    samples = v[arm_v]
    eye_clear = float(min(eye_clear, (m.eyeballs(samples)).min()))
    REPORT["skin"] = {"triangles": int(len(f)), "vertices": int(len(v)), "visible_triangles": int(vis_f.sum()), **topo,
                      "bbox_expected": {"min": [-0.70, -1.10, -0.82], "max": [0.90, 0.50, 0.70]}, "bbox_ok": bool(bb_ok),
                      "edge_lengths": es, "curvature": curv,
                      "arm_body_min_gap": float(gap_d.min()), "gap_design": P["gap"], "hand_eyeball_clearance": eye_clear,
                      "blend_radii": {"finger_root_k": P["k_finger"], "finger_r": P["finger_r"][0], "wrist_k": P["k_wrist"], "wrist_r": P["r_wrist"],
                                      "elbow_k": P["k_elbow"], "elbow_r": P["r_elbow"], "shoulder_k": P["k_shoulder"], "shoulder_r": P["r_shoulder"],
                                      "ratios": {"finger": P["k_finger"] / P["finger_r"][0], "wrist": P["k_wrist"] / P["r_wrist"],
                                                 "elbow": P["k_elbow"] / P["r_elbow"], "shoulder": P["k_shoulder"] / P["r_shoulder"]}}}
    log("edge lengths", json.dumps(es))
    log("curvature", json.dumps(curv))
    log("arm/body min gap %.4f (design %.4f), hand/eyeball clearance %.4f" % (gap_d.min(), P["gap"], eye_clear))
    assert gap_d.min() > P["gap"] * 0.5, "hand and face are (nearly) fused"
    if eye_clear <= 0.002:
        eb = m.eyeballs(samples)
        for i in np.argsort(eb)[:5]:
            log("   eye-near arm vertex", samples[i].round(4), "ball dist %.4f" % eb[i], "arm sdf %.4f body sdf %.4f" % (m.arm(samples[i:i+1])[0], m.body(samples[i:i+1])[0]))
    assert eye_clear > 0.002, "hand intersects an eyeball"

    # ---- silhouette vs figure A (offline; the browser test repeats this on the real render)
    mask = rasterize_mask(v, f, m.eyes, P["eye_r"])
    ref = np.asarray(Image.open(os.path.join(ROOT, "reference", "mask_A.png")).convert("L").resize((mask.shape[1], mask.shape[0]))) > 127
    iou = float((mask & ref).sum() / (mask | ref).sum())
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(ROOT, "reference", "mask_model.png"))
    REPORT["offline_silhouette_iou"] = iou
    log("offline silhouette IoU %.4f" % iou)

    # ---- simulation membrane (Sheen-style GPU surface; replaces the v1/v2 tetrahedral cage)
    import membrane as MB
    sv, sf = build_sim(m, v, f)
    ns = len(sv)
    rings = MB.ordered_rings(ns, sf)
    assert max(len(r) for r in rings) <= MB.MAX_RING, "membrane valence above %d" % MB.MAX_RING
    sn_sdf = sdf_normals(m, sv)
    sn = MB.vertex_normals(sv, rings)                     # the runtime normal pass computes exactly this
    flip = (sn * sn_sdf).sum(1) < 0
    log("  ring orientation: %d of %d normals inward" % (flip.sum(), ns))
    if flip.mean() > 0.5:                                  # faces are wound inward: reverse every ring
        rings = [r[::-1] for r in rings]
        sn = MB.vertex_normals(sv, rings)
        flip = (sn * sn_sdf).sum(1) < 0
    if flip.any():
        bad = np.where(flip)[0]
        log("  inward ring normals at", sv[bad].round(3).tolist()[:12], "dot", (sn * sn_sdf).sum(1)[bad].round(2).tolist())
    REPORT["membrane_inward_ring_normals"] = {"count": int(flip.sum()), "where": "forearm/chest groove lip (hidden by the forearm)"}
    assert flip.sum() <= 20, "ring order gives inward normals"
    slab = m.labels(sv)
    s_group = np.where(slab["fuse"] > 0.02, 2, np.where(slab["is_arm"] > 0.5, 1, 0)).astype(np.uint8)
    adj = np.full((ns, MB.MAX_RING), -1.0, np.float32)
    dist = np.zeros((ns, MB.MAX_RING), np.float32)
    for i, r in enumerate(rings):
        adj[i, :len(r)] = r
        dist[i, :len(r)] = np.linalg.norm(sv[r] - sv[i], axis=1)
    # material: shape recovery (Sheen: accel = 345 (org - pos)) and constraint stiffness per region
    hand_w = slab["hand"]
    sock = (1 - slab["is_arm"]) * (1 - np.clip((slab["eye_dist"] - P["eye_r"] * 1.1) / (P["eye_r"] * 0.8), 0, 1))
    k = np.full(ns, 260.0)                                 # torso
    k = k * (1 - slab["head"]) + 230.0 * slab["head"]
    k = k * (1 - slab["face"]) + 170.0 * slab["face"]      # cheeks / forehead: stretch the most
    k = k * (1 - slab["bellyvol"]) + 150.0 * slab["bellyvol"]
    k = k * (1 - slab["is_arm"]) + 420.0 * slab["is_arm"]
    k = k * (1 - hand_w) + 1100.0 * hand_w                 # hand keeps its shape
    k = k * (1 - sock) + 700.0 * sock                      # eyelids hold the balls
    stiff = np.full(ns, 0.5)
    stiff = stiff * (1 - slab["is_arm"]) + 0.8 * slab["is_arm"]
    stiff = stiff * (1 - hand_w) + 1.0 * hand_w
    stiff = stiff * (1 - sock) + 0.9 * sock
    pinned = (sv[:, 1] < PIN_Y).astype(np.float32)
    # collision flags: body skin near the arm collides with the arm spheres; hand skin near an eye with the balls
    _, carved_s, da_s = m.skin_parts(sv)
    col_body = ((s_group == 0) & (da_s < 0.12)).astype(np.float32)
    col_eye = ((s_group == 1) & (slab["eye_dist"] < P["eye_r"] + 0.12)).astype(np.float32)   # hand skin near an eye
    # (the lids do NOT collide with the balls: the balls ride on the lid ring, that would be a feedback loop; the lids
    #  keep their shape around the ball through the socket shape matching instead)
    # rigid groups (local shape matching in membrane.ts): the two eye sockets and the hand.
    # Each has a ring of membrane vertices (centroid) and three anchors (frame); every member vertex is pulled
    # towards its rest position carried by that frame, with weight w.
    sock_w = np.zeros(ns, np.float32)
    sock_e = np.zeros(ns, np.float32)
    for ei, e in enumerate(m.eyes):
        de = np.linalg.norm(sv - e["center"], axis=1)
        w_ = (s_group != 1) * (1 - np.clip((de - (P["eye_r"] + P["rim_h"] + 0.012)) / 0.05, 0, 1))
        take = w_ > sock_w
        sock_w[take] = w_[take]
        sock_e[take] = ei
    hw = np.clip((hand_w - 0.15) / 0.6, 0, 1).astype(np.float32)
    take = hw > sock_w
    sock_w[take] = hw[take]
    sock_e[take] = len(m.eyes)
    palm_v = np.where((s_group == 1) & (np.linalg.norm(sv - m.palm_c, axis=1) < 0.075))[0]
    rng_h = np.random.default_rng(7)
    hand_ring = sorted(rng_h.choice(palm_v, size=min(24, len(palm_v)), replace=False).tolist())
    qa = sv[palm_v] - m.palm_c
    hand_anchors = [int(palm_v[np.argmax(qa @ m.palm_f)]), int(palm_v[np.argmax(qa @ m.palm_t)]), int(palm_v[np.argmin(qa @ m.palm_t)])]
    col_body[sock_w > 0.5] = 0                             # held lids follow the ball, not the fingers
    REPORT["membrane"] = {"vertices": ns, "triangles": int(len(sf)), "max_valence": int(max(len(r) for r in rings)),
                          "mean_edge": float(dist[dist > 0].mean()), "pinned": int(pinned.sum()),
                          "edge_targets": {"fine": S_FINE, "mid": S_MID, "body": S_BODY, "hidden": S_HIDDEN},
                          "restore_range": [float(k.min()), float(k.max())], "collide_body": int(col_body.sum()), "collide_eye": int(col_eye.sum())}

    # ---- render binding: fine skin -> membrane triangles (Phong + residual in the triangle frame)
    vgroup = np.where(lab["fuse"] > 0.02, 2, np.where(lab["is_arm"] > 0.5, 1, 0))
    bt, bw, bres, bnc, bd = MB.bind_points(v, nrm, sv, sn, sf, s_group, vgroup)
    REPORT["render_binding"] = {"max_distance_to_membrane": float(bd.max()), "p99": float(np.percentile(bd, 99)),
                                "max_residual_normal": float(np.abs(bres[:, 2]).max())}
    log("  render binding: max distance to membrane %.4f (p99 %.4f)" % (bd.max(), np.percentile(bd, 99)))

    # ---- eyes: ring on the lids + gaze
    erings = MB.eye_rings(sv, s_group, m.eyes, P["eye_r"], P["rim_h"])
    eyes_out = []
    lm_p = LM["landmarks"]["eye_right"]["pupil"]
    o, dd = cam.ray(lm_p[0], lm_p[1])
    c0 = m.eyes[0]["center"]
    oc = o - c0
    bq = oc @ dd[0]
    disc = bq * bq - (oc @ oc - P["eye_r"] ** 2)
    hitp = o + dd[0] * (-bq - np.sqrt(max(disc, 0)))
    gaze_r = (hitp - c0) / np.linalg.norm(hitp - c0)
    gaze_r = gaze_r + np.array([0.0, -0.5, 0.0])          # the visible pupil is the top of a larger one tucked behind the fingers
    gaze_r /= np.linalg.norm(gaze_r)
    for i, e in enumerate(m.eyes):
        ang = np.radians((e["az"] - m.eyes[0]["az"]) * 0.75)
        Ry = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
        eyes_out.append({"center": e["center"].tolist(), "normal": e["normal"].tolist(), "up": e["up"].tolist(), "radius": P["eye_r"],
                         "gaze": (Ry @ gaze_r).tolist(), "ring": erings[i]["ring"], "anchors": erings[i]["anchors"]})

    # ---- collision proxies: arm spheres + the two eyeballs (riding on their lid rings)
    pc, pr, pb = MB.arm_proxies(m, sv, s_group)
    for i, e in enumerate(m.eyes):
        pc = np.vstack([pc, e["center"][None, :]])
        # ball + lid, so fingers stay outside the lid - but never larger than the rest clearance of the hand (the rest
        # pose must be an equilibrium)
        hand_d = np.linalg.norm(sv[s_group == 1] - e["center"], axis=1).min()
        pr = np.append(pr, min(P["eye_r"] + P["rim_h"] + 0.002, hand_d - 0.0008))
        pb.append(erings[i]["ring"])
    ptype = np.array([0] * (len(pc) - len(m.eyes)) + [1] * len(m.eyes), np.float32)   # 0 arm sphere, 1 eyeball
    KP = 24
    pids = np.full((len(pc), KP), -1.0, np.float32)
    for i, b in enumerate(pb):
        pids[i, :len(b[:KP])] = b[:KP]
    poff = np.array([pc[i] - sv[[j for j in pb[i][:KP]]].mean(0) for i in range(len(pc))])
    REPORT["proxies"] = {"arm_spheres": int((ptype == 0).sum()), "eyeballs": int((ptype == 1).sum())}
    log("  collision proxies: %d arm spheres + %d eyeballs" % ((ptype == 0).sum(), (ptype == 1).sum()))

    # ---- droplets (bound like the skin)
    dP, dN, dT, dS, dSeed, nbeads = make_droplets(m, v, f, nrm, lab)
    dlab = m.labels(dP)
    dgroup = np.where(dlab["fuse"] > 0.02, 2, np.where(dlab["is_arm"] > 0.5, 1, 0))
    dt_, dw_, dres, dnc, dd_ = MB.bind_points(dP, dN, sv, sn, sf, s_group, dgroup)
    X, N = sv[sf[dt_]], sn[sf[dt_]]
    Tf, Bf, Nf = MB.tri_frame(X, N, dw_)
    dtc = np.stack([(dT * Tf).sum(1), (dT * Bf).sum(1), (dT * Nf).sum(1)], 1)
    REPORT["droplets"] = {"beads": int(nbeads), "streaks": int(len(dP) - nbeads), "radius_min": float(dS[:nbeads, 0].min()), "radius_max": float(dS[:nbeads, 0].max())}
    log("  droplets: %d beads + %d streaks" % (nbeads, len(dP) - nbeads))

    # ---- GLB (render mesh: position = rest, COLOR_0 = belly / hand / eyelid / mouth)
    col = np.stack([lab["belly"], lab["hand"], lab["socket"], lab["mouth"]], axis=1)
    col8 = np.clip(np.round(col * 255), 0, 255).astype(np.uint8)
    tm = trimesh.Trimesh(vertices=v.astype(np.float32), faces=f, vertex_normals=nrm.astype(np.float32), process=False)
    tm.visual = trimesh.visual.ColorVisuals(tm, vertex_colors=col8)
    glb = tm.export(file_type="glb")
    open(os.path.join(OUT, "skin.glb"), "wb").write(glb)

    pk = Pack()
    pk.add("simRest", sv, np.float32)
    pk.add("simNormalRest", sn, np.float32)
    pk.add("simTris", sf, np.uint32)
    pk.add("simAdj", adj, np.float32)
    pk.add("simDist", dist, np.float32)
    pk.add("simRestore", k, np.float32)
    pk.add("simStiff", stiff, np.float32)
    pk.add("simPinned", pinned, np.float32)
    pk.add("simGroup", s_group, np.uint8)
    pk.add("simCollideBody", col_body, np.float32)
    pk.add("simCollideEye", col_eye, np.float32)
    pk.add("simRegionHand", hand_w.astype(np.float32), np.float32)
    pk.add("simSocketW", sock_w, np.float32)
    pk.add("simSocketEye", sock_e, np.float32)
    pk.add("proxyIds", pids, np.float32)
    pk.add("proxyOffset", poff, np.float32)
    pk.add("proxyRadius", pr, np.float32)
    pk.add("proxyType", ptype, np.float32)
    pk.add("proxyRest", pc, np.float32)
    pk.add("skinTri", sf[bt].astype(np.float32), np.float32)
    pk.add("skinBary", bw[:, :2], np.float32)
    pk.add("skinResid", bres, np.float32)
    pk.add("skinNrm", bnc, np.float32)
    pk.add("dropTri", sf[dt_].astype(np.float32), np.float32)
    pk.add("dropBary", dw_[:, :2], np.float32)
    pk.add("dropResid", dres, np.float32)
    pk.add("dropNrm", dnc, np.float32)
    pk.add("dropTan", dtc, np.float32)
    pk.add("dropPos", dP, np.float32)
    pk.add("dropScale", dS, np.float32)
    pk.add("dropSeed", dSeed, np.float32)
    pk.write()

    # ---- scene.json: camera, landmarks (3D), interaction zones, eyes, wall
    sk = lambda uv: m.on_surface(uv, m.skin)[0].tolist()
    top_i = int(np.argmin(np.where(vis_v, w, 9)))
    bel = (lab["belly"] > 0.5) & vis_v
    bel_i = np.where(bel)[0]
    bel_top = bel_i[np.argmin(w[bel_i])]
    tiles = LM["tiles"]
    scene = {
        "camera": cam.as_dict(), "H": H_CHAR, "gap": P["gap"], "pinY": PIN_Y,
        "eyes": eyes_out,
        "rigid": [{"ring": e["ring"], "anchors": e["anchors"]} for e in eyes_out] + [{"ring": hand_ring, "anchors": hand_anchors}],
        "landmarks3d": {"headTop": v[top_i].tolist(), "eyeRight": m.eyes[0]["center"].tolist(), "eyeLeft": m.eyes[1]["center"].tolist(),
                        "palm": (m.palm_c + m.palm_n * P["palm_half"][2]).tolist(), "elbow": m.elbow.tolist(), "wrist": m.wrist.tolist(),
                        "bellyTop": v[bel_top].tolist(), "mouth": m.mouth["pts"][len(m.mouth["pts"]) // 2].tolist()},
        "zones": {"headTop": sk((0.47, 0.215)), "forehead": sk((0.40, 0.30)), "cheekLeft": sk((0.30, 0.335)), "cheekRight": sk((0.63, 0.43)),
                  "hand": sk((0.45, 0.535)), "forearm": sk((0.64, 0.69)), "belly": sk((0.36, 0.86)), "side": sk((0.13, 0.86))},
        "counts": {"skinTriangles": int(len(f)), "skinVertices": int(len(v)), "simVertices": ns, "simTriangles": int(len(sf)),
                   "beads": int(nbeads), "streaks": int(len(dP) - nbeads)},
        "wall": {"z": -1.0, "tilePitchPhoto": tiles.get("pitch_x", 0.194), "groutXPhoto": tiles["vertical_grout_x"],
                 "groutYLeftPhotoWidths": tiles["horizontal_grout_y_left_in_widths"]},
        "handParts": {"wrist": m.wrist.tolist(), "foreDir": m.fore_dir.tolist()},
    }
    json.dump(scene, open(os.path.join(OUT, "scene.json"), "w"), indent=1)
    REPORT["build_seconds"] = round(time.time() - T0, 1)
    json.dump(REPORT, open(os.path.join(ROOT, "docs", "model-report.json"), "w"), indent=1)
    log("done. skin.glb %.1f MB, pack.bin %.1f MB" % (len(glb) / 1e6, len(pk.buf) / 1e6))


if __name__ == "__main__":
    main()
