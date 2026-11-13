"""The one camera shared by the modelling tools and the runtime (exported into assets/scene.json).

World units: the photo window is exactly 1.0 wide (and 1/ASPECT tall) on the plane through the
origin that faces the camera. x right, y up, z towards the camera.
"""
import numpy as np

ASPECT = 0.6930
CAM_POS = np.array([0.0, 0.28, 3.0])
CAM_TARGET = np.array([0.0, 0.0, 0.0])
_DIST = float(np.linalg.norm(CAM_POS - CAM_TARGET))
TAN_HALF_V = (0.5 / ASPECT) / _DIST
FOV_V_DEG = float(np.degrees(2 * np.arctan(TAN_HALF_V)))

_f = (CAM_TARGET - CAM_POS) / _DIST
_r = np.cross(_f, [0.0, 1.0, 0.0])
_r /= np.linalg.norm(_r)
_u = np.cross(_r, _f)
FWD, RIGHT, UP = _f, _r, _u


def project(p):
    """world (N,3) -> photo coords u,v in [0,1] (v down) and view depth."""
    q = np.asarray(p, dtype=np.float64) - CAM_POS
    z = q @ FWD
    x = (q @ RIGHT) / (z * TAN_HALF_V * ASPECT)
    y = (q @ UP) / (z * TAN_HALF_V)
    return 0.5 + 0.5 * x, 0.5 - 0.5 * y, z


def ray(u, v):
    """photo coords -> (origin, unit direction). u, v may be arrays."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    d = (FWD[None, :] + ((u.reshape(-1, 1) - 0.5) * 2 * TAN_HALF_V * ASPECT) * RIGHT[None, :]
         + ((0.5 - v.reshape(-1, 1)) * 2 * TAN_HALF_V) * UP[None, :])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return CAM_POS, d


def as_dict():
    return {"position": CAM_POS.tolist(), "target": CAM_TARGET.tolist(), "fovY": FOV_V_DEG, "aspect": ASPECT}
