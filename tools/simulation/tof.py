"""Simple physical approximation of a ToF/LiDAR depth camera (L515-like) looking at glass.

Per pixel the sensor emits NIR and sees up to two returns along the ray:
  1. the glass front surface (depth d1, specular Fresnel reflection of the emitted light),
  2. the first opaque surface behind the glass (depth d2), reached through the glass twice.
Amplitudes (relative):  a1 = R(theta),  a2 = (1 - R(theta))^2 * rho,  with R the unpolarised Fresnel reflectance
(n = 1.5) at incidence angle theta and rho = 0.5 an assumed albedo of what is behind the glass.
A phase-based ToF measures an amplitude-weighted mix of the path lengths:  d = (a1 d1 + a2 d2) / (a1 + a2).
So the glass is almost transparent at normal incidence (R ~ 4%, depth ~ d2: 'sees through') and becomes the glass
surface at grazing angles (R -> 1).  Opaque pixels return their own depth with amplitude rho.
Pixel footprint: amplitude-weighted 3x3 mixing, which creates flying pixels at depth discontinuities.
Noise: Gaussian, sigma = 1 mm + 0.25% of depth.  Pixels at near-grazing incidence (cos(theta) < 0.1) return nothing.
The model is an approximation (no multipath inside the glass, no glass absorption, no lens flare); it is not tuned
to the real L515 on glass.
"""
import cv2
import numpy as np

N_GLASS = 1.5
RHO = 0.5


def fresnel(cos_i, n=N_GLASS):
    """Unpolarised Fresnel reflectance for light going from air into a medium of index n."""
    cos_i = np.clip(cos_i, 0, 1)
    sin_t = np.sqrt(np.clip(1 - cos_i ** 2, 0, 1)) / n
    cos_t = np.sqrt(np.clip(1 - sin_t ** 2, 0, 1))
    rs = ((cos_i - n * cos_t) / (cos_i + n * cos_t + 1e-9)) ** 2
    rp = ((n * cos_i - cos_t) / (n * cos_i + cos_t + 1e-9)) ** 2
    return 0.5 * (rs + rp)


def view_cos(normal, K, cam_matrix):
    """|cos| of the angle between the surface normal (world) and the viewing ray, per pixel."""
    h, w = normal.shape[:2]
    xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    # Blender camera looks down -Z, +Y up; image y grows downward
    d_cam = np.stack([(xs - K["cx"]) / K["fx"], -(ys - K["cy"]) / K["fy"], -np.ones_like(xs)], -1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    R = np.array(cam_matrix)[:3, :3]
    d_world = d_cam @ R.T
    return np.abs(np.sum(normal * d_world, axis=-1))


def tof_depth(s, seed=0, g=1.0, t_glass=1.0, a_min=0.0):
    """g: glass-return gain (a1 = g R); t_glass: per-pass glass transmission (a2 = rho (t (1-R))^2);
    a_min: pixels whose total return amplitude is below it give no depth. Defaults = the uncalibrated model."""
    rng = np.random.default_rng(seed)
    d1, d2, mask = s["depth_gt"], s["depth_behind"], s["mask"]
    if "_R" not in s:  # cache the geometry-only terms (used by the calibration grid search)
        s["_cos"] = view_cos(s["normal"], s["K"], s["meta"]["cam_matrix_world"])
        s["_R"] = fresnel(s["_cos"])
    cos, R = s["_cos"], s["_R"]
    # per-pixel two-return amplitudes
    a1 = np.where(mask, g * R, RHO).astype(np.float32)
    a2 = np.where(mask, (t_glass * (1 - R)) ** 2 * RHO, 0).astype(np.float32)
    dd2 = np.where(mask & (d2 > 0), d2, d1)  # behind depth only exists where the glass was
    # footprint mixing over 3x3 (amplitude-weighted)
    k = np.ones((3, 3), np.float32)
    num = cv2.filter2D(a1 * d1 + a2 * dd2, -1, k, borderType=cv2.BORDER_REPLICATE)
    den = cv2.filter2D(a1 + a2, -1, k, borderType=cv2.BORDER_REPLICATE)
    d = num / np.maximum(den, 1e-6)
    d = d + rng.normal(0, 1, d.shape).astype(np.float32) * (0.001 + 0.0025 * d)
    d[(cos < 0.1) | (d1 <= 0) | ~np.isfinite(d) | (d <= 0) | ((a1 + a2) < a_min)] = 0
    return d.astype(np.float32)
