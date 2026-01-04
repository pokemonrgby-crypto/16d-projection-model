import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# -----------------------------
# Density choices (kept compatible with your earlier engine)
# -----------------------------
def density_4d(h: float) -> float:
    return (1.0 - h*h) ** (3.0/2.0)

def density_3d(h: float) -> float:
    return (1.0 - h*h) ** (1.0)

VOL_4D_TOTAL, _ = quad(density_4d, -1.0, 1.0)
VOL_3D_TOTAL, _ = quad(density_3d, -1.0, 1.0)

def _clip1(x: float) -> float:
    return float(max(-1.0, min(1.0, x)))

def get_volume_from_coord(coord: float, dim: int) -> float:
    """
    coord in [-1,1] -> cumulative volume ratio in [0,1]
    """
    if coord <= -1.0:
        return 0.0
    if coord >= 1.0:
        return 1.0

    if dim == 4:
        func, total = density_4d, VOL_4D_TOTAL
    elif dim == 3:
        func, total = density_3d, VOL_3D_TOTAL
    else:
        # fallback
        return 0.5

    current_vol, _ = quad(func, -1.0, coord)
    return float(current_vol / total)

def get_coord_from_volume(target_ratio: float, dim: int) -> float:
    """
    target_ratio in [0,1] -> coord in [-1,1] by inversion
    """
    if target_ratio <= 0.0:
        return -1.0
    if target_ratio >= 1.0:
        return 1.0

    if dim == 4:
        func, total = density_4d, VOL_4D_TOTAL
    elif dim == 3:
        func, total = density_3d, VOL_3D_TOTAL
    else:
        return 0.0

    target_vol = target_ratio * total

    def objective(h: float) -> float:
        vol, _ = quad(func, -1.0, h)
        return vol - target_vol

    try:
        return float(brentq(objective, -1.0, 1.0))
    except Exception:
        return 0.0

def normalize_quaternion(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = map(float, q)
    n = (w*w + x*x + y*y + z*z) ** 0.5
    if n == 0.0:
        raise ValueError("Cannot normalize a zero quaternion.")
    return (w/n, x/n, y/n, z/n)

def lerp_angle01(a: float, b: float, t: float) -> float:
    """
    a,b in [0,1). interpolate along shortest arc on a circle.
    """
    a = float(a) % 1.0
    b = float(b) % 1.0
    t = float(t)
    d = (b - a + 0.5) % 1.0 - 0.5
    return (a + t*d) % 1.0

def volumetric_log(w: float, i: float, j: float, k: float) -> tuple[float, float, float]:
    """
    Quaternion (unit) -> volume-like coordinates (v1,v2,v3) in [0,1]^3
    Phase is preserved by atan2(k, j).
    """
    # Assume unit input; still safe to normalize if user passes raw
    w, i, j, k = normalize_quaternion((w, i, j, k))

    v1 = get_volume_from_coord(w, dim=4)

    rem_3d = float(np.sqrt(max(0.0, 1.0 - w*w)))
    if rem_3d < 1e-12:
        return (v1, 0.0, 0.5)

    norm_i = _clip1(i / rem_3d)
    v2 = get_volume_from_coord(norm_i, dim=3)

    # IMPORTANT: protect atan2(0,0) region
    rem_2d = float(np.sqrt(max(0.0, rem_3d*rem_3d - i*i)))
    if rem_2d < 1e-12:
        return (v1, v2, 0.5)

    angle = float(np.arctan2(k, j))  # (-pi, pi]
    v3 = (angle + np.pi) / (2.0 * np.pi)  # [0,1)
    return (float(v1), float(v2), float(v3))

def volumetric_exp(v1: float, v2: float, v3: float) -> tuple[float, float, float, float]:
    """
    (v1,v2,v3) -> Quaternion (unit)
    """
    v1 = float(v1)
    v2 = float(v2)
    v3 = float(v3) % 1.0

    w = get_coord_from_volume(v1, dim=4)
    rem_3d = float(np.sqrt(max(0.0, 1.0 - w*w)))

    norm_i = get_coord_from_volume(v2, dim=3)
    i = float(norm_i * rem_3d)

    rem_2d = float(np.sqrt(max(0.0, rem_3d*rem_3d - i*i)))

    angle = v3 * (2.0 * np.pi) - np.pi
    j = float(rem_2d * np.cos(angle))
    k = float(rem_2d * np.sin(angle))

    return normalize_quaternion((w, i, j, k))

def quaternion_multiply(q1: tuple[float, float, float, float],
                        q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return (float(w), float(x), float(y), float(z))

def quaternion_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    return (w, -x, -y, -z)

def volumetric_mul(vec_a: tuple[float, float, float],
                   vec_b: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Volume-space multiplication (pullback):
    exp(vec_a) and exp(vec_b) -> quaternion multiply -> log(result)
    """
    q1 = volumetric_exp(*vec_a)
    q2 = volumetric_exp(*vec_b)
    q = quaternion_multiply(q1, q2)
    q = normalize_quaternion(q)
    return volumetric_log(*q)

def volumetric_pow(vec: tuple[float, float, float], n: int) -> tuple[float, float, float]:
    """
    Integer power in volume-space by quaternion polar form.
    """
    if not isinstance(n, int):
        raise TypeError("n must be int for volumetric_pow in this demo.")

    if n == 0:
        return volumetric_log(1.0, 0.0, 0.0, 0.0)

    q = volumetric_exp(*vec)

    if n < 0:
        # q^{-1} = conjugate(q) for unit quaternion
        q = quaternion_conjugate(q)
        n = -n

    w, x, y, z = q
    w = float(np.clip(w, -1.0, 1.0))
    theta = float(np.arccos(w))
    sin_theta = float(np.sin(theta))

    if abs(sin_theta) < 1e-12:
        # q is close to +1 or -1
        if w >= 0.0:
            return volumetric_log(1.0, 0.0, 0.0, 0.0)
        # (-1)^n
        if n % 2 == 0:
            return volumetric_log(1.0, 0.0, 0.0, 0.0)
        else:
            return volumetric_log(-1.0, 0.0, 0.0, 0.0)

    ux, uy, uz = x/sin_theta, y/sin_theta, z/sin_theta
    new_theta = n * theta
    new_w = float(np.cos(new_theta))
    new_s = float(np.sin(new_theta))

    new_q = (new_w, ux*new_s, uy*new_s, uz*new_s)
    new_q = normalize_quaternion(new_q)
    return volumetric_log(*new_q)

def lerp_volume(vec_start: tuple[float, float, float],
                vec_end: tuple[float, float, float],
                t: float,
                wrap_v3: bool = True) -> tuple[float, float, float]:
    """
    Interpolation in volume-space.
    v3 is circular; use shortest-arc interpolation if wrap_v3=True.
    """
    a1, a2, a3 = vec_start
    b1, b2, b3 = vec_end
    t = float(t)

    v1 = (1.0 - t)*a1 + t*b1
    v2 = (1.0 - t)*a2 + t*b2
    if wrap_v3:
        v3 = lerp_angle01(a3, b3, t)
    else:
        v3 = (1.0 - t)*a3 + t*b3

    return (float(v1), float(v2), float(v3))
