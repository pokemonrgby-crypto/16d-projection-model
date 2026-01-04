import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from functools import lru_cache

def density_power(h: float, power: float) -> float:
    return (1.0 - h*h) ** power

@lru_cache(maxsize=None)
def total_volume(power: float) -> float:
    f = lambda x: density_power(x, power)
    v, _ = quad(f, -1.0, 1.0)
    return float(v)

def ratio_from_coord(coord: float, power: float) -> float:
    if coord <= -1.0:
        return 0.0
    if coord >= 1.0:
        return 1.0
    f = lambda x: density_power(x, power)
    tot = total_volume(power)
    cur, _ = quad(f, -1.0, coord)
    return float(cur / tot)

def coord_from_ratio(r: float, power: float) -> float:
    if r <= 0.0:
        return -1.0
    if r >= 1.0:
        return 1.0

    tot = total_volume(power)
    target = r * tot
    f = lambda x: density_power(x, power)

    def objective(h: float) -> float:
        cur, _ = quad(f, -1.0, h)
        return cur - target

    try:
        return float(brentq(objective, -1.0, 1.0))
    except Exception:
        return 0.0

def volumetric_log_16d(vec16: tuple[float, ...], start_power: float = 7.5) -> list[float]:
    """
    16D 단위벡터 -> 15개의 ratio (마지막 2차원은 각도)
    start_power=7.5는 '이 엔진이 택한 규칙' (호환 유지 목적).
    """
    if len(vec16) != 16:
        raise ValueError("vec16 must have length 16.")

    coords = np.array(vec16, dtype=float)
    n = np.linalg.norm(coords)
    if n == 0:
        raise ValueError("Cannot log a zero vector.")
    coords /= n

    ratios: list[float] = []
    rem_radius = 1.0
    power = float(start_power)

    # first 14 coords as slice ratios
    for i in range(14):
        val = float(coords[i])
        if rem_radius < 1e-12:
            norm_val = 0.0
        else:
            norm_val = max(-1.0, min(1.0, val / rem_radius))

        ratios.append(ratio_from_coord(norm_val, power))
        rem_radius = float(np.sqrt(max(0.0, rem_radius*rem_radius - val*val)))
        power -= 0.5

    # last 2 coords as angle ratio
    e14, e15 = float(coords[14]), float(coords[15])
    angle = float(np.arctan2(e15, e14))  # (-pi, pi]
    ratios.append((angle + np.pi) / (2.0 * np.pi))
    return ratios

def volumetric_exp_16d(ratios: list[float], start_power: float = 7.5) -> tuple[float, ...]:
    """
    15 ratios -> 16D unit vector
    """
    if len(ratios) != 15:
        raise ValueError("ratios must have length 15.")

    coords = [0.0] * 16
    rem_radius = 1.0
    power = float(start_power)

    for i in range(14):
        r = float(ratios[i])
        norm_val = coord_from_ratio(r, power)
        val = float(norm_val * rem_radius)
        coords[i] = val
        rem_radius = float(np.sqrt(max(0.0, rem_radius*rem_radius - val*val)))
        power -= 0.5

    a = float(ratios[14]) % 1.0
    angle = a * (2.0 * np.pi) - np.pi
    coords[14] = float(rem_radius * np.cos(angle))
    coords[15] = float(rem_radius * np.sin(angle))

    # normalize for safety
    v = np.array(coords, dtype=float)
    v /= np.linalg.norm(v)
    return tuple(float(x) for x in v)

# -----------------------------
# Cayley–Dickson multiplication up to sedenions (16D)
# -----------------------------
def q_mul(a, b):
    w1,x1,y1,z1 = a
    w2,x2,y2,z2 = b
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )

def q_conj(q):
    w,x,y,z = q
    return (w, -x, -y, -z)

def oct_conj(o):
    return (o[0],) + tuple(-x for x in o[1:])

def oct_mul(o1, o2):
    a, b = o1[:4], o1[4:]
    c, d = o2[:4], o2[4:]
    # (a,b)(c,d) = (ac - d* b, da + b c*)
    ac = q_mul(a, c)
    db = q_mul(q_conj(d), b)
    lower = tuple(x - y for x, y in zip(ac, db))
    da = q_mul(d, a)
    bc = q_mul(b, q_conj(c))
    upper = tuple(x + y for x, y in zip(da, bc))
    return lower + upper

def sed_mul(s1: tuple[float, ...], s2: tuple[float, ...]) -> tuple[float, ...]:
    if len(s1) != 16 or len(s2) != 16:
        raise ValueError("sedenion operands must have length 16.")
    A, B = s1[:8], s1[8:]
    C, D = s2[:8], s2[8:]

    AC = oct_mul(A, C)
    DB = oct_mul(oct_conj(D), B)
    lower = tuple(x - y for x, y in zip(AC, DB))

    DA = oct_mul(D, A)
    BC = oct_mul(B, oct_conj(C))
    upper = tuple(x + y for x, y in zip(DA, BC))

    return lower + upper

def normalize_vec(v: tuple[float, ...]) -> tuple[float, ...]:
    a = np.array(v, dtype=float)
    n = np.linalg.norm(a)
    if n == 0:
        raise ValueError("Cannot normalize zero vector.")
    a /= n
    return tuple(float(x) for x in a)

def volume_interpolate_16d(a: tuple[float, ...], b: tuple[float, ...], t: float, start_power: float = 7.5) -> tuple[float, ...]:
    """
    16D에서 'volume coordinate'로 보간 후 복원(정의상 norm이 유지되도록 설계).
    """
    ra = volumetric_log_16d(a, start_power=start_power)
    rb = volumetric_log_16d(b, start_power=start_power)
    rt = [(1.0 - t)*x + t*y for x, y in zip(ra, rb)]
    return volumetric_exp_16d(rt, start_power=start_power)
