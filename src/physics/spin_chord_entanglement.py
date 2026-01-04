import numpy as np

DIMS = 16
RE, IM = 4, 5
DARK_SLICE = slice(8, 16)

def normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return vec / n

def create_16d_electron(dim_idx: int = RE, dark_sigma: float = 0.02, rng=None) -> np.ndarray:
    """
    16D 단위벡터 상태 생성.
    dim_idx 축에 '본질'을 두고, DARK_SLICE에는 작은 노이즈를 섞을 수 있음.
    """
    if rng is None:
        rng = np.random.default_rng()
    vec = np.zeros(DIMS, dtype=float)
    vec[dim_idx] = 1.0
    if dark_sigma > 0:
        vec[DARK_SLICE] = rng.normal(0.0, dark_sigma, 8)
    return normalize(vec)

def apply_spin_rotation(vec: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    스피너(half-angle) 회전: 내부 위상은 angle/2로 회전.
    """
    rad = np.radians(angle_deg)
    theta = rad / 2.0
    c, s = np.cos(theta), np.sin(theta)

    v = vec.copy()
    re, im = v[RE], v[IM]
    v[RE] = re * c - im * s
    v[IM] = re * s + im * c
    return v

def apply_axis_rotation(vec: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    기하학적 축 회전: 관측축 자체를 angle로 회전.
    """
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)

    v = vec.copy()
    re, im = v[RE], v[IM]
    v[RE] = re * c - im * s
    v[IM] = re * s + im * c
    return v

def make_axis_state(angle_deg: float, dark_sigma: float = 0.0, rng=None) -> np.ndarray:
    """
    측정축 벡터 생성(기본: RE-IM 평면 unit circle).
    """
    if rng is None:
        rng = np.random.default_rng()

    v = np.zeros(DIMS, dtype=float)
    rad = np.radians(angle_deg)
    v[RE] = np.cos(rad)
    v[IM] = np.sin(rad)

    if dark_sigma > 0:
        v[DARK_SLICE] = rng.normal(0.0, dark_sigma, 8)

    return normalize(v)

def chord_probability(a: np.ndarray, b: np.ndarray) -> float:
    """
    Unit hypersphere에서 chord 기반 확률:
    P = (||a-b||/2)^2
    """
    chord = np.linalg.norm(a - b)
    return float((chord / 2.0) ** 2)

def measure_entangled_pair_by_axis_chord(axisA: np.ndarray, axisB: np.ndarray, rng) -> tuple[int, int, float]:
    """
    얽힘 "토이" 생성기.

    규칙(정의):
    - P(same) = chord_probability(axisA, axisB)
    - P(opposite) = 1 - P(same)
    - a는 50/50 랜덤
    - b는 same이면 a, opposite이면 -a

    주의:
    - 이 규칙은 axisA와 axisB 둘을 동시에 참조하는 결합 규칙이다.
      (이 레포는 그 사실을 숨기지 않고 '정의'로 둔다.)
    """
    p_same = chord_probability(axisA, axisB)
    same = (rng.random() < p_same)

    a = 1 if rng.random() < 0.5 else -1
    b = a if same else -a
    return a, b, p_same

def estimate_correlation(angleA: float, angleB: float, shots: int = 20000, seed: int = 0, dark_sigma_axes: float = 0.0):
    rng = np.random.default_rng(seed)
    axisA = make_axis_state(angleA, dark_sigma=dark_sigma_axes, rng=rng)
    axisB = make_axis_state(angleB, dark_sigma=dark_sigma_axes, rng=rng)

    prod_sum = 0.0
    last_p_same = 0.0
    for _ in range(shots):
        a, b, p_same = measure_entangled_pair_by_axis_chord(axisA, axisB, rng)
        prod_sum += (a * b)
        last_p_same = p_same

    E_sim = prod_sum / shots

    delta = abs(angleA - angleB) % 360.0
    if delta > 180.0:
        delta = 360.0 - delta

    E_theory = -np.cos(np.radians(delta))
    return float(delta), float(last_p_same), float(E_sim), float(E_theory)
