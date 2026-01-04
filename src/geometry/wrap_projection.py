import numpy as np

def linear_motion(t: np.ndarray, x0: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    16D(일반 ND)에서의 등속 직선운동:
    x(t) = x0 + v*t
    """
    t = np.asarray(t, dtype=float)
    return x0[None, :] + t[:, None] * v[None, :]

def phase_from_state(x: np.ndarray, k: np.ndarray, phi0: float = 0.0) -> np.ndarray:
    """
    위상(말린 축) 정의:
    phi(x) = k · x + phi0
    """
    return x @ k + phi0

def wrapped_projection(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    exp(i*phi) = (cos(phi), sin(phi))
    """
    return np.cos(phi), np.sin(phi)

def observed_wave(phi: np.ndarray) -> np.ndarray:
    """
    관측자가 실수부만 보면 파동처럼 보임: cos(phi)
    """
    return np.cos(phi)
