import numpy as np
from src.volumetric.hypersphere_volume import (
    sed_mul, normalize_vec,
    volume_interpolate_16d
)

def main():
    print("\n--- [Demo] 16D Volume Interpolation vs Naive Interpolation ---")
    rng = np.random.default_rng(0)

    raw = rng.normal(0.0, 1.0, 16)
    base = normalize_vec(tuple(float(x) for x in raw))

    sq = sed_mul(base, base)
    sq = normalize_vec(sq)

    # mid by naive linear interpolation (can shrink norm)
    naive_mid = tuple((a+b)/2.0 for a, b in zip(base, sq))
    naive_norm = float(np.linalg.norm(np.array(naive_mid)))

    # mid by volume interpolation (designed to restore norm=1)
    vol_mid = volume_interpolate_16d(base, sq, t=0.5, start_power=7.5)
    vol_norm = float(np.linalg.norm(np.array(vol_mid)))

    print(f"Base norm: {np.linalg.norm(np.array(base)):.10f}")
    print(f"Square norm: {np.linalg.norm(np.array(sq)):.10f}")
    print(f"Naive mid norm : {naive_norm:.10f}   (can leak/shrink)")
    print(f"Volume mid norm: {vol_norm:.10f}   (restored to ~1 by design)")

if __name__ == "__main__":
    main()
