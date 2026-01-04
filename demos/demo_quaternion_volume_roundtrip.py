import numpy as np

from src.volumetric.quaternion_volume import (
    volumetric_log, volumetric_exp,
    volumetric_mul, volumetric_pow,
    lerp_volume
)

def main():
    print("\n--- [Demo] Quaternion <-> Volume Coordinates Round Trip ---")

    test_cases = [
        ("North Pole", 1, 0, 0, 0),
        ("Deep K(+)", 0, 0, 0, 1),
        ("Deep K(-)", 0, 0, 0, -1),
        ("Mixed", 0.5, -0.5, 0.5, -0.5),
    ]

    for name, w, i, j, k in test_cases:
        norm = np.sqrt(w*w + i*i + j*j + k*k)
        w, i, j, k = w/norm, i/norm, j/norm, k/norm

        print(f"\nTarget: {name}")
        print(f"Original: w={w:.4f}, i={i:.4f}, j={j:.4f}, k={k:.4f}")

        v = volumetric_log(w, i, j, k)
        print(f"  -> Log(v): ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})")

        rw, ri, rj, rk = volumetric_exp(*v)
        print(f"  -> Exp(q): w={rw:.4f}, i={ri:.4f}, j={rj:.4f}, k={rk:.4f}")

        err = abs(w-rw) + abs(i-ri) + abs(j-rj) + abs(k-rk)
        print("  =>", "SUCCESS" if err < 1e-2 else f"FAIL (Error={err:.6f})")

    print("\n--- [Demo] Volume-space multiplication and power ---")
    v_a = (0.5, 0.5, 0.5)
    v_b = (0.5, 0.5, 0.5)

    v_mul = volumetric_mul(v_a, v_b)
    print(f"A {v_a} * B {v_b} -> {v_mul}")

    v_sq = volumetric_pow(v_a, 2)
    v_cube = volumetric_pow(v_a, 3)
    print(f"A^2 -> {v_sq}")
    print(f"A^3 -> {v_cube}")

    print("\n--- [Critical Test] i*i = -1 ---")
    vol_i = volumetric_log(0, 1, 0, 0)
    vol_ii = volumetric_mul(vol_i, vol_i)
    restored = volumetric_exp(*vol_ii)
    print(f"vol(i)   = {vol_i}")
    print(f"vol(i*i) = {vol_ii}")
    print(f"Exp(vol(i*i)) = {restored}  (Expected near (-1,0,0,0))")

    print("\n--- [Animation Test] Smooth transition (+k -> -k) with circular v3 interpolation ---")
    start = volumetric_log(0, 0, 0, 1)
    end   = volumetric_log(0, 0, 0, -1)
    print(f"Start v = {start}")
    print(f"End   v = {end}")

    for t in np.linspace(0, 1, 11):
        vt = lerp_volume(start, end, float(t), wrap_v3=True)
        w,i,j,k = volumetric_exp(*vt)
        print(f"t={t:.1f} | j={j:+.3f}, k={k:+.3f} | norm={np.sqrt(w*w+i*i+j*j+k*k):.4f}")

if __name__ == "__main__":
    main()
