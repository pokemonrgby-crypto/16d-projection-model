import numpy as np

from src.physics.spin_chord_entanglement import (
    create_16d_electron, apply_spin_rotation,
    chord_probability, apply_axis_rotation,
    estimate_correlation
)

from src.physics.spin_chord_entanglement import DIMS, RE, IM

def test_spin_720():
    print("\n=== [Test 1] Spin Stability: 720 Degrees ===")
    rng = np.random.default_rng(0)
    e = create_16d_electron(rng=rng, dark_sigma=0.02)
    start = e.copy()

    steps = 24
    last = None
    for step in range(steps + 1):
        deg = (step / steps) * 720.0
        r = apply_spin_rotation(e, deg)
        last = r
        marker = ""
        if abs(deg - 360.0) < 1e-9:
            marker = " <--- flipped (-1 phase)"
        if abs(deg - 720.0) < 1e-9:
            marker = " <--- returned (+1)"

        print(f"{deg:6.1f}° | RE={r[RE]:+.4f} | IM={r[IM]:+.4f} | norm={np.linalg.norm(r):.8f}{marker}")

    diff = np.linalg.norm(last - start)
    print("=>", "SUCCESS" if diff < 1e-9 else f"NOTE: diff={diff}")

def test_rosetta_chord():
    print("\n=== [Test 2] Rosetta: chord^2 = sin^2(theta/2) ===")

    base = np.zeros(DIMS)
    base[RE] = 1.0

    angles = np.linspace(0, 180, 19)
    for deg in angles:
        qm = np.sin(np.radians(deg)/2.0)**2
        rot = apply_axis_rotation(base, deg)
        p = chord_probability(base, rot)
        gap = abs(qm - p)
        print(f"{deg:6.1f}° | chordProb={p:.8f} | sin^2={qm:.8f} | gap={gap:.3e}")

def test_chsh():
    print("\n=== [Test 3] CHSH (toy entanglement generator) ===")
    shots = 30000

    # One standard-ish set (you can change)
    a0, a1 = 0.0, 90.0
    b0, b1 = 45.0, 315.0

    pairs = [
        ("E(a0,b0)", a0, b0, 1),
        ("E(a0,b1)", a0, b1, 2),
        ("E(a1,b0)", a1, b0, 3),
        ("E(a1,b1)", a1, b1, 4),
    ]

    E = {}
    for name, aa, bb, seed in pairs:
        delta, p_same, E_sim, E_th = estimate_correlation(aa, bb, shots=shots, seed=seed, dark_sigma_axes=0.0)
        E[name] = E_sim
        print(f"{name:10s} | Δ={delta:6.1f}° | P(same)={p_same:.8f} | E_sim={E_sim:+.6f} | E_th={E_th:+.6f}")

    S = abs(E["E(a0,b0)"] + E["E(a0,b1)"] + E["E(a1,b0)"] - E["E(a1,b1)"])
    print(f"\nCHSH S ≈ {S:.6f}")
    print("=>", "S > 2 (violation in this toy model)" if S > 2.0 else "S <= 2")

def main():
    test_spin_720()
    test_rosetta_chord()
    test_chsh()

if __name__ == "__main__":
    main()
