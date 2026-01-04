import numpy as np
import matplotlib.pyplot as plt

from src.geometry.wrap_projection import linear_motion, phase_from_state, wrapped_projection, observed_wave

def main():
    print("\n--- [Demo] Linear Motion -> Rotation/Wave by Wrapped Projection ---")

    t = np.linspace(0.0, 20.0, 1000)

    # ND state: x(t) = x0 + v*t
    D = 16
    x0 = np.zeros(D)
    v = np.zeros(D)
    v[0] = 1.0  # simplest: move along one axis

    X = linear_motion(t, x0, v)

    # Choose a phase readout direction k (what axis is "wrapped")
    k = np.zeros(D)
    k[0] = 1.0

    phi = phase_from_state(X, k, phi0=0.0)
    c, s = wrapped_projection(phi)
    wave = observed_wave(phi)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(t, X[:, 0], "k-", linewidth=2, label="16D reality (one axis)")
    plt.title("1) Reality (16D): linear motion")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(c, s, "r-", linewidth=2, label="wrapped projection")
    plt.title("2) Topology: wrapped axis -> circle")
    plt.xlabel("cos(phi)")
    plt.ylabel("sin(phi)")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.text(0, 0, "Observer", ha="center", va="center")

    plt.subplot(2, 1, 2)
    plt.plot(t, wave, "b-", linewidth=2, label="observed wave = cos(phi)")
    plt.axhline(0.0, color="k", linestyle="--", alpha=0.3)
    plt.title("3) Observation (3D): wave from wrapped linear phase")
    plt.xlabel("t")
    plt.ylabel("amplitude")
    plt.ylim(-2, 2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\n[Result]")
    print("- Input: linear motion in higher-D")
    print("- Projection: wrapped phase exp(i*phi)")
    print("- Output: rotation (circle) and wave (cos)")

if __name__ == "__main__":
    main()
