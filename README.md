# 16D Projection Model (Toy / Experimental)

## 1) Hypothesis (Single World Assumption)
- The universe is 16-dimensional.
- Observers live in a projected 3D view.
- Time corresponds to the real axis; the other axes act like imaginary/space directions.

This repository is **not a formal proof** of physics.  
It is a **model definition + reproducible simulations**.

---

## 2) Core Definitions (What the code actually assumes)
### (D1) Linear motion in 16D
A free state moves linearly:
x(t) = x0 + v*t

### (D2) Wrapped / periodic projection
A chosen phase is read as:
phi(x) = k·x + phi0
Observed signal:
(cos(phi), sin(phi))  (and cos(phi) looks like a wave)

### (D3) Chord-based probability on a unit hypersphere
Probability is defined as:
P = (||a - b|| / 2)^2
For a unit sphere, this equals sin^2(theta/2).

### (D4) Quaternion "volumetric coordinates"
We provide a bijective encode/decode (except measure-zero edge cases) between:
- unit quaternions (w,i,j,k) on S^3
- volume-like coordinates (v1,v2,v3) in [0,1)^3

Multiplication in volume-space is defined by pullback:
mul(vA, vB) = log( exp(vA) * exp(vB) )

---

## 3) How to run (Google Colab / Local)
### Install
```bash
pip install -r requirements.txt
````

### Run demos

```bash
python -m demos.demo_linear_to_rotation
python -m demos.demo_quaternion_volume_roundtrip
python -m demos.demo_spin_chord_chsh
python -m demos.demo_sedenion_volume_interpolation
```

---

## 4) Demos (What you will see)

* Linear motion (16D) -> rotation / wave (3D projection)
* Quaternion volumetric encode/decode round-trip (phase preserved via atan2)
* 720° spinor loop + chord probability = sin^2(theta/2)
* CHSH violation in an axis-chord entanglement toy generator (explicitly nonlocal axis-coupling rule)
* 16D "volume interpolation" vs naive interpolation (norm leak vs norm preserved by design)

---

## 5) Notes / Limitations

* This repo focuses on **reproducible outputs** and clear definitions.
* Any claim beyond the model definitions must be treated as a hypothesis until experimentally validated.

