"""Quickstart example: generate surfaces, extract features, discover formulas.

Run with:
    python examples/quickstart.py
"""

from math_discovery.surface_data_gen import generate_sphere, generate_torus, generate_klein_bottle
from math_discovery.common import compute_betti, classify_statement, parse_statement


def main():
    # ── 1. Generate surfaces and inspect their topology ──────────────
    print("=== Surface Generation ===\n")

    sphere = generate_sphere(subdivisions=1)
    print(f"Sphere (1 subdivision):")
    print(f"  V={sphere.V}, E={sphere.E}, F={sphere.F}")
    print(f"  Euler characteristic: chi = V - E + F = {sphere.chi}")
    print(f"  Betti numbers: b0={sphere.b0}, b1={sphere.b1}, b2={sphere.b2}")
    print()

    torus = generate_torus(width=4, height=4)
    print(f"Torus (4x4 grid):")
    print(f"  V={torus.V}, E={torus.E}, F={torus.F}")
    print(f"  Euler characteristic: chi = {torus.chi}")
    print(f"  Betti numbers: b0={torus.b0}, b1={torus.b1}, b2={torus.b2}")
    print()

    klein = generate_klein_bottle(width=4, height=4)
    print(f"Klein bottle (4x4 grid):")
    print(f"  V={klein.V}, E={klein.E}, F={klein.F}")
    print(f"  Euler characteristic: chi = {klein.chi}")
    print(f"  Betti numbers: b0={klein.b0}, b1={klein.b1}, b2={klein.b2}")
    print()

    # ── 2. Verify Betti numbers from boundary matrices ───────────────
    print("=== Betti Number Verification ===\n")

    d1 = sphere.dense_d1()
    d2 = sphere.dense_d2()
    b0, b1, b2 = compute_betti(d1, d2)
    print(f"Sphere Betti (recomputed from matrices): b0={b0}, b1={b1}, b2={b2}")
    assert (b0, b1, b2) == (1, 0, 1), "Sphere should have b=(1,0,1)"

    d1 = torus.dense_d1()
    d2 = torus.dense_d2()
    b0, b1, b2 = compute_betti(d1, d2)
    print(f"Torus Betti (recomputed from matrices):  b0={b0}, b1={b1}, b2={b2}")
    assert (b0, b1, b2) == (1, 2, 1), "Torus should have b=(1,2,1)"
    print()

    # ── 3. Parse and classify mathematical formulas ──────────────────
    print("=== Formula Classification ===\n")

    formulas = [
        "height_d1 - width_d1 + width_d2 = 2",   # Euler char (sphere)
        "null_d1 - rank_d2 = 2",                   # b1 = 2 (torus)
        "height_d1 - rank_d1 = 1",                 # b0 = 1 (connected)
    ]

    for text in formulas:
        statement = parse_statement(text)
        concepts = classify_statement(statement)
        print(f"  '{text}'")
        print(f"    concepts: {concepts}")

        # Evaluate against surfaces
        sphere_features = sphere.features()
        torus_features = torus.features()
        print(f"    holds on sphere: {statement.evaluate(sphere_features)}")
        print(f"    holds on torus:  {statement.evaluate(torus_features)}")
        print()

    print("Done! See README.md for full usage guide.")


if __name__ == "__main__":
    main()
