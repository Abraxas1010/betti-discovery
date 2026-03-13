"""Core tests for betti-discovery.

Run with: pytest tests/
"""

import numpy as np
from math_discovery.common import (
    f2_rank,
    compute_betti,
    parse_statement,
    classify_statement,
    SurfaceData,
)
from math_discovery.surface_data_gen import (
    generate_sphere,
    generate_torus,
    generate_klein_bottle,
    generate_disjoint_union,
)
from math_discovery.feature_extractor import extract_features
from math_discovery.conjecturing_agent import ConjecturingAgent
from math_discovery.skeptical_agent import SkepticalAgent
from math_discovery.config import CAConfig, SAConfig


# ── GF(2) linear algebra ────────────────────────────────────────────

def test_f2_rank_identity():
    eye = np.eye(3, dtype=np.uint8)
    assert f2_rank(eye) == 3


def test_f2_rank_zero():
    zero = np.zeros((3, 3), dtype=np.uint8)
    assert f2_rank(zero) == 0


def test_f2_rank_dependent():
    mat = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    # Row 3 = Row 1 XOR Row 2 over GF(2)
    assert f2_rank(mat) == 2


# ── Surface generation ───────────────────────────────────────────────

def test_sphere_betti():
    for subdivisions in [0, 1, 2]:
        s = generate_sphere(subdivisions=subdivisions)
        assert s.b0 == 1, f"sphere({subdivisions}): b0={s.b0}"
        assert s.b1 == 0, f"sphere({subdivisions}): b1={s.b1}"
        assert s.b2 == 1, f"sphere({subdivisions}): b2={s.b2}"
        assert s.chi == 2, f"sphere({subdivisions}): chi={s.chi}"


def test_torus_betti():
    for w, h in [(3, 4), (4, 4), (5, 6)]:
        t = generate_torus(width=w, height=h)
        assert t.b0 == 1, f"torus({w},{h}): b0={t.b0}"
        assert t.b1 == 2, f"torus({w},{h}): b1={t.b1}"
        assert t.b2 == 1, f"torus({w},{h}): b2={t.b2}"
        assert t.chi == 0, f"torus({w},{h}): chi={t.chi}"


def test_klein_bottle_betti():
    # Over GF(2): H_1(Klein) = (Z/2)^2, so b1=2; H_2 = Z/2, so b2=1
    k = generate_klein_bottle(width=4, height=4)
    assert k.b0 == 1
    assert k.b1 == 2
    assert k.b2 == 1
    assert k.chi == 0


def test_disjoint_union_betti():
    s1 = generate_sphere(subdivisions=0)
    s2 = generate_sphere(subdivisions=0)
    union = generate_disjoint_union([s1, s2])
    assert union.b0 == 2  # two components
    assert union.b1 == 0
    assert union.b2 == 2


def test_euler_characteristic():
    """V - E + F should always equal chi for all surface types."""
    for surface in [
        generate_sphere(0),
        generate_sphere(1),
        generate_torus(4, 4),
        generate_klein_bottle(4, 4),
    ]:
        assert surface.V - surface.E + surface.F == surface.chi


# ── Features ─────────────────────────────────────────────────────────

def test_feature_consistency():
    sphere = generate_sphere(1)
    features = extract_features(sphere)
    assert features["height_d1"] == float(sphere.V)
    assert features["width_d1"] == float(sphere.E)
    assert features["width_d2"] == float(sphere.F)
    assert features["null_d1"] == features["width_d1"] - features["rank_d1"]
    assert features["null_d2"] == features["width_d2"] - features["rank_d2"]


# ── Serialization roundtrip ──────────────────────────────────────────

def test_surface_roundtrip():
    original = generate_torus(4, 4)
    payload = original.to_dict()
    restored = SurfaceData.from_dict(payload)
    assert restored.V == original.V
    assert restored.E == original.E
    assert restored.F == original.F
    assert restored.b0 == original.b0
    assert restored.b1 == original.b1
    assert restored.b2 == original.b2
    assert restored.chi == original.chi


# ── Formula parsing and classification ───────────────────────────────

def test_parse_euler():
    s = parse_statement("height_d1 - width_d1 + width_d2 = 2")
    concepts = classify_statement(s)
    assert "chi" in concepts


def test_parse_b1():
    s = parse_statement("null_d1 - rank_d2 = 2")
    concepts = classify_statement(s)
    assert "b1" in concepts


def test_parse_b0():
    s = parse_statement("height_d1 - rank_d1 = 1")
    concepts = classify_statement(s)
    assert "b0" in concepts


def test_formula_evaluation():
    sphere = generate_sphere(1)
    features = sphere.features()
    euler = parse_statement("height_d1 - width_d1 + width_d2 = 2")
    assert euler.evaluate(features) is True


# ── Agent basics ─────────────────────────────────────────────────────

def test_ca_produces_candidates():
    sphere = generate_sphere(1)
    features = [sphere.features()]
    attention = np.ones(1, dtype=float)
    agent = ConjecturingAgent(CAConfig(), seed=42)
    candidate = agent.step(features, attention)
    assert candidate.statement.render()  # non-empty


def test_sa_attention_update():
    sa = SkepticalAgent(10, SAConfig())
    rows = [{"height_d1": float(i), "width_d1": 0.0, "height_d2": 0.0,
             "width_d2": 0.0, "rank_d1": 0.0, "rank_d2": 0.0,
             "null_d1": 0.0, "null_d2": 0.0} for i in range(10)]
    attention = sa.step("height_d1 - width_d1 = 0", rows, 0.5)
    assert abs(float(attention.sum()) - 1.0) < 1e-6


def test_ca_export_load_roundtrip():
    agent = ConjecturingAgent(CAConfig(), seed=42)
    agent.feature_preferences["null_d1"] = 0.5
    agent.concept_counts["b1"] = 3
    state = agent.export_state()

    agent2 = ConjecturingAgent(CAConfig(), seed=42)
    agent2.load_state(state)
    assert agent2.feature_preferences["null_d1"] == 0.5
    assert agent2.concept_counts["b1"] == 3
