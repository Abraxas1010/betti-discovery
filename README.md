<img src="assets/Apoth3osis.webp" alt="Apoth3osis — Formal Mathematics and Verified Software" width="140"/>

<sub><strong>Our tech stack is ontological:</strong><br>
<strong>Hardware — Physics</strong><br>
<strong>Software — Mathematics</strong><br><br>
<strong>Our engineering workflow is simple:</strong> discover, build, grow, learn & teach</sub>

---

<sub>
<strong>Acknowledgment</strong><br>
We humbly thank the collective intelligence of humanity for providing the technology and culture we cherish. We do our best to properly reference the authors of the works utilized herein, though we may occasionally fall short. Our formalization acts as a reciprocal validation—confirming the structural integrity of their original insights while securing the foundation upon which we build. In truth, all creative work is derivative; we stand on the shoulders of those who came before, and our contributions are simply the next link in an unbroken chain of human ingenuity.
</sub>

---

# Betti Discovery — Automated Algebraic-Topological Invariant Discovery from Triangulated Surfaces

[![License: Apoth3osis License Stack v1](https://img.shields.io/badge/License-Apoth3osis%20License%20Stack%20v1-blue.svg)](LICENSE.md)

A multi-agent system that automatically discovers provable formulas for **Betti numbers** and **Euler characteristic** from raw triangulated surface data. Two cooperative AI agents — a Conjecturing Agent and a Skeptical Agent — play a discovery game over combinatorial surfaces, proposing candidate integer-linear equations and verifying them against formally proved Lean 4 theorems.

## What is this?

Betti Discovery generates triangulated surfaces (spheres, tori, Klein bottles), computes GF(2) chain-complex features from their boundary matrices, and uses reinforcement learning to discover the classical algebraic-topological invariants — without being told what they are.

## How it works

The system has five layers that build on each other:

### 1. Surface Generation

Deterministic combinatorial generators produce triangulated closed surfaces:

| Surface | Construction | Betti numbers | Euler char |
|---------|-------------|---------------|------------|
| **Sphere** | Octahedron + loop subdivision | (1, 0, 1) | 2 |
| **Torus** | Rectangular grid, periodic identification | (1, 2, 1) | 0 |
| **Klein bottle** | Grid with orientation-reversing glue | (1, 2, 1) over GF(2) | 0 |
| **Disjoint union** | Block-diagonal composition | additive | additive |

```python
from math_discovery.surface_data_gen import generate_sphere, generate_torus

sphere = generate_sphere(subdivisions=1)
print(f"Sphere: V={sphere.V}, E={sphere.E}, F={sphere.F}, chi={sphere.chi}")
# Sphere: V=18, E=48, F=32, chi=2

torus = generate_torus(width=4, height=4)
print(f"Torus: b0={torus.b0}, b1={torus.b1}, b2={torus.b2}")
# Torus: b0=1, b1=2, b2=1
```

### 2. GF(2) Chain Complex Features

From each surface's boundary matrices `d1` (vertices x edges) and `d2` (edges x faces), we extract 8 features using Gaussian elimination over the two-element field:

```
C₂ --d2--> C₁ --d1--> C₀

height_d1 = V       width_d1  = E       rank_d1           null_d1 = E - rank_d1
height_d2 = E       width_d2  = F       rank_d2           null_d2 = F - rank_d2
```

The **relationships** between these 8 numbers encode the topology:
- `V - E + F = chi` (Euler characteristic)
- `null(d1) - rank(d2) = b1` (first Betti number: independent cycles)
- `V - rank(d1) = b0` (zeroth Betti number: connected components)

But the agents don't know this. They have to discover it.

### 3. Multi-Agent Discovery Game

Two agents cooperate in an episodic game:

**Conjecturing Agent (CA)** searches ~20,000 candidate linear integer equations, scoring them by:
- Empirical fit (low residual on sampled data patches)
- Learned feature preferences (per-feature REINFORCE weights)
- Concept bonuses (extra score for chi/b1/b0-shaped formulas)
- Diminishing novelty (prevents fixation on already-discovered concepts)

**Skeptical Agent (SA)** maintains an attention distribution over the dataset, steering the CA toward counterexample-rich regions — data points where current proposals fail.

### 4. Certified Provability Oracle

When the CA proposes a formula, it's checked against **5 formally verified Lean 4 theorems** via deterministic template matching:

| Template | Formula | Profile |
|----------|---------|---------|
| `sphereEuler` | V - E + F = 2 | sphere |
| `torusEuler` | V - E + F = 0 | torus |
| `vanishingMiddleBetti` | null(d1) - rank(d2) = 0 | sphere |
| `bettiOneValue2` | null(d1) - rank(d2) = 2 | torus/Klein |
| `twoComponentB0` | V - rank(d1) = 2 | disjoint union |

No neural network, no heuristic — if the formula matches a theorem template exactly, it scores 1.0 provability. The Lean proofs are in [`lean/RankNullityPremises.lean`](lean/RankNullityPremises.lean).

### 5. Training & Evaluation

**REINFORCE training** over 24 episodes updates the CA's feature preferences and the SA's focus parameters. The reward signal:
- +0.75 for discovering chi, +0.45 for b1
- +0.90 + 0.15 × provability for successful episodes
- -0.25 baseline penalty for failures

**4-variant ablation** isolates component contributions:

| Variant | Oracle | SA | Tests |
|---------|--------|----|-------|
| Only CA | yes | no | CA baseline |
| M0 | yes | yes | Full system |
| M1 | no | yes | Oracle value |
| M2 | yes | no | SA value |

## Quick Start

### Install

```bash
pip install -e .
# or just: pip install numpy  (the only dependency)
```

### Run the example

```bash
python examples/quickstart.py
```

### Generate → Train → Evaluate (CLI)

```bash
# Generate 24 triangulated surfaces
python -m math_discovery generate --dataset D0 --count 24

# Train for 24 episodes
python -m math_discovery train --n-episodes 24 --json

# Run ablation evaluation (trains + evaluates)
python -m math_discovery evaluate --from-training --json
```

### Python API

```python
from math_discovery.config import TrainingConfig, EnvConfig
from math_discovery.run_training import train
from math_discovery.evaluate import evaluate_models

# Train
config = TrainingConfig(n_episodes=24, dataset="D0")
env, checkpoint = train(config)
print(f"CA learned preferences: {env.ca.export_state()['feature_preferences']}")

# Evaluate
results = evaluate_models("D0", n_eval_episodes=8, seed=42, from_training=True)
for name, metrics in results["evaluations"]["untrained"]["variants"].items():
    print(f"{name}: proved_concept = {metrics['metrics']['proved_concept_pct']:.1%}")
```

## Datasets

| Name | Surface types | Difficulty |
|------|--------------|------------|
| D0 | sphere, torus | Easiest — orientable only |
| D1 | + Klein bottle | Adds non-orientable |
| D2 | + disjoint unions | Adds disconnected |
| D3 | All four types | Full complexity |

## Project Structure

```
math_discovery/
├── common.py             # GF(2) algebra, formula representation, parsing
├── config.py             # All configuration dataclasses
├── surface_data_gen.py   # Deterministic surface generators
├── feature_extractor.py  # 8-feature extraction from boundary matrices
├── conjecturing_agent.py # CA: formula search + REINFORCE learning
├── skeptical_agent.py    # SA: attention steering toward counterexamples
├── mathworld_env.py      # Discovery game environment + provability oracle
├── run_training.py       # REINFORCE training loop
├── evaluate.py           # 4-variant ablation harness
└── __main__.py           # CLI dispatcher

lean/
├── RankNullityPremises.lean  # Formal Lean 4 theorems (5 templates)
└── SRTranslation.lean        # Symbolic regression AST bridge

examples/
└── quickstart.py         # Self-contained demo
```

## Current Results (Honest)

The untrained system discovers b1-type formulas at **87.5%** proved-concept rate — the multi-agent game + certified oracle works well out of the box.

The trained system currently scores **62.5%** — REINFORCE training creates a b1 preference bias that makes chi discovery harder. This is an open research problem documented in the project.

**What works:**
- Surface generation and Betti number computation are mathematically correct
- The certified oracle is sound (backed by verified Lean theorems)
- The SA demonstrably improves discovery (M0 > Only CA)

**What doesn't yet:**
- Training degrades chi discovery (b1 dominance in reward signal)
- The CA cannot discover chi from a trained state (0% chi in 24 training episodes)

## Lean 4 Formal Bridge

The `lean/` directory contains two files that connect the Python discovery system to formal mathematics:

- **`RankNullityPremises.lean`** — 5 theorems proved in Lean 4 using Mathlib, over arbitrary division rings and finite-dimensional modules. Each theorem's proof is `omega` after unfolding definitions.
- **`SRTranslation.lean`** — An AST for symbolic regression expressions, with `render` and `featureArity` functions.

These are reference/companion files. The Python system uses template matching against the theorem shapes; the Lean files prove those shapes are correct.

## Dependencies

- **Python 3.11+**
- **NumPy** (the only runtime dependency)
- **pytest** (optional, for running tests)

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

[Apoth3osis License Stack v1](LICENSE.md)
