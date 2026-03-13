"""Conjecturing Agent (CA) — proposes candidate mathematical formulas.

The CA searches over ~20,000 linear integer forms to find equations that
hold (or nearly hold) on subsets of the data.  It uses:

- **Patch sampling**: attention-weighted selection of nearby data points
- **Analytic search**: exhaustive evaluation of candidate forms on each patch
- **Softmax scaffolding**: stochastic selection among top candidates
- **Per-feature REINFORCE**: lightweight policy gradient on feature preferences

When the system discovers a concept (chi, b1, b0), it updates preferences
for the canonical features of that concept, reinforcing future proposals
in the same direction.

Example::

    from math_discovery.conjecturing_agent import ConjecturingAgent
    from math_discovery.config import CAConfig

    agent = ConjecturingAgent(CAConfig(), seed=42)
    # agent.step(feature_rows, attention_weights) -> CandidateStatement
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from math_discovery.common import (
    AtomicFormula,
    FEATURE_NAMES,
    Statement,
    classify_statement,
    iter_linear_forms,
    json_dumps,
    parse_statement,
    softmax,
)
from math_discovery.config import CAConfig, DatasetConfig, default_backend
from math_discovery.feature_extractor import extract_dataset_features
from math_discovery.surface_data_gen import generate_dataset


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@dataclass(slots=True)
class CandidateStatement:
    """A candidate formula proposed by the CA, with scoring metadata."""
    statement: Statement
    residual: float
    adjusted_score: float
    patch_indices: list[int]
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement.render(),
            "residual": self.residual,
            "adjusted_score": self.adjusted_score,
            "patch_indices": self.patch_indices,
            "backend": self.backend,
            "concepts": sorted(classify_statement(self.statement)),
        }


class AnalyticBackend:
    """Exhaustive search over all candidate linear forms."""

    def __init__(self, config: CAConfig):
        self.config = config
        self.linear_forms = iter_linear_forms(
            feature_names=config.feature_names,
            max_terms=config.max_terms,
            coefficient_values=config.coefficient_values,
        )

    def search(
        self,
        feature_rows: list[dict[str, float]],
        weights: np.ndarray,
        patch_indices: list[int],
        feature_preferences: dict[str, float],
        concept_counts: dict[str, int],
    ) -> list[CandidateStatement]:
        if not patch_indices:
            return []
        patch = [feature_rows[index] for index in patch_indices]
        patch_weights = weights[np.array(patch_indices, dtype=int)]
        if float(patch_weights.sum()) <= 0.0:
            patch_weights = np.full(len(patch_indices), 1.0 / max(1, len(patch_indices)), dtype=float)
        else:
            patch_weights = patch_weights / float(patch_weights.sum())

        candidates: dict[tuple[tuple[str, int], int], CandidateStatement] = {}
        for linear_form in self.linear_forms:
            if len(linear_form) < 2:
                continue
            values = np.array(
                [sum(coeff * float(row[name]) for name, coeff in linear_form) for row in patch],
                dtype=float,
            )
            constant = int(np.round(np.average(values, weights=patch_weights)))
            residual = float(np.average(np.abs(values - constant), weights=patch_weights))
            if residual > self.config.residual_tolerance:
                continue
            atom = AtomicFormula(coefficients=linear_form, rhs=constant)
            truth_values = np.array(
                [
                    1.0 if abs(sum(coeff * float(row[name]) for name, coeff in linear_form) - constant) <= 1e-6 else 0.0
                    for row in feature_rows
                ],
                dtype=float,
            )
            if float(truth_values.sum()) in {0.0, float(len(truth_values))}:
                continue
            preference = sum(feature_preferences.get(name, 0.0) for name, _ in linear_form)
            concepts = classify_statement(Statement(atoms=(atom,)))
            concept_bonus = 0.04 * len(concepts)
            novelty_bonus = sum(0.12 / (1 + concept_counts.get(concept, 0)) for concept in concepts)
            complexity_penalty = 0.01 * atom.complexity()
            adjusted_score = float(-residual + preference + concept_bonus + novelty_bonus - complexity_penalty)
            key = (atom.coefficients, atom.rhs)
            existing = candidates.get(key)
            candidate = CandidateStatement(
                statement=Statement(atoms=(atom,)),
                residual=residual,
                adjusted_score=adjusted_score,
                patch_indices=patch_indices,
                backend="analytic",
            )
            if existing is None or candidate.adjusted_score > existing.adjusted_score:
                candidates[key] = candidate

        ordered = sorted(
            candidates.values(),
            key=lambda c: (c.adjusted_score, -c.residual),
            reverse=True,
        )
        return ordered[: self.config.max_candidates]


class ConjecturingAgent:
    """The Conjecturing Agent: proposes formulas, learns from rewards.

    The CA maintains per-feature preference weights that are updated via
    REINFORCE.  Positive rewards for concept-bearing formulas strengthen
    the canonical features of those concepts.
    """

    def __init__(self, config: CAConfig, seed: int = 42):
        self.config = config
        self.backend_name = default_backend(config.backend)
        self.rng = np.random.default_rng(seed)
        self.feature_preferences = dict(config.feature_preferences)
        self.concept_counts = {"chi": 0, "b1": 0, "b0": 0}
        self.analytic_backend = AnalyticBackend(config)
        self.last_policy: dict[str, Any] | None = None

    def _sample_patch(
        self,
        feature_rows: list[dict[str, float]],
        attention_weights: np.ndarray,
    ) -> list[int]:
        total = len(feature_rows)
        sample_size = min(total, self.config.patch_size)
        if sample_size <= 0:
            return []
        weights = attention_weights.astype(float)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            weights = np.full(total, 1.0 / max(1, total), dtype=float)
        else:
            weights = weights / total_weight
        anchor = int(self.rng.choice(total, p=weights))
        matrix = np.array(
            [[float(row[feature]) for feature in self.config.feature_names] for row in feature_rows],
            dtype=float,
        )
        scales = np.maximum(matrix.std(axis=0, ddof=0), 1.0)
        normalized = matrix / scales
        distances = np.linalg.norm(normalized - normalized[anchor], axis=1)
        jitter = self.rng.uniform(0.0, 1e-6, size=distances.shape[0])
        order = np.argsort(distances + jitter)
        patch = sorted(int(index) for index in order[:sample_size].tolist())
        return patch

    def _candidate_backend(self) -> str:
        if self.backend_name == "pysr" and not _module_available("pysr"):
            return "analytic"
        if self.backend_name == "gplearn" and not _module_available("gplearn"):
            return "analytic"
        return self.backend_name

    def generate_atomic_formulae(
        self,
        feature_rows: list[dict[str, float]],
        attention_weights: np.ndarray,
    ) -> list[CandidateStatement]:
        backend = self._candidate_backend()
        candidates: list[CandidateStatement] = []
        for _ in range(self.config.n_patches):
            patch_indices = self._sample_patch(feature_rows, attention_weights)
            if backend == "analytic":
                candidates.extend(
                    self.analytic_backend.search(
                        feature_rows,
                        attention_weights,
                        patch_indices,
                        self.feature_preferences,
                        self.concept_counts,
                    )
                )
            else:
                raise NotImplementedError(f"backend {backend} not yet implemented")
        dedup: dict[str, CandidateStatement] = {}
        for candidate in candidates:
            key = candidate.statement.render()
            if key not in dedup or candidate.adjusted_score > dedup[key].adjusted_score:
                dedup[key] = candidate
        return sorted(dedup.values(), key=lambda c: c.adjusted_score, reverse=True)

    def scaffold(self, candidates: list[CandidateStatement]) -> CandidateStatement | None:
        """Stochastic selection via softmax over adjusted scores."""
        if not candidates:
            return None
        scores = np.array([c.adjusted_score for c in candidates], dtype=float)
        probabilities = softmax(scores)
        selected_index = int(self.rng.choice(len(candidates), p=probabilities))
        selected = candidates[selected_index]
        self.last_policy = {
            "selected_index": selected_index,
            "probability": float(probabilities[selected_index]),
            "features": sorted(selected.statement.used_features()),
            "concepts": sorted(classify_statement(selected.statement)),
        }
        return selected

    def step(
        self,
        feature_rows: list[dict[str, float]],
        attention_weights: np.ndarray,
    ) -> CandidateStatement:
        """Run one full CA step: generate candidates, then select one."""
        candidates = self.generate_atomic_formulae(feature_rows, attention_weights)
        selected = self.scaffold(candidates)
        if selected is None:
            fallback = CandidateStatement(
                statement=parse_statement("width_d1 - height_d2 = 0"),
                residual=1.0,
                adjusted_score=-1.0,
                patch_indices=[],
                backend="analytic",
            )
            self.last_policy = {
                "selected_index": 0,
                "probability": 1.0,
                "features": sorted(fallback.statement.used_features()),
                "concepts": sorted(classify_statement(fallback.statement)),
            }
            return fallback
        return selected

    def apply_reward(self, reward: float) -> None:
        """Update feature preferences via REINFORCE.

        - Concept-bearing successes update the canonical features of that concept.
        - Non-concept failures are true no-ops (no preference drift).
        - Non-concept successes update non-protected features.
        """
        if not self.last_policy:
            return
        scale = float(self.config.learning_rate * reward)
        concept_features: dict[str, tuple[str, ...]] = {
            "chi": ("height_d1", "width_d1", "width_d2"),
            "b1": ("null_d1", "rank_d2"),
            "b0": ("height_d1", "rank_d1"),
        }
        protected_features = {feature for values in concept_features.values() for feature in values}
        features: set[str] = set()
        for concept in self.last_policy.get("concepts", []):
            features.update(concept_features.get(concept, ()))
        if not features:
            if reward <= 0.0:
                return  # non-concept failure -> true no-op
            features.update(self.last_policy.get("features", []))
            features.difference_update(protected_features)
        if not features:
            return
        for feature in sorted(features):
            self.feature_preferences[feature] = self.feature_preferences.get(feature, 0.0) + scale
        if reward > 0.0:
            for concept in self.last_policy.get("concepts", []):
                if concept in self.concept_counts:
                    self.concept_counts[concept] += 1

    def export_state(self) -> dict[str, Any]:
        return {
            "backend": self._candidate_backend(),
            "feature_preferences": dict(sorted(self.feature_preferences.items())),
            "concept_counts": dict(sorted(self.concept_counts.items())),
        }

    def load_state(self, payload: dict[str, Any]) -> None:
        feature_preferences = payload.get("feature_preferences") or {}
        for feature in self.config.feature_names:
            if feature in feature_preferences:
                self.feature_preferences[feature] = float(feature_preferences[feature])
        concept_counts = payload.get("concept_counts") or {}
        for concept in self.concept_counts:
            if concept in concept_counts:
                self.concept_counts[concept] = int(concept_counts[concept])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate mathematical statements from surface features.")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--dataset", default="D0", choices=["D0", "D1", "D2", "D3"])
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="auto", choices=["auto", "analytic", "pysr", "gplearn"])
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.data is not None:
        payload = json.loads(args.data.read_text(encoding="utf-8"))
        if "features" not in payload:
            payload = extract_dataset_features(payload)
    else:
        surfaces = generate_dataset(DatasetConfig(dataset=args.dataset, count=args.count, seed=args.seed))
        payload = extract_dataset_features(
            {"dataset": args.dataset, "count": len(surfaces), "surfaces": [s.to_dict() for s in surfaces]}
        )

    feature_rows = [row["features"] for row in payload["features"]]
    attention = np.full(len(feature_rows), 1.0 / max(1, len(feature_rows)), dtype=float)
    agent = ConjecturingAgent(CAConfig(backend=args.backend), seed=args.seed)

    outputs = []
    for _ in range(args.n_steps):
        candidate = agent.step(feature_rows, attention)
        outputs.append(candidate.to_dict())
        agent.apply_reward(0.05)

    report = {
        "ok": True,
        "backend": agent._candidate_backend(),
        "dataset": payload["dataset"],
        "steps": outputs,
        "agent_state": agent.export_state(),
    }
    if args.json:
        print(json_dumps(report))
    else:
        print(f"Backend: {report['backend']}")
        for step, output in enumerate(outputs, start=1):
            print(f"[{step}] {output['statement']} :: residual={output['residual']:.4f}")


if __name__ == "__main__":
    main()
