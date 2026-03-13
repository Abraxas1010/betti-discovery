"""Skeptical Agent (SA) — steers attention toward counterexamples.

The SA cooperates with the Conjecturing Agent by maintaining a probability
distribution (attention vector) over the dataset.  When a formula fails on
certain data points, the SA increases attention on those points, guiding
the CA toward harder regions of the data.

Key mechanisms:
- **Focus scaling**: amplifies attention on data points where the current
  formula fails, proportional to provability score
- **Relaxation blending**: prevents attention collapse by mixing with
  a uniform distribution
- **Reward-driven adaptation**: focus_scale and relaxation evolve via
  the same reward signal as the CA

Example::

    from math_discovery.skeptical_agent import SkepticalAgent
    from math_discovery.config import SAConfig

    sa = SkepticalAgent(n_datapoints=24, config=SAConfig())
    new_attention = sa.step("height_d1 - width_d1 + width_d2 = 2", feature_rows, 0.8)
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from math_discovery.common import json_dumps, normalized_entropy, parse_statement
from math_discovery.config import DatasetConfig, SAConfig
from math_discovery.feature_extractor import extract_dataset_features
from math_discovery.surface_data_gen import generate_dataset


class SkepticalAgent:
    """Attention-steering agent that focuses on counterexamples."""

    def __init__(self, n_datapoints: int, config: SAConfig):
        self.config = config
        self.n_datapoints = n_datapoints
        self.attention = np.full(n_datapoints, 1.0 / max(1, n_datapoints), dtype=float)
        self.last_policy: dict[str, Any] | None = None

    def step(
        self,
        statement_text: str,
        feature_rows: list[dict[str, float]],
        provability_score: float,
    ) -> np.ndarray:
        """Update attention: increase weight on data points where the statement fails."""
        statement = parse_statement(statement_text)
        truth_values = np.array([1.0 if statement.evaluate(row) else 0.0 for row in feature_rows], dtype=float)
        false_mask = 1.0 - truth_values
        focused = self.attention.copy()
        if float(false_mask.sum()) > 0.0:
            focused *= 1.0 + self.config.focus_scale * max(provability_score, 0.1) * false_mask
        uniform = np.full_like(focused, 1.0 / max(1, focused.size), dtype=float)
        blended = (1.0 - self.config.relaxation) * focused + self.config.relaxation * uniform
        blended = np.clip(blended, self.config.min_attention, None)
        blended /= float(blended.sum())
        self.last_policy = {
            "false_count": int(false_mask.sum()),
            "entropy_before": normalized_entropy(self.attention),
            "entropy_after": normalized_entropy(blended),
            "statement": statement_text,
        }
        self.attention = blended
        return self.attention

    def apply_reward(self, reward: float) -> None:
        """Adapt focus parameters: positive reward increases focus, negative relaxes."""
        delta = self.config.learning_rate * reward
        self.config.focus_scale = float(np.clip(self.config.focus_scale + delta, 0.2, 4.0))
        self.config.relaxation = float(np.clip(self.config.relaxation - 0.5 * delta, 0.01, 0.4))

    def export_state(self) -> dict[str, Any]:
        return {
            "focus_scale": self.config.focus_scale,
            "relaxation": self.config.relaxation,
            "entropy": normalized_entropy(self.attention),
        }

    def load_state(self, payload: dict[str, Any]) -> None:
        if "focus_scale" in payload:
            self.config.focus_scale = float(payload["focus_scale"])
        if "relaxation" in payload:
            self.config.relaxation = float(payload["relaxation"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate the skeptical agent's attention dynamics.")
    parser.add_argument("--n-datapoints", type=int, default=48)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--statement", default="height_d1 - width_d1 + width_d2 = 2")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    dataset = generate_dataset(DatasetConfig(dataset="D0", count=args.n_datapoints, seed=42))
    payload = extract_dataset_features(
        {"dataset": "D0", "count": len(dataset), "surfaces": [s.to_dict() for s in dataset]}
    )
    rows = [row["features"] for row in payload["features"]]
    agent = SkepticalAgent(len(rows), SAConfig())

    history = []
    for step in range(args.n_steps):
        provability = min(0.9, 0.15 + 0.05 * step)
        attention = agent.step(args.statement, rows, provability)
        history.append({
            "step": step + 1,
            "entropy": normalized_entropy(attention),
            "max_weight": float(attention.max()),
            "false_count": agent.last_policy["false_count"] if agent.last_policy else 0,
        })
        agent.apply_reward(0.1)

    if args.json:
        print(json_dumps({"ok": True, "steps": history, "final_state": agent.export_state()}))
    else:
        for row in history:
            print(row)


if __name__ == "__main__":
    main()
