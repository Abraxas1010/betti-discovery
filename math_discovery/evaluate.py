"""Four-variant ablation evaluation harness.

Compares the full system (M0) against ablated variants to measure
the contribution of each component:

    ┌──────────┬────────┬─────────────────────┬──────────────┐
    │ Variant  │ Oracle │ Skeptical Agent      │ What it tests│
    ├──────────┼────────┼─────────────────────┼──────────────┤
    │ Only CA  │ yes    │ no (uniform attn)    │ CA baseline  │
    │ M0       │ yes    │ yes                  │ Full system  │
    │ M1       │ no     │ yes                  │ Oracle value │
    │ M2       │ yes    │ no                   │ SA value     │
    └──────────┴────────┴─────────────────────┴──────────────┘

Runs untrained evaluation, then optionally trains and re-evaluates to
measure the effect of REINFORCE learning.

Usage::

    python -m math_discovery.evaluate --from-training --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from math_discovery.common import json_dumps
from math_discovery.config import RESULT_ROOT, DatasetConfig, EnvConfig, TrainingConfig, ensure_directories
from math_discovery.feature_extractor import extract_dataset_features
from math_discovery.mathworld_env import MathWorldEnv
from math_discovery.run_training import compute_ca_reward, train
from math_discovery.surface_data_gen import generate_dataset


def _load_rows(dataset: str, count: int, seed: int) -> list[dict[str, Any]]:
    surfaces = generate_dataset(DatasetConfig(dataset=dataset, count=count, seed=seed))
    extracted = extract_dataset_features(
        {"dataset": dataset, "count": len(surfaces), "surfaces": [surface.to_dict() for surface in surfaces]}
    )
    return extracted["features"]


def _variant_config(name: str, dataset: str) -> tuple[EnvConfig, bool]:
    config = EnvConfig(dataset=dataset)
    use_sa = True
    if name == "Only CA":
        config.ca_config.patch_size = 24
        config.ca_config.n_patches = 1
        use_sa = False
    elif name == "M1":
        config.use_provability_oracle = False
        config.fixed_provability_score = 0.5
    elif name == "M2":
        use_sa = False
    return config, use_sa


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(results))
    contains_chi = sum(1 for r in results if "chi" in r["concepts"])
    contains_b1 = sum(1 for r in results if "b1" in r["concepts"])
    proved_concept = sum(
        1 for r in results
        if r["success"] and any(c in {"chi", "b1"} for c in r["concepts"])
    )
    successes = sum(1 for r in results if r["success"])
    unique_atomic = sorted({r["statement"] for r in results})
    return {
        "contains_chi_pct": contains_chi / total,
        "contains_b1_pct": contains_b1 / total,
        "proved_concept_pct": proved_concept / total,
        "success_pct": successes / total,
        "unique_atomic_formulae": len(unique_atomic),
    }


def load_checkpoint(env: MathWorldEnv, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    env.ca.load_state(payload.get("ca_state") or {})
    env.sa.load_state(payload.get("sa_state") or {})
    return payload


def _evaluate_suite(
    rows: list[dict[str, Any]],
    dataset: str,
    n_eval_episodes: int,
    seed: int,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    variants = ["Only CA", "M0", "M1", "M2"]
    report: dict[str, Any] = {
        "dataset": dataset,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "variants": {},
    }
    for variant_index, name in enumerate(variants):
        config, use_sa = _variant_config(name, dataset)
        env = MathWorldEnv(rows, config, seed=seed + variant_index)
        checkpoint_meta: dict[str, Any] | None = None
        if checkpoint_path is not None:
            checkpoint_meta = load_checkpoint(env, checkpoint_path)
        results = []
        for _ in range(n_eval_episodes):
            episode = env.run_episode(use_sa=use_sa)
            results.append(episode.to_dict())
        report["variants"][name] = {
            "checkpoint_loaded": checkpoint_path is not None,
            "checkpoint_algorithm": checkpoint_meta.get("algorithm") if checkpoint_meta else None,
            "metrics": _metrics(results),
            "episodes": results,
        }
    return report


def evaluate_models(
    dataset: str,
    n_eval_episodes: int,
    seed: int,
    checkpoint_path: Path | None = None,
    train_episodes: int = 24,
    from_training: bool = False,
) -> dict[str, Any]:
    """Run the full ablation evaluation."""
    rows = _load_rows(dataset, 24, seed)
    report: dict[str, Any] = {
        "dataset": dataset,
        "n_eval_episodes": n_eval_episodes,
        "evaluations": {
            "untrained": _evaluate_suite(rows, dataset, n_eval_episodes, seed),
        },
    }
    effective_checkpoint = checkpoint_path
    if from_training and effective_checkpoint is None:
        _, training_payload = train(
            TrainingConfig(
                dataset=dataset,
                n_episodes=train_episodes,
                seed=seed,
                env_config=EnvConfig(dataset=dataset),
            )
        )
        effective_checkpoint = Path(training_payload["checkpoint_path"])
        report["training"] = {
            "episodes": train_episodes,
            "checkpoint_path": str(effective_checkpoint),
            "algorithm": training_payload["checkpoint"]["algorithm"],
        }
    if effective_checkpoint is not None:
        report["evaluations"]["trained"] = _evaluate_suite(
            rows, dataset, n_eval_episodes, seed, checkpoint_path=effective_checkpoint,
        )
        untrained_m0 = report["evaluations"]["untrained"]["variants"]["M0"]["metrics"]["proved_concept_pct"]
        trained_m0 = report["evaluations"]["trained"]["variants"]["M0"]["metrics"]["proved_concept_pct"]
        only_ca = report["evaluations"]["untrained"]["variants"]["Only CA"]["metrics"]["proved_concept_pct"]
        report["comparisons"] = {
            "m0_trained_gt_untrained": trained_m0 > untrained_m0,
            "m0_untrained_gt_only_ca": untrained_m0 > only_ca,
            "values": {
                "M0_trained": trained_m0,
                "M0_untrained": untrained_m0,
                "Only_CA_untrained": only_ca,
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate math discovery ablations.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--from-training", action="store_true")
    parser.add_argument("--train-episodes", type=int, default=24)
    parser.add_argument("--dataset", default="D0", choices=["D0", "D1", "D2", "D3"])
    parser.add_argument("--n-eval-episodes", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    ensure_directories()
    report = evaluate_models(
        args.dataset, args.n_eval_episodes, args.seed,
        checkpoint_path=args.checkpoint,
        train_episodes=args.train_episodes,
        from_training=args.from_training,
    )
    output_path = args.output or (RESULT_ROOT / f"{args.dataset}_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_dumps(report) + "\n", encoding="utf-8")

    summary = {
        label: {name: payload["metrics"] for name, payload in section["variants"].items()}
        for label, section in report["evaluations"].items()
    }
    final = {
        "ok": True,
        "output": str(output_path),
        "summary": summary,
        "comparisons": report.get("comparisons"),
    }
    if args.json:
        print(json_dumps(final))
    else:
        print(f"Evaluation results -> {output_path}")
        for label, variants in summary.items():
            print(f"\n  {label}:")
            for name, metrics in variants.items():
                print(f"    {name}: proved_concept={metrics['proved_concept_pct']:.1%}")


if __name__ == "__main__":
    main()
