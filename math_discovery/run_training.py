"""REINFORCE training loop for the multi-agent discovery game.

Trains the Conjecturing Agent and Skeptical Agent over multiple episodes.
The reward structure incentivizes discovering provable concept-bearing
formulas, with bonuses for chi (Euler characteristic) discovery:

    base reward = -0.25  (failure penalty)
    + 0.75 if chi concept found
    + 0.45 if b1 concept found
    + 0.90 + 0.15 * provability  if episode succeeds
    + 0.05 * max(0, 4 - steps)   (speed bonus)

Usage::

    python -m math_discovery.run_training --n-episodes 24 --json
"""

from __future__ import annotations

import argparse
from typing import Any

from math_discovery.common import json_dumps
from math_discovery.config import MODEL_ROOT, DatasetConfig, EnvConfig, TrainingConfig, ensure_directories
from math_discovery.feature_extractor import extract_dataset_features
from math_discovery.mathworld_env import EpisodeResult, MathWorldEnv
from math_discovery.surface_data_gen import generate_dataset


def compute_ca_reward(result: EpisodeResult) -> float:
    """Compute the Conjecturing Agent's reward for an episode."""
    reward = -0.25
    if "chi" in result.concepts:
        reward += 0.75
    if "b1" in result.concepts:
        reward += 0.45
    if result.success:
        reward += 0.9 + (0.15 * result.provability)
        reward += 0.05 * max(0, 4 - result.steps)
    return reward


def compute_sa_reward(result: EpisodeResult, game_mode: str) -> float:
    """Compute the Skeptical Agent's reward (cooperative or competitive)."""
    reward = compute_ca_reward(result)
    return reward if game_mode == "cooperative" else -reward


def _load_rows(dataset: str, count: int, seed: int) -> list[dict[str, Any]]:
    surfaces = generate_dataset(DatasetConfig(dataset=dataset, count=count, seed=seed))
    extracted = extract_dataset_features(
        {"dataset": dataset, "count": len(surfaces), "surfaces": [surface.to_dict() for surface in surfaces]}
    )
    return extracted["features"]


def train(config: TrainingConfig) -> tuple[MathWorldEnv, dict[str, Any]]:
    """Run the full training loop and save a checkpoint."""
    ensure_directories()
    rows = _load_rows(config.dataset, DatasetConfig(dataset=config.dataset, seed=config.seed).count, config.seed)
    env = MathWorldEnv(rows, config.env_config, seed=config.seed)
    history = []
    for episode_index in range(config.n_episodes):
        result = env.run_episode()
        ca_reward = compute_ca_reward(result)
        sa_reward = compute_sa_reward(result, config.game_mode)
        if config.algorithm == "maddpg":
            joint_reward = 0.5 * (ca_reward + sa_reward)
            env.ca.apply_reward(joint_reward)
            env.sa.apply_reward(joint_reward)
        else:
            env.ca.apply_reward(ca_reward)
            env.sa.apply_reward(sa_reward)
        history.append({
            "episode": episode_index + 1,
            "result": result.to_dict(),
            "ca_reward": ca_reward,
            "sa_reward": sa_reward,
        })

    checkpoint = {
        "algorithm": config.algorithm,
        "game_mode": config.game_mode,
        "dataset": config.dataset,
        "episodes": config.n_episodes,
        "ca_state": env.ca.export_state(),
        "sa_state": env.sa.export_state(),
        "history": history,
    }
    checkpoint_path = MODEL_ROOT / "latest.json"
    checkpoint_path.write_text(json_dumps(checkpoint) + "\n", encoding="utf-8")
    return env, {"checkpoint_path": str(checkpoint_path), "checkpoint": checkpoint}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multi-agent discovery system.")
    parser.add_argument("--dataset", default="D0", choices=["D0", "D1", "D2", "D3"])
    parser.add_argument("--n-episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--game-mode", default="cooperative", choices=["cooperative", "competitive"])
    parser.add_argument("--algorithm", default="reinforce", choices=["reinforce", "maddpg"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = TrainingConfig(
        dataset=args.dataset,
        n_episodes=args.n_episodes,
        seed=args.seed,
        game_mode=args.game_mode,
        algorithm=args.algorithm,
        env_config=EnvConfig(dataset=args.dataset),
    )
    _, payload = train(config)
    report = {
        "ok": True,
        "checkpoint_path": payload["checkpoint_path"],
        "episodes": config.n_episodes,
        "algorithm": config.algorithm,
    }
    if args.json:
        print(json_dumps(report))
    else:
        print(f"Training complete: {config.n_episodes} episodes, checkpoint -> {payload['checkpoint_path']}")


if __name__ == "__main__":
    main()
