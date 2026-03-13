"""Configuration dataclasses and directory management.

All output paths default to ``./output/`` relative to the current working
directory.  Override by setting the ``BETTI_DISCOVERY_OUTPUT`` environment
variable or by passing ``output_root`` to ``ensure_directories()``.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

from math_discovery.common import FEATURE_NAMES


def _default_output_root() -> Path:
    env = os.environ.get("BETTI_DISCOVERY_OUTPUT")
    if env:
        return Path(env)
    return Path.cwd() / "output"


OUTPUT_ROOT = _default_output_root()
SURFACE_ROOT = OUTPUT_ROOT / "surfaces"
EPISODE_ROOT = OUTPUT_ROOT / "episodes"
MODEL_ROOT = OUTPUT_ROOT / "models"
RESULT_ROOT = OUTPUT_ROOT / "results"
TMP_ROOT = OUTPUT_ROOT / "tmp"


def dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def default_backend(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    if dependency_available("pysr"):
        return "pysr"
    if dependency_available("gplearn"):
        return "gplearn"
    return "analytic"


def ensure_directories() -> None:
    for path in (SURFACE_ROOT, EPISODE_ROOT, MODEL_ROOT, RESULT_ROOT, TMP_ROOT):
        path.mkdir(parents=True, exist_ok=True)


# ── Dataset generation config ────────────────────────────────────────

@dataclass(slots=True)
class DatasetConfig:
    """Controls which surfaces are generated and how many."""
    dataset: str = "D0"
    count: int = 24
    seed: int = 42
    sphere_subdivisions: tuple[int, ...] = (0, 1, 2)
    torus_grids: tuple[tuple[int, int], ...] = ((3, 4), (4, 4), (4, 5), (5, 6))
    klein_grids: tuple[tuple[int, int], ...] = ((3, 4), (4, 4), (4, 5))


# ── Agent configs ────────────────────────────────────────────────────

@dataclass(slots=True)
class CAConfig:
    """Conjecturing Agent configuration."""
    backend: str = "auto"
    n_patches: int = 8
    patch_size: int = 6
    max_terms: int = 4
    coefficient_values: tuple[int, ...] = (-2, -1, 1, 2)
    max_candidates: int = 24
    residual_tolerance: float = 0.25
    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_preferences: dict[str, float] = field(
        default_factory=lambda: {feature: 0.0 for feature in FEATURE_NAMES}
    )
    learning_rate: float = 0.05


@dataclass(slots=True)
class SAConfig:
    """Skeptical Agent configuration."""
    focus_scale: float = 1.5
    relaxation: float = 0.12
    learning_rate: float = 0.05
    min_attention: float = 1e-4


# ── Environment & training configs ───────────────────────────────────

@dataclass(slots=True)
class EnvConfig:
    """MathWorld environment configuration."""
    dataset: str = "D0"
    max_steps: int = 4
    provability_threshold: float = 0.95
    use_provability_oracle: bool = True
    fixed_provability_score: float = 0.5
    premise_profile: str = "auto"
    ca_config: CAConfig = field(default_factory=CAConfig)
    sa_config: SAConfig = field(default_factory=SAConfig)


@dataclass(slots=True)
class TrainingConfig:
    """REINFORCE training loop configuration."""
    dataset: str = "D0"
    n_episodes: int = 24
    seed: int = 42
    game_mode: str = "cooperative"
    algorithm: str = "reinforce"
    checkpoint_interval: int = 6
    env_config: EnvConfig = field(default_factory=EnvConfig)


@dataclass(slots=True)
class EvalConfig:
    """Ablation evaluation configuration."""
    dataset: str = "D0"
    n_eval_episodes: int = 8
    train_episodes: int = 24
    seed: int = 42
    env_config: EnvConfig = field(default_factory=EnvConfig)
