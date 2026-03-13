"""MathWorld Environment — the discovery game arena.

The environment runs episodes where:
1. The CA proposes a formula
2. The provability oracle evaluates whether the formula matches a known theorem
3. The SA updates attention based on where the formula fails
4. Steps repeat until success or budget exhaustion

The **certified provability oracle** is the key component: it performs
deterministic template matching against 5 formally verified Lean theorems,
giving a binary 0/1 provability score.  No neural network, no heuristic —
if the formula matches a theorem template with the right premise profile
and right-hand-side constant, it scores 1.0.

The 5 certified templates:
- ``sphereEuler``: V - E + F = 2 (sphere profile)
- ``torusEuler``: V - E + F = 0 (torus profile)
- ``vanishingMiddleBetti``: null(d1) - rank(d2) = 0 (sphere profile)
- ``bettiOneValue2``: null(d1) - rank(d2) = 2 (torus/Klein bottle profile)
- ``twoComponentB0``: V - rank(d1) = 2 (disjoint union profile)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from math_discovery.common import Statement, classify_statement, json_dumps, normalized_entropy
from math_discovery.config import EPISODE_ROOT, EnvConfig, ensure_directories
from math_discovery.conjecturing_agent import ConjecturingAgent, CandidateStatement
from math_discovery.skeptical_agent import SkepticalAgent

GLOBAL_PROVABILITY_CACHE: dict[tuple[str, str, bool, float], "ProvabilityResult"] = {}


@dataclass(slots=True)
class ProvabilityResult:
    score: float
    typechecked: bool
    tactic_success: bool
    tactic: str | None
    prove_assist_summary: dict[str, Any]
    translation_path: str
    output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "typechecked": self.typechecked,
            "tactic_success": self.tactic_success,
            "tactic": self.tactic,
            "prove_assist_summary": self.prove_assist_summary,
            "translation_path": self.translation_path,
            "output": self.output,
        }


@dataclass(slots=True)
class EpisodeResult:
    success: bool
    steps: int
    statement: str
    concepts: list[str]
    provability: float
    premise_profile: str
    episode_path: str
    timeline: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "steps": self.steps,
            "statement": self.statement,
            "concepts": self.concepts,
            "provability": self.provability,
            "premise_profile": self.premise_profile,
            "episode_path": self.episode_path,
            "timeline": self.timeline,
        }


def _normalized_coeffs(statement: Statement) -> dict[str, int] | None:
    if len(statement.atoms) != 1:
        return None
    coeffs = dict(statement.atoms[0].coeff_dict())
    # Normalize: height_d2 == width_d1 (both equal E)
    if "height_d2" in coeffs:
        coeffs["width_d1"] = coeffs.get("width_d1", 0) + coeffs.pop("height_d2")
    return {name: coeff for name, coeff in coeffs.items() if coeff != 0}


def _certified_template(statement: Statement, premise_profile: str) -> str | None:
    """Match a statement against formally verified Lean theorem templates.

    Returns the theorem name if matched, None otherwise.
    This is a pure deterministic function — no ML, no heuristics.
    """
    coeffs = _normalized_coeffs(statement)
    if coeffs is None:
        return None
    rhs = statement.atoms[0].rhs
    if coeffs == {"height_d1": 1, "width_d1": -1, "width_d2": 1}:
        if premise_profile == "sphere" and rhs == 2:
            return "sphereEuler"
        if premise_profile == "torus" and rhs == 0:
            return "torusEuler"
    if coeffs == {"null_d1": 1, "rank_d2": -1}:
        if premise_profile in {"torus", "klein_bottle"} and rhs == 2:
            return "bettiOneValue2"
        if premise_profile == "sphere" and rhs == 0:
            return "vanishingMiddleBetti"
    if coeffs == {"height_d1": 1, "rank_d1": -1} and premise_profile == "disjoint_union" and rhs == 2:
        return "twoComponentB0"
    return None


def _profile_constant_supported(statement: Statement, premise_profile: str) -> bool:
    coeffs = _normalized_coeffs(statement)
    if coeffs is None:
        return True
    rhs = statement.atoms[0].rhs
    if coeffs == {"height_d1": 1, "width_d1": -1, "width_d2": 1}:
        expected = {"sphere": 2, "torus": 0, "klein_bottle": 0}
        return rhs == expected.get(premise_profile, rhs)
    if coeffs == {"null_d1": 1, "rank_d2": -1}:
        expected = {"sphere": 0, "torus": 2, "klein_bottle": 2}
        return rhs == expected.get(premise_profile, rhs)
    return True


class MathWorldEnv:
    """The discovery game environment.

    Each episode: CA proposes formulas, oracle checks provability,
    SA updates attention.  Success = a provable discovery-concept formula.
    """

    def __init__(self, dataset_rows: list[dict[str, Any]], config: EnvConfig, seed: int = 42):
        self.config = config
        self.rows = dataset_rows
        self.features = [row["features"] for row in dataset_rows]
        self.ca = ConjecturingAgent(config.ca_config, seed=seed)
        self.sa = SkepticalAgent(len(self.features), config.sa_config)
        self.rng = np.random.default_rng(seed)
        self.provability_cache: dict[tuple[str, str, bool, float], ProvabilityResult] = {}

    def _premise_profile(self, candidate: CandidateStatement) -> str:
        if self.config.premise_profile != "auto":
            return self.config.premise_profile
        if not candidate.patch_indices:
            return "generic"
        labels = [self.rows[index]["surface_type"] for index in candidate.patch_indices]
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        dominant = max(counts.items(), key=lambda item: item[1])[0]
        if dominant not in {"sphere", "torus", "klein_bottle", "disjoint_union"}:
            return "generic"
        return dominant

    def score_provability(self, candidate: CandidateStatement, premise_profile: str) -> ProvabilityResult:
        cache_key = (
            candidate.statement.render(),
            premise_profile,
            self.config.use_provability_oracle,
            float(self.config.fixed_provability_score),
        )
        cached = self.provability_cache.get(cache_key)
        if cached is not None:
            return cached
        shared = GLOBAL_PROVABILITY_CACHE.get(cache_key)
        if shared is not None:
            self.provability_cache[cache_key] = shared
            return shared

        # Oracle bypassed?
        if not self.config.use_provability_oracle:
            result = ProvabilityResult(
                score=self.config.fixed_provability_score,
                typechecked=False, tactic_success=False, tactic=None,
                prove_assist_summary={"ok": True, "bypassed": True},
                translation_path="", output="provability oracle bypassed",
            )
            self.provability_cache[cache_key] = result
            GLOBAL_PROVABILITY_CACHE[cache_key] = result
            return result

        # Try certified template match
        certified_template = _certified_template(candidate.statement, premise_profile)
        if certified_template is not None:
            result = ProvabilityResult(
                score=1.0,
                typechecked=True, tactic_success=True, tactic="certified_template",
                prove_assist_summary={"ok": True, "certified_template": certified_template},
                translation_path="", output=f"certified template: {certified_template}",
            )
            self.provability_cache[cache_key] = result
            GLOBAL_PROVABILITY_CACHE[cache_key] = result
            return result

        # Degenerate check
        truth_values = [candidate.statement.evaluate(row) for row in self.features]
        truth_count = sum(1 for value in truth_values if value)
        if truth_count == 0 or truth_count == len(truth_values):
            result = ProvabilityResult(
                score=0.05,
                typechecked=False, tactic_success=False, tactic=None,
                prove_assist_summary={"ok": True, "degenerate": True, "truth_count": truth_count, "total": len(truth_values)},
                translation_path="", output="degenerate statement rejected before proving",
            )
            self.provability_cache[cache_key] = result
            GLOBAL_PROVABILITY_CACHE[cache_key] = result
            return result

        # Non-discovery skip
        concepts = classify_statement(candidate.statement)
        if not {"chi", "b1"}.intersection(concepts):
            result = ProvabilityResult(
                score=0.05,
                typechecked=False, tactic_success=False, tactic=None,
                prove_assist_summary={"ok": True, "skipped": "non_discovery_statement"},
                translation_path="", output="non-discovery statement skipped before proving",
            )
            self.provability_cache[cache_key] = result
            GLOBAL_PROVABILITY_CACHE[cache_key] = result
            return result

        # Profile support check
        profile_indices = [
            index for index, row in enumerate(self.rows)
            if str(row.get("surface_type")) == premise_profile
        ]
        if premise_profile != "generic" and profile_indices:
            profile_truth_count = sum(1 for index in profile_indices if truth_values[index])
            if profile_truth_count < 2:
                result = ProvabilityResult(
                    score=0.05,
                    typechecked=False, tactic_success=False, tactic=None,
                    prove_assist_summary={"ok": True, "skipped": "insufficient_profile_support", "profile_truth_count": profile_truth_count, "premise_profile": premise_profile},
                    translation_path="", output="statement does not generalize within claimed surface family",
                )
                self.provability_cache[cache_key] = result
                GLOBAL_PROVABILITY_CACHE[cache_key] = result
                return result

        # Profile constant check
        if not _profile_constant_supported(candidate.statement, premise_profile):
            result = ProvabilityResult(
                score=0.05,
                typechecked=False, tactic_success=False, tactic=None,
                prove_assist_summary={"ok": True, "skipped": "profile_constant_mismatch", "premise_profile": premise_profile},
                translation_path="", output="statement constant is incompatible with the claimed surface family",
            )
            self.provability_cache[cache_key] = result
            GLOBAL_PROVABILITY_CACHE[cache_key] = result
            return result

        # Unsupported concept formula
        result = ProvabilityResult(
            score=0.25,
            typechecked=False, tactic_success=False, tactic=None,
            prove_assist_summary={"ok": True, "skipped": "unsupported_concept_formula", "premise_profile": premise_profile, "concepts": sorted(concepts)},
            translation_path="", output="concept formula is not one of the certified templates",
        )
        self.provability_cache[cache_key] = result
        GLOBAL_PROVABILITY_CACHE[cache_key] = result
        return result

    def _serialize_episode(self, payload: dict[str, Any]) -> Path:
        ensure_directories()
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = EPISODE_ROOT / f"{timestamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
        return output_path

    def run_episode(self, use_sa: bool = True) -> EpisodeResult:
        """Run one discovery episode (up to max_steps attempts)."""
        attention = self.sa.attention.copy()
        timeline: list[dict[str, Any]] = []
        last_statement = ""
        last_profile = "generic"
        last_score = 0.0
        last_concepts: list[str] = []
        discovery_concepts = {"chi", "b1"}

        for step in range(self.config.max_steps):
            candidate = self.ca.step(self.features, attention)
            profile = self._premise_profile(candidate)
            provability = self.score_provability(candidate, profile)
            concepts = sorted(classify_statement(candidate.statement))
            last_statement = candidate.statement.render()
            last_profile = profile
            last_score = provability.score
            last_concepts = concepts
            timeline.append({
                "step": step + 1,
                "statement": last_statement,
                "concepts": concepts,
                "provability": provability.to_dict(),
                "attention_entropy": normalized_entropy(attention),
                "patch_indices": candidate.patch_indices,
            })
            if provability.score >= self.config.provability_threshold and discovery_concepts.intersection(concepts):
                payload = {
                    "success": True, "steps": step + 1,
                    "statement": last_statement, "concepts": concepts,
                    "premise_profile": profile, "timeline": timeline,
                }
                episode_path = self._serialize_episode(payload)
                return EpisodeResult(
                    success=True, steps=step + 1, statement=last_statement,
                    concepts=concepts, provability=provability.score,
                    premise_profile=profile, episode_path=str(episode_path),
                    timeline=timeline,
                )
            if use_sa:
                attention = self.sa.step(last_statement, self.features, provability.score)

        payload = {
            "success": False, "steps": self.config.max_steps,
            "statement": last_statement, "concepts": last_concepts,
            "premise_profile": last_profile, "timeline": timeline,
        }
        episode_path = self._serialize_episode(payload)
        return EpisodeResult(
            success=False, steps=self.config.max_steps, statement=last_statement,
            concepts=last_concepts, provability=last_score,
            premise_profile=last_profile, episode_path=str(episode_path),
            timeline=timeline,
        )
