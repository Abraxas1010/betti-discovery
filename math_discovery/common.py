"""Core data structures, GF(2) linear algebra, and formula representation.

This module provides:
- GF(2) Gaussian elimination for computing matrix rank over the two-element field
- Betti number computation from boundary matrices of a chain complex
- SurfaceData: a compact representation of a triangulated surface with its
  incidence matrices stored in sparse column-ones format
- AtomicFormula / Statement: symbolic linear integer formulas over 8 features
  derived from boundary matrix dimensions and ranks
- Parser and classifier for formula expressions
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any

import numpy as np


# ── Feature names ────────────────────────────────────────────────────
# These correspond 1:1 to the 8 measurable quantities from a
# two-stage chain complex  C2 --d2--> C1 --d1--> C0  over GF(2):
#
#   height_d1 = #rows(d1) = dim(C0) = V  (vertex count)
#   width_d1  = #cols(d1) = dim(C1) = E  (edge count)
#   height_d2 = #rows(d2) = dim(C1) = E  (always equals width_d1)
#   width_d2  = #cols(d2) = dim(C2) = F  (face count)
#   rank_d1   = rank(d1) over GF(2)
#   rank_d2   = rank(d2) over GF(2)
#   null_d1   = nullity(d1) = E - rank_d1
#   null_d2   = nullity(d2) = F - rank_d2

FEATURE_NAMES: tuple[str, ...] = (
    "height_d1",
    "width_d1",
    "height_d2",
    "width_d2",
    "rank_d1",
    "rank_d2",
    "null_d1",
    "null_d2",
)

SURFACE_TYPES: tuple[str, ...] = (
    "sphere",
    "torus",
    "klein_bottle",
    "disjoint_union",
)


# ── Utilities ────────────────────────────────────────────────────────

def json_dumps(payload: Any) -> str:
    """Deterministic JSON serialization for reproducible artifacts."""
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    if values.size == 0:
        return values
    centered = values - float(np.max(values))
    exp = np.exp(centered)
    total = float(exp.sum())
    if total <= 0.0:
        return np.full_like(values, 1.0 / max(1, values.size), dtype=float)
    return exp / total


def normalized_entropy(weights: np.ndarray) -> float:
    """Shannon entropy normalized to [0, 1].  Used to measure attention spread."""
    if weights.size == 0:
        return 0.0
    safe = np.clip(weights.astype(float), 1e-12, 1.0)
    entropy = float(-(safe * np.log(safe)).sum())
    max_entropy = math.log(weights.size) if weights.size > 1 else 1.0
    if max_entropy <= 0.0:
        return 0.0
    return entropy / max_entropy


# ── GF(2) linear algebra ────────────────────────────────────────────

def block_diag(matrices: list[np.ndarray]) -> np.ndarray:
    """Block-diagonal composition of GF(2) matrices."""
    if not matrices:
        return np.zeros((0, 0), dtype=np.uint8)
    row_total = sum(int(mat.shape[0]) for mat in matrices)
    col_total = sum(int(mat.shape[1]) for mat in matrices)
    out = np.zeros((row_total, col_total), dtype=np.uint8)
    row_cursor = 0
    col_cursor = 0
    for matrix in matrices:
        rows, cols = matrix.shape
        out[row_cursor:row_cursor + rows, col_cursor:col_cursor + cols] = matrix
        row_cursor += rows
        col_cursor += cols
    return out


def dense_to_col_ones(matrix: np.ndarray) -> list[list[int]]:
    """Convert a dense GF(2) matrix to sparse column-ones representation.

    Each column becomes a list of row indices where the entry is 1 (mod 2).
    This is how SurfaceData stores its boundary matrices compactly.
    """
    rows, cols = matrix.shape
    col_ones: list[list[int]] = []
    for col in range(cols):
        ones = [row for row in range(rows) if int(matrix[row, col]) % 2 == 1]
        col_ones.append(ones)
    return col_ones


def col_ones_to_dense(rows: int, cols: int, col_ones: list[list[int]]) -> np.ndarray:
    """Convert sparse column-ones back to a dense GF(2) matrix."""
    dense = np.zeros((rows, cols), dtype=np.uint8)
    if len(col_ones) != cols:
        raise ValueError(f"expected {cols} columns, got {len(col_ones)}")
    for col, ones in enumerate(col_ones):
        for row in ones:
            if row < 0 or row >= rows:
                raise ValueError(f"row {row} out of bounds for {rows} rows")
            dense[row, col] = 1
    return dense


def f2_rank(matrix: np.ndarray) -> int:
    """Compute the rank of a matrix over GF(2) via Gaussian elimination.

    This is the core linear-algebra primitive.  All Betti number
    computations reduce to this: rank over the two-element field.
    """
    work = (matrix.astype(np.uint8) % 2).copy()
    rows, cols = work.shape
    pivot_row = 0
    rank = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        candidates = np.where(work[pivot_row:, col] == 1)[0]
        if candidates.size == 0:
            continue
        pivot = pivot_row + int(candidates[0])
        if pivot != pivot_row:
            work[[pivot_row, pivot]] = work[[pivot, pivot_row]]
        for row in range(rows):
            if row != pivot_row and work[row, col] == 1:
                work[row] ^= work[pivot_row]
        pivot_row += 1
        rank += 1
    return rank


def compute_betti(d1: np.ndarray, d2: np.ndarray) -> tuple[int, int, int]:
    """Compute Betti numbers (b0, b1, b2) from boundary matrices d1, d2.

    Given the chain complex  C2 --d2--> C1 --d1--> C0  over GF(2):
      b0 = dim(C0) - rank(d1)           = #connected components
      b1 = nullity(d1) - rank(d2)       = #independent 1-cycles
      b2 = nullity(d2)                  = #independent 2-cycles (cavities)

    For a closed orientable surface of genus g:  b0=1, b1=2g, b2=1.
    """
    v_count, e_count = d1.shape
    _, f_count = d2.shape
    rank_d1 = f2_rank(d1)
    rank_d2 = f2_rank(d2)
    null_d1 = e_count - rank_d1
    null_d2 = f_count - rank_d2
    b0 = v_count - rank_d1
    b1 = null_d1 - rank_d2
    b2 = null_d2
    return int(b0), int(b1), int(b2)


# ── Surface data ─────────────────────────────────────────────────────

@dataclass(slots=True)
class SurfaceData:
    """A triangulated surface stored as sparse boundary matrices.

    Attributes:
        d1_cols: Boundary map d1 (vertices x edges) in column-ones format.
        d2_cols: Boundary map d2 (edges x faces) in column-ones format.
        V, E, F: Vertex, edge, face counts.
        b0, b1, b2: Betti numbers.
        chi: Euler characteristic = V - E + F.
        surface_type: One of "sphere", "torus", "klein_bottle", "disjoint_union".
        metadata: Arbitrary extra data (e.g. grid dimensions, subdivisions).
    """
    d1_cols: list[list[int]]
    d2_cols: list[list[int]]
    V: int
    E: int
    F: int
    b0: int
    b1: int
    b2: int
    chi: int
    surface_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def dense_d1(self) -> np.ndarray:
        """Reconstruct dense boundary matrix d1 (V x E)."""
        return col_ones_to_dense(self.V, self.E, self.d1_cols)

    def dense_d2(self) -> np.ndarray:
        """Reconstruct dense boundary matrix d2 (E x F)."""
        return col_ones_to_dense(self.E, self.F, self.d2_cols)

    def features(self) -> dict[str, float]:
        """Extract the 8 chain-complex features for this surface."""
        rank_d1 = f2_rank(self.dense_d1())
        rank_d2 = f2_rank(self.dense_d2())
        return {
            "height_d1": float(self.V),
            "width_d1": float(self.E),
            "height_d2": float(self.E),
            "width_d2": float(self.F),
            "rank_d1": float(rank_d1),
            "rank_d2": float(rank_d2),
            "null_d1": float(self.E - rank_d1),
            "null_d2": float(self.F - rank_d2),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "d1_cols": self.d1_cols,
            "d2_cols": self.d2_cols,
            "V": self.V,
            "E": self.E,
            "F": self.F,
            "b0": self.b0,
            "b1": self.b1,
            "b2": self.b2,
            "chi": self.chi,
            "surface_type": self.surface_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SurfaceData":
        return cls(
            d1_cols=[[int(v) for v in col] for col in payload["d1_cols"]],
            d2_cols=[[int(v) for v in col] for col in payload["d2_cols"]],
            V=int(payload["V"]),
            E=int(payload["E"]),
            F=int(payload["F"]),
            b0=int(payload["b0"]),
            b1=int(payload["b1"]),
            b2=int(payload["b2"]),
            chi=int(payload["chi"]),
            surface_type=str(payload["surface_type"]),
            metadata=dict(payload.get("metadata") or {}),
        )


# ── Symbolic formula representation ─────────────────────────────────

@dataclass(frozen=True, slots=True)
class AtomicFormula:
    """A single linear integer equation over the 8 features.

    Example: height_d1 - width_d1 + width_d2 = 2  (Euler characteristic of a sphere)
    """
    coefficients: tuple[tuple[str, int], ...]
    rhs: int

    def coeff_dict(self) -> dict[str, int]:
        return {name: coeff for name, coeff in self.coefficients}

    def render(self) -> str:
        parts: list[str] = []
        for index, feature in enumerate(FEATURE_NAMES):
            coeff = self.coeff_dict().get(feature)
            if coeff is None or coeff == 0:
                continue
            magnitude = abs(coeff)
            sign = "-" if coeff < 0 else "+"
            body = feature if magnitude == 1 else f"{magnitude} * {feature}"
            if not parts:
                parts.append(body if coeff > 0 else f"- {body}")
            else:
                parts.append(f"{sign} {body}")
        lhs = " ".join(parts) if parts else "0"
        return f"{lhs} = {self.rhs}"

    def complexity(self) -> int:
        return sum(abs(coeff) for _, coeff in self.coefficients) + 1

    def used_features(self) -> set[str]:
        return {name for name, coeff in self.coefficients if coeff != 0}

    def evaluate(self, features: dict[str, float]) -> float:
        total = 0.0
        for name, coeff in self.coefficients:
            total += float(coeff) * float(features[name])
        return total

    def residual(self, features: dict[str, float]) -> float:
        return self.evaluate(features) - float(self.rhs)

    def matches_coefficients(self, target: dict[str, int]) -> bool:
        return self.coeff_dict() == target


@dataclass(frozen=True, slots=True)
class Statement:
    """A conjunction of atomic formulas (e.g. "V - E + F = 2 and null_d1 - rank_d2 = 0")."""
    atoms: tuple[AtomicFormula, ...]

    def render(self) -> str:
        return " and ".join(atom.render() for atom in self.atoms)

    def complexity(self) -> int:
        return sum(atom.complexity() for atom in self.atoms)

    def used_features(self) -> set[str]:
        features: set[str] = set()
        for atom in self.atoms:
            features.update(atom.used_features())
        return features

    def evaluate(self, features: dict[str, float], tolerance: float = 1e-6) -> bool:
        return all(abs(atom.residual(features)) <= tolerance for atom in self.atoms)


# ── Formula parsing ──────────────────────────────────────────────────

TOKEN_RE = re.compile(r"\s*(and|=|\+|-|\*|\(|\)|\d+|[A-Za-z_][A-Za-z0-9_]*)")


def _tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        raise ValueError(f"cannot parse expression: {text!r}")
    return tokens


def _parse_linear_side(tokens: list[str]) -> dict[str, int]:
    coeffs: dict[str, int] = {}
    sign = 1
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "+":
            sign = 1
            index += 1
            continue
        if token == "-":
            sign = -1
            index += 1
            continue
        coeff = sign
        if token.isdigit():
            coeff = sign * int(token)
            if index + 2 < len(tokens) and tokens[index + 1] == "*" and re.match(r"[A-Za-z_]", tokens[index + 2]):
                feature = tokens[index + 2]
                coeffs[feature] = coeffs.get(feature, 0) + coeff
                index += 3
                sign = 1
                continue
            raise ValueError("constant-only terms are not supported on the left-hand side")
        feature = token
        coeffs[feature] = coeffs.get(feature, 0) + coeff
        index += 1
        sign = 1
    return {name: value for name, value in coeffs.items() if value != 0}


def parse_atomic_formula(text: str) -> AtomicFormula:
    """Parse a string like "height_d1 - width_d1 + width_d2 = 2" into an AtomicFormula."""
    if "=" not in text:
        raise ValueError(f"formula must contain '=': {text!r}")
    left_text, right_text = [part.strip() for part in text.split("=", 1)]
    tokens = _tokenize(left_text)
    coeffs = _parse_linear_side(tokens)
    rhs = int(right_text)
    ordered = tuple((name, coeffs[name]) for name in FEATURE_NAMES if name in coeffs)
    return AtomicFormula(coefficients=ordered, rhs=rhs)


def parse_statement(text: str) -> Statement:
    """Parse a conjunction like "V - E + F = 2 and null_d1 - rank_d2 = 0"."""
    atoms = tuple(parse_atomic_formula(part.strip()) for part in text.split(" and ") if part.strip())
    if not atoms:
        raise ValueError(f"statement must contain at least one atomic formula: {text!r}")
    return Statement(atoms=atoms)


def classify_statement(statement: Statement) -> set[str]:
    """Identify which mathematical concepts a statement captures.

    Returns a set of concept names:
    - "chi": Euler characteristic (V - E + F = constant)
    - "b1": First Betti number (nullity(d1) - rank(d2) = constant)
    - "b0": Zeroth Betti number (V - rank(d1) = constant)
    """
    def normalize(coeffs: dict[str, int]) -> dict[str, int]:
        normalized = dict(coeffs)
        if "height_d2" in normalized:
            normalized["width_d1"] = normalized.get("width_d1", 0) + normalized.pop("height_d2")
        return {name: coeff for name, coeff in normalized.items() if coeff != 0}

    concepts: set[str] = set()
    for atom in statement.atoms:
        coeffs = normalize(atom.coeff_dict())
        if coeffs == {"height_d1": 1, "width_d1": -1, "width_d2": 1}:
            concepts.add("chi")
        if coeffs == {"null_d1": 1, "rank_d2": -1}:
            concepts.add("b1")
        if coeffs == {"height_d1": 1, "rank_d1": -1}:
            concepts.add("b0")
    return concepts


# ── Candidate enumeration ────────────────────────────────────────────

def choose_combinations(items: tuple[str, ...], max_terms: int) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for size in range(1, max_terms + 1):
        out.extend(combinations(items, size))
    return out


def iter_linear_forms(
    feature_names: tuple[str, ...],
    max_terms: int,
    coefficient_values: tuple[int, ...],
) -> list[tuple[tuple[str, int], ...]]:
    """Enumerate all candidate linear integer forms up to max_terms features.

    With 8 features, max_terms=4, coefficients=(-2,-1,1,2), this produces
    ~20,288 candidate forms — the search space the conjecturing agent explores.
    """
    forms: list[tuple[tuple[str, int], ...]] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    for combo in choose_combinations(feature_names, max_terms):
        for coeffs in product(coefficient_values, repeat=len(combo)):
            if all(coeff == 0 for coeff in coeffs):
                continue
            candidate = tuple((name, int(coeff)) for name, coeff in zip(combo, coeffs) if coeff != 0)
            if not candidate:
                continue
            gcd = 0
            for _, coeff in candidate:
                gcd = math.gcd(gcd, abs(coeff))
            if gcd > 1:
                candidate = tuple((name, coeff // gcd) for name, coeff in candidate)
            ordered = tuple((name, coeff) for name in FEATURE_NAMES if any(name == existing for existing, _ in candidate) for existing, coeff in candidate if existing == name)
            if ordered in seen:
                continue
            seen.add(ordered)
            forms.append(ordered)
    return forms
