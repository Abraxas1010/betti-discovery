"""Extract rank/nullity features from surface boundary matrices.

Given a triangulated surface, this module computes 8 numeric features
from its two boundary matrices (d1, d2):

    ┌─────────────┬─────────────────────────────────┐
    │ Feature     │ Meaning                         │
    ├─────────────┼─────────────────────────────────┤
    │ height_d1   │ #rows of d1 = V (vertices)      │
    │ width_d1    │ #cols of d1 = E (edges)          │
    │ height_d2   │ #rows of d2 = E (edges)          │
    │ width_d2    │ #cols of d2 = F (faces)           │
    │ rank_d1     │ GF(2) rank of d1                 │
    │ rank_d2     │ GF(2) rank of d2                 │
    │ null_d1     │ E - rank_d1 (nullity of d1)      │
    │ null_d2     │ F - rank_d2 (nullity of d2)      │
    └─────────────┴─────────────────────────────────┘

These features are the observable input to the conjecturing agent.
Mathematical relationships between them encode topological invariants:
  - V - E + F = chi  (Euler characteristic)
  - null_d1 - rank_d2 = b1  (first Betti number)
  - V - rank_d1 = b0  (zeroth Betti number)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from math_discovery.common import SurfaceData, f2_rank, json_dumps


def extract_features(surface: SurfaceData) -> dict[str, float]:
    """Extract the 8 chain-complex features from a single surface."""
    d1 = surface.dense_d1()
    d2 = surface.dense_d2()
    rank_d1 = f2_rank(d1)
    rank_d2 = f2_rank(d2)
    return {
        "height_d1": float(surface.V),
        "width_d1": float(surface.E),
        "height_d2": float(surface.E),
        "width_d2": float(surface.F),
        "rank_d1": float(rank_d1),
        "rank_d2": float(rank_d2),
        "null_d1": float(surface.E - rank_d1),
        "null_d2": float(surface.F - rank_d2),
    }


def extract_dataset_features(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract features for an entire dataset of surfaces."""
    features = []
    for surface_payload in payload["surfaces"]:
        surface = SurfaceData.from_dict(surface_payload)
        features.append(
            {
                "surface_type": surface.surface_type,
                "features": extract_features(surface),
                "ground_truth": {
                    "b0": surface.b0,
                    "b1": surface.b1,
                    "b2": surface.b2,
                    "chi": surface.chi,
                },
                "metadata": surface.metadata,
            }
        )
    return {
        "dataset": payload["dataset"],
        "count": len(features),
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract rank/nullity features from surface incidence matrices.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON to stdout.")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    features = extract_dataset_features(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_dumps(features) + "\n", encoding="utf-8")
    report = {
        "ok": True,
        "dataset": features["dataset"],
        "count": features["count"],
        "output": str(args.output) if args.output else None,
        "sample": features["features"][: min(3, len(features["features"]))],
    }
    if args.json:
        print(json_dumps(report))
    else:
        print(f"Extracted {features['count']} feature rows from {args.input}")
        for row in report["sample"]:
            print(row)


if __name__ == "__main__":
    main()
