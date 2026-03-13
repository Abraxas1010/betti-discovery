"""Deterministic combinatorial generators for triangulated surfaces.

Generates four surface types:
- **Sphere**: octahedral base with configurable subdivision
- **Torus**: rectangular grid with periodic identification (both directions)
- **Klein bottle**: rectangular grid with orientation-reversing identification
- **Disjoint union**: block-diagonal composition of two surfaces

All surfaces are triangulated (every face is a triangle), and their
boundary matrices are computed over GF(2) to yield exact Betti numbers.

Example::

    from math_discovery.surface_data_gen import generate_sphere, generate_torus

    sphere = generate_sphere(subdivisions=1)
    print(f"Sphere: V={sphere.V}, E={sphere.E}, F={sphere.F}, chi={sphere.chi}")
    # Sphere: V=18, E=48, F=32, chi=2

    torus = generate_torus(width=4, height=4)
    print(f"Torus: b0={torus.b0}, b1={torus.b1}, b2={torus.b2}")
    # Torus: b0=1, b1=2, b2=1
"""

from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np

from math_discovery.common import (
    SurfaceData,
    block_diag,
    compute_betti,
    dense_to_col_ones,
    f2_rank,
    json_dumps,
)
from math_discovery.config import DatasetConfig, SURFACE_ROOT, ensure_directories


def build_surface_from_faces(
    faces: list[tuple[int, int, int]],
    surface_type: str,
    metadata: dict[str, Any] | None = None,
) -> SurfaceData:
    """Build a SurfaceData from a list of triangular faces.

    Each face is a triple of vertex indices.  Edges and boundary matrices
    are computed automatically.
    """
    metadata = dict(metadata or {})
    vertex_count = max(max(face) for face in faces) + 1 if faces else 0
    edge_index: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    for face in faces:
        tri_edges = (
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[0], face[2]))),
        )
        for edge in tri_edges:
            if edge not in edge_index:
                edge_index[edge] = len(edges)
                edges.append(edge)
    edge_count = len(edges)
    face_count = len(faces)

    d1 = np.zeros((vertex_count, edge_count), dtype=np.uint8)
    for edge_id, (left, right) in enumerate(edges):
        d1[left, edge_id] = 1
        d1[right, edge_id] = 1

    d2 = np.zeros((edge_count, face_count), dtype=np.uint8)
    for face_id, face in enumerate(faces):
        tri_edges = (
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[0], face[2]))),
        )
        for edge in tri_edges:
            d2[edge_index[edge], face_id] = 1

    b0, b1, b2 = compute_betti(d1, d2)
    return SurfaceData(
        d1_cols=dense_to_col_ones(d1),
        d2_cols=dense_to_col_ones(d2),
        V=vertex_count,
        E=edge_count,
        F=face_count,
        b0=b0,
        b1=b1,
        b2=b2,
        chi=vertex_count - edge_count + face_count,
        surface_type=surface_type,
        metadata=metadata,
    )


def _subdivide_faces(faces: list[tuple[int, int, int]], next_vertex: int) -> tuple[list[tuple[int, int, int]], int]:
    """Loop subdivision: each triangle becomes 4 triangles."""
    midpoint_map: dict[tuple[int, int], int] = {}
    refined: list[tuple[int, int, int]] = []

    def midpoint(a: int, b: int) -> int:
        nonlocal next_vertex
        key = tuple(sorted((a, b)))
        if key in midpoint_map:
            return midpoint_map[key]
        midpoint_map[key] = next_vertex
        next_vertex += 1
        return midpoint_map[key]

    for a, b, c in faces:
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ac = midpoint(a, c)
        refined.extend(
            [
                (a, ab, ac),
                (ab, b, bc),
                (ac, bc, c),
                (ab, bc, ac),
            ]
        )
    return refined, next_vertex


def generate_sphere(subdivisions: int = 1) -> SurfaceData:
    """Generate a triangulated sphere by subdividing an octahedron.

    Args:
        subdivisions: Number of loop-subdivision passes.  0 = raw octahedron (6V, 12E, 8F).

    Returns:
        SurfaceData with b0=1, b1=0, b2=1, chi=2.
    """
    faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),
        (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4),
    ]
    next_vertex = 6
    for _ in range(max(0, subdivisions)):
        faces, next_vertex = _subdivide_faces(faces, next_vertex)
    return build_surface_from_faces(
        faces, "sphere", metadata={"subdivisions": int(subdivisions)},
    )


def _torus_vertex(i: int, j: int, width: int, height: int) -> int:
    return (j % height) * width + (i % width)


def generate_torus(width: int = 4, height: int = 4) -> SurfaceData:
    """Generate a triangulated torus from a rectangular grid with periodic identification.

    Args:
        width: Grid width (number of quads along one axis).
        height: Grid height (number of quads along the other axis).

    Returns:
        SurfaceData with b0=1, b1=2, b2=1, chi=0.
    """
    faces: list[tuple[int, int, int]] = []
    for j in range(height):
        for i in range(width):
            a = _torus_vertex(i, j, width, height)
            b = _torus_vertex(i + 1, j, width, height)
            c = _torus_vertex(i, j + 1, width, height)
            d = _torus_vertex(i + 1, j + 1, width, height)
            faces.append((a, b, c))
            faces.append((b, d, c))
    return build_surface_from_faces(
        faces, "torus", metadata={"width": int(width), "height": int(height)},
    )


def _klein_vertex(i: int, j: int, width: int, height: int) -> int:
    wraps = i // width
    i_mod = i % width
    j_mod = j % height
    if wraps % 2 == 1:
        j_mod = (-j_mod) % height
    return j_mod * width + i_mod


def generate_klein_bottle(width: int = 4, height: int = 4) -> SurfaceData:
    """Generate a triangulated Klein bottle with orientation-reversing identification.

    Args:
        width: Grid width.
        height: Grid height.

    Returns:
        SurfaceData with b0=1, b1=1, b2=0, chi=0.
    """
    faces: list[tuple[int, int, int]] = []
    for j in range(height):
        for i in range(width):
            a = _klein_vertex(i, j, width, height)
            b = _klein_vertex(i + 1, j, width, height)
            c = _klein_vertex(i, j + 1, width, height)
            d = _klein_vertex(i + 1, j + 1, width, height)
            faces.append((a, b, c))
            faces.append((b, d, c))
    return build_surface_from_faces(
        faces, "klein_bottle", metadata={"width": int(width), "height": int(height)},
    )


def generate_disjoint_union(surfaces: list[SurfaceData]) -> SurfaceData:
    """Block-diagonal disjoint union of multiple surfaces."""
    d1 = block_diag([surface.dense_d1() for surface in surfaces])
    d2 = block_diag([surface.dense_d2() for surface in surfaces])
    b0, b1, b2 = compute_betti(d1, d2)
    return SurfaceData(
        d1_cols=dense_to_col_ones(d1),
        d2_cols=dense_to_col_ones(d2),
        V=int(d1.shape[0]),
        E=int(d1.shape[1]),
        F=int(d2.shape[1]),
        b0=b0,
        b1=b1,
        b2=b2,
        chi=int(d1.shape[0] - d1.shape[1] + d2.shape[1]),
        surface_type="disjoint_union",
        metadata={"components": [surface.surface_type for surface in surfaces]},
    )


def _sample_connected(surface_type: str, config: DatasetConfig, rng: np.random.Generator) -> SurfaceData:
    if surface_type == "sphere":
        subdivisions = int(rng.choice(config.sphere_subdivisions))
        return generate_sphere(subdivisions=subdivisions)
    if surface_type == "torus":
        width, height = config.torus_grids[int(rng.integers(0, len(config.torus_grids)))]
        return generate_torus(width=width, height=height)
    if surface_type == "klein_bottle":
        width, height = config.klein_grids[int(rng.integers(0, len(config.klein_grids)))]
        return generate_klein_bottle(width=width, height=height)
    raise ValueError(f"unsupported connected surface type: {surface_type}")


def generate_dataset(config: DatasetConfig) -> list[SurfaceData]:
    """Generate a batch of surfaces according to a dataset configuration.

    Datasets:
    - D0: spheres + tori (simplest, orientable only)
    - D1: + Klein bottles (adds non-orientable)
    - D2: + disjoint unions (adds disconnected)
    - D3: all four types
    """
    rng = np.random.default_rng(config.seed)
    if config.dataset == "D0":
        palette = ("sphere", "torus")
    elif config.dataset == "D1":
        palette = ("sphere", "torus", "klein_bottle")
    elif config.dataset == "D2":
        palette = ("sphere", "torus", "disjoint_union")
    elif config.dataset == "D3":
        palette = ("sphere", "torus", "klein_bottle", "disjoint_union")
    else:
        raise ValueError(f"unknown dataset: {config.dataset}")

    surfaces: list[SurfaceData] = []
    palette_cycle = cycle(palette)
    for _ in range(config.count):
        surface_type = next(palette_cycle)
        if surface_type == "disjoint_union":
            left = _sample_connected(str(rng.choice(("sphere", "torus", "klein_bottle"))), config, rng)
            right = _sample_connected(str(rng.choice(("sphere", "torus", "klein_bottle"))), config, rng)
            surface = generate_disjoint_union([left, right])
        else:
            surface = _sample_connected(surface_type, config, rng)
        surfaces.append(surface)
    return surfaces


def save_dataset(dataset: str, surfaces: list[SurfaceData], output_path: Path) -> dict[str, Any]:
    """Serialize a surface dataset to JSON."""
    ensure_directories()
    payload = {
        "dataset": dataset,
        "count": len(surfaces),
        "surfaces": [surface.to_dict() for surface in surfaces],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate triangulated surface datasets.")
    parser.add_argument("--dataset", default="D0", choices=["D0", "D1", "D2", "D3"])
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=SURFACE_ROOT / "D0.json")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON to stdout.")
    args = parser.parse_args()

    config = DatasetConfig(dataset=args.dataset, count=args.count, seed=args.seed)
    surfaces = generate_dataset(config)
    save_dataset(args.dataset, surfaces, args.output)

    if args.json:
        summaries = [{"type": s.surface_type, "V": s.V, "E": s.E, "F": s.F, "chi": s.chi} for s in surfaces[:5]]
        print(json_dumps({"ok": True, "dataset": args.dataset, "count": len(surfaces), "sample": summaries}))
    else:
        print(f"Generated {len(surfaces)} surfaces for {args.dataset} -> {args.output}")
        for s in surfaces[:5]:
            print(f"  {s.surface_type}: V={s.V} E={s.E} F={s.F} chi={s.chi} b=({s.b0},{s.b1},{s.b2})")


if __name__ == "__main__":
    main()
