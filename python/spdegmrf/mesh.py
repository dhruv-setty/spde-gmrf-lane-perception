"""Structured triangular meshes and point location."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    """A 2-D triangular mesh.

    nodes: (n, 2) float array of node coordinates.
    tris:  (m, 3) int array of node indices, counter-clockwise.
    """

    nodes: np.ndarray
    tris: np.ndarray

    @property
    def n(self) -> int:
        return int(self.nodes.shape[0])

    def relabel(self, perm: np.ndarray) -> "Mesh":
        """Return the same mesh with node k of the result equal to node perm[k] of this mesh."""
        perm = np.asarray(perm, dtype=np.int64)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(perm.size)
        return Mesh(self.nodes[perm].copy(), inv[self.tris])


def rect_mesh(nx: int, ny: int, lx: float | None = None, ly: float | None = None) -> Mesh:
    """Regular nx by ny node lattice on [0, lx] x [0, ly], each cell split into two triangles.

    With unit spacing (the default) the P1 stiffness matrix equals the five-point lattice
    Laplacian, because the couplings across the cell diagonals cancel.
    """
    lx = float(nx - 1) if lx is None else float(lx)
    ly = float(ny - 1) if ly is None else float(ly)
    xs = np.linspace(0.0, lx, nx)
    ys = np.linspace(0.0, ly, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = a + 1
            c = a + nx
            d = c + 1
            tris.append((a, b, d))
            tris.append((a, d, c))
    return Mesh(nodes, np.asarray(tris, dtype=np.int64))


def scrambled(mesh: Mesh, seed: int = 0) -> Mesh:
    """Relabel nodes in a random order, as an unstructured mesh generator may produce."""
    rng = np.random.default_rng(seed)
    return mesh.relabel(rng.permutation(mesh.n))


def barycentric(mesh: Mesh, pts: np.ndarray, eps: float = 1e-12):
    """Locate each point in a triangle and return (triangle index, barycentric weights).

    Raises ValueError for a point outside the mesh.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    v0 = mesh.nodes[mesh.tris[:, 0]]
    v1 = mesh.nodes[mesh.tris[:, 1]]
    v2 = mesh.nodes[mesh.tris[:, 2]]
    det = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    tri_idx = np.empty(len(pts), dtype=np.int64)
    weights = np.empty((len(pts), 3))
    for s, p in enumerate(pts):
        l1 = ((p[0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (p[1] - v0[:, 1])) / det
        l2 = ((v1[:, 0] - v0[:, 0]) * (p[1] - v0[:, 1]) - (p[0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])) / det
        l0 = 1.0 - l1 - l2
        inside = np.flatnonzero((l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps))
        if inside.size == 0:
            raise ValueError(f"point {s} at {p.tolist()} lies outside the mesh")
        t = int(inside[0])
        w = np.array([l0[t], l1[t], l2[t]])
        w = np.clip(w, 0.0, None)
        tri_idx[s] = t
        weights[s] = w / w.sum()
    return tri_idx, weights
