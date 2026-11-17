"""Operation counts (multiply-accumulates) for each stage, derived from the matrices themselves.

Every host row is computed from the sparsity pattern of the matrix it operates on, so each
count links back to a matrix: Q(theta), the posterior precision Q + P^T P / sigma^2, its
Cholesky factor L, the projector P, and the int8 transition matrix A^T on the device.

Counts, for a lower Cholesky factor whose column j has eta_j entries below the diagonal:
  factorisation      sum_j eta_j (eta_j + 1) / 2 MACs, plus n square roots and sum_j eta_j divisions
  log|Q|             n logarithms of the factor diagonal, no MACs
  two triangular     2 * (nnz(L) - n) MACs, plus 2n divisions
  sparse product     nnz(M) MACs for M x

Usage (from python/):  python tools/mac_table.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spdegmrf.cholesky import SymbolicCholesky  # noqa: E402
from spdegmrf.estimate import sample_field  # noqa: E402
from spdegmrf.export import route_sites, true_theta  # noqa: E402
from spdegmrf.fem import assemble  # noqa: E402
from spdegmrf.hmm import HmmParams  # noqa: E402
from spdegmrf.lattice import lattice_precision  # noqa: E402
from spdegmrf.mesh import rect_mesh  # noqa: E402
from spdegmrf.projector import projector  # noqa: E402
from spdegmrf.spde import SpdePrecision  # noqa: E402

K = 4  # regimes
D = 2  # observation channels


def factor_counts(M: sp.spmatrix, order="rcm") -> dict:
    sym = SymbolicCholesky(M, order)
    eta = np.array([c.size - 1 for c in sym.cols], dtype=np.int64)
    return {
        "n": int(sym.n),
        "nnz_lower_M": int(sp.tril(sp.csc_matrix(M)).nnz),
        "nnz_L": int(sym.nnz),
        "factor_macs": int(np.sum(eta * (eta + 1) // 2)),
        "solve_macs": int(2 * (sym.nnz - sym.n)),
    }


def device_counts() -> dict:
    """Per-sample counts for hmm_step in firmware/src/hmm_filter.c, read from its loops."""
    fc = K * K                      # hmm_fc_s8: (in + offset) * w for each of K outputs, K inputs
    emission = K * D * 2            # (y - mean) * inv_sigma, then z * z, per regime and channel
    update = K + K                  # pk * exp(-delta), then 255 * u
    return {
        "fc_macs": fc,
        "emission_macs": emission,
        "update_macs": update,
        "total_macs": fc + emission + update,
        "requantisations": K,       # one per FC output
        "lut_lookups": K,           # exp2 table, 256 x uint16
        "divisions": K,             # normalisation to sum 255
        "lifted_offset_reads": 1,   # one int16 P x_hat value for the current site
    }


def build(nodes: int = 16, lx: float = 1.5, m: int = 200, sigma: float = 15.0, n_sites: int = 64,
          seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    mesh = rect_mesh(nodes, nodes, lx, lx)
    Cd, G = assemble(mesh)
    spde = SpdePrecision(Cd, G)
    kappa, tau = true_theta()
    Q = spde.Q(kappa, tau).tocsc()
    sample_field(spde, kappa, tau, rng, size=8)  # same random stream as export.run_pipeline
    sensors = rng.uniform(0.0, lx, size=(m, 2))
    P = projector(mesh, sensors).tocsr()
    Qp = (Q + (P.T @ P) / sigma ** 2).tocsc()
    P_dev = projector(mesh, route_sites(n_sites, lx)).tocsr()

    fq = factor_counts(Q)
    fqp = factor_counts(Qp)
    lat20 = lattice_precision(20).tocsc()
    perm = np.random.default_rng(0).permutation(lat20.shape[0])
    # As in lattice.fill_in: RCM is applied to the scrambled matrix.
    f20 = factor_counts(lat20[perm][:, perm], "rcm")
    f20s = factor_counts(lat20, perm)

    host = [
        {"stage": "Assemble Q(theta) values", "matrix": "Q (16x16 mesh)", "n": fq["n"],
         "nnz": int(Q.nnz), "macs": int(3 * Q.nnz), "per": "theta (multiplies)",
         "note": "tau^2 (kappa^4 c + 2 kappa^2 g + k) per stored entry, 3 multiplies (spde.values)"},
        {"stage": "Sparse product Q x", "matrix": "Q (16x16 mesh)", "n": fq["n"], "nnz": int(Q.nnz),
         "macs": int(Q.nnz), "per": "product", "note": "cuSPARSE SpMV or SpMM"},
        {"stage": "Cholesky of Q", "matrix": "L of Q, RCM", "n": fq["n"], "nnz": fq["nnz_L"],
         "macs": fq["factor_macs"], "per": "theta", "note": "cuDSS refactorisation; log|Q| from diag(L)"},
        {"stage": "Cholesky of Q + P^T P / sigma^2", "matrix": "L of posterior precision, RCM",
         "n": fqp["n"], "nnz": fqp["nnz_L"], "macs": fqp["factor_macs"], "per": "theta",
         "note": "cuDSS refactorisation"},
        {"stage": "Posterior mean solve", "matrix": "L of posterior precision", "n": fqp["n"],
         "nnz": fqp["nnz_L"], "macs": fqp["solve_macs"], "per": "right-hand side",
         "note": "forward and back substitution"},
        {"stage": "Sensor product P^T y", "matrix": "P (200 sensors)", "n": int(P.shape[0]),
         "nnz": int(P.nnz), "macs": int(P.nnz), "per": "right-hand side",
         "note": "three barycentric weights per row"},
        {"stage": "Lift x_hat to device sites", "matrix": "P_dev (64 sites)", "n": int(P_dev.shape[0]),
         "nnz": int(P_dev.nnz), "macs": int(P_dev.nnz), "per": "re-estimation",
         "note": "offsets P x_hat, quantised to int16 for the device"},
        {"stage": "Cholesky, 20x20 lattice, RCM", "matrix": "L of Q (Figure 3)", "n": f20["n"],
         "nnz": f20["nnz_L"], "macs": f20["factor_macs"], "per": "theta", "note": "fill-in experiment"},
        {"stage": "Cholesky, 20x20 lattice, scrambled (seed 0)", "matrix": "L of Q (Figure 3)",
         "n": f20s["n"], "nnz": f20s["nnz_L"], "macs": f20s["factor_macs"], "per": "theta",
         "note": "same matrix, random node order"},
    ]
    params = HmmParams.default(P_dev @ np.zeros(Q.shape[0]))
    dev = device_counts()
    device = [
        {"stage": "Prediction A^T alpha_(t-1)", "matrix": f"A^T ({K}x{K}, int8)",
         "nnz": int(np.count_nonzero(params.A)), "macs": dev["fc_macs"],
         "note": "arm_fully_connected_s8 or portable C99 kernel; 4 requantisations"},
        {"stage": "Emission b_k(y_t)", "matrix": f"mu, 1/sigma ({K}x{D}) and P x_hat", "nnz": K * D,
         "macs": dev["emission_macs"], "note": "one lifted offset read per sample; 4 table lookups"},
        {"stage": "Update and normalise", "matrix": "belief alpha_t (4, int8)", "nnz": K,
         "macs": dev["update_macs"], "note": "4 divisions; rescale prevents underflow"},
        {"stage": "Total per sample", "matrix": "", "nnz": None, "macs": dev["total_macs"],
         "note": "independent of the mesh size n"},
    ]
    return {"host": host, "device": device, "device_detail": dev,
            "mesh": {"nodes_per_side": nodes, "n": fq["n"], "sensors": m, "device_sites": n_sites}}


def markdown(t: dict) -> str:
    out = ["| Stage | Matrix | n | nnz | MACs | Per |", "|---|---|---|---|---|---|"]
    for r in t["host"]:
        out.append(f"| {r['stage']} | {r['matrix']} | {r['n']:,} | {r['nnz']:,} | {r['macs']:,} | {r['per']} |")
    out += ["", "| Stage | Matrix | Entries | MACs |", "|---|---|---|---|"]
    for r in t["device"]:
        nnz = "" if r["nnz"] is None else f"{r['nnz']:,}"
        out.append(f"| {r['stage']} | {r['matrix']} | {nnz} | {r['macs']:,} |")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    t = build()
    print(markdown(t))
    if a.json:
        a.json.write_text(json.dumps(t, indent=2))


if __name__ == "__main__":
    main()
