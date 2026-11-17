"""Regenerate the Figure 3 comparison: Cholesky factor patterns, scrambled order versus RCM.

  python python/tools/fill_in_figure.py docs/spde-fill-in.png   (requires matplotlib)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spdegmrf.cholesky import SymbolicCholesky  # noqa: E402
from spdegmrf.lattice import fill_in, lattice_precision  # noqa: E402


def main(out: str, seed: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Q = lattice_precision(20)
    perm = np.random.default_rng(seed).permutation(Q.shape[0])
    Qs = Q[perm][:, perm]
    scr = SymbolicCholesky(Qs, "natural")
    rcm = SymbolicCholesky(Qs, "rcm")
    r = fill_in(20, seed)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, sym, title in ((axes[0], scr, "Scrambled order"), (axes[1], rcm, "Reverse Cuthill-McKee")):
        F = sym.factor(Qs)
        ax.spy(F.to_sparse(), markersize=0.3, color="black")
        ax.set_title(f"{title}: nnz(L) = {sym.nnz:,}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"20 x 20 lattice, n = {r.n}, nnz(Q) = {r.q_nnz:,}; RCM removes {100 * r.reduction:.0f}% "
                 f"of the factor nonzeros (seed {seed})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}: scrambled {r.scrambled_nnz}, rcm {r.rcm_nnz}, reduction {r.reduction:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "spde-fill-in.png")
