"""README figures computed by the Python reference in python/spdegmrf.

Every array plotted here is produced by the repository's own code with the fixed seeds the
tests and the export pipeline use. Nothing is drawn by hand. Run from the repository root:

  python docs/capture/plots.py docs/screenshots

Writes PNG files and plots.log (the numbers behind each figure) into the output directory.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import matplotlib.tri as mtri  # noqa: E402

from spdegmrf import hmm, quant  # noqa: E402
from spdegmrf.cholesky import SymbolicCholesky  # noqa: E402
from spdegmrf.estimate import Estimator, sample_field  # noqa: E402
from spdegmrf.export import route_sites, run_pipeline, true_theta  # noqa: E402
from spdegmrf.fem import assemble  # noqa: E402
from spdegmrf.lattice import fill_in, lattice_precision  # noqa: E402
from spdegmrf.mesh import rect_mesh  # noqa: E402
from spdegmrf.projector import projector  # noqa: E402
from spdegmrf.spde import SpdePrecision  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/screenshots")
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "plots.log", "w")


def log(msg: str) -> None:
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"wrote {name}")


log(f"numpy {np.__version__}, matplotlib {matplotlib.__version__}, python {sys.version.split()[0]}")

# ---------------------------------------------------------------------------------------
# 1. Estimated field. The steps of spdegmrf.export.run_pipeline (seed 2026), repeated here
#    in the same order so that the simulated true field is available for comparison; the
#    estimate is then checked against run_pipeline itself.
# ---------------------------------------------------------------------------------------
t0 = time.time()
seed, nodes, lx, m, R, sigma, n_sites = 2026, 16, 1.5, 200, 8, 15.0, 64
rng = np.random.default_rng(seed)
mesh = rect_mesh(nodes, nodes, lx, lx)
Cd, G = assemble(mesh)
spde = SpdePrecision(Cd, G)
kappa, tau = true_theta()
X = sample_field(spde, kappa, tau, rng, size=R)
sensors = rng.uniform(0.0, lx, size=(m, 2))
P = projector(mesh, sensors)
Y = P @ X + sigma * rng.standard_normal((m, R))
est = Estimator(spde, P, sigma)
k_hat, t_hat, ll_hat = est.fit(Y, kappa0=2.0, tau0=0.005)
x_hat = est.posterior_mean(k_hat, t_hat, Y[:, -1])
sites = route_sites(n_sites, lx)
P_dev = projector(mesh, sites)
offsets = P_dev @ x_hat
ref = run_pipeline()
assert np.isclose(ref.kappa_hat, k_hat) and np.isclose(ref.tau_hat, t_hat)
assert np.allclose(ref.site_offset, offsets)
log(f"field: n={mesh.n} nodes, {m} sensors, R={R} replicates, sigma={sigma} mm; "
    f"kappa true={kappa:.4f} hat={k_hat:.4f} ({abs(k_hat / kappa - 1) * 100:.1f}%); "
    f"tau true={tau:.6f} hat={t_hat:.6f} ({abs(t_hat / tau - 1) * 100:.1f}%); "
    f"matches run_pipeline: yes; nnz(P)={P.nnz}, nnz(P_dev)={P_dev.nnz}; "
    f"lifted offsets {offsets.min():.1f} to {offsets.max():.1f} mm")

tri = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.tris)
x_true = X[:, -1]
vmax = float(np.max(np.abs(np.concatenate([x_true, x_hat]))))
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), constrained_layout=True)
ax = axes[0]
ax.triplot(tri, color="0.75", lw=0.4)
sc = ax.scatter(sensors[:, 0], sensors[:, 1], c=Y[:, -1], cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=14,
                edgecolors="k", linewidths=0.3)
ax.set_title(f"Observations y = P x + e at {m} sensors\n(replicate {R} of {R}, sigma = {sigma:g} mm)", fontsize=10)
ax = axes[1]
tc = ax.tripcolor(tri, x_true, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud")
ax.set_title(f"Simulated field x (kappa = {kappa:.3f}, tau = {tau:.5f})", fontsize=10)
ax = axes[2]
ax.tripcolor(tri, x_hat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud")
ax.plot(sites[:, 0], sites[:, 1], color="k", lw=0.8)
ax.scatter(sites[:, 0], sites[:, 1], c=offsets, cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=22, edgecolors="k",
           linewidths=0.6, zorder=3)
ax.set_title(f"Posterior mean x_hat (kappa_hat = {k_hat:.3f}, tau_hat = {t_hat:.5f})\n"
             f"with lifted offsets P_dev x_hat at {n_sites} route sites", fontsize=10)
for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlim(0, lx)
    ax.set_ylim(0, lx)
    ax.set_xlabel("km")
axes[0].set_ylabel("km")
fig.colorbar(tc, ax=axes, shrink=0.85, label="mm")
fig.suptitle(f"16 x 16 estimation mesh, n = {mesh.n}; spdegmrf.export.run_pipeline, seed {seed}", fontsize=11)
save(fig, "plot_field_estimate.png")
log(f"field figure: {time.time() - t0:.1f} s")

# ---------------------------------------------------------------------------------------
# 2. Marginal log-likelihood surface over (kappa, tau), from Estimator.loglik, with the
#    symbolic analysis made once and reused for every grid point.
# ---------------------------------------------------------------------------------------
t0 = time.time()
f0 = est.factorisations
lk = np.linspace(np.log(kappa) - 0.6, np.log(kappa) + 0.6, 25)
lt = np.linspace(np.log(tau) - 0.6, np.log(tau) + 0.6, 25)
L = np.array([[est.loglik(np.exp(a), np.exp(b), Y) for a in lk] for b in lt])
log(f"loglik grid: 25 x 25 points, {est.factorisations - f0} numeric factorisations, symbolic analyses "
    f"{est.analyses}; max on grid {L.max():.2f}; loglik at estimate {ll_hat:.2f}")
fig, ax = plt.subplots(figsize=(6.6, 5.4))
cs = ax.contourf(np.exp(lk), np.exp(lt), L - L.max(), levels=np.linspace(-200, 0, 21), cmap="viridis",
                 extend="min")
ax.contour(np.exp(lk), np.exp(lt), L - L.max(), levels=[-20, -10, -5, -2], colors="w", linewidths=0.6)
ax.plot(kappa, tau, "w+", ms=14, mew=2, label="true (simulation)")
ax.plot(k_hat, t_hat, "rx", ms=10, mew=2, label="Nelder-Mead estimate")
ax.set_xlabel("kappa")
ax.set_ylabel("tau")
ax.legend(loc="upper right", fontsize=9)
fig.colorbar(cs, ax=ax, label="log p(y | kappa, tau) minus grid maximum")
ax.set_title("Marginal log-likelihood, Estimator.loglik, 16 x 16 mesh", fontsize=10)
save(fig, "plot_loglik_surface.png")
log(f"loglik figure: {time.time() - t0:.1f} s")

# ---------------------------------------------------------------------------------------
# 3. Sparsity of Q and of its Cholesky factor (10 by 10 and 20 by 20 lattices).
# ---------------------------------------------------------------------------------------
Q10 = lattice_precision(10)
Q20 = lattice_precision(20)
perm = np.random.default_rng(0).permutation(400)
Qs = Q20[perm][:, perm]
scr = SymbolicCholesky(Qs, "natural")
rcm = SymbolicCholesky(Qs, "rcm")
r0 = fill_in(20, 0)
log(f"sparsity: nnz(Q10)={Q10.nnz}, max per row {np.diff(Q10.tocsc().indptr).max()}; nnz(Q20)={Q20.nnz}; "
    f"nnz(L) scrambled seed 0={scr.nnz}, rcm={rcm.nnz}")
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
axes[0].spy(Q10, markersize=1.2, color="k")
axes[0].set_title(f"Q, 10 x 10 lattice: n = 100, nnz = {Q10.nnz:,}", fontsize=10)
axes[1].spy(scr.factor(Qs).to_sparse(), markersize=0.3, color="k")
axes[1].set_title(f"L, 20 x 20 lattice, scrambled (seed 0): nnz = {scr.nnz:,}", fontsize=10)
axes[2].spy(rcm.factor(Qs).to_sparse(), markersize=0.3, color="k")
axes[2].set_title(f"L, 20 x 20 lattice, RCM: nnz = {rcm.nnz:,}", fontsize=10)
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
save(fig, "plot_sparsity.png")

# ---------------------------------------------------------------------------------------
# 4. Scrambled-order factor nonzeros over many random relabellings (Figure 3 counts).
# ---------------------------------------------------------------------------------------
t0 = time.time()
N_SEEDS = 1000
STUDY = 37615
vals = np.array([SymbolicCholesky(Q20, np.random.default_rng(s).permutation(400)).nnz for s in range(N_SEEDS)])
rcm_vals = np.array([SymbolicCholesky(Q20[q][:, q], "rcm").nnz
                     for q in (np.random.default_rng(s).permutation(400) for s in range(N_SEEDS))])
rcm_counts = {int(v): int(c) for v, c in zip(*np.unique(rcm_vals, return_counts=True))}
below = float(np.mean(vals <= STUDY))
log(f"scrambled nnz(L) over seeds 0..{N_SEEDS - 1} (numpy default_rng permutation): min {vals.min()}, "
    f"5th pct {np.percentile(vals, 5):.0f}, median {np.median(vals):.0f}, mean {vals.mean():.1f}, "
    f"95th pct {np.percentile(vals, 95):.0f}, max {vals.max()}; seeds 0,1,2: {vals[:3].tolist()}; "
    f"fraction <= {STUDY}: {below:.3f}; exact matches of {STUDY}: {int(np.sum(vals == STUDY))}; "
    f"RCM applied to each scrambled matrix, nnz(L): count of seeds per value {rcm_counts}; reduction at {STUDY}: {1 - 11378 / STUDY:.3f}; "
    f"{time.time() - t0:.1f} s")
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.hist(vals, bins=50, color="0.55", edgecolor="0.3")
ax.axvline(vals[0], color="tab:blue", lw=1.6, label=f"this repository, seed 0: {vals[0]:,}")
ax.axvline(STUDY, color="tab:red", lw=1.6, ls="--", label=f"case study value: {STUDY:,}")
ax.axvline(rcm_vals[0], color="tab:green", lw=1.6, ls=":", label=f"RCM, seed 0: {rcm_vals[0]:,}")
ax.set_xlabel("nnz(L), Cholesky factor of Q on the 20 x 20 lattice (diagonal included)")
ax.set_ylabel("number of seeds")
ax.set_title(f"Scrambled node order: {N_SEEDS:,} uniform random relabellings (seeds 0 to {N_SEEDS - 1})\n"
             f"{100 * below:.1f}% of relabellings give {STUDY:,} or fewer", fontsize=10)
ax.legend(fontsize=9, loc="upper left")
save(fig, "plot_fill_in_seeds.png")

# ---------------------------------------------------------------------------------------
# 5. Forward filter: float reference against the int8 filter (the data of
#    python/tests/test_forward_filter.py, seed 7, T = 5000; first 600 samples shown).
# ---------------------------------------------------------------------------------------
rng = np.random.default_rng(7)
p = hmm.HmmParams.default(rng.normal(0.0, 40.0, 64))
truth, fsites, Yf = hmm.simulate(p, 5000, rng)
fb = hmm.forward_filter(p, Yf, fsites)
qm = quant.quantise(p)
filt = quant.Int8Filter(qm)
qb, modes = filt.run(quant.quantise_obs(Yf), fsites)
ag = quant.agreement(fb, qb, truth)
log(f"forward filter (seed 7, T={ag.T}): argmax agreement {ag.argmax_agreement:.4f}, mean |belief error| "
    f"{ag.mean_abs_belief_error:.4f}, max {ag.max_abs_belief_error:.4f}, accuracy vs simulated regime float "
    f"{ag.float_accuracy_vs_truth:.4f} int8 {ag.int8_accuracy_vs_truth:.4f}, floor hits {filt.floor_hits}")
T = 600
t = np.arange(T)
qn = quant.dequant_belief(np.asarray(qb))
qn = qn / qn.sum(1, keepdims=True)
colours = ["#4c9f70", "#e0b23a", "#e07b39", "#c03a3a"]
fig, axes = plt.subplots(4, 1, figsize=(12, 9.2), sharex=True, constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.2, 0.5, 1, 1]})
ax = axes[0]
ax.plot(t, Yf[:T, 0], lw=0.7, color="tab:blue", label="lateral offset y_1 (mm)")
ax.plot(t, Yf[:T, 1], lw=0.7, color="tab:purple", label="SDLP window y_2 (mm)")
ax.set_ylabel("mm")
ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.set_title("Simulated observations, seed 7 (python/tests/test_forward_filter.py)", fontsize=10)
ax = axes[1]
ax.step(t, truth[:T], where="post", color="k", lw=1.0, label="simulated regime")
ax.step(t, np.asarray(modes)[:T] + 0.12, where="post", color="tab:red", lw=0.8, label="int8 mode (offset 0.12)")
ax.set_yticks(range(4))
ax.set_yticklabels(hmm.REGIMES, fontsize=8)
ax.legend(fontsize=8, loc="upper left", ncol=2)
for ax, B, title in ((axes[2], fb, "Float forward filter, posterior P(S_t = k | y_1..t)"),
                     (axes[3], qn, "int8 forward filter (quant.Int8Filter), belief (alpha + 128) / 255, normalised")):
    ax.stackplot(t, B[:T].T, colors=colours, labels=hmm.REGIMES, lw=0)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("probability")
axes[2].legend(fontsize=8, loc="upper right", ncol=4)
axes[3].set_xlabel("sample t")
fig.suptitle(f"Forward filter over the first {T} of {ag.T} samples; over all {ag.T}: argmax agreement "
             f"{100 * ag.argmax_agreement:.2f}%, mean |belief error| {ag.mean_abs_belief_error:.4f}", fontsize=11)
save(fig, "plot_forward_filter.png")
LOG.close()
