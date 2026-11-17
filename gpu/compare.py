"""Compare spde_gpu output with the CPU reference. Requires a CUDA build of spde_gpu.

  python -m spdegmrf.export --gpu-case case.bin
  ./gpu/build/spde_gpu case.bin > gpu.csv
  python gpu/compare.py gpu.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from spdegmrf.export import run_pipeline  # noqa: E402

res = run_pipeline()
worst = 0.0
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        k, t, g = float(row["kappa"]), float(row["tau"]), float(row["loglik"])
        c = res.estimator.loglik(k, t, res.Y_train)
        worst = max(worst, abs(g - c) / max(1.0, abs(c)))
        print(f"kappa={k:g} tau={t:g} gpu={g:.10g} cpu={c:.10g}")
print(f"max relative difference: {worst:.3e}")
sys.exit(0 if worst < 1e-8 else 1)
