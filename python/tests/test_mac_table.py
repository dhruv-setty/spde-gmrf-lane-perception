"""The operation-count table is derived from the matrices and agrees with the fill-in results."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "mac_table", Path(__file__).resolve().parents[1] / "tools" / "mac_table.py")
mac_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mac_table)


def test_counts_link_to_matrices():
    t = mac_table.build()
    rows = {r["stage"]: r for r in t["host"]}
    assert rows["Cholesky, 20x20 lattice, RCM"]["nnz"] == 11378
    assert rows["Cholesky, 20x20 lattice, scrambled (seed 0)"]["nnz"] == 45258
    assert rows["Sparse product Q x"]["macs"] == rows["Sparse product Q x"]["nnz"]
    assert rows["Sensor product P^T y"]["nnz"] == 3 * rows["Sensor product P^T y"]["n"]
    assert rows["Lift x_hat to device sites"]["nnz"] == 3 * 64
    # Posterior mean solve: two triangular sweeps over the strictly lower part of L.
    post = rows["Posterior mean solve"]
    assert post["macs"] == 2 * (post["nnz"] - post["n"])


def test_device_count_is_independent_of_mesh():
    small = mac_table.build(nodes=8)["device_detail"]["total_macs"]
    large = mac_table.build(nodes=16)["device_detail"]["total_macs"]
    assert small == large == 40
