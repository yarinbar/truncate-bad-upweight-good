import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks/static_experiments.ipynb"


def nbconvert(args, cwd, tmp_path):
    """Run nbconvert with a `python3` kernel that is this interpreter, whatever `python3` the machine defines."""
    # Why: a machine-wide JUPYTER_PATH can map `python3` to another interpreter (e.g. one without matplotlib).
    spec = tmp_path / "jupyter/kernels/python3"
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python 3", "language": "python"}))
    env = {**os.environ, "JUPYTER_PATH": str(tmp_path / "jupyter")}
    return subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", *args],
                          cwd=cwd, env=env, capture_output=True, text=True)


def execute(workdir):
    shutil.copytree(ROOT / "notebooks", workdir / "notebooks", ignore=shutil.ignore_patterns("figs"))
    res = nbconvert(["--ExecutePreprocessor.timeout=1800", "--output", "executed.ipynb", "static_experiments.ipynb"],
                    workdir / "notebooks", workdir)
    assert res.returncode == 0, res.stderr[-4000:]
    return json.loads((workdir / "notebooks/executed.ipynb").read_text())


def all_text(nb):
    chunks = []
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            chunks.append("".join(out.get("text", "")))
            chunks.append("".join(out.get("data", {}).get("text/plain", "")))
    return "\n".join(chunks)


@pytest.mark.slow
def test_notebook_reproduces_b3_numbers_and_writes_paper_figures(tmp_path):
    nb = execute(tmp_path)
    text = all_text(nb)
    assert "0.9325" in text and "0.882" in text
    figs = {p.name for p in (tmp_path / "notebooks/figs").glob("*.svg")}
    assert {"fixed_threshold_static.svg", "fixed_threshold_beta_star.svg",
            "region_agreement_bottom_vs_top_Skywork-Llama_ultrafeedback.svg",
            "benefit_vs_cost_lambda_Skywork-Llama_ultrafeedback.svg"} <= figs
    assert len([f for f in figs if f.startswith(("region_agreement", "benefit_vs_cost"))]) == 8


def test_notebook_fails_clearly_without_data(tmp_path):
    shutil.copytree(ROOT / "notebooks", tmp_path / "notebooks", ignore=shutil.ignore_patterns("figs", "*.npz"))
    res = nbconvert(["--output", "x.ipynb", "static_experiments.ipynb"], tmp_path / "notebooks", tmp_path)
    assert res.returncode != 0
    assert "scripts/export_static_data.py" in res.stderr


def _panels(source):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PANELS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("no PANELS assignment")


def test_notebook_panels_match_export_script():
    cells = "\n".join("".join(c["source"]) for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code")
    assert _panels(cells) == _panels((ROOT / "scripts/export_static_data.py").read_text())
