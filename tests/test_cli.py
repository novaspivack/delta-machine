import os
import subprocess
import sys
from pathlib import Path


def test_cli_headless_execution():
    root = Path(__file__).resolve().parents[1]
    parent_dir = root.parent
    scenario_dir = root / "scenarios"
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    paths = [str(root / "src"), str(parent_dir)]
    if pythonpath:
        paths.append(pythonpath)
    env["PYTHONPATH"] = ":".join(paths)
    cmd = [
        sys.executable,
        "-m",
        "delta_machine.cli",
        "headless",
        "--scenario",
        "basic_dsac.yaml",
        "--steps",
        "5",
        "--scenario-dir",
        str(scenario_dir),
    ]
    result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

