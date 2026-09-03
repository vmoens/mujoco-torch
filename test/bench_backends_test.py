import json
import subprocess
import sys
from pathlib import Path


def test_table_groups_devices_and_rejects_duplicate_cells(tmp_path):
    script = Path(__file__).parents[1] / "examples" / "bench_backends.py"
    results = tmp_path / "results.jsonl"
    rows = [
        {
            "env": "humanoid",
            "device": "cpu",
            "backend": "torch",
            "mode": "default",
            "batch_size": 16,
            "steps_per_s": 100,
        },
        {
            "env": "humanoid",
            "device": "cuda:0",
            "backend": "torch",
            "mode": "default",
            "batch_size": 16,
            "steps_per_s": 200,
        },
    ]
    results.write_text("".join(f"{json.dumps(row)}\n" for row in rows))

    rendered = subprocess.run(
        [sys.executable, script, "--table", results], check=True, capture_output=True, text=True
    ).stdout

    assert "device=cpu" in rendered
    assert "device=cuda:0" in rendered
    assert "| mujoco-torch compile | 100 |" in rendered
    assert "| mujoco-torch compile | 200 |" in rendered

    with results.open("a") as file:
        file.write(f"{json.dumps(rows[0])}\n")
    duplicate = subprocess.run([sys.executable, script, "--table", results], capture_output=True, text=True)

    assert duplicate.returncode != 0
    assert "duplicate steps_per_s cell" in duplicate.stderr
