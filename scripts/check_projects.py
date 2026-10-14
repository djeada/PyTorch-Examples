"""Copy each lesson outside the repository and run its numerical checks in isolation."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    projects = sorted((root / "projects").glob("[0-9][0-9]_*"))
    if not projects or [project.name[:2] for project in projects] != [
        f"{i:02}" for i in range(1, len(projects) + 1)
    ]:
        raise RuntimeError("Expected nonempty, consecutively numbered project folders.")
    for project in projects:
        if not all(
            (project / name).is_file() for name in ("main.py", "README.md", "requirements.txt")
        ):
            raise RuntimeError(f"Missing standalone project files: {project.name}")
        with TemporaryDirectory(prefix="pytorch-course-") as temporary:
            copied = Path(temporary) / project.name
            shutil.copytree(
                project, copied, ignore=shutil.ignore_patterns("__pycache__", ".venv", "outputs")
            )
            for seed in args.seeds:
                # -I excludes the repository and PYTHONPATH from module lookup.
                # Dependencies come from this Python environment's installed packages.
                command = [
                    sys.executable,
                    "-I",
                    str(copied / "main.py"),
                    "--check",
                    "--seed",
                    str(seed),
                ]
                result = subprocess.run(
                    command, cwd=copied, capture_output=True, text=True, timeout=120
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"{project.name} failed with seed {seed}:\n{result.stdout}\n{result.stderr}"
                    )
                metrics = json.loads(result.stdout)
                if not isinstance(metrics, dict) or not metrics:
                    raise RuntimeError(f"{project.name} did not report metrics")
                print(
                    f"PASS {project.name} (seed {seed}): {json.dumps(metrics, sort_keys=True)}",
                    flush=True,
                )
    print(f"All {len(projects)} independent projects passed with {len(args.seeds)} seeds.")


if __name__ == "__main__":
    main()
