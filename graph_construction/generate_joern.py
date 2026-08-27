"""
Usage:
    python generate_joern.py --projects linux
    python generate_joern.py --projects FFmpeg OpenSSL PHP-SRC ImageMagick
    python generate_joern.py --projects all
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import (
    get_all_projects,
    get_joern_script,
    get_joern_targets_file,
)
from pipeline_utils import DATA_DIR, joern_output_dir


def collect_joern_scan_targets(projects: list[str], testcases: list[str] | None = None) -> list[tuple[str, str]]:
    """
    Walk the data directory and collect (srcDir, outDir) pairs.

    outDir is the mirrored path under joern_output/ (see
    pipeline_utils.joern_output_dir) — deliberately outside the source tree so
    importCode() cannot re-ingest Joern's own output. Includes any directory
    holding at least one .c OR .h file.

    If testcases is provided, only include paths whose test-case directory
    name (the path component directly under project_data_dir) matches one
    of the given names.
    """
    targets: list[tuple[str, str]] = []
    for project in projects:
        project_data_dir = Path(DATA_DIR) / project
        if not project_data_dir.is_dir():
            print(f"[WARN] data dir not found, skipping: {project_data_dir}")
            continue
        for variant in ("before", "after", "fixing"):
            for src_dir in sorted(project_data_dir.rglob(variant)):
                # Include header-only directories too. 74 of the 822 linux cases
                # have a fix that touches only .h files; filtering on *.c alone
                # silently dropped them from the dataset. Joern's c2cpg parses
                # headers fine (verified: a header-only dir exports 25 methods).
                if not (src_dir.is_dir()
                        and (any(src_dir.rglob("*.c")) or any(src_dir.rglob("*.h")))):
                    continue

                if testcases:
                    # test-case dir is the path component directly under project_data_dir
                    rel_parts = src_dir.relative_to(project_data_dir).parts
                    test_name = rel_parts[0] if rel_parts else None
                    if test_name not in testcases:
                        continue

                out_dir = joern_output_dir(src_dir)
                targets.append((str(src_dir), out_dir))
    return targets

def run(projects: list[str], testcases: list[str] | None = None) -> None:
    joern_script = get_joern_script()
    targets_file = get_joern_targets_file()
    mapping_dir  = targets_file.parent   # graph_construction/mapping/

    targets = collect_joern_scan_targets(projects, testcases)
    if not targets:
        print("No targets found — nothing to do.")
        return

    # Write targets file: genJoern.sc expects "srcDir;outDir" per line
    targets_file.parent.mkdir(parents=True, exist_ok=True)
    targets_file.write_text(
        "\n".join(f"{src};{out}" for src, out in targets) + "\n",
        encoding="utf-8"
    )
    print(f"Wrote {len(targets)} targets to {targets_file}")

    # Run joern once from mapping/ so genJoern.sc finds joern_targets.txt there
    cmd = ["joern", "--script", str(joern_script)]
    print(f"Running: {' '.join(cmd)}")
    print(f"CWD    : {mapping_dir}")

    result = subprocess.run(cmd, cwd=str(mapping_dir))
    if result.returncode != 0:
        print(f"[ERROR] Joern exited with code {result.returncode}")
    else:
        print("[OK] Joern completed")


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Joern CPG dot files for before/after/fixing source directories."
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        default=get_all_projects(),
        help="Projects to process (default: all configured projects).",
    )
    parser.add_argument(
        "--testcases",
        nargs="+",
        default=None,
        help="If given, only regenerate joern output for these specific test-case names (e.g. test1261 test818 test147).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    run(projects=args.projects, testcases=args.testcases)