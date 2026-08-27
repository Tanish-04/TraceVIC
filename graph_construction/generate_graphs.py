"""
Handles graph.json generation for BOTH:

  Mode 1 (--mode fix):
    Generate graph.json for the FIXING COMMIT of each test case.
    Run after generate_joern.py.

  Mode 2 (--mode history):
    Generate graph.json for ALL HISTORY COMMITS in commits.json.
    Run after generate_joern.py with --mode history.

Output:
    <test_dir>/<commit_sha>/graph.json

Usage:
    # Fix commit graphs
    python generate_graphs.py --mode fix --projects linux
    python generate_graphs.py --mode fix --projects FFmpeg OpenSSL
    python generate_graphs.py --mode fix --projects all

    # History commit graphs
    python generate_graphs.py --mode history --projects linux
    python generate_graphs.py --mode history --projects FFmpeg OpenSSL PHP-SRC ImageMagick
    python generate_graphs.py --mode history --projects all
"""

import gc
import json
import os
import sys
import argparse
import traceback
from pathlib import Path

GRAPH_CONSTRUCTION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GRAPH_CONSTRUCTION_DIR))

from project.union_project import CUnionProject
from parse_patch.patch import Patch

from config_loader import (
    get_all_projects,
    get_project_compile_db,
)
from pipeline_utils import (
    DATA_DIR,
    joern_output_dir
)

def joern_non_empty(joern_dir: str) -> bool:
    return os.path.isdir(joern_dir) and len(os.listdir(joern_dir)) > 0


def is_assembly_only(after_dir: str) -> bool:
    if not os.path.isdir(after_dir):
        return False
    names = [n for n in os.listdir(after_dir)
             if os.path.isfile(os.path.join(after_dir, n))]
    has_c_or_h = any(n.endswith(".c") or n.endswith(".h") for n in names)
    has_S      = any(n.endswith(".S") for n in names)
    return has_S and not has_c_or_h

def is_addition_only_patch(patch_path: str) -> bool:
    try:
        p = Patch(patch_path)   # pass path, not content
        has_add = any(
            pl.lineno > 0
            for plines in p.addMap.values()
            for pl in plines
        )
        has_del = any(
            pl.lineno > 0
            for plines in p.delMap.values()
            for pl in plines
        )
        return has_add and not has_del
    except Exception as e:
        print(f"    [WARN] is_addition_only_patch exception: {e}")
        return False


def append_graph_error_log(label: str, commit_sha: str, error_msg: str):
    try:
        log_path = GRAPH_CONSTRUCTION_DIR / "graph_errors.txt"
        with open(log_path, "a") as fh:
            fh.write(f"\n{'='*80}\n")
            fh.write(f"Label    : {label}\n")
            fh.write(f"Commit   : {commit_sha}\n")
            fh.write(f"Error    : {error_msg}\n")
            fh.write(f"Traceback:\n{traceback.format_exc()}\n")
    except Exception:
        pass


def generate_graph(commit_dir: str, commit_sha: str, label: str,
                   addition_also: bool = False) -> bool:
    """
    Run CUnionProject on commit_dir and write graph.json.
    Returns True on success, False on skip/failure.

    addition_also
        Build the BASELINE variant of an addition-only case and write it to
        ``graph_addition.json`` instead of ``graph.json``.

        NOTE ON THE NAME: graph_addition.json is the graph for an addition-only
        case built WITHOUT the SEM-SZZ anchor logic -- it is the opposite of
        what the filename suggests.  Normally an addition-only fix goes through
        run_sem_szz_analysis(), which synthesises PatchLine(isDel=True) ANCHOR
        lines (nearest non-noise neighbour of the insertion, plus the enclosing
        function signature) so V-SZZ has something to trace.  Those lines were
        never deleted.  That is correct and deliberate for TempoVIC, but it is a
        TempoVIC contribution, so feeding it to the NeuralSZZ/B-SZZ baseline
        would hand the baseline our own algorithm: published B-SZZ has nothing
        to blame on an addition-only commit.  With this flag the standard
        deletion path runs instead, which yields ZERO isDel nodes (verified on
        test1014/1016/1017/1032) -- the honest baseline behaviour.

        graph.json is never touched, so both variants coexist.
    """
    after_dir    = os.path.join(commit_dir, "after")
    fixing_dir   = os.path.join(commit_dir, "fixing")
    before_joern = joern_output_dir(os.path.join(commit_dir, "before"))
    after_joern  = joern_output_dir(os.path.join(commit_dir, "after"))
    graph_path   = os.path.join(
        commit_dir, "graph_addition.json" if addition_also else "graph.json"
    )

    if os.path.exists(graph_path):
        print(f"    [SKIP] graph already exists: {graph_path}")
        return True

    if not joern_non_empty(before_joern) and not joern_non_empty(after_joern):
        print("    [SKIP] both joern dirs empty — run joern step first")
        return False

    patch_files = os.listdir(fixing_dir) if os.path.exists(fixing_dir) else []
    if not patch_files:
        print("    [SKIP] no patch file in fixing/")
        return False

    if is_assembly_only(after_dir):
        print("    [SKIP] assembly-only (.S) — no C/H files to graph")
        return False

    patch_path    = os.path.join(fixing_dir, patch_files[0])
    addition_only = is_addition_only_patch(patch_path)

    print(f"    [DEBUG] patch file: {patch_files[0]}")
    print(f"    [DEBUG] addition_only={addition_only}")

    if addition_also and not addition_only:
        # graph.json is already built on the standard deletion path for these,
        # so a graph_addition.json would be a byte-identical duplicate.
        print("    [SKIP] --addition-also: not an addition-only patch, "
              "graph.json already is the baseline variant")
        return True

    if addition_only and addition_also:
        print("    [INFO] addition-only patch — SEM-SZZ DISABLED "
              "(baseline variant → graph_addition.json)")
        addition_only = False
    elif addition_only:
        print("    [INFO] addition-only patch — applying SEM-SZZ analysis")

    try:
        union_project = CUnionProject(commit_dir, addition_only=addition_only)
        union_project.write_json(graph_path)
        node_count = len(union_project.get_final_nodes())
        print(f"    [OK] {node_count} nodes → {os.path.basename(graph_path)}")
        return True
    except KeyboardInterrupt:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"    [FAIL] {error_msg[:150]}")
        append_graph_error_log(label, commit_sha, error_msg)
        return False


def apply_compile_db_env(project: str) -> None:
    try:
        compile_db_path = get_project_compile_db(project)
        os.environ["GRAPHGEN_COMPILE_DB"] = compile_db_path
        print(f"\n[INFO] Compile DB: {compile_db_path}")
    except (KeyError, FileNotFoundError) as e:
        os.environ.pop("GRAPHGEN_COMPILE_DB", None)
        print(f"\n[WARN] No compile DB for {project}: {e}")


def run_fix(projects: list, testcases, addition_also: bool = False) -> None:
    total = ok = failed = skip = 0
    batch_count = 0

    for project in projects:
        apply_compile_db_env(project)
        project_dir = os.path.join(DATA_DIR, project)

        if not os.path.isdir(project_dir):
            print(f"\n[WARN] project dir not found: {project_dir}")
            continue

        test_names = sorted(
            d for d in os.listdir(project_dir)
            if os.path.isdir(os.path.join(project_dir, d))
        )

        if testcases:
            test_names = [t for t in test_names if t in testcases]
            if not test_names:
                print(f"\n[WARN] none of the requested testcases found under {project_dir}")
                continue


        print(f"\n{'='*70}")
        print(f"PROJECT: {project}  ({len(test_names)} test cases)")
        print(f"{'='*70}")

        proj_ok = proj_fail = proj_skip = 0

        for test_name in test_names:
            test_dir  = os.path.join(project_dir, test_name)
            info_path = os.path.join(test_dir, "info.json")

            if not os.path.exists(info_path):
                proj_skip += 1
                continue

            with open(info_path) as fh:
                info = json.load(fh)

            fix_sha = info.get("fix")
            if not fix_sha:
                proj_skip += 1
                continue

            commit_dir = os.path.join(test_dir, fix_sha)
            if not os.path.isdir(commit_dir):
                print(f"\n  {test_name}: [SKIP] fix commit dir not found")
                proj_skip += 1
                continue

            print(f"\n  {test_name}  fix={fix_sha[:12]}")

            result = generate_graph(commit_dir, fix_sha, test_name,
                                    addition_also=addition_also)
            if result:
                proj_ok += 1
                batch_count += 1
                if batch_count >= 10:
                    gc.collect()
                    batch_count = 0
            else:
                proj_fail += 1

        total  += len(test_names)
        ok     += proj_ok
        failed += proj_fail
        skip   += proj_skip

        print(f"\n  {project} summary — "
              f"ok={proj_ok}  fail={proj_fail}  skip={proj_skip}")

    print_run_summary("FIX", total, ok, failed, skip)

def run_history(projects: list, testcases) -> None:
    """Mode 2 — generate graph.json for all history commits."""
    total_tests   = 0
    total_commits = ok_commits = fail_commits = skip_commits = 0
    batch_count   = 0

    for project in projects:
        apply_compile_db_env(project)
        project_dir = os.path.join(DATA_DIR, project)

        if not os.path.isdir(project_dir):
            print(f"\n[WARN] project dir not found: {project_dir}")
            continue

        test_names = sorted(
            d for d in os.listdir(project_dir)
            if os.path.isdir(os.path.join(project_dir, d))
        )

        if testcases:
            test_names = [t for t in test_names if t in testcases]
            if not test_names:
                print(f"\n[WARN] none of the requested testcases found under {project_dir}")
                continue

        print(f"\n{'='*70}")
        print(f"PROJECT: {project}  ({len(test_names)} test cases)")
        print(f"{'='*70}")

        for test_name in test_names:
            test_dir     = os.path.join(project_dir, test_name)
            commits_path = os.path.join(test_dir, "commits.json")

            if not os.path.exists(commits_path):
                print(f"\n  {test_name}: [SKIP] missing commits.json")
                continue

            with open(commits_path) as fh:
                commits_data = json.load(fh)

            history_commits = commits_data.get("all_commits_in_history", [])
            if not history_commits:
                print(f"\n  {test_name}: [SKIP] no history commits")
                continue

            print(f"\n  {test_name}  history={len(history_commits)}")
            total_tests += 1
            test_ok = test_fail = test_skip = 0

            for commit_sha in history_commits:
                commit_dir = os.path.join(test_dir, commit_sha)
                print(f"    {commit_sha[:12]}")

                if not os.path.isdir(commit_dir):
                    print("      [SKIP] commit dir not found")
                    test_skip += 1
                    continue

                result = generate_graph(
                    commit_dir, commit_sha, f"{test_name}/{commit_sha[:12]}")
                if result:
                    test_ok += 1
                    batch_count += 1
                    if batch_count >= 10:
                        gc.collect()
                        batch_count = 0
                else:
                    test_fail += 1

            total_commits  += len(history_commits)
            ok_commits     += test_ok
            fail_commits   += test_fail
            skip_commits   += test_skip

            print(f"    → ok={test_ok}  fail={test_fail}  skip={test_skip}")

    print(f"\n{'='*70}")
    print("STEP 3 (HISTORY) COMPLETE")
    print(f"{'='*70}")
    print(f"  Test cases       : {total_tests}")
    print(f"  Total commits    : {total_commits}")
    print(f"  Generated (ok)   : {ok_commits}")
    print(f"  Failed           : {fail_commits}")
    print(f"  Skipped          : {skip_commits}")
    if total_commits > 0:
        print(f"  Success rate     : {ok_commits*100/total_commits:.1f}%")


def print_run_summary(mode: str, total: int, ok: int, failed: int, skip: int):
    print(f"\n{'='*70}")
    print(f"STEP 3 ({mode}) COMPLETE")
    print(f"{'='*70}")
    print(f"  Total test cases : {total}")
    print(f"  Generated (ok)   : {ok}")
    print(f"  Failed           : {failed}")
    print(f"  Skipped          : {skip}")
    if total > 0:
        print(f"  Success rate     : {ok*100/total:.1f}%")


if __name__ == "__main__":
    _all_projects = get_all_projects()

    parser = argparse.ArgumentParser(
        description="Generate graphs — fix commit (step 3) or history commits (step 7)"
    )
    parser.add_argument(
        "--mode", choices=["fix", "history"], default="fix",
        help="'fix' = fixing commit only (step 3), "
             "'history' = all V-SZZ history commits (step 7)"
    )
    parser.add_argument(
        "--projects", nargs="+",
        default=_all_projects,
        choices=_all_projects,
        help="Which projects to process (default: all configured projects)"
    )
    parser.add_argument(
        "--addition-also", action="store_true",
        help="BASELINE variant for addition-only cases: skip SEM-SZZ anchor "
             "extraction and write graph_addition.json (graph.json untouched). "
             "Use for NeuralSZZ/B-SZZ, which must not receive SEM-SZZ's "
             "synthesised isDel anchor lines. Cases that are not addition-only "
             "are skipped -- their graph.json is already the baseline variant. "
             "--mode fix only."
    )
    parser.add_argument(
    "--testcases", nargs="+", default=None,
    help="If given, only regenerate graphs for these specific test-case names (e.g. test1261 test818 test147)."
    )
    args = parser.parse_args()

    if args.mode == "fix":
        run_fix(projects=args.projects, testcases=args.testcases,
                addition_also=args.addition_also)
    else:
        if args.addition_also:
            parser.error("--addition-also applies to --mode fix only "
                         "(SEM-SZZ runs on the fixing commit, not history commits)")
        run_history(projects=args.projects, testcases=args.testcases)
