"""
Run B-SZZ on each test case's fixing commit graph.json to trace
deletion lines back and find candidate inducing commits.

Requires:
  - graph.json must exist and be non-empty for the fixing commit

Output per test case:
  <test_dir>/<fix_sha>/graph_bszz.json

Usage:
    python run_bszz.py --projects linux
    python run_bszz.py --projects linux FFmpeg
"""

import os
import sys
import json
import argparse
import subprocess
import traceback

from config import (
    SZZ_DIR, REPOS_DIR, GRAPH_DATA_DIR, PROJECT_TO_REPO,
    get_available_projects,
)

sys.path.insert(0, SZZ_DIR)

from szz.b_szz import BaseSZZ
from szz.core.abstract_szz import ImpactedFile


def file_exists_in_commit(repo_dir: str, commit: str, path: str) -> bool:
    """Return True if path exists in the tree of commit."""
    try:
        r = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{path}"],
            cwd=repo_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return r.returncode == 0
    except Exception:
        return False


# {(repo_dir, commit) -> {underscore_name: git_path}}, or None if ls-tree failed.
#
# The tree index is built ONCE per commit.  It used to be rebuilt per NODE:
# every node in a case shares one fix_sha, so a graph averaging ~10 nodes ran
# ~10 redundant `git ls-tree -r` scans of the whole kernel tree (~7,200 calls
# across 755 cases instead of 755).  That mattered for more than speed --  on a
# loaded machine the 30 s timeout fires, resolve returns None, and the caller
# silently falls back to fName.replace("_", "/"), which is WRONG for any real
# filename containing an underscore (net_xfrm_xfrm_user.c -> net/xfrm/xfrm/user.c).
# The blame then fails and the node is written as commits=["ERROR"].
_TREE_CACHE = {}


def _tree_index(repo_dir: str, commit: str):
    """{underscore_name: git_path} for .c/.h files at `commit`. Cached per commit."""
    key = (repo_dir, commit)
    if key in _TREE_CACHE:
        return _TREE_CACHE[key]

    index = None
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit],
            cwd=repo_dir,
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            index = {}
            for git_path in result.stdout.strip().split("\n"):
                if git_path.endswith(".c") or git_path.endswith(".h"):
                    index[git_path.replace("/", "_")] = git_path
        else:
            print(f"    [WARN] git ls-tree failed rc={result.returncode} at {commit[:12]}")
    except Exception as exc:
        print(f"    [WARN] git ls-tree {type(exc).__name__} at {commit[:12]}: {exc}")

    _TREE_CACHE[key] = index
    return index


def resolve_file_path(repo_dir: str, commit: str, underscore_name: str) -> str:
    """Convert underscore-format filename back to real git path."""
    index = _tree_index(repo_dir, commit)
    if index is None:
        return None
    return index.get(underscore_name)


def run_bszz_node(node: dict, fix_sha: str, fname: str,
                  szz: BaseSZZ, repo_dir: str,
                  ground_truth: list) -> dict:
    """Run B-SZZ on a single deletion node. Mutates and returns node."""
    beg_line  = node["lineBeg"]
    end_line  = node["lineEnd"]
    mod_lines = list(range(beg_line, end_line + 1))
    impacted  = ImpactedFile(fname, mod_lines)

    node["rootcause"] = False
    node["commits"]   = []

    parent = f"{fix_sha}^"
    if not file_exists_in_commit(repo_dir, parent, fname):
        print(f"    [SKIP node] {fname}:{beg_line}-{end_line} (file not in parent)")
        node["commits"] = ["FILE_NOT_FOUND"]
        return node

    bug_commits = szz.find_bic(fix_sha, [impacted])
    bug_cids    = [c.hexsha for c in bug_commits]
    node["commits"] = bug_cids

    for cid in bug_cids:
        for gt in ground_truth:
            if cid.startswith(gt) or gt.startswith(cid):
                node["rootcause"] = True
                print(f"    ✓ Match: {cid[:12]}")
                break
        if node["rootcause"]:
            break

    print(f"    {fname}:{beg_line}-{end_line} → "
          f"{len(bug_cids)} commit(s)  "
          f"rootcause={node['rootcause']}")
    return node


def run_bszz_on_test(test_dir: str, test_name: str, repo_name: str) -> bool:
    """Run B-SZZ on the fixing commit graph for a single test case."""
    info_path = os.path.join(test_dir, "info.json")
    if not os.path.exists(info_path):
        print(f"    [SKIP] missing info.json")
        return False

    with open(info_path) as fh:
        info = json.load(fh)

    fix_sha = info.get("fix")
    if not fix_sha:
        print(f"    [SKIP] no 'fix' key in info.json")
        return False

    commit_dir = os.path.join(test_dir, fix_sha)

    # graph_addition.json is the BASELINE variant, written only for
    # addition-only cases by `generate_graphs.py --mode fix --addition-also`.
    # It exists for exactly those cases (183 of 755 on the linux set); every
    # other case falls through to graph.json, which is already built on the
    # standard deletion path and is therefore identical to what the baseline
    # would produce.
    #
    # Why this matters: for an addition-only fix, graph.json's isDel nodes are
    # SEM-SZZ ANCHOR lines that were never deleted -- TempoVIC synthesises them
    # so V-SZZ has something to trace.  Blaming those would give B-SZZ our own
    # algorithm for free; published B-SZZ cannot handle an addition-only commit
    # at all.  The baseline variant has no isDel nodes, so such a case yields no
    # candidates -- the honest result.
    addition_path = os.path.join(commit_dir, "graph_addition.json")
    graph_path    = os.path.join(commit_dir, "graph.json")

    if os.path.exists(addition_path):
        graph_path = addition_path
        print(f"    [BASELINE] using graph_addition.json (SEM-SZZ excluded)")
    elif not os.path.exists(graph_path):
        print(f"    [SKIP] graph.json missing")
        return False

    with open(graph_path) as fh:
        graph = json.load(fh)

    if len(graph) == 0:
        print(f"    [SKIP] graph.json is empty (0 nodes)")
        return False

    repo_dir      = os.path.join(REPOS_DIR, repo_name)
    ground_truth  = info.get("induce", [])
    output_path   = os.path.join(commit_dir, "graph_bszz.json")

    # Resumable: an interrupted sweep must not redo completed cases.
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"    [SKIP] graph_bszz.json already exists")
        return True

    # use_temp_dir=False is REQUIRED, not an optimisation.  The default (True)
    # does mkdtemp(dir=os.getcwd()) + copytree() of the whole repo -- 15 GB for
    # linux -- and BaseSZZ is constructed once PER TEST CASE, so a 755-case
    # sweep would copy ~11 TB into the working directory.  Safe because SZZ is
    # read-only here: find_bic() only calls _blame(), and the one mutating
    # helper (_set_working_tree_to_commit, which does head.reset(working_tree=True))
    # has no call sites anywhere in the szz package.
    szz = BaseSZZ(repo_name, None, REPOS_DIR, use_temp_dir=False)

    final_nodes     = []
    found_rootcause = False
    del_count = add_count = 0

    for node in graph:
        if node["isDel"]:
            # Resolve only for deletion nodes -- add nodes never reach B-SZZ.
            fname = resolve_file_path(repo_dir, fix_sha, node["fName"])
            if not fname:
                # Only correct when the basename has no underscore; report it
                # rather than letting a bad path turn into a silent ERROR node.
                fname = node["fName"].replace("_", "/")
                print(f"    [WARN] unresolved path, guessing {node['fName']} -> {fname}")
            try:
                node = run_bszz_node(
                    node, fix_sha, fname, szz, repo_dir, ground_truth)

                if node.get("rootcause"):
                    found_rootcause = True

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"    [ERROR] {fname}:{node['lineBeg']}-{node['lineEnd']}: {e}")
                node["commits"] = ["ERROR"]

            del_count += 1
        else:
            add_count += 1

        final_nodes.append(node)

    with open(output_path, "w") as fh:
        json.dump(final_nodes, fh, indent=2)

    print(f"    del={del_count} add={add_count} "
          f"rootcause={found_rootcause} → graph_bszz.json")
    return True


def run(projects: list, dry_run: bool = False,
        testcases: list = None, exclude: list = None):
    total = ok = failed = skip = rootcause_found = 0
    only = set(testcases) if testcases else None
    skip_set = set(exclude) if exclude else set()

    for project in projects:
        project_dir = str(GRAPH_DATA_DIR / project)
        repo_name   = PROJECT_TO_REPO[project]

        if not os.path.isdir(project_dir):
            print(f"\n[WARN] project dir not found: {project_dir}")
            continue

        test_names = sorted(
            d for d in os.listdir(project_dir)
            if os.path.isdir(os.path.join(project_dir, d))
            and (only is None or d in only)
            and d not in skip_set
        )

        print(f"\n{'='*70}")
        print(f"PROJECT: {project}  ({len(test_names)} test cases)  repo: {repo_name}")
        print(f"{'='*70}")

        proj_ok = proj_fail = proj_skip = 0

        for test_name in test_names:
            test_dir = os.path.join(project_dir, test_name)
            print(f"\n  {test_name}")

            if dry_run:
                print(f"    [DRY-RUN] would run B-SZZ")
                proj_ok += 1
                continue

            try:
                result = run_bszz_on_test(test_dir, test_name, repo_name)
                if result:
                    proj_ok += 1
                    with open(os.path.join(test_dir, "info.json")) as fh:
                        info = json.load(fh)
                    fix_sha  = info.get("fix", "")
                    out_path = os.path.join(test_dir, fix_sha, "graph_bszz.json")
                    if os.path.exists(out_path):
                        with open(out_path) as fh:
                            nodes = json.load(fh)
                        if any(n.get("rootcause") for n in nodes):
                            rootcause_found += 1
                else:
                    proj_skip += 1
            except KeyboardInterrupt:
                print("\n[INTERRUPTED]")
                raise
            except Exception as e:
                print(f"    [ERROR] {e}")
                traceback.print_exc()
                proj_fail += 1

        total  += len(test_names)
        ok     += proj_ok
        failed += proj_fail
        skip   += proj_skip

        print(f"\n  {project} summary — ok={proj_ok}  fail={proj_fail}  skip={proj_skip}")

    print(f"\n{'='*70}")
    print(f"B-SZZ COMPLETE")
    print(f"{'='*70}")
    print(f"  Total test cases  : {total}")
    print(f"  Processed (ok)    : {ok}")
    print(f"  Skipped           : {skip}")
    print(f"  Failed            : {failed}")
    print(f"  Root cause found  : {rootcause_found}")
    if ok > 0:
        print(f"  GT match rate     : {rootcause_found*100/ok:.1f}%")


if __name__ == "__main__":
    available = get_available_projects()

    parser = argparse.ArgumentParser(
        description="Run B-SZZ on fixing commit graphs"
    )
    parser.add_argument(
        "--projects", nargs="+",
        default=available,
        help=f"Which projects to process (available: {available})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without running SZZ"
    )
    parser.add_argument(
        "--testcases", nargs="+", default=None,
        help="Only process these test-case names (e.g. test1014 test872)"
    )
    parser.add_argument(
        "--exclude", nargs="+", default=None,
        help="Skip these test-case names. Use when a case's graph_addition.json "
             "is not built yet -- without it the case would silently fall back "
             "to the SEM-SZZ-contaminated graph.json, and because this script "
             "skips existing graph_bszz.json it would never be retried."
    )
    args = parser.parse_args()

    run(projects=args.projects, dry_run=args.dry_run,
        testcases=args.testcases, exclude=args.exclude)