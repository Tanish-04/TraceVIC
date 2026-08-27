"""
Requires:
  - graph.json must exist and be non-empty for the fixing commit
  - Run after generate_graphs.py with --mode fix.

Output per test case:
    <test_dir>/<fix_sha>/graph_vszz.json

Usage:
    python history_commits_retrieval.py --projects linux
    python history_commits_retrieval.py --projects FFmpeg OpenSSL PHP-SRC ImageMagick
    python history_commits_retrieval.py --projects all
"""

import os
import sys
import json
import argparse
import subprocess
import traceback
from pathlib import Path

GRAPH_CONSTRUCTION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GRAPH_CONSTRUCTION_DIR))

from config_loader import get_all_projects, get_pyszz_path
from pipeline_utils import (
    DATA_DIR,
    REPOS_DIR,
    PROJECT_TO_REPO,
    resolve_underscore_name_to_git_path,
)

# V-SZZ imports 
sys.path.insert(0, str(get_pyszz_path()))

from szz.my_szz_ast import MySZZAST
from szz.b_szz import BaseSZZ
from szz.core.abstract_szz import ImpactedFile


def file_exists_in_commit_tree(repo_dir: str, commit: str, path: str) -> bool:
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



def bszz_history_results(szz, fix_sha: str, fname: str, mod_lines: list) -> list:
    """Adapt B-SZZ blame output into V-SZZ's find_bic() return shape.

    V-SZZ yields, per blamed line:
        {"line_num", "line_str", "file_path",
         "previous_commits": [(sha, line_num, code, file_path), ...]}   # the CHAIN

    B-SZZ has no chain -- it is ONE `git blame` hop.  BaseSZZ.find_bic() also
    discards the BlameData and keeps only SHAs, so we call _blame() directly
    with the arguments find_bic() uses (b_szz.py) -- identical blame, identical
    commit set -- and keep line_num/line_str/file_path.  find_history_node()
    matches a chain entry into that commit's graph.json by line-number range
    FIRST, so the real historical line number matters: the fix-time number is
    wrong for the older file (measured 164 lines off on test1004) and can
    silently match the WRONG node.

    Blame entries for the same fix line are collapsed into ONE
    `previous_commits` list at successive depths, so multiple blame commits stay
    distinct rankable candidates.  Emitting them as separate chains would put
    them all at temporal_pos=1, where build_tp_to_commit()'s first-chain-wins
    map (`if tp not in tp_map`) would drop every one after the first while its
    nodes still entered the graph tagged tp=1 and got mislabelled.  Affects 1.1%
    of deletion lines.  Note these depths carry no chronological meaning for
    B-SZZ -- the commits are alternatives, not a history.
    """
    blame_data = szz._blame(
        rev=f"{fix_sha}^",
        file_path=fname,
        modified_lines=mod_lines,
        ignore_revs_file_path=None,
        ignore_whitespaces=False,
        skip_comments=False,
    )

    by_line = {}
    for entry in blame_data:
        rec = by_line.setdefault(entry.line_num, {
            "line_num":         entry.line_num,
            "line_str":         entry.line_str,
            "file_path":        entry.file_path,
            "previous_commits": [],
            "_seen":            set(),
        })
        sha = entry.commit.hexsha
        if sha in rec["_seen"]:
            continue
        rec["_seen"].add(sha)
        rec["previous_commits"].append(
            (sha, entry.line_num, entry.line_str, entry.file_path))

    out = []
    for rec in by_line.values():
        rec.pop("_seen", None)
        out.append(rec)
    return out


def run_vszz_on_test(test_dir: str, test_name: str, repo_name: str,
                     szz_kind: str = "vszz") -> bool:
    """
    Run V-SZZ on the fixing commit graph for a single test case.
    Writes graph_vszz.json next to graph.json.
    Returns True on success, False on skip/failure.
    """
    info_path = os.path.join(test_dir, "info.json")
    if not os.path.exists(info_path):
        print("    [SKIP] missing info.json")
        return False

    with open(info_path) as fh:
        info = json.load(fh)

    fix_sha = info.get("fix")
    if not fix_sha:
        print("    [SKIP] no 'fix' key in info.json")
        return False

    commit_dir = os.path.join(test_dir, fix_sha)
    graph_path = os.path.join(commit_dir, "graph.json")

    # For B-SZZ, prefer graph_addition.json when it exists (the 183 addition-only
    # cases). graph.json's isDel nodes there are SEM-SZZ ANCHOR lines that were
    # never deleted -- a TempoVIC contribution. Tracing them with B-SZZ would
    # smuggle SEM-SZZ back into an arm defined as "no SEM-SZZ", and the case
    # would look solvable when published B-SZZ cannot handle it at all.
    if szz_kind == "bszz":
        addition_path = os.path.join(commit_dir, "graph_addition.json")
        if os.path.exists(addition_path):
            graph_path = addition_path
            print("    [BSZZ] using graph_addition.json (SEM-SZZ excluded)")

    if not os.path.exists(graph_path):
        print("    [SKIP] graph.json missing — run step 3 first")
        return False

    with open(graph_path) as fh:
        graph = json.load(fh)

    if len(graph) == 0:
        print("    [SKIP] graph.json is empty (0 nodes)")
        return False

    repo_dir    = os.path.join(REPOS_DIR, repo_name)
    out_name    = "graph_vszz.json" if szz_kind == "vszz" else "graph_bszz_chains.json"
    output_path = os.path.join(commit_dir, out_name)

    # Skip if already processed
    if os.path.exists(output_path):
        print(f"    [SKIP] {out_name} already exists")
        return True

    if szz_kind == "vszz":
        szz = MySZZAST(
            repo_full_name=repo_name,
            repo_url=None,
            repos_dir=REPOS_DIR,
            use_temp_dir=False,
            ast_map_path=None
        )
    else:
        # B-SZZ: one blame hop, no chain. Everything downstream of
        # history_results is shared with the V-SZZ path.
        szz = BaseSZZ(repo_name, None, REPOS_DIR, use_temp_dir=False)

    ground_truth    = info.get("induce", [])
    final_nodes     = []
    found_rootcause = False
    del_count = add_count = 0

    for node in graph:
        fname = resolve_underscore_name_to_git_path(repo_dir, fix_sha, node["fName"])
        if not fname:
            fname = node["fName"].replace("_", "/")

        beg_line = node["lineBeg"]
        end_line = node["lineEnd"]
        mod_lines = list(range(beg_line, end_line + 1))
        impacted  = ImpactedFile(fname, mod_lines)

        node["rootcause"]          = False
        node["commits"]            = []
        node["history_chains"]     = []
        node["introducer_commits"] = []

        if node["isDel"]:
            try:
                parent = f"{fix_sha}^"
                if not file_exists_in_commit_tree(repo_dir, parent, fname):
                    print(f"    [SKIP node] {fname}:{beg_line}-{end_line} "
                          f"(file not in parent)")
                    node["commits"] = ["FILE_NOT_FOUND"]
                    final_nodes.append(node)
                    del_count += 1
                    continue

                if szz_kind == "vszz":
                    history_results = szz.find_bic(fix_sha, [impacted])
                else:
                    history_results = bszz_history_results(
                        szz, fix_sha, fname, mod_lines)
                all_commits        = set()
                introducer_commits = set()

                for line_history in history_results:
                    previous_commits = line_history["previous_commits"]
                    chain_commits    = [c[0] for c in previous_commits]
                    all_commits.update(chain_commits)

                    if previous_commits:
                        introducer_commits.add(previous_commits[-1][0])

                    node["history_chains"].append({
                        "fix_line": line_history["line_num"],
                        "code":     line_history["line_str"][:100],
                        "history": [
                            {
                                "commit":    c[0][:12],
                                "file_path": c[3] if len(c) > 3 else line_history.get("file_path", ""),
                                "line_num":  c[1],
                                "code":      c[2][:80] if len(c) > 2 else ""
                            }
                            for c in previous_commits
                        ],
                        "introducer": previous_commits[-1][0][:12] if previous_commits else "UNKNOWN"
                    })

                node["commits"]            = list(all_commits)
                node["introducer_commits"] = list(introducer_commits)

                candidates = all_commits | introducer_commits
                for candidate in candidates:
                    for gt in ground_truth:
                        if candidate.startswith(gt) or gt.startswith(candidate):
                            found_rootcause   = True
                            node["rootcause"] = True
                            print(f"    ✓ Match: {candidate[:12]}")
                            break
                    if node["rootcause"]:
                        break

                print(f"    {fname}:{beg_line}-{end_line} → "
                      f"{len(introducer_commits)} introducer(s)  "
                      f"rootcause={node['rootcause']}")

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"    [ERROR] {fname}:{beg_line}-{end_line}: {e}")
                node["commits"] = ["ERROR"]

            del_count += 1

        else:
            add_count += 1

        final_nodes.append(node)

    with open(output_path, "w") as fh:
        json.dump(final_nodes, fh, indent=2)

    print(f"    del={del_count} add={add_count} "
          f"rootcause={found_rootcause} → {out_name}")
    return True


def run(projects: list, szz_kind: str = "vszz") -> None:
    total           = 0
    ok              = 0
    failed          = 0
    skip            = 0
    rootcause_found = 0

    for project in projects:
        project_dir = os.path.join(DATA_DIR, project)
        repo_name   = PROJECT_TO_REPO[project]

        if not os.path.isdir(project_dir):
            print(f"\n[WARN] project dir not found: {project_dir}")
            continue

        test_names = sorted(
            d for d in os.listdir(project_dir)
            if os.path.isdir(os.path.join(project_dir, d))
        )

        print(f"\n{'='*70}")
        print(f"PROJECT: {project}  ({len(test_names)} test cases)  repo: {repo_name}")
        print(f"{'='*70}")

        proj_ok = proj_fail = proj_skip = 0

        for test_name in test_names:
            test_dir = os.path.join(project_dir, test_name)
            print(f"\n  {test_name}")

            try:
                result = run_vszz_on_test(test_dir, test_name, repo_name,
                                          szz_kind=szz_kind)
                if result:
                    proj_ok += 1
                    info_path = os.path.join(test_dir, "info.json")
                    with open(info_path) as fh:
                        info = json.load(fh)
                    fix_sha   = info.get("fix", "")
                    vszz_path = os.path.join(
                        test_dir, fix_sha,
                        "graph_vszz.json" if szz_kind == "vszz"
                        else "graph_bszz_chains.json")
                    if os.path.exists(vszz_path):
                        with open(vszz_path) as fh:
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

        print(f"\n  {project} summary — "
              f"ok={proj_ok}  fail={proj_fail}  skip={proj_skip}")

    print(f"\n{'='*70}")
    print("STEP 4 COMPLETE")
    print(f"{'='*70}")
    print(f"  Total test cases  : {total}")
    print(f"  Processed (ok)    : {ok}")
    print(f"  Skipped           : {skip}")
    print(f"  Failed            : {failed}")
    print(f"  Root cause found  : {rootcause_found}")
    if ok > 0:
        print(f"  GT match rate     : {rootcause_found*100/ok:.1f}%")


if __name__ == "__main__":
    all_project_names = get_all_projects()

    parser = argparse.ArgumentParser(
        description="Run V-SZZ (or B-SZZ) on fixing commit graphs"
    )
    parser.add_argument(
        "--szz", choices=["vszz", "bszz"], default="vszz",
        help="Chain backend. 'vszz' (default) walks the full history and writes "
             "graph_vszz.json -- TempoVIC's normal candidates. 'bszz' takes ONE "
             "blame hop and writes graph_bszz_chains.json, for the "
             "TempoVIC-pipeline-with-B-SZZ-candidates arm; it also prefers "
             "graph_addition.json so SEM-SZZ anchors are excluded."
    )
    parser.add_argument(
        "--projects", nargs="+",
        default=all_project_names,
        choices=all_project_names,
        help="Which projects to process (default: all configured projects)"
    )
    args = parser.parse_args()

    run(projects=args.projects, szz_kind=args.szz)