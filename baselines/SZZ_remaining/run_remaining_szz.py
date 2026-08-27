"""
Driver to run the remaining SZZ methods (R-SZZ, L-SZZ, RA-SZZ, TSE-SZZ) from
method_replication_scripts/ against the TempoVIC linux CVE dataset that has
been copied into this project (datasets/tempovic/, repos/linux).

Lives entirely inside replication_package/ -- does not import or write
anything under /mnt/data/TempoVIC/. See Decisions.md for why.

Output format mirrors TempoVIC's baselines/SZZ results layout so evaluate.py
there could later diff against it if wanted:
    results/<method>-szz/linux/run1/<method>-linux.json  -> {fix_sha: [inducing_shas]}
    results/<method>-szz/linux/run1/<method>-linux-progress.json -> [completed fix_shas]

Usage:
    venv/bin/python run_remaining_szz.py --method r   --time run1
    venv/bin/python run_remaining_szz.py --method l   --time run1
    venv/bin/python run_remaining_szz.py --method ra  --time run1
    venv/bin/python run_remaining_szz.py --method tse --time run1
"""
import argparse
import json
import logging as log
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(HERE, "method_replication_scripts")
sys.path.insert(0, SCRIPTS_DIR)
# ra_szz.py hardcodes PATH_TO_REFMINER as "../tools/RefactoringMiner-2.0/..."
# relative to CWD, assuming it's run from method_replication_scripts/. Match
# that assumption rather than patch vendored code.
os.chdir(SCRIPTS_DIR)

DEFAULT_DATASET = os.path.join(HERE, "datasets", "tempovic", "verified_cve_with_versions_C.json")
REPOS_DIR = os.path.join(HERE, "repos")
RESULTS_DIR = os.path.join(HERE, "results")

# Project -> clone URL. Only used by the MASZZ-family constructors, which accept
# a URL but never fetch when the repo already exists under repos_dir.
REPO_URLS = {
    "linux":       "https://github.com/torvalds/linux.git",
    "FFmpeg":      "https://github.com/FFmpeg/FFmpeg.git",
    "ImageMagick": "https://github.com/ImageMagick/ImageMagick.git",
    "OpenSSL":     "https://github.com/openssl/openssl.git",
    "PHP-SRC":     "https://github.com/php/php-src.git",
}

# Set by main() from --project / --dataset. Module-level because the run_*
# helpers below were written against these names; keeping them avoids
# rewriting vendored-adjacent code paths that are already validated.
DATASET_FILE = DEFAULT_DATASET
REPO_FULL_NAME = "linux"
REPO_URL = REPO_URLS["linux"]

EXTS = ["c", "h"]


def load_fix_commits(project=REPO_FULL_NAME):
    """Return list of (cve_id, fix_commit_sha) pairs for one project, deduplicated by fix commit."""
    with open(DATASET_FILE) as f:
        data = json.load(f)

    pairs = []
    seen = set()
    for entry in data:
        if entry.get("project") != project:
            continue
        cve = entry.get("cve_id")
        for fd in entry.get("fixing_details", []):
            sha = fd.get("fixing_commit")
            if sha and sha not in seen:
                seen.add(sha)
                pairs.append((cve, sha))
    return pairs


def get_paths(method, time, project=None):
    project = project or REPO_FULL_NAME
    out_dir = os.path.join(RESULTS_DIR, f"{method}-szz", project, time)
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"{method}-{project}.json")
    progress_file = os.path.join(out_dir, f"{method}-{project}-progress.json")
    log_file = os.path.join(out_dir, f"{method}-{project}-log.txt")
    return output_file, progress_file, log_file


def load_json_or(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def run_ma_family(method, fix_commits, output, completed):
    """R-SZZ / L-SZZ, both subclass MASZZ, both take a single persistent instance
    (repos_dir points at our already-cloned repo -> AbstractSZZ uses it in place,
    no clone/copy needed since repo_dir already exists)."""
    if method == "r":
        from r_szz import RSZZ as Cls
    elif method == "l":
        from l_szz import LSZZ as Cls
    else:
        raise ValueError(method)

    szz = Cls(repo_full_name=REPO_FULL_NAME, repo_url=REPO_URL, repos_dir=REPOS_DIR)
    # SAFETY: AbstractSZZ.__del__ rmtree's the *repos_dir itself* when
    # use_temp_dir (default True, not exposed by this subclass's __init__) is
    # truthy at cleanup time -- see Decisions.md incident log. Disable it so
    # our persistent repo copy survives process exit.
    szz.use_temp_dir = False

    for i, (cve, commit) in enumerate(fix_commits):
        if commit in completed:
            continue
        print(f"[{method}-szz] ({i+1}/{len(fix_commits)}) {cve} fix={commit}")
        try:
            imp_files = szz.get_impacted_files(
                fix_commit_hash=commit, file_ext_to_parse=EXTS, only_deleted_lines=True
            )
            bic = szz.find_bic(
                fix_commit_hash=commit, impacted_files=imp_files, ignore_revs_file_path=None
            )
            output[commit] = [c.hexsha for c in bic if c]
        except Exception as e:
            log.error(f"FAILED on {commit}: {e}")
            traceback.print_exc()
            output[commit] = {"error": str(e)}
        completed.append(commit)
        yield commit


def run_ra_szz(fix_commits, output, completed):
    from ra_szz import RASZZ

    szz = RASZZ(repo_full_name=REPO_FULL_NAME, repo_url=REPO_URL, repos_dir=REPOS_DIR)
    szz.use_temp_dir = False  # see Decisions.md incident log

    for i, (cve, commit) in enumerate(fix_commits):
        if commit in completed:
            continue
        print(f"[ra-szz] ({i+1}/{len(fix_commits)}) {cve} fix={commit}")
        try:
            imp_files = szz.get_impacted_files(
                fix_commit_hash=commit, file_ext_to_parse=EXTS, only_deleted_lines=True
            )
            bic = szz.find_bic(
                fix_commit_hash=commit, impacted_files=imp_files, ignore_revs_file_path=None
            )
            output[commit] = [c.hexsha for c in bic if c]
        except Exception as e:
            log.error(f"FAILED on {commit}: {e}")
            traceback.print_exc()
            output[commit] = {"error": str(e)}
        completed.append(commit)
        yield commit


def run_tse_szz(fix_commits, output, completed):
    import tse_szz
    from pydriller import Git

    repo_path = os.path.join(REPOS_DIR, REPO_FULL_NAME)
    git_repo = Git(repo_path)

    for i, (cve, commit) in enumerate(fix_commits):
        if commit in completed:
            continue
        print(f"[tse-szz] ({i+1}/{len(fix_commits)}) {cve} fix={commit}")
        try:
            fix_commit_obj = git_repo.get_commit(commit)
            vccs = tse_szz.get_contributing_commits(git_repo, fix_commit_obj)
            output[commit] = list(vccs.keys())
        except Exception as e:
            log.error(f"FAILED on {commit}: {e}")
            traceback.print_exc()
            output[commit] = {"error": str(e)}
        completed.append(commit)
        yield commit
    git_repo.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["r", "l", "ra", "tse"])
    parser.add_argument("--time", default="run1")
    parser.add_argument("--limit", type=int, default=None, help="only process first N commits (smoke test)")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--project", default="linux",
                        help="project key in the dataset AND directory name under repos/")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="dataset JSON (default: the 755-case linux set)")
    args = parser.parse_args()

    global DATASET_FILE, REPO_FULL_NAME, REPO_URL
    # This module os.chdir()s to SCRIPTS_DIR at import (ra_szz needs it), so a
    # relative --dataset would resolve against the wrong directory. Resolve it
    # against the package root, which is what the caller means.
    DATASET_FILE = (args.dataset if os.path.isabs(args.dataset)
                    else os.path.join(HERE, args.dataset))
    if not os.path.exists(DATASET_FILE):
        sys.exit(f"dataset not found: {DATASET_FILE}")
    REPO_FULL_NAME = args.project
    REPO_URL = REPO_URLS.get(args.project, "")
    repo_path = os.path.join(REPOS_DIR, REPO_FULL_NAME)
    if not os.path.isdir(repo_path):
        sys.exit(f"repo not found: {repo_path}")

    output_file, progress_file, log_file = get_paths(args.method, args.time, args.project)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log.basicConfig(
        level=log.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[log.FileHandler(log_file), log.StreamHandler(sys.stdout)],
    )

    fix_commits = load_fix_commits(args.project)
    if args.limit:
        fix_commits = fix_commits[: args.limit]
    print(f"Total distinct {args.project} fix commits to process: {len(fix_commits)}")

    output = load_json_or(output_file, {})
    completed = load_json_or(progress_file, [])
    print(f"Already completed: {len(completed)}")

    if args.method in ("r", "l"):
        gen = run_ma_family(args.method, fix_commits, output, completed)
    elif args.method == "ra":
        gen = run_ra_szz(fix_commits, output, completed)
    elif args.method == "tse":
        gen = run_tse_szz(fix_commits, output, completed)
    else:
        raise ValueError(args.method)

    n_since_save = 0
    for _ in gen:
        n_since_save += 1
        if n_since_save >= args.save_every:
            save_json(output, output_file)
            save_json(completed, progress_file)
            n_since_save = 0

    save_json(output, output_file)
    save_json(completed, progress_file)
    print(f"Done. Wrote {output_file}")


if __name__ == "__main__":
    main()
