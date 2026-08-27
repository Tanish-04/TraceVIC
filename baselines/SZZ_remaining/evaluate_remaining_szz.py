"""
Evaluate R-SZZ / L-SZZ / RA-SZZ / TSE-SZZ results (produced by
run_remaining_szz.py) against the same ground truth used for TempoVIC's own
B-SZZ/AG-SZZ/MA-SZZ/V-SZZ baseline numbers, using the identical metric
definitions as TempoVIC/baselines/SZZ/evaluate.py (12-char SHA prefix match,
global precision/recall/F1 across the dataset) so the numbers are directly
comparable in the same table. Re-implemented here (not imported) to keep
replication_package fully self-contained -- see Decisions.md.

Usage:
    venv/bin/python evaluate_remaining_szz.py --method r
    venv/bin/python evaluate_remaining_szz.py --method l
    venv/bin/python evaluate_remaining_szz.py --method ra
    venv/bin/python evaluate_remaining_szz.py --method tse
    venv/bin/python evaluate_remaining_szz.py --all   # prints + saves a combined summary table
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(HERE, "datasets", "tempovic", "verified_cve_with_versions_C.json")
RESULTS_DIR = os.path.join(HERE, "results")

# Set by main(); module-level so evaluate()/load_* keep their existing shape.
DATASET_FILE = DEFAULT_DATASET
PROJECTS = ["linux"]

METHODS = ["tse", "l", "r", "ra"]
METHOD_LABEL = {"tse": "TSE-SZZ (VCC-SZZ)", "l": "L-SZZ", "r": "R-SZZ", "ra": "RA-SZZ"}


def load_ground_truth(projects=None):
    projects = set(projects or PROJECTS)
    with open(DATASET_FILE) as f:
        data = json.load(f)
    # fix_commit -> set of ground-truth inducing commit shas (full length)
    gt_by_fix = {}
    cve_by_fix = {}
    for entry in data:
        if entry.get("project") not in projects:
            continue
        cve = entry.get("cve_id")
        for fd in entry.get("fixing_details", []):
            fix = fd.get("fixing_commit")
            if not fix:
                continue
            gt = {ic["commit_id"] for ic in fd.get("inducing_commits", []) if ic.get("commit_id")}
            gt_by_fix[fix] = gt
            cve_by_fix[fix] = cve
    return gt_by_fix, cve_by_fix


def load_results(method, time="run1", projects=None):
    """Merge one method's per-project result files into a single fix->preds map."""
    merged, paths, missing = {}, [], []
    for project in (projects or PROJECTS):
        path = os.path.join(RESULTS_DIR, f"{method}-szz", project, time,
                            f"{method}-{project}.json")
        paths.append(path)
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path) as f:
            merged.update(json.load(f))
    if missing and not merged:
        return None, ", ".join(missing)
    return merged, ", ".join(paths)


def evaluate(method, time="run1", projects=None):
    projects = projects or PROJECTS
    gt_by_fix, cve_by_fix = load_ground_truth(projects)
    results, path = load_results(method, time, projects)

    label = METHOD_LABEL[method]
    print(f"\n{'=' * 90}")
    print(f"  {label}")
    print(f"{'=' * 90}")

    if results is None:
        print(f"  [ERROR] results file not found: {path}")
        return None

    total_cves = 0
    total_matches = 0
    total_gt = 0
    total_szz = 0
    szz_failures = 0
    no_gt_count = 0
    not_yet_run = 0
    identified_cves = 0   # CVE-level: at least one GT VIC among the predictions

    for fix, gt_commits in gt_by_fix.items():
        if fix not in results:
            not_yet_run += 1
            continue

        raw = results[fix]
        if isinstance(raw, dict) and "error" in raw:
            szz_failures += 1
            continue

        szz_commits = set(raw) if isinstance(raw, list) else {raw}
        szz_commits.discard(None)

        if len(gt_commits) == 0:
            no_gt_count += 1
            continue

        total_cves += 1
        gt_short = {c[:12] for c in gt_commits}
        szz_short = {c[:12] for c in szz_commits if c}
        matches = len(gt_short & szz_short)

        total_matches += matches
        total_gt += len(gt_commits)
        total_szz += len(szz_commits)
        if matches > 0:
            identified_cves += 1

    precision = total_matches / total_szz if total_szz > 0 else 0.0
    recall = total_matches / total_gt if total_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    # F-beta with beta=2: weights recall twice as heavily as precision.
    beta = 2
    f2 = ((1 + beta**2) * precision * recall / (beta**2 * precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"  {'Fix commits with results:':<30} {len(results):>8} / {len(gt_by_fix)}")
    print(f"  {'Not yet processed:':<30} {not_yet_run:>8}  (run still in progress if > 0)")
    print(f"  {'CVEs evaluated:':<30} {total_cves:>8}")
    print(f"  {'Identified CVEs (>=1 GT hit):':<30} {identified_cves:>8}")
    print(f"  {'SZZ failures (errored):':<30} {szz_failures:>8}")
    print(f"  {'No ground truth:':<30} {no_gt_count:>8}")
    print(f"  {'Total GT commits:':<30} {total_gt:>8}")
    print(f"  {'Total SZZ-predicted commits:':<30} {total_szz:>8}")
    print(f"  {'Matched commits:':<30} {total_matches:>8}")
    print(f"  {'Precision:':<30} {precision:>8.4f}")
    print(f"  {'Recall:':<30} {recall:>8.4f}")
    print(f"  {'F1:':<30} {f1:>8.4f}")
    print(f"  {'F2:':<30} {f2:>8.4f}")

    return {
        "method": method,
        "label": label,
        "results_file_present": len(results),
        "not_yet_processed": not_yet_run,
        "cves_evaluated": total_cves,
        "identified_cves": identified_cves,
        "szz_failures": szz_failures,
        "no_gt": no_gt_count,
        "total_gt_commits": total_gt,
        "total_szz_commits": total_szz,
        "matched_commits": total_matches,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "complete": not_yet_run == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--time", default="run1")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--save", default=os.path.join(RESULTS_DIR, "evaluation_summary.json"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="dataset JSON (default: the 755-case linux set)")
    parser.add_argument("--projects", nargs="+", default=["linux"],
                        help="project keys to pool over (default: linux)")
    args = parser.parse_args()

    global DATASET_FILE, PROJECTS
    DATASET_FILE = (args.dataset if os.path.isabs(args.dataset)
                    else os.path.join(HERE, args.dataset))
    PROJECTS = args.projects

    if args.all or not args.method:
        summary = []
        for m in METHODS:
            r = evaluate(m, args.time, PROJECTS)
            if r:
                summary.append(r)

        print(f"\n{'=' * 90}")
        print("  SUMMARY TABLE (append to your document)")
        print(f"{'=' * 90}")
        print(f"  {'Method':<22} {'Status':<12} {'CVEs':<7} {'IdCVE':<7} {'Precision':<11} {'Recall':<9} {'F1':<8} {'F2':<8}")
        print(f"  {'-' * 94}")
        for r in summary:
            status = "COMPLETE" if r["complete"] else f"PARTIAL ({r['results_file_present']}/{r['results_file_present']+r['not_yet_processed']})"
            print(f"  {r['label']:<22} {status:<12} {r['cves_evaluated']:<7} {r['identified_cves']:<7} {r['precision']:<11.4f} {r['recall']:<9.4f} {r['f1']:<8.4f} {r['f2']:<8.4f}")

        with open(args.save, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved machine-readable summary to {args.save}")
    else:
        evaluate(args.method, args.time, PROJECTS)


if __name__ == "__main__":
    main()
