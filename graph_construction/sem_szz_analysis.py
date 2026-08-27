"""
sem_szz_analysis.py
--------------------
Anchor-line extraction for addition-only fixing commits.

APPROACH (replaces the previous SEM-SZZ state-comparison implementation)
-------------------------------------------------------------------------
Analysis of 9 real CVEs confirms that every addition-only fix follows the
same universal pattern:

    [anchor operation]        <- already present in before/
    [missing companion]       <- absent in before/, ADDED by the fix
    [downstream operation]    <- already present in before/

The vulnerability is always an ABSENT statement, never a wrong one.
SEM-SZZ state-comparison cannot detect absence, which is why the previous
implementation found root-causes in only 3/304 cases.

The correct approach is:
  1. For each group of added lines in the fix, find the nearest non-noise
     line ABOVE and BELOW the insertion point in the after/ file.
  2. Also extract the enclosing function signature as a fallback anchor.
     Analysis of failing cases (CVE-2017-0750, CVE-2016-9576, CVE-2016-5728,
     CVE-2016-3689) shows two failure modes:
       A) all immediate neighbors are noise (large addition deep in body)
       B) neighbors exist but predate or postdate the VIC; the function
          signature is introduced by the VIC and virtually never modified
          afterwards, making it the most reliable anchor.
  3. Filter out noise lines (goto, return -ERRNO, labels, braces, etc.).
  4. Cap at MAX_ANCHORS_PER_FILE to prevent V-SZZ candidate explosion.
  5. Map anchor line numbers from after/ coordinates to before/ coordinates.
  6. Return as PatchLine(isDel=True) -- the existing V-SZZ pipeline traces
     these anchor lines back through history to find the VIC.

WHY THIS FINDS THE VIC
-----------------------
The VIC is always the commit that introduced the anchor operation without
its required companion.  V-SZZ on the anchor line traces exactly to that
commit.  Phase 2 then ranks the resulting candidate pool.

PUBLIC INTERFACE
----------------
run_sem_szz_analysis() has the same signature as before so union_project.py
requires zero changes.  The joern path and n_hops parameters are kept for
API compatibility but are no longer used.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Set, Tuple

from parse_patch.patch import Patch
from parse_patch.patch_line import PatchLine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Maximum anchor lines forwarded to V-SZZ per file.
# Each block contributes up to 3 anchors (above, below, signature).
# After deduplication this stays manageable; Phase 2 ranks the candidates.
MAX_ANCHORS_PER_FILE = 6

# How many lines up/down to search from a block boundary for a neighbor anchor.
SEARCH_RADIUS = 20

# How many lines upward to scan when searching for enclosing function signature.
SIGNATURE_SCAN_RADIUS = 300


# ---------------------------------------------------------------------------
# Public entry point  (identical signature to the old implementation)
# ---------------------------------------------------------------------------

def run_sem_szz_analysis(
    before_path:       str,
    after_path:        str,
    patch_parser:      Patch,
    before_joern_path: str,       # kept for API compatibility -- not used
    after_joern_path:  str,       # kept for API compatibility -- not used
    n_hops:            int = 3,   # kept for API compatibility -- not used
) -> Dict[str, List[PatchLine]]:
    """
    Extract anchor lines from an addition-only fixing commit.

    For each changed file, identifies the lines in before/ that the fix
    inserted its companion next to.  These are the "anchor" lines -- the
    operations that were written without their required guard/cleanup/check.

    Returns
    -------
    Dict mapping file_key -> list of PatchLine(isDel=True, lineno=<before/ line>)
    consumed by CProject.override_lines when addition_mode=True.
    """
    results: Dict[str, List[PatchLine]] = {}

    for file_key, add_plines in patch_parser.addMap.items():
        if not add_plines:
            continue
        if not (file_key.endswith(".c") or file_key.endswith(".h")):
            continue

        add_line_numbers = sorted(
            {pl.lineno for pl in add_plines if pl.lineno > 0}
        )
        if not add_line_numbers:
            continue

        # Locate the after/ source file
        after_file = _resolve_file(after_path, file_key)
        if after_file is None:
            logger.debug("Anchor: skipping %s -- not found in after/", file_key)
            continue

        try:
            with open(after_file, encoding="utf-8", errors="ignore") as fh:
                after_lines = fh.readlines()
        except OSError:
            continue

        add_set     = set(add_line_numbers)
        total_lines = len(after_lines)

        # Build after/ -> before/ line number mapping
        line_cur2pre = _compute_cur2pre_mapping(add_set, total_lines)

        # Group added lines into contiguous insertion blocks
        blocks = _group_into_blocks(add_line_numbers)

        # For each block collect anchor candidates
        anchor_candidates: List[int] = []   # in after/ coordinates

        for block_start, block_end in blocks:

            # 1. Nearest non-noise neighbor ABOVE the block
            above = _find_anchor(
                after_lines, block_start - 1, add_set,
                direction="up",
            )

            # 2. Nearest non-noise neighbor BELOW the block
            below = _find_anchor(
                after_lines, block_end + 1, add_set,
                direction="down",
                total_lines=total_lines,
            )

            # Prefer above first: in Sub-family B (missing cleanup) the
            # anchor (resource acquisition) is always above the insertion.
            # In Sub-family A (missing guard) the dangerous call is below.
            if above is not None:
                anchor_candidates.append(above)
            if below is not None:
                anchor_candidates.append(below)

            # 3. Enclosing function signature (fallback anchor).
            #    The VIC always introduces the function that contains the
            #    missing check/cleanup.  The signature is rarely modified
            #    after introduction, so git blame on it reliably reaches
            #    the VIC even when body lines have been touched later.
            sig = _find_enclosing_function_signature(
                after_lines, block_start, add_set
            )
            if sig is not None:
                anchor_candidates.append(sig)

        # Deduplicate preserving order (above-first, then below, then sig)
        seen:    Set[int]  = set()
        deduped: List[int] = []
        for ln in anchor_candidates:
            if ln not in seen:
                seen.add(ln)
                deduped.append(ln)

        # Cap to control V-SZZ candidate pool size
        anchors = deduped[:MAX_ANCHORS_PER_FILE]

        if not anchors:
            logger.debug("Anchor: no valid anchors found for %s", file_key)
            continue

        # Map anchor lines from after/ coordinates to before/ coordinates
        before_line_nums: List[int] = []
        for cur_ln in anchors:
            pre_ln = line_cur2pre.get(cur_ln)
            if pre_ln is not None and pre_ln > 0:
                before_line_nums.append(pre_ln)

        if before_line_nums:
            deduped_before = sorted(set(before_line_nums))
            # Normalise key to underscore-flat format ("kernel/exit.c" ->
            # "kernel_exit.c") to match the us_key lookup in project.py.
            us_key = file_key.replace("/", "_")
            logger.info(
                "Anchor: %s -> %d anchor line(s) %s (before/ coords)",
                us_key, len(deduped_before), deduped_before,
            )
            results[us_key] = _to_patch_lines(deduped_before)

    return results


# ---------------------------------------------------------------------------
# Neighbor anchor search
# ---------------------------------------------------------------------------

def _find_anchor(
    after_lines: List[str],
    start_line:  int,
    add_set:     Set[int],
    direction:   str,
    total_lines: int = 0,
) -> Optional[int]:
    """
    Walk away from a block boundary until a non-noise, non-added line is found.
    Returns the 1-indexed line number in after/ coordinates, or None.
    """
    if direction == "up":
        candidates = range(
            start_line,
            max(0, start_line - SEARCH_RADIUS) - 1,
            -1,
        )
    else:
        end = total_lines if total_lines > 0 else start_line + SEARCH_RADIUS
        candidates = range(
            start_line,
            min(end, start_line + SEARCH_RADIUS) + 1,
        )

    for ln in candidates:
        if ln <= 0:
            break
        if total_lines > 0 and ln > total_lines:
            break
        if ln in add_set:
            continue
        if ln > len(after_lines):
            continue
        content = after_lines[ln - 1].strip()
        if not _is_noise(content):
            return ln

    return None


# ---------------------------------------------------------------------------
# Enclosing function signature extraction
# ---------------------------------------------------------------------------

# Control-flow keywords that open a block but are NOT a function signature
_CTRL_FLOW = frozenset({
    'if', 'else', 'while', 'for', 'switch', 'do',
    'return', 'goto', 'case', 'default',
})


def _is_function_signature(line: str) -> bool:
    """
    Return True if this line looks like a C function definition signature.

    Criteria (all must hold):
      - Contains '(' (has a parameter list)
      - Does NOT end with ';' (that would be a prototype or call statement)
      - Does NOT end with ',' (continuation argument)
      - First token is not a control-flow keyword
      - Is not a comment or preprocessor line
    """
    if not line or '(' not in line:
        return False
    stripped = line.rstrip()
    if stripped.endswith(';') or stripped.endswith(','):
        return False
    if line.startswith('//') or line.startswith('*') or line.startswith('/*'):
        return False
    if line.startswith('#'):
        return False
    first = line.split()[0].lstrip('(').lstrip('*') if line.split() else ''
    if first in _CTRL_FLOW:
        return False
    return True


def _find_enclosing_function_signature(
    after_lines: List[str],
    block_start:  int,
    add_set:      Set[int],
) -> Optional[int]:
    """
    Scan upward from block_start using brace-depth tracking to find the
    opening '{' of the enclosing function, then return the function
    signature line just above (or on) that opening brace.

    Brace depth convention when scanning UPWARD:
      - Each '{' encountered DECREASES depth (we are leaving a block)
      - Each '}' encountered INCREASES depth (we are entering a block)
    When depth goes negative we have passed the function's opening '{'.

    Returns 1-indexed line number in after/ coordinates, or None.
    """
    depth = 0

    for ln in range(block_start - 1, max(0, block_start - SIGNATURE_SCAN_RADIUS - 1), -1):
        if ln <= 0 or ln > len(after_lines):
            continue

        stripped = after_lines[ln - 1].strip()

        # Update brace depth (scanning upward: { decreases, } increases)
        depth += stripped.count('}') - stripped.count('{')

        # When depth goes negative we have passed the function opening brace.
        # Search a short window starting from this line (inclusive) upward
        # for the function signature.  Starting from ln (not ln-1) handles
        # the common single-line case: "static int foo(...) {"
        if depth < 0:
            for sig_ln in range(ln, max(0, ln - 15), -1):
                if sig_ln <= 0 or sig_ln > len(after_lines):
                    continue
                if sig_ln in add_set:
                    continue
                sig_line = after_lines[sig_ln - 1].strip()
                if not sig_line:
                    continue
                if _is_function_signature(sig_line):
                    return sig_ln
            break

    return None


# ---------------------------------------------------------------------------
# Noise filter
# ---------------------------------------------------------------------------

_NOISE_PATTERNS: List[re.Pattern] = [
    re.compile(r'^goto\s+\w+\s*;$'),                       # goto label;
    re.compile(r'^return\s+[-\w]+\s*;$'),                  # return -ERRNO; / return 0;
    re.compile(r'^(ret|err|rc|retval)\s*=\s*-\w+\s*;$'),  # ret = -ENOMEM;
    re.compile(r'^(ret|err|rc|retval)\s*=\s*\w+\s*;$'),   # ret = val;
    re.compile(r'^[a-zA-Z_]\w*\s*:\s*$'),                  # label:
    re.compile(r'^(case\s+.+|default\s*):\s*$'),           # case X: / default:
    re.compile(r'^break\s*;$'),                            # break;
    re.compile(r'^continue\s*;$'),                         # continue;
]

_NOISE_EXACT: Set[str] = {
    '{', '}', '};', ')', ');',
    'else', 'else {', '} else {', '} else', '} else{',
    'do {', 'do{',
}


def _is_noise(line: str) -> bool:
    """
    Return True if this line is structural/error-handling boilerplate and
    should NOT be used as a neighbor anchor.
    """
    if not line:
        return True
    if line in _NOISE_EXACT:
        return True
    if (line.startswith('//') or line.startswith('*')
            or line.startswith('/*') or line.startswith('#')):
        return True
    for pattern in _NOISE_PATTERNS:
        if pattern.match(line):
            return True
    return False


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _group_into_blocks(line_numbers: List[int]) -> List[Tuple[int, int]]:
    """
    Group sorted line numbers into contiguous (start, end) pairs.
    Example: [953, 954, 955, 960, 961] -> [(953, 955), (960, 961)]
    """
    if not line_numbers:
        return []
    blocks: List[Tuple[int, int]] = []
    start = end = line_numbers[0]
    for ln in line_numbers[1:]:
        if ln == end + 1:
            end = ln
        else:
            blocks.append((start, end))
            start = end = ln
    blocks.append((start, end))
    return blocks


def _compute_cur2pre_mapping(
    add_set:     Set[int],
    total_lines: int,
) -> Dict[int, int]:
    """
    Build a mapping  after/ line number -> before/ line number.
    Non-added line N in after/ maps to N minus the count of added lines
    at positions <= N.  Added lines have no before/ counterpart.
    """
    cur2pre: Dict[int, int] = {}
    pre_line = 1
    for cur_line in range(1, total_lines + 1):
        if cur_line not in add_set:
            cur2pre[cur_line] = pre_line
            pre_line += 1
    return cur2pre


def _resolve_file(base_path: str, file_key: str) -> Optional[str]:
    """
    Find the source file under base_path.
    Tries full nested path, then underscore-flattened, then basename only.
    """
    full = os.path.join(base_path, file_key)
    if os.path.exists(full):
        return full
    flat_us = os.path.join(base_path, file_key.replace("/", "_"))
    if os.path.exists(flat_us):
        return flat_us
    flat = os.path.join(base_path, os.path.basename(file_key))
    if os.path.exists(flat):
        return flat
    return None


def _to_patch_lines(line_numbers: List[int]) -> List[PatchLine]:
    """
    Wrap line numbers as PatchLine(isDel=True) for the V-SZZ pipeline.
    isDel=True means 'trace this line back through history via V-SZZ'.
    """
    return [
        PatchLine("", isDel=True, isAdd=False, isBg=False, lineno=ln)
        for ln in line_numbers
        if ln > 0
    ]