#!/usr/bin/env bash
# audit-citations.sh — sanity-check the file:line citations in this skill's references/.
#
# WHAT IT CATCHES (fully automatic):
#   - a cited file that no longer exists (renamed/moved/deleted)
#   - a cited line number that is now past the end of the file (gross drift / truncation)
#   - an ambiguous basename that resolves to >1 path in the repo
#
# WHAT IT CANNOT CATCH (needs a human eyeball):
#   - content drift where the line still exists but the code moved a few lines
#     (e.g. transformer.cc gaining instrumentation shifting every anchor by ~8).
#   For that, the script PRINTS the current content of each cited line so you can
#   scan one screen instead of hand-grepping every citation. A "verified on DATE"
#   stamp is worth nothing the moment the file is touched again — only a fresh run is.
#
# Usage:  bash scripts/audit-citations.sh [-q]
#   -q   quiet: only print FAIL / AMBIGUOUS lines (good for CI / a fast gate)
#
# Exit status: number of hard failures (missing file / out-of-range line), capped at 255.

set -uo pipefail

QUIET=0
[[ "${1:-}" == "-q" ]] && QUIET=1

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)" || {
  echo "not in a git repo" >&2; exit 255; }
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF_DIR="$SKILL_DIR/references"
[[ -d "$REF_DIR" ]] || { echo "no references/ dir at $REF_DIR" >&2; exit 255; }

# Include roots a bare `metal/primitives.h` / `softmax.cc` citation may resolve against.
ROOTS=("" "src/" "include/ctranslate2/" "include/" "python/" "tests/")

# Cache the repo file list once for the basename fallback.
ALL_FILES="$(git -C "$REPO" ls-files)"

resolve() { # echoes a repo-relative path for $1, or nothing if unresolved/ambiguous
  local cite="$1" r hits
  for r in "${ROOTS[@]}"; do
    [[ -f "$REPO/$r$cite" ]] && { echo "$r$cite"; return 0; }
  done
  # fallback: match by path suffix across tracked files
  hits="$(grep -E "(^|/)$(printf '%s' "$cite" | sed 's/[.[\*^$/]/\\&/g')$" <<<"$ALL_FILES")"
  [[ "$(wc -l <<<"$hits" | tr -d ' ')" == "1" && -n "$hits" ]] && { echo "$hits"; return 0; }
  [[ -n "$hits" ]] && return 2   # ambiguous
  return 1                       # unresolved
}

fails=0 ambig=0 ok=0
shopt -s nullglob
for md in "$REF_DIR"/*.md; do
  printed_header=0
  # Pull every  <file>.<ext>:<line>[-<line>]  token (ext set matches the codebase).
  while IFS= read -r cite; do
    [[ -z "$cite" ]] && continue
    file="${cite%%:*}"; lines="${cite#*:}"; start="${lines%%-*}"
    path="$(resolve "$file")"; rc=$?
    label=""
    if [[ $rc -eq 2 ]]; then
      ambig=$((ambig+1)); label="AMBIG"; content="(matches multiple tracked files)"
    elif [[ $rc -ne 0 || -z "$path" ]]; then
      fails=$((fails+1)); label="MISS "; content="(no such file in repo)"
    else
      local_n="$(wc -l <"$REPO/$path")"; local_n="${local_n//[^0-9]/}"
      if [[ "$start" =~ ^[0-9]+$ ]] && (( start > local_n )); then
        fails=$((fails+1)); label="RANGE"; content="(line $start > $local_n EOF) $path"
      else
        ok=$((ok+1)); label="ok   "
        content="$(sed -n "${start}p" "$REPO/$path" | sed 's/^[[:space:]]*//' | cut -c1-80)"
      fi
    fi
    if [[ "$label" == "ok   " && $QUIET -eq 1 ]]; then continue; fi
    if [[ $printed_header -eq 0 ]]; then echo; echo "=== ${md#$REF_DIR/} ==="; printed_header=1; fi
    printf "  %s %-34s %s\n" "$label" "$cite" "$content"
  done < <(grep -hoE "[A-Za-z0-9_/.-]+\.(cc|cu|mm|h|py|metal):[0-9]+(-[0-9]+)?" "$md" | sort -u)
done

echo
echo "citations ok=$ok  fail=$fails  ambiguous=$ambig"
echo "NB: 'ok' = file exists & line in range. CONTENT drift is not auto-checked — eyeball the printed lines."
(( fails > 255 )) && fails=255
exit "$fails"
