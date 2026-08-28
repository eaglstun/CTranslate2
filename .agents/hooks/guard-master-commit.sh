#!/usr/bin/env bash
# PreToolUse guard: refuse `git commit` while HEAD is on a protected branch
# (master/main). The recurring mistake this prevents is work landing directly on
# the default branch when it should have been on a feature branch.
#
# Contract: reads the PreToolUse JSON event on stdin. Exit 0 = allow. Exit 2 =
# block the tool call and feed stderr back to Claude as the reason.
#
# Scope is deliberately narrow: only `git commit` is blocked (that is the moment
# work lands). Amends, merges, and everything else are untouched. To bypass for a
# legitimate commit-to-master, switch to a branch first (`git switch -c <name>`)
# or temporarily disable this hook in .claude/settings.json.

set -euo pipefail

PROTECTED_RE='^(master|main)$'

input="$(cat)"

tool="$(printf '%s' "$input" | jq -r '.tool_name // empty')"
[ "$tool" = "Bash" ] || exit 0

cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // empty')"

# Only care about actual commits. Match `git commit` as a word, tolerating
# leading env/`cd`, pipes, and `&&` chains. Skip if there's no commit verb.
printf '%s' "$cmd" | grep -Eq '(^|[^[:alnum:]_])git([[:space:]]+-[^[:space:]]+)*[[:space:]]+commit([[:space:]]|$)' || exit 0

# Resolve the branch of the repo the command will run in. Fall back to the CWD.
branch="$(git branch --show-current 2>/dev/null || true)"

if printf '%s' "$branch" | grep -Eq "$PROTECTED_RE"; then
  cat >&2 <<EOF
BLOCKED: you are about to commit directly onto '$branch', a protected branch.

Work is not supposed to land on the default branch. Create a feature branch first:

    git switch -c <descriptive-branch-name>

then re-run the commit. If this really must go on '$branch' (e.g. a release
merge), disable this guard in .claude/settings.json first.
EOF
  exit 2
fi

exit 0
