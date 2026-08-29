#!/usr/bin/env bash
# Downstream validation harness for the Metal backend
# (see docs/metal/history/int8-plan.md).
#
# Pipeline: build the C++ lib from THIS worktree -> cmake --install to a pinned
# per-run prefix -> rebuild the Python wheel against it -> force-reinstall into
# each consumer venv -> run each consumer's canonical job -> diff vs golden.
#
# Goldens are captured with fp16 on Metal from the same build; int8 candidates
# are compared against them with a quant-error tolerance (tests/downstream/
# projects.json documents metric + tolerance per consumer).
#
# Usage:
#   scripts/validate-downstream.sh --capture-goldens   # (re)record fp16 goldens
#   scripts/validate-downstream.sh                     # validate int8 vs goldens
#   Flags: --skip-build  --skip-install  --only NAME  --compute-type CT
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
DS="$ROOT/tests/downstream"
CFG="$DS/projects.json"
PY3=python3

MODE=validate
SKIP_BUILD=0
SKIP_INSTALL=0
ONLY=""
COMPUTE=int8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --capture-goldens) MODE=capture; COMPUTE=float16 ;;
    --skip-build)      SKIP_BUILD=1 ;;
    --skip-install)    SKIP_INSTALL=1 ;;
    --only)            ONLY="$2"; shift ;;
    --compute-type)    COMPUTE="$2"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

cfg() { $PY3 -c "import json,os,sys; c=json.load(open('$CFG')); $1"; }

PREFIX=$(cfg "print(os.path.expanduser(c['prefix']))")
WHEEL_PY=$(cfg "print(c['wheel_python'])")

echo "== downstream harness: mode=$MODE compute_type=$COMPUTE prefix=$PREFIX"

# ---------------------------------------------------------------- build+install
if [[ $SKIP_BUILD -eq 0 ]]; then
  echo "== building C++ lib"
  cmake --build "$ROOT/build" -j 10 || exit 1
  echo "== installing to $PREFIX"
  cmake --install "$ROOT/build" --prefix "$PREFIX" > /dev/null || exit 1
  echo "== building wheel (CTRANSLATE2_ROOT=$PREFIX)"
  rm -rf "$ROOT/python/dist"
  (cd "$ROOT/python" && CTRANSLATE2_ROOT="$PREFIX" "$WHEEL_PY" setup.py -q bdist_wheel) || exit 1
fi
WHEEL=$(ls -t "$ROOT"/python/dist/ctranslate2-*.whl 2>/dev/null | head -1)
if [[ -z "$WHEEL" ]]; then echo "no wheel in python/dist — build first" >&2; exit 1; fi
echo "== wheel: $WHEEL"

# ------------------------------------------------------------- venvs + reinstall
fix_rpath() {
  local py="$1" site ext
  site=$("$py" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  for ext in "$site"/ctranslate2/_ext*.so; do
    [[ -e "$ext" ]] || continue
    if ! otool -l "$ext" | grep -q "path $PREFIX/lib"; then
      install_name_tool -add_rpath "$PREFIX/lib" "$ext"
      codesign -f -s - "$ext" 2>/dev/null || true
    fi
  done
}

if [[ $SKIP_INSTALL -eq 0 ]]; then
  for vname in $(cfg "print('\n'.join(c['venvs']))"); do
    vpath=$(cfg "print(c['venvs']['$vname'])")
    if [[ ! -x "$vpath/bin/python" ]]; then
      bsrc=$(cfg "print(c.get('bootstrap',{}).get('$vname',{}).get('editable',''))")
      bpy=$(cfg "print(c.get('bootstrap',{}).get('$vname',{}).get('python','3.10'))")
      if [[ -z "$bsrc" ]]; then echo "venv $vname missing and no bootstrap entry" >&2; exit 1; fi
      echo "== bootstrapping venv $vname ($vpath, python $bpy)"
      uv venv --python "$bpy" "$vpath" || exit 1
      uv pip install --python "$vpath/bin/python" -e "$bsrc" || exit 1
    fi
    echo "== installing wheel into $vname"
    uv pip install --python "$vpath/bin/python" --force-reinstall "$WHEEL" || exit 1
    fix_rpath "$vpath/bin/python"
    "$vpath/bin/python" -c "import ctranslate2 as c; print('   ', '$vname', 'ct2', c.__version__, c.get_supported_compute_types('metal'))" || exit 1
  done
fi

# ------------------------------------------------------------------ run + diff
mkdir -p "$DS/goldens" "$DS/results"
VERDICTS="$DS/results/verdicts.jsonl"
[[ $MODE == validate ]] && : > "$VERDICTS"
FAILED=0
N=$(cfg "print(len(c['consumers']))")

for ((i = 0; i < N; i++)); do
  name=$(cfg "print(c['consumers'][$i]['name'])")
  [[ -n "$ONLY" && "$name" != "$ONLY" ]] && continue
  venv=$(cfg "print(c['venvs'][c['consumers'][$i]['venv']])")
  driver=$(cfg "print(c['consumers'][$i]['driver'])")
  metric=$(cfg "print(c['consumers'][$i]['metric'])")
  tol=$(cfg "print(c['consumers'][$i]['tolerance'])")
  golden=$(cfg "print(c['consumers'][$i]['golden'])")
  pass_golden=$(cfg "print(int(c['consumers'][$i].get('pass_golden', False)))")
  IFS=$'\x1f' read -r -a dargs <<< "$(cfg "print('\x1f'.join(c['consumers'][$i]['args']))")"

  extra=()
  if [[ $MODE == capture ]]; then
    out="$DS/$golden"
  else
    out="$DS/results/$name.$COMPUTE.json"
    [[ $pass_golden -eq 1 ]] && extra=(--golden "$DS/$golden")
    if [[ ! -f "$DS/$golden" ]]; then
      echo "== $name: NO GOLDEN ($golden) — run --capture-goldens first" >&2
      echo "{\"consumer\": \"$name\", \"pass\": false, \"error\": \"missing golden\"}" >> "$VERDICTS"
      FAILED=1
      continue
    fi
  fi

  echo "== running $name ($COMPUTE)"
  if ! (cd "$ROOT" && "$venv/bin/python" "$DS/$driver" \
        --compute-type "$COMPUTE" --output "$out" "${dargs[@]}" ${extra[@]+"${extra[@]}"}); then
    echo "== $name: DRIVER FAILED" >&2
    if [[ $MODE == validate ]]; then
      echo "{\"consumer\": \"$name\", \"pass\": false, \"error\": \"driver failed\"}" >> "$VERDICTS"
      FAILED=1
    fi
    continue
  fi

  if [[ $MODE == validate ]]; then
    if ! $PY3 "$DS/compare.py" --metric "$metric" --tolerance "$tol" \
          --golden "$DS/$golden" --candidate "$out" | tee -a "$VERDICTS"; then
      FAILED=1
    fi
  fi
done

if [[ $MODE == capture ]]; then
  echo "== goldens captured to $DS/goldens/"
  exit 0
fi

echo "== verdicts:"
cat "$VERDICTS"
exit $FAILED
