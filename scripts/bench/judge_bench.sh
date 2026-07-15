#!/bin/bash
# Quality check for speed_bench.sh's output.
#
# For every <prefix>_00001_.mp4 that speed_bench.sh produced under output/,
# reports:
#   - mouth_range (lip motion; pass >= 0.02, see analyze_mouth.py)
#   - YAVG brightness (fade-to-black check; pass: min >= 50 and the first/last
#     5 frames are within 10% of the clip's median brightness)
#   - a side-by-side comparison image for eyeballing motion/quality
#
# Usage: bash scripts/bench/judge_bench.sh
set -uo pipefail

ROOT="${LTX_MAC_ROOT:-$HOME/work/ltx_mac}"
export PATH=/opt/homebrew/bin:$PATH
PY="$ROOT/venv/bin/python"
OUT="$ROOT/output"
T=/tmp/bench_judge
mkdir -p "$T"

PREFIXES=(warm448 bench3s448 bench2s448 bench4s384 benchfp16 bench2s384 bench2s320)

echo "=== Clips found ==="
present=()
for p in "${PREFIXES[@]}"; do
  f="$OUT/${p}_00001_.mp4"
  if [ -f "$f" ]; then
    present+=("$p")
    echo "  OK   $p"
  else
    echo "  MISS $p (not generated yet)"
  fi
done

if [ ${#present[@]} -eq 0 ]; then
  echo "No clips found. Run speed_bench.sh first."
  exit 1
fi

echo ""
echo "=== Mouth motion (mouth_range, pass >= 0.02) ==="
for p in "${present[@]}"; do
  f="$OUT/${p}_00001_.mp4"
  line=$("$PY" "$ROOT/scripts/analyze_mouth.py" "$f" | tail -1)
  echo "$p	$line"
done

echo ""
echo "=== Brightness (YAVG; pass: min>=50 and first5/last5 within 10% of median) ==="
for p in "${present[@]}"; do
  f="$OUT/${p}_00001_.mp4"
  ffmpeg -i "$f" -vf signalstats,metadata=print:key=lavfi.signalstats.YAVG -f null - 2>&1 |
    grep YAVG | awk -F= '{print $2}' | F="$p" python3 -c "
import sys, os
v = [float(x) for x in sys.stdin]
name = os.environ.get('F', '?')
if not v:
    print(f'{name}\tNO_YAVG_DATA')
else:
    mn = min(v)
    med = sorted(v)[len(v)//2]
    f5 = sum(v[:5]) / min(5, len(v))
    l5 = sum(v[-5:]) / min(5, len(v))
    ok = mn >= 50 and abs(f5 - med) <= med * 0.1 and abs(l5 - med) <= med * 0.1
    print(f'{name}\tmin={mn:.0f} median={med:.0f} first5={f5:.0f} last5={l5:.0f} ok={ok}')
"
done

echo ""
echo "=== Elapsed time (from speed_bench.sh logs) ==="
for p in "${present[@]}"; do
  log="$OUT/logs/${p}.log"
  if [ -f "$log" ]; then
    el=$(grep -o 'elapsed=[0-9]*s' "$log" | tail -1)
    echo "$p	${el:-N/A}"
  else
    echo "$p	(no log at $log)"
  fi
done

echo ""
echo "=== Side-by-side comparison ==="
inputs=()
filters=()
i=0
labels=()
for p in "${present[@]}"; do
  f="$OUT/${p}_00001_.mp4"
  inputs+=(-i "$f")
  filters+=("[$i:v]select=eq(n\\,30),scale=224:400[v$i]")
  labels+=("[v$i]")
  i=$((i+1))
done
filtercomplex=$(IFS=';'; echo "${filters[*]}")
labelcat=$(IFS=''; echo "${labels[*]}")
ffmpeg -y -v error "${inputs[@]}" \
  -filter_complex "${filtercomplex};${labelcat}hstack=${#present[@]}" \
  -frames:v 1 "$T/5up.png"
echo "-> $T/5up.png (left to right: ${present[*]})"
