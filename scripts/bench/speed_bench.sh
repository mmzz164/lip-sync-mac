#!/bin/bash
# Speed benchmark driver for the Mac lip-sync pipeline.
#
# Usage:
#   bash scripts/bench/speed_bench.sh                       # all configs
#   bash scripts/bench/speed_bench.sh bench3s448 benchfp16   # only these configs
#
# Configs whose output already exists (output/<prefix>_00001_.mp4) are
# skipped automatically, so if you interrupt this script, re-running the
# same command picks up where it left off.
#
# Add a config by adding one line to CONFIGS (name|sigmas|width|height|fp16(0/1)).
# fp16=1 configs restart ComfyUI with --force-fp16 before running.
#
# Required env vars (no sane generic default, since they depend on your setup):
#   TTS_MODEL_PATH   path to a Qwen3-TTS model directory (fine-tuned or Base)
#   TTS_SPEAKER      speaker id for --speaker (custom_voice mode). If your
#                    setup uses voice-clone instead, edit warmup()/run_config()
#                    below to pass --ref-audio/--ref-text instead of
#                    --tts-model/--speaker.
#   BENCH_IMAGE      reference image filename, expected under input/
set -uo pipefail

ROOT="${LTX_MAC_ROOT:-$HOME/work/ltx_mac}"
export PATH=/opt/homebrew/bin:$PATH
PY="$ROOT/venv/bin/python"
OUT="$ROOT/output"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

: "${TTS_MODEL_PATH:?set TTS_MODEL_PATH to a Qwen3-TTS model dir}"
: "${TTS_SPEAKER:?set TTS_SPEAKER to a speaker id supported by that model}"
: "${BENCH_IMAGE:?set BENCH_IMAGE to a reference image filename under input/}"

BENCH_TEXT="This is a local benchmark on the MacBook."
GUIDE_STRENGTH=0.94
IMG_COMPRESSION=24
SEED=42

SIG4="1.0, 0.98125, 0.909375, 0.421875, 0.0"   # 4 steps
SIG3="1.0, 0.98125, 0.725, 0.0"                # 3 steps
SIG2="1.0, 0.909375, 0.0"                       # 2 steps

# name|sigmas|width|height|fp16(0/1)
CONFIGS=(
  "warm448|$SIG4|448|800|0"
  "bench3s448|$SIG3|448|800|0"
  "bench2s448|$SIG2|448|800|0"
  "bench4s384|$SIG4|384|672|0"
  "benchfp16|$SIG4|448|800|1"
  "bench2s384|$SIG2|384|672|0"
  "bench2s320|$SIG2|320|576|0"
)

want=("$@")

is_wanted() {
  local name="$1"
  [ ${#want[@]} -eq 0 ] && return 0
  local w
  for w in "${want[@]}"; do [ "$w" = "$name" ] && return 0; done
  return 1
}

already_done() {
  local matches
  matches=$(ls "$OUT"/"$1"_*.mp4 2>/dev/null)
  [ -n "$matches" ]
}

comfy_running() {
  curl -s -m 2 "http://127.0.0.1:8188/queue" >/dev/null 2>&1
}

stop_comfy() {
  pkill -f "main.py --listen 127.0.0.1 --port 8188" 2>/dev/null
  local i
  for i in $(seq 1 20); do comfy_running || break; sleep 1; done
  sleep 1
}

start_comfy() {
  local extra="$1"
  stop_comfy
  echo "[bench] starting ComfyUI ${extra:+(with $extra)}..."
  nohup bash "$ROOT/run_comfyui_mac.sh" $extra > /tmp/comfy_mac.log 2>&1 &
  local i
  for i in $(seq 1 90); do
    comfy_running && { echo "[bench] ComfyUI HTTP up after ${i}s"; return 0; }
    sleep 2
  done
  echo "[bench] ERROR: ComfyUI did not come up within 180s"
  return 1
}

warmup() {
  local prefix="$1"
  echo "[bench] warmup run ($prefix) to force model load into memory..."
  rm -f "$OUT/${prefix}_"*.mp4 "$OUT/${prefix}_workflow.json"
  "$PY" "$ROOT/scripts/generate_lipsync_fast.py" \
    --text "Hi." --image "$BENCH_IMAGE" \
    --tts-model "$TTS_MODEL_PATH" --speaker "$TTS_SPEAKER" \
    --auto-duration --width 448 --height 800 \
    --guide-strength "$GUIDE_STRENGTH" --img-compression "$IMG_COMPRESSION" \
    --seed "$SEED" --prefix "$prefix" \
    > "$LOGDIR/${prefix}.log" 2>&1
  rm -f "$OUT/${prefix}_"*.mp4 "$OUT/${prefix}_workflow.json"
}

run_config() {
  local name="$1" sigmas="$2" width="$3" height="$4"
  echo "[bench] === $name (sigmas='$sigmas' ${width}x${height}) ==="
  rm -f "$OUT/${name}_"*.mp4   # defensive: avoid matching a stale result from an earlier run
  LTX_SIGMAS="$sigmas" "$PY" "$ROOT/scripts/generate_lipsync_fast.py" \
    --text "$BENCH_TEXT" --image "$BENCH_IMAGE" \
    --tts-model "$TTS_MODEL_PATH" --speaker "$TTS_SPEAKER" \
    --auto-duration --width "$width" --height "$height" \
    --guide-strength "$GUIDE_STRENGTH" --img-compression "$IMG_COMPRESSION" \
    --seed "$SEED" --prefix "$name" \
    2>&1 | tee "$LOGDIR/${name}.log"
  local el
  el=$(grep -o 'elapsed=[0-9]*s' "$LOGDIR/${name}.log" | tail -1)
  echo "[bench] RESULT $name ${el:-NO_ELAPSED (check the log / did it time out?)}"
}

need_fp16_group=0
need_plain_group=0
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r name sigmas width height fp16 <<< "$entry"
  is_wanted "$name" || continue
  already_done "$name" && continue
  if [ "$fp16" = "1" ]; then need_fp16_group=1; else need_plain_group=1; fi
done

if [ "$need_plain_group" = "1" ]; then
  start_comfy "" || exit 1
  warmup "warmup0"
  for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r name sigmas width height fp16 <<< "$entry"
    [ "$fp16" = "1" ] && continue
    is_wanted "$name" || continue
    if already_done "$name"; then
      echo "[bench] SKIP $name (already done)"
      continue
    fi
    run_config "$name" "$sigmas" "$width" "$height"
  done
fi

if [ "$need_fp16_group" = "1" ]; then
  start_comfy "--force-fp16" || exit 1
  warmup "warmupfp16"
  for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r name sigmas width height fp16 <<< "$entry"
    [ "$fp16" = "1" ] || continue
    is_wanted "$name" || continue
    if already_done "$name"; then
      echo "[bench] SKIP $name (already done)"
      continue
    fi
    run_config "$name" "$sigmas" "$width" "$height"
  done
fi

echo "[bench] all done. outputs:"
ls -la "$OUT"/*_00001_.mp4 2>/dev/null
