# lip-sync-mac

Generate short lip-synced talking-head video clips **entirely on a single Mac**
(Apple Silicon) from one portrait image, one line of text, and a reference
voice. No GPU server required.

Pipeline: **text -> Qwen3-TTS speech -> LTX-2.3 (ComfyUI) image-to-video with
audio conditioning -> lip-synced MP4.**

This was built and tuned on a MacBook Pro M4 Pro (48GB unified memory). It
should work on any Apple Silicon Mac with enough RAM to hold the models
(~20-25GB resident during generation, more if you don't close other apps);
see [Performance](#performance) for what to expect on less memory.

## Demo

https://github.com/user-attachments/assets/f525624f-b61e-4423-9348-3978e7f2cf1b

Turn the sound on — lip-sync is the whole point, and no still or GIF can show
whether the mouth actually matches the speech. Same file as
[`assets/demo.mp4`](assets/demo.mp4) (8.4s, 320x576, 359 KB).

It says *"This clip was generated entirely on a MacBook. Both the voice and the
lip motion are synthetic."*, which is literally true of it: nothing in the video
is a recording of anyone saying those words.

Generated with the defaults on an M4 Pro in ~5 minutes (298s), from the files in
`assets/`. Reproduce it exactly — `--seed` covers the TTS take as well as the
video, so you should get the same clip back:

```bash
cp assets/demo_portrait.png assets/demo_ref.wav assets/demo_ref.txt "$LTX_MAC_ROOT/input/"
"$PY" generate_lipsync_fast.py \
  --text "This clip was generated entirely on a MacBook. Both the voice and the lip motion are synthetic." \
  --image demo_portrait.png \
  --ref-audio demo_ref.wav \
  --prompt "A pilot in a white uniform speaking clearly at the camera, natural lip sync, dark studio background, medium shot" \
  --auto-duration --seed 42 --prefix demo
```

No `--ref-text` here because `demo_ref.txt` sits next to `demo_ref.wav`: the
transcript is a property of the reference clip, so keep the two together and
the flag becomes unnecessary. Pass `--ref-text` only for a clip that has no
sidecar.

**Demo asset credits.** The portrait is
["Portrait Pilot" by Elliott Chau](https://stocksnap.io/photo/SW0YN0Z5T0)
(StockSnap, CC0), cropped to 320x576. The reference voice is clip `LJ001-0009`
of the [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) (public
domain, read by Linda Johnson), resampled to 24kHz mono. `demo_ref.txt` is that
clip's own transcript, straight from the dataset — LJ Speech ships audio already
paired with exact text, which is the whole reason the clone has something
trustworthy to work from.

> A note if you swap in your own portrait: a permissive image licence covers
> copyright, not likeness. CC0/stock terms say nothing about whether the person
> shown agreed to be animated saying words they never said, and much of what
> turns up under "free portrait" is private individuals or minors. Use a face
> you have permission to use, or a synthetic one.

## How it works

```
   text  ─────────────┐
                       ▼
  reference voice ─► Qwen3-TTS (Apple MPS) ─► speech .wav
                                                   │
  portrait image ───────────────────────────────► │
                                                   ▼
                              ComfyUI + LTX-2.3 (image-to-video, audio-conditioned)
                                                   │
                                                   ▼
                                          lip-synced .mp4
```

- **TTS**: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 1.7B, run on the
  Mac's GPU via PyTorch MPS. Supports voice cloning from a short reference
  clip, or a fine-tuned custom speaker if you've trained one.
- **Video**: [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) driven
  through [ComfyUI](https://github.com/comfyanonymous/ComfyUI)'s API. The
  DiT model runs as a GGUF quantization via
  [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) so it fits
  comfortably in unified memory (a full fp8 checkpoint is ~28GB and not
  worth it on a Mac; the GGUF is ~16-18GB).
- Everything runs locally over `127.0.0.1`; no data leaves the machine.

## Requirements

- Apple Silicon Mac, 32GB+ unified memory recommended (48GB tested).
- macOS with Homebrew.
- Python 3.12 (a `uv venv` is assumed below, but any venv tool works).
- ~40GB free disk space for models.
- The [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
  to fetch model weights (`uv tool install huggingface_hub`, or
  `pip install huggingface_hub` — gives you the `hf` command). No account or
  token needed: every repo used here is public and ungated.

## Setup

```bash
# 1. System tools
brew install ffmpeg sox uv

# 2. ComfyUI + the GGUF loader node
export LTX_MAC_ROOT="$HOME/work/ltx_mac"   # pick any path; scripts default to this
mkdir -p "$LTX_MAC_ROOT" && cd "$LTX_MAC_ROOT"
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git ComfyUI/custom_nodes/ComfyUI-GGUF
# ComfyUI's --base-directory flag looks for custom_nodes directly under the
# base dir, not under ComfyUI/ - this symlink is required, not cosmetic.
ln -s ComfyUI/custom_nodes custom_nodes

# 3. Python environment
uv venv --python 3.12 venv
uv pip install --python venv/bin/python torch torchvision torchaudio
uv pip install --python venv/bin/python -r ComfyUI/requirements.txt
uv pip install --python venv/bin/python -r ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt
uv pip install --python venv/bin/python -r requirements.txt   # this repo's requirements.txt

# 4. Copy this repo's scripts + launcher into place
#    (clone this repo somewhere, then:)
cp -r /path/to/lip-sync-mac/scripts "$LTX_MAC_ROOT/scripts"
cp /path/to/lip-sync-mac/run_comfyui_mac.sh "$LTX_MAC_ROOT/"
mkdir -p "$LTX_MAC_ROOT"/{input,output,models/tts,models/diffusion_models,models/text_encoders,models/vae,models/checkpoints}
```

### Model weights

About 35GB total, from three Hugging Face repos. None are gated, so no token or
licence click-through is needed. Run this from `$LTX_MAC_ROOT`:

```bash
cd "$LTX_MAC_ROOT"

# 1. The DiT, as a GGUF quantization (16.4GB - the big one)
hf download unsloth/LTX-2.3-GGUF \
  distilled-1.1/ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf \
  --local-dir /tmp/ltxdl
mv /tmp/ltxdl/distilled-1.1/ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf models/diffusion_models/

# 2. Video VAE (1.5GB) -> models/vae/
hf download unsloth/LTX-2.3-GGUF vae/ltx-2.3-22b-dev_video_vae.safetensors --local-dir /tmp/ltxdl
mv /tmp/ltxdl/vae/ltx-2.3-22b-dev_video_vae.safetensors models/vae/

# 3. Audio VAE (0.4GB) and text-encoder connectors (2.3GB) -> models/checkpoints/
#    Note the destination: see the warning below, it is not the obvious one.
hf download unsloth/LTX-2.3-GGUF \
  vae/ltx-2.3-22b-dev_audio_vae.safetensors \
  text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
  --local-dir /tmp/ltxdl
mv /tmp/ltxdl/vae/ltx-2.3-22b-dev_audio_vae.safetensors models/checkpoints/
mv /tmp/ltxdl/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors models/checkpoints/

# 4. Gemma text encoder (9.5GB). Different repo - it is NOT in the unsloth one.
hf download eraRelentless/Gemma_3_12B_it_fp4 gemma_3_12B_it_fp4_mixed.safetensors \
  --local-dir /tmp/ltxdl
mv /tmp/ltxdl/gemma_3_12B_it_fp4_mixed.safetensors models/text_encoders/

# 5. Qwen3-TTS: base model (4.5GB) + its tokenizer (0.7GB)
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/tts/Qwen3-TTS-12Hz-1.7B-Base
hf download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/tts/Qwen3-TTS-Tokenizer-12Hz

rm -rf /tmp/ltxdl
```

`hf download --local-dir` keeps the repo's own subfolder layout, which is why
each step moves the file afterwards rather than downloading straight into place.

The result should look like this — the script defaults expect these exact paths
(override with the `LTX_*` environment variables in
`scripts/generate_lipsync_fast.py` if you put them elsewhere):

```
models/
├── diffusion_models/ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf
├── vae/ltx-2.3-22b-dev_video_vae.safetensors
├── text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
├── checkpoints/ltx-2.3-22b-dev_audio_vae.safetensors
├── checkpoints/ltx-2.3-22b-distilled_embeddings_connectors.safetensors
└── tts/
    ├── Qwen3-TTS-12Hz-1.7B-Base/
    └── Qwen3-TTS-Tokenizer-12Hz/
```

> **Important:** the audio VAE and the connectors go in `models/checkpoints/`,
> *not* in `vae/` or `text_encoders/` where their names — and the folders they
> sit in on Hugging Face — both suggest. ComfyUI's `LTXVAudioVAELoader` and
> `LTXAVTextEncoderLoader` only populate their dropdown from
> `models/checkpoints/`, so the obvious folder produces a `value_not_in_list`
> error at generation time.

A fine-tuned or cloned voice is optional; `generate_lipsync_fast.py` works
out of the box with Qwen3-TTS's built-in voice-cloning mode: drop a few seconds
of reference audio in `input/` as `voice.wav`, with exactly what it says in
`voice.txt` next to it. `assets/` has a usable pair if you just want to see the
thing run.

## Usage

Start ComfyUI in one terminal (first launch takes a few minutes to load
models):

```bash
bash run_comfyui_mac.sh
# check it's up: curl -s http://127.0.0.1:8188/queue
```

In another terminal, generate a clip:

```bash
export PATH=/opt/homebrew/bin:$PATH
PY="$LTX_MAC_ROOT/venv/bin/python"
cd "$LTX_MAC_ROOT/scripts"

# put your portrait under $LTX_MAC_ROOT/input/, e.g. person.png
"$PY" generate_lipsync_fast.py \
  --text "Hello, this is a test of the lip-sync pipeline." \
  --image person.png \
  --ref-audio voice_sample.wav \
  --auto-duration \
  --seed 42 --prefix demo
```

The reference clip needs its exact transcript, so that Qwen3-TTS knows what the
sample is saying. Put it in `voice_sample.txt` beside the wav and it is picked
up automatically; otherwise pass `--ref-text "<transcript>"`. A transcript that
does not match the audio degrades the cloned voice with no error, which is why
one of the two is required.

The output lands at `$LTX_MAC_ROOT/output/demo_00001_.mp4`.

`--seed` covers both the TTS take and the video noise, so the same seed
reproduces the whole clip. Omit it and Qwen3-TTS samples a different take
(different wording emphasis, different length) every run, which is usually not
what you want when comparing two settings.

If you have a fine-tuned custom-voice speaker instead of a reference clip,
use `--tts-model <path> --speaker <name>` in place of `--ref-audio`.

### Retrying a bad clip

LTX occasionally locks onto the still reference image and mostly ignores the
audio (a "dead" clip with barely any mouth motion), or fades to black during
speech. `regen_seg.py` re-rolls the seed a few times and keeps the best
result by mouth motion + end-of-clip brightness:

```bash
"$PY" regen_seg.py \
  --text "Hello, this is a test of the lip-sync pipeline." \
  --prefix demo \
  --image person.png \
  --ref-audio voice_sample.wav \
  --prompt "A person speaking clearly at the camera, natural lip sync, medium shot" \
  --seeds 123,777,2024
```

It takes the same voice arguments as `generate_lipsync_fast.py`, so swap in
`--tts-model <path> --speaker <name>` if you have a fine-tuned speaker. The
winning clip is copied to `output/<prefix>_00001_.mp4`.

### Checking quality

```bash
"$PY" analyze_mouth.py output/demo_00001_.mp4
```

Prints a `mouth_range` value (spread between the 90th and 10th percentile of
mouth openness across the clip). In practice:

- **>= 0.02**: acceptable, mouth is clearly moving.
- **< 0.02**: likely a "dead" clip - LTX rendered a mostly-static face. Retry
  with a different seed (see above) rather than trying to fix it with
  parameters; see [Tuning](#tuning) for why.

Also worth checking for a fade-to-black at the very end of the clip (common
LTX artifact): `ffmpeg -i clip.mp4 -vf signalstats,metadata=print:key=lavfi.signalstats.YAVG -f null -`
and look for a luminance (`YAVG`) drop in the last few frames.

## Tuning

Two knobs control how closely the output follows the reference image vs. how
freely LTX can animate it:

- `--guide-strength` (default 0.94): how tightly the video is pinned to the
  reference frame. Lower = more freedom to move, but also more freedom to
  drift.
- `--img-compression` (default 24): how much the reference image is
  abstracted before conditioning. Higher = more freedom to move.

**Both knobs can cause the generated face to drift into looking like a
different person if pushed too far** - this is more likely with clean,
high-detail reference images (e.g. a studio headshot with a flat background)
than with casual photos, where the model already has more "wiggle room" for
different faces. We hit this directly: lowering `guide-strength` to 0.80
produced a visibly different face partway through the clip, and raising
`img-compression` to 30-40 produced a completely different person, on the
same source photo where a "dead" (mouth barely moving) clip came out at the
defaults. If you're not getting motion, **try a different `--seed` before
you touch these two flags** - it's usually the safer fix, and it's what
`regen_seg.py` automates.

`--width`/`--height` and the sampler step count (`LTX_SIGMAS` env var, see
comments in `generate_lipsync_fast.py`) trade quality for speed - see
[Performance](#performance). The defaults (320x576, 2 steps) are the
benchmarked fast config; raise `--width`/`--height` for quality, keeping both
a multiple of 32.

`--force-fp16` on ComfyUI's launch (in `run_comfyui_mac.sh`) is **not** a
free speedup on Apple Silicon: in testing it made generation ~37% *slower*
than the bf16 default, the opposite of its effect on CUDA. Left as bf16 by
default for that reason.

## Performance

Benchmarked on an M4 Pro, 48GB, for a ~3.5s clip. Reproduce with:

```bash
export TTS_MODEL_PATH=<path-to-tts-model> TTS_SPEAKER=<speaker-id> BENCH_IMAGE=person.png
bash scripts/bench/speed_bench.sh        # generates every config, skipping ones already done
bash scripts/bench/judge_bench.sh        # prints mouth_range/brightness and a comparison image
```

| Sampler steps | Resolution | Time (warm) | vs. 4-step/448x800 baseline |
|---|---|---|---|
| 4 | 448x800 | ~355s | 1x |
| 3 | 448x800 | ~352s | ~1.01x (see note) |
| 2 | 448x800 | ~227s | ~1.56x |
| 4 | 384x672 | ~262s | ~1.36x |
| 2 | 384x672 | ~157s | ~2.26x |
| 2 | 320x576 | ~137s | ~2.59x (**default**) |
| 4 | 448x800, `--force-fp16` | ~488s | 0.73x (slower!) |

> The 355s baseline was measured in an earlier session than the other rows;
> compare warm runs to warm runs. The 3-step row is an outlier — stepping 4->2
> saves ~128s, so 4->3 saving only ~3s does not fit, and interpolation would
> predict ~290s. Treat 3-step as un-measured rather than as "no faster than
> 4-step", and re-run `speed_bench.sh bench3s448` if you care about it.

Lower step counts trade off lip-sync robustness for speed - fewer steps make
"dead" (non-moving) clips and identity drift somewhat more likely, so always
run `analyze_mouth.py` and eyeball the result. For a portrait-crop PiP
overlay (roughly 280px tall in a 1080p composite), 320x576 is visually
indistinguishable from 448x800 or higher.

A single Mac cannot compete with a multi-GPU server for large batch jobs
(expect on the order of 25-40x slower per clip vs. a modern datacenter GPU at
equivalent quality settings) - this pipeline is meant for one-off clips,
touch-ups, or offline/overnight batches, not bulk production.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `value_not_in_list` for audio VAE / text-encoder connectors | Both `LTXVAudioVAELoader` and `LTXAVTextEncoderLoader` populate their dropdown from `models/checkpoints/` only. Move those two files there. Easy to get wrong because Hugging Face ships them under `vae/` and `text_encoders/`, which is where they look like they belong. |
| `UnetLoaderGGUF` node not found | Missing the `custom_nodes -> ComfyUI/custom_nodes` symlink; ComfyUI's `--base-directory` looks for `custom_nodes` directly under the base dir. |
| Can't find `gemma_3_12B_it_fp4_mixed.safetensors` in the LTX GGUF repo | It isn't there. `unsloth/LTX-2.3-GGUF` ships the DiT, the VAEs and the connectors, but not the Gemma text encoder — that one comes from `eraRelentless/Gemma_3_12B_it_fp4`. See [Model weights](#model-weights). |
| `RuntimeError: invalid low watermark ratio 1.4` on startup | You changed `PYTORCH_MPS_HIGH_WATERMARK_RATIO` without also lowering `PYTORCH_MPS_LOW_WATERMARK_RATIO` below it. `run_comfyui_mac.sh` sets both (0.85/0.75) for this reason. |
| `brew: command not found` / `sox not found` over SSH | Non-interactive shells don't have `/opt/homebrew/bin` on `PATH` by default. `export PATH=/opt/homebrew/bin:$PATH` before running anything. |
| A wall of `speaker_encoder.*` weight warnings when loading a fine-tuned TTS model | Harmless - `generate_custom_voice` doesn't use the speaker encoder. |
| `submit_and_wait` reports `TIMEOUT` but the clip shows up in `output/` moments later | The default wait is 1800s; a long clip on Apple Silicon can exceed it while still finishing. ComfyUI keeps working after the script gives up - check `output/` before assuming failure. |
| `cv2.CascadeClassifier` missing / `analyze_mouth.py` crashes on import | Some `opencv-python-headless` 5.x wheels ship without Haar cascade support. Pin to 4.x (`requirements.txt` already does this). |
| Generated face doesn't look like the reference person | See [Tuning](#tuning) - you likely pushed `--guide-strength` down or `--img-compression` up too far. Revert to defaults and change `--seed` instead. |
| Cloned voice sounds off, but nothing errored | Check that the transcript actually matches the reference audio, word for word. Qwen3-TTS trusts it and will not complain: a stale `voice.txt`, or a `--ref-text` left over from another clip, just quietly degrades the voice. Trim the wav and the text together, or use a source that ships both (LJ Speech, Common Voice). |
| `pkill -f "port 8188"`-style patterns kill your own shell too | `pkill -f` matches the full command line, including whatever invoked it (e.g. a wrapping `bash -c "... port 8188 ..."`). Match on something more specific, like the exact `main.py --listen 127.0.0.1 --port 8188` invocation, or check with `pgrep -fa` first. |

## License

MIT - see [LICENSE](LICENSE).
