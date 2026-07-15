#!/usr/bin/env python3
"""Regenerate ONE lip-sync clip by trying several seeds and keeping the best.

Some LTX seeds fade the clip to black DURING audible speech (not just in the
tail), so no trim can hide it without cutting words. Some seeds also barely
animate the mouth at all (see lipsync_utils.mouth_range - this happens more
often with a still, high-detail reference image than with a casual photo).
This script re-rolls the seed and keeps the first clip that (a) still moves
the mouth and (b) stays bright through the -30dB audible speech end.

Reuses lipsync_utils (U) / generate_lipsync_fast (G) so voice + look stay
identical. Only the video seed is re-rolled: --tts-seed pins the speech, because
Qwen3-TTS samples a different take (and a different length) every run otherwise,
and a replacement clip that changed length would no longer fit its slot.
"""
import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lipsync_utils as U
import generate_lipsync_fast as G
INPUT_DIR = U.INPUT_DIR
OUTPUT_DIR = U.OUTPUT_DIR


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def yavg(path):
    p = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vf",
              "signalstats,metadata=print:key=lavfi.signalstats.YAVG", "-f", "null", "-"])
    return [float(x) for x in re.findall(r"YAVG=([0-9.]+)", p.stderr + p.stdout)]


def fps_of(path):
    o = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path]).stdout.strip()
    n, d = o.split("/")
    return float(n) / float(d)


def audible_end(path, db=30):
    p = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
              "-af", f"silencedetect=noise=-{db}dB:d=0.15", "-f", "null", "-"])
    s = re.findall(r"silence_start: ([0-9.]+)", p.stderr + p.stdout)
    return float(s[-1]) if s else None


def bright_at_end(path):
    """Ratio of luminance at the -30dB audible end vs the speech-region median."""
    y = yavg(path)
    fps = fps_of(path)
    if not y:
        return 0.0
    ae = audible_end(path, 30) or (len(y) / fps * 0.8)
    lo = max(1, int(round(ae * fps)))
    med = statistics.median(y[:lo]) if lo > 1 else statistics.median(y)
    idx = min(len(y) - 1, lo)
    return y[idx] / med if med else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--prefix", required=True, help="output filename prefix, e.g. clip01")
    ap.add_argument("--tts-model", default=None, help="Path to a fine-tuned Qwen3-TTS model directory. Defaults to the Base model.")
    ap.add_argument("--speaker", default=None, help="Speaker name for custom_voice mode. Omit to clone from --ref-audio instead.")
    ap.add_argument("--ref-audio", default=None, help="Reference audio in input/ for voice cloning (used when --speaker is omitted)")
    ap.add_argument("--ref-text", default=None, help="Exact transcript of --ref-audio. Optional if a matching <name>.txt sits beside the audio.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--language", default="English")
    ap.add_argument("--guide-strength", type=float, default=0.94)
    ap.add_argument("--img-compression", type=int, default=24)
    ap.add_argument("--width", type=int, default=320)   # 320x576 keeps generation fast; raise to 448x800/576x1024 for higher quality.
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--tail-pause", type=float, default=0.5)
    ap.add_argument("--seeds", default="123,777,2024,31337,55")
    ap.add_argument("--tts-seed", type=int, default=42, help="Seeds the TTS take. Fixed by default: an unseeded take changes length between runs, so a regenerated clip would no longer fit the slot it replaces. Pass the --seed you used originally to get that exact take back.")
    ap.add_argument("--min-mouth-range", type=float, default=0.02)
    ap.add_argument("--min-bright", type=float, default=0.85)
    args = ap.parse_args()

    ref_audio_path = None
    ref_text = args.ref_text
    if not args.speaker:
        if not args.ref_audio:
            sys.exit("need either --speaker (custom_voice) or --ref-audio (voice clone)")
        ref_audio_path = args.ref_audio if os.path.isabs(args.ref_audio) else os.path.join(INPUT_DIR, args.ref_audio)
        if not os.path.exists(ref_audio_path):
            sys.exit(f"missing: {ref_audio_path}")
        ref_text = args.ref_text or U.ref_text_for(ref_audio_path)
        if not ref_text:
            sys.exit(f"no transcript for {os.path.basename(ref_audio_path)}: pass --ref-text, "
                     f"or put it in {os.path.splitext(ref_audio_path)[0] + '.txt'}")

    tts_name = f"{args.prefix}_tts.wav"
    tts_path = os.path.join(INPUT_DIR, tts_name)
    # Same TTS entry point as the single-shot CLI, so a clip regenerated here
    # keeps the voice it was originally generated with.
    dur = G.tts_voice_clone(args.text, ref_audio_path, ref_text, args.language, tts_path,
                            tts_model_path=args.tts_model, speaker=args.speaker,
                            seed=args.tts_seed)
    nf = U.frames_for_audio(dur + args.tail_pause)
    vdur = U.video_dur_for_frames(nf)
    U.pad_audio_to(tts_path, vdur)
    print(f"[TTS] {tts_name} dur={dur:.2f}s video={vdur:.2f}s", flush=True)

    best = None  # (ok, mouth, path, seed)
    for sd in [int(x) for x in args.seeds.split(",")]:
        vpfx = f"{args.prefix}rg{sd}"
        wf = G.build_workflow(
            image_name=args.image, audio_name=tts_name, prompt_text=args.prompt,
            duration=vdur, seed=sd, prefix=vpfx, width=args.width, height=args.height,
            guide_strength=args.guide_strength, img_compression=args.img_compression)
        U.clean_prefix(vpfx)
        pid = U.submit(wf)
        out = U.wait_for(pid)
        mr = U.mouth_range(out) or 0.0
        ratio = bright_at_end(out)
        ok = ratio >= args.min_bright
        print(f"[LTX] seed={sd} {os.path.basename(out)} mouth={mr:.4f} bright@end={ratio:.2f} ok={ok}", flush=True)
        score = (1 if ok else 0, mr)
        if best is None or score > (best[0], best[1]):
            best = (score[0], mr, out, sd)
        if ok and mr >= args.min_mouth_range:
            break

    ok, mr, out, sd = best
    canonical = os.path.join(OUTPUT_DIR, f"{args.prefix}_00001_.mp4")
    shutil.copyfile(out, canonical)
    print(f"BEST seed={sd} bright_ok={bool(ok)} mouth={mr:.4f} -> {canonical}", flush=True)


if __name__ == "__main__":
    main()
