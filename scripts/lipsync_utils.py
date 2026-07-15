#!/usr/bin/env python3
"""Shared helpers for driving ComfyUI and post-processing lip-sync clips.

These are the reusable building blocks behind generate_lipsync_fast.py's
single-shot CLI: submitting a workflow and waiting for it by prompt_id
(instead of globbing for the newest file, which can return a stale clip from
a previous run), keeping audio/video length in sync on the LTX frame grid,
and a lightweight OpenCV mouth-openness metric used to detect "dead" clips
where the model ignored the audio and barely animated the mouth.

Used by generate_lipsync_fast.py and regen_seg.py; feel free to import these
directly if you're writing your own batch driver.

This module deliberately imports nothing from the other scripts: they import
it, not the other way round.
"""
import glob
import json
import math
import os
import time
import urllib.request

LTX_ROOT = os.path.expanduser(os.environ.get("LTX_MAC_ROOT", "~/work/ltx_mac"))
COMFYUI_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
OUTPUT_DIR = os.environ.get("LTX_OUTPUT_DIR", os.path.join(LTX_ROOT, "output"))
INPUT_DIR = os.environ.get("LTX_INPUT_DIR", os.path.join(LTX_ROOT, "input"))
FPS = 24


# ---------------------------------------------------------------------------
# ComfyUI submit / wait-by-prompt-id
# ---------------------------------------------------------------------------
def submit(wf):
    data = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(f"{COMFYUI_URL}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read()).get("prompt_id")


def wait_for(pid, timeout=1800, progress=False):
    """Block until the prompt finishes; return the produced mp4 path (from job outputs).

    Set progress=True to print a dot per poll, for interactive single-clip runs
    where the alternative is minutes of silence.
    """
    import sys
    start = time.time()
    ticks = 0
    while time.time() - start < timeout:
        if progress:
            ticks += 1
            sys.stdout.write("." if ticks % 12 else f" {int(time.time()-start)}s\n")
            sys.stdout.flush()
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFYUI_URL}/history/{pid}").read())
        except Exception:
            time.sleep(3)
            continue
        if pid in h:
            entry = h[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"prompt {pid} failed: {json.dumps(status)[:500]}")
            outs = entry.get("outputs", {})
            # SaveVideo is node 190; fall back to any node carrying media
            order = ["190"] + [k for k in outs if k != "190"]
            for nid in order:
                o = outs.get(nid, {})
                for key in ("videos", "gifs", "images"):
                    if o.get(key):
                        item = o[key][0]
                        sub = item.get("subfolder", "")
                        return os.path.join(OUTPUT_DIR, sub, item["filename"])
            # outputs present but no media entry yet -> give ComfyUI a moment
            time.sleep(2)
        time.sleep(3)
    raise TimeoutError(f"prompt {pid} did not finish within {timeout}s")


def clean_prefix(prefix):
    for f in glob.glob(os.path.join(OUTPUT_DIR, f"{prefix}_*.mp4")):
        try:
            os.remove(f)
        except OSError:
            pass


def frames_for_audio(tts_dur):
    """Smallest LTX-valid frame count (8n+1) whose duration >= the speech.

    LTX requires (frames-1) % 8 == 0 and internally rounds up to that grid.
    Snapping the requested video duration to some other grid first (e.g. the
    nearest 0.5s) and letting LTX round up separately makes the video run
    longer than the audio, and the model stretches the mouth motion across
    that extra time -> progressive lip/voice desync. Matching frames to the
    audio length directly removes that stretch.
    """
    raw = max(1.0, min(tts_dur, 30.0)) * FPS
    n = max(1, math.ceil((raw - 1) / 8))
    return n * 8 + 1


def video_dur_for_frames(num_frames):
    # build_workflow computes num_frames = int(FPS*duration)+1, so invert it.
    return (num_frames - 1) / FPS


def pad_audio_to(path, target_dur):
    """Append trailing silence so the audio spans >= target_dur.

    TrimAudioDuration only trims (never pads), so without this the audio latent
    would be shorter than the video latent and the two would not align in time.
    """
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(path)
    target = int(round(target_dur * sr))
    if audio.shape[0] >= target:
        return
    pad = target - audio.shape[0]
    if audio.ndim == 1:
        audio = np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])
    else:
        audio = np.concatenate([audio, np.zeros((pad, audio.shape[1]), dtype=audio.dtype)])
    sf.write(path, audio, sr)


def mouth_range(path):
    """Lip-motion proxy for a finished clip: spread (p90-p10) of mouth openness.

    LTX sometimes locks onto the still reference image and ignores the audio for a
    given seed (more likely at high guide_strength), producing a clip whose face
    barely moves - i.e. it isn't lip-syncing. A healthy talking clip has a clear
    openness spread (~0.03-0.04); a dead clip is ~0.01. Uses plain OpenCV Haar +
    a dark lip-gap proxy (same method as analyze_mouth.py). Returns None if
    OpenCV is unavailable or no face is found, so the caller can skip the check.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(path)
    vals = []
    last = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, 1.2, 5, minSize=(120, 120))
        if len(faces):
            last = max(faces, key=lambda b: b[2] * b[3])
        if last is None:
            continue
        x, y, w, h = last
        mx0, mx1 = x + int(0.30 * w), x + int(0.70 * w)
        my0, my1 = y + int(0.66 * h), y + int(0.92 * h)
        roi = gray[my0:my1, mx0:mx1]
        if roi.size == 0:
            continue
        thr = roi.mean() - 0.6 * roi.std()
        gap = int(((roi < thr).mean(axis=1) > 0.35).sum())
        vals.append(gap / float(h))
    cap.release()
    if not vals:
        return None
    a = np.array(vals)
    return float(np.percentile(a, 90) - np.percentile(a, 10))
