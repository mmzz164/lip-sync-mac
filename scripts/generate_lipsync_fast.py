#!/usr/bin/env python3
"""End-to-end Qwen3-TTS + LTX-2.3 lip-sync video generator.

Pipeline:
  1. Qwen3-TTS Voice Clone: text + reference wav -> synthesized wav (24kHz mono)
  2. ComfyUI API: image + wav -> LTX-2.3 I2V with audio conditioning
  3. Output: lip-synced MP4 in <LTX_MAC_ROOT>/output/

Usage:
  python generate_lipsync_fast.py \
    --text "I am a software engineer, and I love writing code every day." \
    --image character.png \
    --ref-audio voice_sample.wav \
    --ref-text "<exact transcript of voice_sample.wav>" \
    --duration 5 \
    --prompt "A young person in a bright office, speaking clearly at the camera, natural lip sync"
"""
import argparse, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lipsync_utils as U

LTX_ROOT = U.LTX_ROOT
COMFYUI_URL = U.COMFYUI_URL
OUTPUT_DIR = U.OUTPUT_DIR
INPUT_DIR = U.INPUT_DIR
FPS = U.FPS
# 320x576 at the 2-step default is the benchmarked config (see README
# "Performance"); raise for quality, keeping both dimensions multiples of 32.
DEFAULT_WIDTH, DEFAULT_HEIGHT = 320, 576
NEG_PROMPT = (
    "blurry, low quality, still frame, ugly, distorted face, "
    "text, letters, words, japanese characters, chinese characters, korean characters, english text, "
    "hiragana, katakana, kanji, subtitles, captions, closed captions, japanese subtitles, "
    "on-screen text, chyron, lower-third graphic, text overlay, caption bar, banner text, ticker tape, "
    "watermark, signature, logo overlay, typography, speech bubble, UI text, extra signage, "
    "illegible writing, scribbles, random text, sign overlay, label, nameplate"
)

QWEN3_TTS_LOCAL = os.environ.get("QWEN3_TTS_LOCAL", os.path.join(LTX_ROOT, "models/tts/Qwen3-TTS-12Hz-1.7B-Base"))


PROMPTS_DIR = os.path.join(LTX_ROOT, "models/tts/prompts")

# --- Model setup for Apple Silicon -----------------------------------------
# A full fp8 all-in-one checkpoint (~28GB, loaded via CheckpointLoaderSimple)
# runs fine on a CUDA server but is unnecessarily heavy for a 48GB Mac. Here we
# use a GGUF-quantized DiT plus separate VAE/text-encoder checkpoints instead
# (loaded piecewise, ~16-18GB total). Because the distilled-1.1 GGUF is
# already a self-distilled checkpoint, the separate ~7GB distillation LoRA is
# not needed. Set LTX_LORA if you're using a non-distilled ("dev") GGUF that
# still needs the LoRA (LoraLoaderModelOnly is inserted only when set).
GGUF_MODEL = os.environ.get("LTX_GGUF", "ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf")
VIDEO_VAE = os.environ.get("LTX_VIDEO_VAE", "ltx-2.3-22b-dev_video_vae.safetensors")
AUDIO_VAE = os.environ.get("LTX_AUDIO_VAE", "ltx-2.3-22b-dev_audio_vae.safetensors")
TEXT_ENCODER = os.environ.get("LTX_TEXT_ENCODER", "gemma_3_12B_it_fp4_mixed.safetensors")
CONNECTORS = os.environ.get("LTX_CONNECTORS", "ltx-2.3-22b-distilled_embeddings_connectors.safetensors")
TE_DEVICE = os.environ.get("LTX_TE_DEVICE", "default")  # set to "cpu" if fp4 text-encoder OOMs on MPS
LORA_NAME = os.environ.get("LTX_LORA", "")
LORA_STRENGTH = float(os.environ.get("LTX_LORA_STRENGTH", "0.5"))
# Default sampler schedule: 2 steps. Benchmarked on an M4 Pro (48GB) at
# 320x576: mouth-motion and brightness checks pass and it is noticeably
# faster than more steps, at some quality risk (see README "Tuning" section).
# Fall back to more steps for higher quality:
#   3 steps: LTX_SIGMAS="1.0, 0.98125, 0.725, 0.0"
#   4 steps: LTX_SIGMAS="1.0, 0.98125, 0.909375, 0.421875, 0.0"
#   8 steps (server-grade quality): LTX_SIGMAS="1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
SIGMAS = os.environ.get("LTX_SIGMAS") or "1.0, 0.909375, 0.0"


def load_voice_prompt(path_or_name):
    """Restore a saved voice embedding (.safetensors) into a 1-item VoiceClonePromptItem list."""
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
    from safetensors.torch import load_file
    from safetensors import safe_open
    if not path_or_name.endswith(".safetensors"):
        path_or_name = path_or_name + ".safetensors"
    path = path_or_name if os.path.isabs(path_or_name) else os.path.join(PROMPTS_DIR, path_or_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"voice prompt not found: {path}")
    tensors = load_file(path)
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata() or {}
    item = VoiceClonePromptItem(
        ref_code=tensors.get("ref_code"),
        ref_spk_embedding=tensors["ref_spk_embedding"],
        x_vector_only_mode=metadata.get("x_vector_only_mode", "False") == "True",
        icl_mode=metadata.get("icl_mode", "False") == "True",
        ref_text=metadata.get("ref_text"),
    )
    print(f"[TTS] loaded voice prompt: {path}")
    return [item]


def tts_voice_clone(text, ref_audio_path, ref_text, language, out_wav,
                    voice_prompt=None, tts_model_path=None,
                    speaker=None, instruct=None, use_voice_design=False,
                    pitch_shift_semitones=0.0, seed=None):
    """Run TTS, switching between clone / custom_voice / voice_design based on the given args.

    Qwen3-TTS samples autoregressively, so without a seed the same text yields a
    different take (and a different length) every run. Seeding it here is what
    makes --seed reproduce a whole clip rather than just the video noise.
    """
    import torch, soundfile as sf
    from qwen_tts import Qwen3TTSModel
    if seed is not None:
        torch.manual_seed(seed)
    t0 = time.time()
    model_path = tts_model_path or QWEN3_TTS_LOCAL
    print(f"[TTS] loading {model_path}")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=os.environ.get("TTS_DEVICE", "mps"),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print(f"[TTS] loaded in {time.time()-t0:.1f}s, generating...")
    t1 = time.time()
    gen_kwargs = dict(text=text, language=language)
    if use_voice_design:
        # Voice Design requires --instruct to describe the target voice.
        if not instruct:
            raise ValueError("VoiceDesign mode requires --instruct (voice description)")
        gen_kwargs["instruct"] = instruct
        wavs, sr = model.generate_voice_design(**gen_kwargs)
    else:
        if instruct:
            gen_kwargs["instruct"] = instruct
        if speaker:
            gen_kwargs["speaker"] = speaker
            wavs, sr = model.generate_custom_voice(**gen_kwargs)
        elif voice_prompt is not None:
            gen_kwargs["voice_clone_prompt"] = voice_prompt
            wavs, sr = model.generate_voice_clone(**gen_kwargs)
        else:
            gen_kwargs["ref_audio"] = ref_audio_path
            gen_kwargs["ref_text"] = ref_text
            wavs, sr = model.generate_voice_clone(**gen_kwargs)
    audio_out = wavs[0]
    if pitch_shift_semitones and abs(pitch_shift_semitones) > 1e-3:
        import librosa
        print(f"[TTS] applying pitch shift {pitch_shift_semitones:+.1f} semitones")
        audio_out = librosa.effects.pitch_shift(audio_out, sr=sr, n_steps=pitch_shift_semitones)
    sf.write(out_wav, audio_out, sr)
    dur = len(audio_out) / sr
    print(f"[TTS] wrote {out_wav}  duration={dur:.2f}s sr={sr}  gen_time={time.time()-t1:.1f}s")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return dur


def build_workflow(image_name, audio_name, prompt_text, duration, seed, prefix, width, height,
                   guide_strength=0.94, img_compression=24):
    # LTX returns ceil(length/8)*8 + 1 frames, so asking for 8n+1 (the shape the
    # output takes) rounds up a whole latent step and hands back 8 extra frames:
    # length=25 -> 33, 145 -> 153. Ask for the 8n below it instead and the count
    # comes back exactly. Measured on an M4 Pro across 24/25, 48/49, 144/145.
    num_frames = int(FPS * duration)
    wf = {
        "100": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": GGUF_MODEL}},
        "101": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": TEXT_ENCODER, "ckpt_name": CONNECTORS, "device": TE_DEVICE}},
        "102": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": AUDIO_VAE}},
        "103": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "110": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": ["101", 0]}},
        "111": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG_PROMPT, "clip": ["101", 0]}},
        "112": {"class_type": "LTXVConditioning", "inputs": {"positive": ["110", 0], "negative": ["111", 0], "frame_rate": float(FPS)}},
        "120": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "130": {"class_type": "LTXVPreprocess", "inputs": {"image": ["120", 0], "img_compression": img_compression}},
        "121": {"class_type": "LoadAudio", "inputs": {"audio": audio_name}},
        "122": {"class_type": "TrimAudioDuration", "inputs": {"audio": ["121", 0], "start_index": 0, "duration": duration}},
        "140": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width, "height": height, "length": num_frames, "batch_size": 1}},
        "141": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["122", 0], "audio_vae": ["102", 0]}},
        "142": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 8, "height": 8}},
        "143": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["141", 0], "mask": ["142", 0]}},
        "150": {"class_type": "LTXVAddGuide", "inputs": {"positive": ["112", 0], "negative": ["112", 1], "vae": ["103", 0], "latent": ["140", 0], "image": ["130", 0], "frame_idx": 0, "strength": guide_strength}},
        "151": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["150", 2], "audio_latent": ["143", 0]}},
        "160": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS}},
        "161": {"class_type": "CFGGuider", "inputs": {"model": ["100", 0], "positive": ["150", 0], "negative": ["150", 1], "cfg": 1.0}},
        "162": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}},
        "163": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "164": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["163", 0], "guider": ["161", 0], "sampler": ["162", 0], "sigmas": ["160", 0], "latent_image": ["151", 0]}},
        "170": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["164", 0]}},
        "171": {"class_type": "VAEDecode", "inputs": {"samples": ["170", 0], "vae": ["103", 0]}},
        "172": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["170", 1], "audio_vae": ["102", 0]}},
        "180": {"class_type": "CreateVideo", "inputs": {"images": ["171", 0], "fps": float(FPS), "audio": ["172", 0]}},
        "190": {"class_type": "SaveVideo", "inputs": {"video": ["180", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}},
    }
    if LORA_NAME:
        wf["105"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["100", 0], "lora_name": LORA_NAME, "strength_model": LORA_STRENGTH}}
        wf["161"]["inputs"]["model"] = ["105", 0]
    return wf


def submit_and_wait(wf, timeout=1800):
    """Queue the workflow and return the mp4 this specific job produced.

    Waits on the prompt_id rather than globbing output/ for the newest matching
    file: a rerun with an already-used --prefix would match last run's clip on
    the first poll and report it as a fresh success.
    """
    start = time.time()
    try:
        pid = U.submit(wf)
        print(f"[LTX] queued prompt_id={pid}")
    except Exception as e:
        print(f"[LTX] submit error: {e}")
        if hasattr(e, "read"):
            print(e.read().decode()[:2000])
        return None
    try:
        out = U.wait_for(pid, timeout=timeout, progress=True)
    except (RuntimeError, TimeoutError) as e:
        print(f"\n[LTX] {e}")
        return None
    print(f"\n[LTX] output: {out}  elapsed={time.time()-start:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Text to speak")
    ap.add_argument("--image", required=True, help="Character image in input/ (or an absolute path)")
    ap.add_argument("--ref-audio", default=None, help="Reference audio in input/ for voice cloning (ignored if --voice-prompt/--speaker given)")
    ap.add_argument("--ref-text", default=None, help="Exact transcript of --ref-audio. Optional if a matching <name>.txt sits beside the audio; one of the two is required, since a transcript that does not match quietly degrades the cloned voice.")
    ap.add_argument("--voice-prompt", default=None, help="Saved voice embedding name/path (.safetensors) in models/tts/prompts/. If given, overrides --ref-audio/--ref-text.")
    ap.add_argument("--tts-model", default=None, help="Path to a fine-tuned Qwen3-TTS model directory. Defaults to the Base model.")
    ap.add_argument("--speaker", default=None, help="Speaker name for custom_voice mode (FT speaker or preset). When given, uses generate_custom_voice.")
    ap.add_argument("--instruct", default=None, help="Natural-language voice/style instruction for TTS")
    ap.add_argument("--voice-design", action="store_true", help="Use VoiceDesign API (requires --tts-model pointing to VoiceDesign model and --instruct)")
    ap.add_argument("--pitch-shift", type=float, default=0.0, help="Post-process pitch shift in semitones (+3 = 3 semitones higher, -2 = lower)")
    ap.add_argument("--language", default="English", choices=["Auto", "English", "Japanese", "Chinese"])
    ap.add_argument("--duration", type=float, default=5.0, help="Video duration in seconds (ignored if --auto-duration)")
    ap.add_argument("--auto-duration", action="store_true", help="Match video duration to the generated TTS audio length (plus a 0.2s tail), snapped up to LTX's frame grid")
    ap.add_argument("--prompt", default="A person speaking clearly at the camera, natural lip sync, medium shot")
    ap.add_argument("--seed", type=int, default=None, help="Seeds both the TTS take and the video noise. Omitted = random, and the clip is not reproducible.")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--guide-strength", type=float, default=0.94, help="LTXVAddGuide.strength. 1.0=tight to ref image (stiff), 0.7-0.85=looser (more movement, more identity drift)")
    ap.add_argument("--img-compression", type=int, default=24, help="LTXVPreprocess.img_compression. higher=more pixel abstraction=more freedom (and more identity drift)")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    prefix = args.prefix or f"lipsync_{int(time.time())}"
    tts_wav_name = f"{prefix}_tts.wav"
    tts_wav_path = os.path.join(INPUT_DIR, tts_wav_name)

    voice_prompt_items = None
    ref_audio_path = None
    ref_text = args.ref_text
    if args.voice_design:
        if not args.instruct:
            sys.exit("--voice-design requires --instruct")
    elif args.speaker:
        pass  # custom_voice mode: nothing else to resolve
    elif args.voice_prompt:
        voice_prompt_items = load_voice_prompt(args.voice_prompt)
    else:
        if not args.ref_audio:
            sys.exit("need one of --ref-audio, --voice-prompt, --speaker, or --voice-design")
        ref_audio_path = args.ref_audio if os.path.isabs(args.ref_audio) else os.path.join(INPUT_DIR, args.ref_audio)
        if not os.path.exists(ref_audio_path):
            sys.exit(f"missing: {ref_audio_path}")
        ref_text = args.ref_text or U.ref_text_for(ref_audio_path)
        if not ref_text:
            sys.exit(f"no transcript for {os.path.basename(ref_audio_path)}: pass --ref-text, "
                     f"or put it in {os.path.splitext(ref_audio_path)[0] + '.txt'}")
    img_abs = args.image if os.path.isabs(args.image) else os.path.join(INPUT_DIR, args.image)
    if not os.path.exists(img_abs):
        sys.exit(f"missing: {img_abs}")

    print("=" * 60)
    print(f" text      : {args.text!r}")
    print(f" image     : {args.image}")
    print(f" voice     : {args.speaker or args.voice_prompt or args.ref_audio or 'voice-design'}")
    print(f" language  : {args.language}")
    print(f" duration  : {'auto' if args.auto_duration else str(args.duration) + 's'}")
    print(f" seed      : {seed}")
    print(f" prefix    : {prefix}")
    print("=" * 60)

    tts_dur = tts_voice_clone(args.text, ref_audio_path, ref_text, args.language, tts_wav_path,
                              voice_prompt=voice_prompt_items,
                              tts_model_path=args.tts_model,
                              speaker=args.speaker,
                              instruct=args.instruct,
                              use_voice_design=args.voice_design,
                              pitch_shift_semitones=args.pitch_shift,
                              seed=seed)
    # Snap the video length straight onto LTX's 8n+1 frame grid (see
    # lipsync_utils.frames_for_audio). Rounding to some intermediate grid and
    # letting LTX round up again on top leaves the video longer than the
    # speech, and the model stretches the mouth motion to fill it -> the lips
    # drift further out of sync the longer the clip runs.
    requested = (tts_dur + 0.2) if args.auto_duration else args.duration
    video_dur = U.video_dur_for_frames(U.frames_for_audio(requested))
    if args.auto_duration:
        print(f"[LTX] auto duration: tts={tts_dur:.2f}s -> video={video_dur:.2f}s")
    elif abs(video_dur - args.duration) > 0.01:
        print(f"[LTX] duration {args.duration}s -> {video_dur:.2f}s (LTX frame grid)")
    # TrimAudioDuration only ever trims, so a clip shorter than the video would
    # leave the audio latent shorter than the video latent and the two would
    # not line up in time.
    U.pad_audio_to(tts_wav_path, video_dur)

    wf = build_workflow(
        image_name=args.image,
        audio_name=tts_wav_name,
        prompt_text=args.prompt,
        duration=video_dur,
        seed=seed,
        prefix=prefix,
        width=args.width,
        height=args.height,
        guide_strength=args.guide_strength,
        img_compression=args.img_compression,
    )

    # save workflow for debugging / reruns
    wf_path = os.path.join(OUTPUT_DIR, f"{prefix}_workflow.json")
    with open(wf_path, "w") as f:
        json.dump(wf, f, indent=2)
    print(f"[LTX] workflow saved: {wf_path}")

    out_video = submit_and_wait(wf)
    if out_video:
        print(f"\n=== DONE ===")
        print(f"video: {out_video}")
        print(f"view : {COMFYUI_URL}/view?filename={os.path.basename(out_video)}")
    else:
        sys.exit("FAILED")


if __name__ == "__main__":
    main()
