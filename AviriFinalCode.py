#!/usr/bin/env python3
"""
Text + single‐face portrait → edge‐tts → SadTalker → final talking‐head mp4
 • Auto 5‐point frontal alignment (eyes/nose/mouth)
 • Face crop + generous margin
 • rembg “alpha_matting” grey background
 • Optional SRT subtitles (not shown here, but easily re‐added)
"""

import os
import sys
import uuid
import argparse
import asyncio
import subprocess
import io

# ─── External packages ─────────────────────────────────────────────────────────
import edge_tts
import face_recognition
import numpy as np
from PIL import Image
from rembg import remove
import face_alignment
from skimage import transform as trans
# ────────────────────────────────────────────────────────────────────────────────

# ─── ① Voice selection ─────────────────────────────────────────────────────────
VOICE_MAP = {
    "india": {
        "female": "en-IN-NeerjaNeural",
        "male":   "en-IN-PrabhatNeural",
    },
}
DEFAULT_VOICE = {
    "female": "en-US-JennyNeural",
    "male":   "en-US-GuyNeural",
}
def select_voice(nationality: str, gender: str) -> str:
    nat, gen = nationality.lower(), gender.lower()
    return VOICE_MAP.get(nat, {}).get(gen, DEFAULT_VOICE.get(gen, DEFAULT_VOICE["female"]))
# ────────────────────────────────────────────────────────────────────────────────

# ─── ② Frontal‐face alignment helper ────────────────────────────────────────────
# Uses 5 landmarks: left‐eye, right‐eye, nose tip, left mouth, right mouth
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

def align_face_rgb(rgb_np: np.ndarray, out_size: int = 512) -> np.ndarray:
    """
    Take an H×W×3 RGB image, detect 5‐point landmarks,
    compute a similarity transform to frontal, and
    return a out_size×out_size uint8 RGB.
    """
    preds = fa.get_landmarks(rgb_np)
    if preds is None:
        # fallback: center‐crop + resize
        h, w = rgb_np.shape[:2]
        m = min(h, w)
        crop = rgb_np[(h-m)//2:(h-m)//2+m, (w-m)//2:(w-m)//2+m]
        return np.array(Image.fromarray(crop).resize((out_size, out_size), Image.BICUBIC))

    pts = preds[0][:, :2]
    # source: 5 chosen points
    src = np.stack([pts[36], pts[45], pts[30], pts[48], pts[54]])
    # destination: standardized positions in a 112×112 crop
    dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32) * (out_size / 112)

    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    warped = trans.warp(rgb_np, tform.inverse,
                        output_shape=(out_size, out_size),
                        preserve_range=True)
    return warped.astype(np.uint8)
# ────────────────────────────────────────────────────────────────────────────────

# ─── ③ Pre‐process: align → face‐crop → grey BG via rembg ────────────────────────
def preprocess_face(img_path: str, work_dir: str, margin: float = 0.45) -> str:
    # 1) load & align
    orig = face_recognition.load_image_file(img_path)           # BGR‐ordered array
    aligned = align_face_rgb(orig, out_size=512)               # now frontal 512×512

    # 2) detect face box on aligned image
    locs = face_recognition.face_locations(aligned, model="hog")
    if not locs:
        raise RuntimeError("No face detected after alignment.")
    top, right, bottom, left = locs[0]
    h, w = aligned.shape[:2]
    # 3) expand by margin
    dh, dw = int((bottom - top) * margin), int((right - left) * margin)
    t = max(0, top - dh);    b = min(h, bottom + dh)
    l = max(0, left - dw);   r = min(w, right + dw)
    face_crop = aligned[t:b, l:r]

    # 4) background removal → RGBA
    pil_crop = Image.fromarray(face_crop)
    buf = io.BytesIO(); pil_crop.save(buf, format="PNG")
    rgba_bytes = remove(
        buf.getvalue(),
        alpha_matting=True,
        alpha_matting_foreground_threshold=230,
        alpha_matting_background_threshold=5,
        alpha_matting_erode_size=0
    )
    face_rgba = Image.open(io.BytesIO(rgba_bytes)).convert("RGBA")

    # 5) composite over solid grey
    grey_bg = Image.new("RGBA", face_rgba.size, (128, 128, 128, 255))
    grey_bg.paste(face_rgba, mask=face_rgba.split()[3])

    # 6) ensure final is exactly 512×512 RGB
    final = grey_bg.convert("RGB").resize((512, 512), Image.BICUBIC)
    out_path = os.path.join(work_dir, "processed_face.png")
    final.save(out_path)
    print(f"[Pre] Saved aligned & cropped face → {out_path}")
    return out_path
# ────────────────────────────────────────────────────────────────────────────────

# ─── ④ edge‐tts: text → speech.wav ──────────────────────────────────────────────
async def generate_speech(text: str, voice: str, out_path: str):
    print(f"[TTS] voice={voice}")
    await edge_tts.Communicate(text, voice).save(out_path)
    print(f"[TTS] Saved audio → {out_path}")

def run_tts(text: str, voice: str, out_path: str):
    try:
        asyncio.run(generate_speech(text, voice, out_path))
    except Exception as e:
        print("[ERROR][TTS]", e); sys.exit(1)
# ────────────────────────────────────────────────────────────────────────────────

# ─── ⑤ SadTalker: image + wav → silent mp4 ─────────────────────────────────────
def run_sadtalker(image_path: str, wav_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "inference.py",
        "--source_image", image_path,
        "--driven_audio", wav_path,
        "--enhancer", "gfpgan",
        "--checkpoint_dir", "checkpoints",
        "--result_dir", out_dir,
    ]
    print("[SadTalker] ", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("[ERROR][SadTalker]", e); sys.exit(1)

    mp4s = [f for f in os.listdir(out_dir) if f.lower().endswith(".mp4")]
    if not mp4s:
        print("[ERROR][SadTalker] No output .mp4"); sys.exit(1)
    return os.path.join(out_dir, mp4s[0])
# ────────────────────────────────────────────────────────────────────────────────

# ─── ⑥ Merge audio + video ──────────────────────────────────────────────────────
def merge_audio(video_path: str, audio_path: str, out_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", out_path
    ]
    print("[ffmpeg] Merging →", out_path)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("[ERROR][ffmpeg]\n", e.stdout.decode()); sys.exit(1)
    print("[ffmpeg] Done.")
# ────────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Text+Image → talking-head mp4")
    p.add_argument("text",  help="Script to speak")
    p.add_argument("image", help="Portrait image file")
    p.add_argument("--gender", choices=["male","female"], default="female")
    p.add_argument("--nat",    default="")
    args = p.parse_args()

    if not os.path.isfile(args.image):
        print("[ERROR] Image missing:", args.image); sys.exit(1)

    work = f"tmp_{uuid.uuid4().hex}"
    os.makedirs(work, exist_ok=True)

    wav      = os.path.join(work, "speech.wav")
    vid_dir  = os.path.join(work, "video"); os.makedirs(vid_dir, exist_ok=True)
    final_mp4 = os.path.join(vid_dir, "final_with_audio.mp4")

    # 1) TTS
    voice = select_voice(args.nat, args.gender)
    run_tts(args.text, voice, wav)

    # 2) Preprocess & alignment
    processed = preprocess_face(args.image, work)

    # 3) SadTalker silent video
    silent = run_sadtalker(processed, wav, vid_dir)

    # 4) Merge
    merge_audio(silent, wav, final_mp4)

    print(f"\n[✅] Complete! → {final_mp4}")

if __name__ == "__main__":
    main()
