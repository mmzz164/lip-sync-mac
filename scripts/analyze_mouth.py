#!/usr/bin/env python3
"""Measure mouth-openness of a talking-head clip with plain OpenCV (no models).

Pipeline (robust for a fixed, frontal, still-head avatar):
  1. Haar face detect -> face box (reuse previous box if a frame misses).
  2. Mouth ROI = lower-center of the face box.
  3. Openness proxy per frame = vertical extent of the dark lip-gap band inside
     the ROI (rows that are substantially darker than the ROI mean), normalized
     by face height. Closed mouth -> thin lip line (~0). Open -> tall dark gap.

Prints one TSV line per clip: name frames mean max p90 open%.
Usage: analyze_mouth.py clip1.mp4 [clip2.mp4 ...]
"""
import sys
import cv2
import numpy as np

FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
OPEN_THR = 0.02  # normalized openness above this counts as "mouth open"


def openness_series(path):
    cap = cv2.VideoCapture(path)
    vals = []
    last = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = FACE.detectMultiScale(gray, 1.2, 5, minSize=(120, 120))
        if len(faces):
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            last = (x, y, w, h)
        elif last is not None:
            x, y, w, h = last
        else:
            continue
        mx0, mx1 = x + int(0.30 * w), x + int(0.70 * w)
        my0, my1 = y + int(0.66 * h), y + int(0.92 * h)
        roi = gray[my0:my1, mx0:mx1]
        if roi.size == 0:
            continue
        thr = roi.mean() - 0.6 * roi.std()
        dark_frac_per_row = (roi < thr).mean(axis=1)
        gap_rows = int((dark_frac_per_row > 0.35).sum())
        vals.append(gap_rows / float(h))
    cap.release()
    return np.array(vals)


def main():
    for p in sys.argv[1:]:
        a = openness_series(p)
        name = p.split("/")[-1]
        if a.size == 0:
            print(f"{name}\tNO_FACE")
            continue
        base = np.percentile(a, 10)   # closed-mouth baseline within the clip
        peak = np.percentile(a, 90)   # typical open peak (robust to outliers)
        print(f"{name}\tframes={a.size}\tmax={a.max():.4f}\tp90={peak:.4f}\t"
              f"base(p10)={base:.4f}\trange(p90-p10)={peak-base:.4f}")


if __name__ == "__main__":
    main()
