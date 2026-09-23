"""
RoadGuardian AI - combined video inference (tracking + anti-flicker + smoothing)
Runs the hazard model and the pothole model on the same video using
ByteTrack for stable IDs, a hold buffer so boxes don't blink off during
brief misses, and exponential smoothing on box coordinates so the box
edges don't jitter frame to frame even while actively detected.

Usage (defaults match the folder structure this script ships in):
    python detect.py

Custom paths:
    python detect.py --video video/myclip.mp4 --hazard models/hazard_best.pt --pothole models/pothole_best.pt --output output/result.mp4
"""

import argparse
import os
import cv2
from ultralytics import YOLO

HAZARD_COLOR = (0, 165, 255)   # orange (BGR)
POTHOLE_COLOR = (0, 0, 255)    # red (BGR)


class TrackMemory:
    """Holds each track's box for a few frames after detection is briefly
    lost (stops flicker), and smooths box coordinates over time so the
    box doesn't jitter/wobble even while continuously detected."""

    def __init__(self, hold_frames, smooth_alpha):
        self.hold_frames = hold_frames
        self.smooth_alpha = smooth_alpha  # 0-1. Lower = smoother/slower to move, higher = snappier/more jitter.
        self.tracks = {}  # id -> {box, cls, conf, last_seen, seen_this_frame}

    def _smooth_box(self, old_box, new_box):
        a = self.smooth_alpha
        return tuple(int(a * n + (1 - a) * o) for n, o in zip(new_box, old_box))

    def update(self, results, frame_idx):
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes
            for i in range(len(boxes)):
                tid = int(boxes.id[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                new_box = (x1, y1, x2, y2)
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])

                if tid in self.tracks:
                    smoothed_box = self._smooth_box(self.tracks[tid]["box"], new_box)
                else:
                    smoothed_box = new_box  # first time seeing this ID, nothing to smooth against yet

                self.tracks[tid] = {
                    "box": smoothed_box,
                    "cls": cls_id,
                    "conf": conf,
                    "last_seen": frame_idx,
                    "seen_this_frame": True,
                }

        stale_ids = []
        for tid, t in self.tracks.items():
            if t["last_seen"] != frame_idx:
                t["seen_this_frame"] = False
            if frame_idx - t["last_seen"] > self.hold_frames:
                stale_ids.append(tid)
        for tid in stale_ids:
            del self.tracks[tid]

        return self.tracks


def draw_tracks(frame, tracks, color, model_names, id_prefix):
    for tid, t in tracks.items():
        x1, y1, x2, y2 = t["box"]
        label = f"{id_prefix}{tid} {model_names[t['cls']]} {t['conf']:.2f}"

        thickness = 2 if t["seen_this_frame"] else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="RoadGuardian AI combined video inference")
    parser.add_argument("--video", default="video/input_video.mp4", help="Path to input video")
    parser.add_argument("--hazard", default="models/hazard_best.pt", help="Path to hazard model weights")
    parser.add_argument("--pothole", default="models/pothole_best.pt", help="Path to pothole model weights")
    parser.add_argument("--output", default="output/result.mp4", help="Path to write annotated output video")
    parser.add_argument("--conf", type=float, default=0.15,
                         help="Confidence threshold for both models (lower = catches more, but more false positives)")
    parser.add_argument("--hold", type=int, default=10,
                         help="Frames to keep showing a box after the model briefly loses it")
    parser.add_argument("--smooth", type=float, default=0.4,
                         help="Box smoothing strength, 0-1. Lower = steadier/less jitter but slower to react, higher = snappier but more wobble")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video} — put your video in the video/ folder.")

    use_hazard = os.path.exists(args.hazard)
    use_pothole = os.path.exists(args.pothole)
    if not use_hazard and not use_pothole:
        raise FileNotFoundError(
            "Neither model was found. Put hazard_best.pt and/or pothole_best.pt in the models/ folder."
        )
    if not use_hazard:
        print(f"[!] Hazard model not found at {args.hazard} — running with pothole model only.")
    if not use_pothole:
        print(f"[!] Pothole model not found at {args.pothole} — running with hazard model only.")

    hazard_model = YOLO(args.hazard) if use_hazard else None
    pothole_model = YOLO(args.pothole) if use_pothole else None

    hazard_memory = TrackMemory(args.hold, args.smooth) if use_hazard else None
    pothole_memory = TrackMemory(args.hold, args.smooth) if use_pothole else None

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    print(f"Processing {total_frames} frames at {width}x{height}, {fps:.1f} fps...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if hazard_model is not None:
            hazard_results = hazard_model.track(
                frame, conf=args.conf, persist=True, tracker="bytetrack.yaml", verbose=False
            )[0]
            tracks = hazard_memory.update(hazard_results, frame_idx)
            frame = draw_tracks(frame, tracks, HAZARD_COLOR, hazard_model.names, id_prefix="H")

        if pothole_model is not None:
            pothole_results = pothole_model.track(
                frame, conf=args.conf, persist=True, tracker="bytetrack.yaml", verbose=False
            )[0]
            tracks = pothole_memory.update(pothole_results, frame_idx)
            frame = draw_tracks(frame, tracks, POTHOLE_COLOR, pothole_model.names, id_prefix="P")

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total_frames} frames done")

    cap.release()
    writer.release()
    print(f"Done. Output saved to: {args.output}")


if __name__ == "__main__":
    main()
