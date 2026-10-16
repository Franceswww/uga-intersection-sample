"""
track_clip.py - detect and track road users in a video clip (YOLO + BoT-SORT).

    python track_clip.py --video clips/11_SanfordCedar.mp4 --out results/11_SanfordCedar
    python track_clip.py --video clips/11_SanfordCedar.mp4 --out results/11_SanfordCedar --annotate

Writes <out>/trajectories.csv, one row per detected object per processed frame:
    frame      frame index in the clip (1-based)
    time_sec   frame / fps
    track_id   id kept by the tracker while the object stays in view
    class      Pedestrian / Bicycle / Passenger Vehicle / Motorcycle / Bus / Truck
    conf       detector confidence
    foot_x, foot_y   bottom-centre of the box = the point on the ground (pixels)
    x1, y1, x2, y2   bounding box (pixels)
and, with --annotate, <out>/annotated.mp4 (processed frames only).

Defaults are the settings used for the UGA 15-intersection study: every 2nd frame,
imgsz 1920, confidence 0.10 (low threshold on purpose: weak detections feed the
tracker's second association stage and reduce ID breaks), BoT-SORT with a 4 s buffer.
The clips are 4K at ~15 fps; one 15-minute clip takes ~10 minutes on an RTX 4090.
"""
import argparse
import csv
import os
import time

import cv2
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = {0: "Pedestrian", 1: "Bicycle", 2: "Passenger Vehicle", 3: "Motorcycle", 5: "Bus", 7: "Truck"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolov8m.pt", help="ultralytics weights; downloaded on first use")
    ap.add_argument("--tracker", default=os.path.join(HERE, "botsort.yaml"))
    ap.add_argument("--skip", type=int, default=2, help="process every n-th frame")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--device", default="0", help="GPU index, or 'cpu'")
    ap.add_argument("--annotate", action="store_true", help="also write annotated.mp4")
    ap.add_argument("--max-seconds", type=float, default=None, help="stop after this many seconds of video (for a quick test)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model = YOLO(a.model)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{os.path.basename(a.video)}: {W}x{H} @ {fps:.2f} fps, {total} frames; skip {a.skip}, imgsz {a.imgsz}, conf {a.conf}")

    writer = cv2.VideoWriter(os.path.join(a.out, "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), max(fps / a.skip, 1.0), (W, H)) if a.annotate else None
    f = open(os.path.join(a.out, "trajectories.csv"), "w", newline="", encoding="utf-8")
    wr = csv.writer(f); wr.writerow(["frame", "time_sec", "track_id", "class", "conf", "foot_x", "foot_y", "x1", "y1", "x2", "y2"])

    frame_no, processed, t0 = 0, 0, time.time()
    while True:
        if not cap.grab():
            break
        frame_no += 1
        if frame_no % a.skip:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = frame_no / fps
        if a.max_seconds and t > a.max_seconds:
            break
        res = model.track(frame, persist=True, tracker=a.tracker, conf=a.conf, iou=0.45, agnostic_nms=True,
                          classes=list(CLASS_NAMES), imgsz=a.imgsz, device=a.device, verbose=False)[0]
        if res.boxes is not None and res.boxes.id is not None:
            for i in range(len(res.boxes)):
                x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
                wr.writerow([frame_no, round(t, 3), int(res.boxes.id[i]), CLASS_NAMES.get(int(res.boxes.cls[i]), "unknown"),
                             round(float(res.boxes.conf[i]), 3), int((x1 + x2) / 2), int(y2), round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
        if writer is not None:
            writer.write(res.plot(line_width=2))
        processed += 1
        if processed % 200 == 0:
            el = time.time() - t0
            print(f"  frame {frame_no}/{total} ({100 * frame_no / max(total, 1):.0f}%)  {processed / el:.1f} processed frames/s")
    cap.release(); f.close()
    if writer is not None:
        writer.release()
    print(f"done: {processed} frames in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
