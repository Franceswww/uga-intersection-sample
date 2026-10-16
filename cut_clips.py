"""
cut_clips.py - cut the same 15-minute window from every intersection's day file (stream copy).

    python cut_clips.py --date 2026-03-23 --start 13:00 --minutes 15 --out clips

The NVR files start at 00:00 and are cut without re-encoding, so each clip begins at the
last keyframe before the requested time (up to ~66 s early); the true start of each clip is
written to clips/manifest.csv. Hooper's file needs the stsd patch first (see README).
"""
import argparse
import csv
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

NAS = Path(r"\\172.22.112.233\incoming")
FF = r"C:\Users\qw28137\venvs\uga-yolo\Lib\site-packages\static_ffmpeg\bin\win32"
SITES = [  # no, tag, NAS folder, camera used by the zone files, zones stem
    (1, "BaxterHull", "Baxter at Hull (CSCS) - UGA-C.1.5", "02", "Baxter_at_Hull"),
    (2, "BocockFounders", "Bocock Street at Founders (CSCS) - UGA-C.2.34", "01", "Bocock_Street"),
    (3, "BookstoreSW", "Bookstore Southwest (CSCS) - UGA-C.4.28", "02", "Bookstore_Southwest"),
    (4, "LumpkinBaldwin", "Lumpkin at Baldwin (CSCS) - UGA-C.1.11", "02", "Lumpkin_at_Baldwin"),
    (5, "LumpkinBaxter", "Lumpkin at Baxter (CSCS) - UGA-C.1.12", "02", "Lumpkin_at_Baxter"),
    (6, "LumpkinCedar", "Lumpkin at Cedar (CSCS) - UGA-C.2.4", "04", "Lumpkin_at_Cedar"),
    (7, "LumpkinNoName", "Lumpkin at No Name Park (CSCS) - UGA-C.1.13", "02", "Lumpkin_at_No_Name_Park"),
    (8, "LumpkinTate", "Lumpkin at Tate  (CSCS) - UGA-C.2.3", "02", "Lumpkin_at_Tate"),
    (9, "MLCQuad", "MLC Georgia Quad (CSCS) - UGA-C.2.29", None, "MLC_Georgia_Quad"),
    (10, "SanfordBaldwin", "Sanford at Baldwin (CSCS) - UGA-C.2.6", "02", "Sanford_at_Baldwin"),
    (11, "SanfordCedar", "Sanford at Cedar (CSCS) - UGA-C.2.5", "02", "Sanford_at_Cedar"),
    (12, "SanfordHooper", "Sanford at Hooper Street (CSCS) - UGA-C.6.20", "01", "Sanford_at_Hooper_Street"),
    (13, "SanfordNB", "Sanford Drive at Northbound (CSCS) -UGA-C.5.6", "02", "Sanford_Drive_at_Northbound"),
    (14, "SanfordPhysics", "Sanford Drive at Physics Parking Lot (CSCS) - UGA-C.5.5", "03", "Sanford_Drive_at_Physics_Parking_lot"),
    (15, "StadiumWest", "Stadium West (CSCS) - UGA-C.2.7", "03", "Stadium_West"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-03-23"); ap.add_argument("--start", default="13:00"); ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--out", default="clips"); ap.add_argument("--sites", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = dt.date.fromisoformat(a.date); folder = f"{d.strftime('%B')} {d.day}, {d.year}"
    h, m = a.start.split(":"); t0 = int(h) * 3600 + int(m) * 60
    rows = []
    for no, tag, site, cam, zstem in SITES:
        if a.sites and str(no) not in a.sites.split(","):
            continue
        files = sorted((NAS / site / folder).glob("*.mp4"))
        if cam:
            files = [f for f in files if f"- {cam}-" in f.name]
        if not files:
            print(f"{no:2d} {tag}: no file"); continue
        src = files[0]
        # the file's own start (creation time) is a few seconds before midnight; find it
        pr = subprocess.run([f"{FF}\\ffprobe.exe", "-v", "error", "-show_entries", "format_tags=creation_time", "-of", "json", str(src)], capture_output=True, text=True).stdout
        try:
            ct = dt.datetime.fromisoformat(json.loads(pr)["format"]["tags"]["creation_time"].replace("Z", "+00:00")).astimezone(dt.timezone(dt.timedelta(hours=-4)))
            start_s = (ct - ct.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
            start_s = start_s - 86400 if start_s > 43200 else start_s
        except Exception:
            start_s = 0.0
        seek = t0 - start_s
        dst = os.path.join(a.out, f"{no:02d}_{tag}.mp4")
        r = subprocess.run([f"{FF}\\ffmpeg.exe", "-hide_banner", "-loglevel", "error", "-ss", f"{seek:.3f}", "-i", str(src), "-t", f"{a.minutes * 60:.0f}",
                            "-c", "copy", "-avoid_negative_ts", "make_zero", "-y", dst], capture_output=True, text=True)
        if r.returncode or not os.path.exists(dst):
            print(f"{no:2d} {tag}: ffmpeg failed ({r.stderr.strip()[:120]})"); continue
        # where the clip really starts: the keyframe ffmpeg snapped back to
        pk = subprocess.run([f"{FF}\\ffprobe.exe", "-v", "error", "-read_intervals", f"{max(seek - 80, 0):.0f}%{seek:.3f}", "-show_packets", "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout
        keys = [float(l.split(",")[0]) for l in pk.splitlines() if l.endswith("K__") or ",K" in l]
        kf = max([k for k in keys if k <= seek + 0.5], default=seek)
        real = start_s + kf
        rows.append(dict(no=no, site=tag, clip=os.path.basename(dst), camera=cam or "single", zones=f"{zstem}.json", requested_start=a.start,
                         actual_start=f"{int(real // 3600):02d}:{int(real % 3600 // 60):02d}:{real % 60:04.1f}", lead_s=round(seek - kf, 1), size_MB=round(os.path.getsize(dst) / 1e6, 1)))
        print(f"{no:2d} {tag:15s} starts {rows[-1]['actual_start']} ({rows[-1]['lead_s']} s before {a.start})  {rows[-1]['size_MB']} MB")
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
