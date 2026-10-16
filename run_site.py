"""
run_site.py - the study's full chain on one clip: track -> clean/stitch -> counts.

    python run_site.py --clip clips/11_SanfordCedar.mp4 --site Sanford_at_Cedar --out results/11_SanfordCedar
    python run_site.py --clip clips/04_LumpkinBaldwin.mp4 --site Lumpkin_at_Baldwin --out results/04_LumpkinBaldwin

Stages (each is skipped when its output already exists; --force redoes all):
  1. track_clip.py        YOLOv8m + BoT-SORT                        -> <out>/trajectories.csv
  2. stitch_tracks.py     the study's cleaner: spike repair, teleport and time-gap splits,
                          duplicate merge, frame-border filter, fragment re-identification
                                                                    -> <out>/clean_trajectories.csv
  3. flow_typed.py        zone semantics -> vehicle movements (entry arm -> exit arm) and
                          pedestrian trips with destinations          -> <out>/results/*
stitch_tracks.py and flow_typed.py are the study's own files, unchanged, so the numbers
match the study's. Two sites carry validated options (Lumpkin at Baldwin: --mouth-cut-arm E
and --gate-dir-share 0.6); they are applied automatically from SITE_FLAGS.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITE_FLAGS = {"Lumpkin_at_Baldwin": dict(mouth_cut_arm="E", gate_dir_share="0.6")}


def run(cmd, label):
    print(f"\n=== {label} ===\n    " + " ".join(str(c) for c in cmd[1:]), flush=True)
    rc = subprocess.run([str(c) for c in cmd]).returncode
    if rc:
        sys.exit(f"[{label}] failed with exit code {rc}")


def has_rows(p):
    return os.path.exists(p) and os.path.getsize(p) > 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True); ap.add_argument("--site", required=True, help="zone file stem, e.g. Sanford_at_Cedar")
    ap.add_argument("--out", required=True); ap.add_argument("--force", action="store_true")
    ap.add_argument("--annotate", action="store_true"); ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    zones = os.path.join(HERE, "zones", f"{a.site}.json"); names = os.path.join(HERE, "zones", f"{a.site}_names.json")
    raw = os.path.join(a.out, "trajectories.csv"); clean = os.path.join(a.out, "clean_trajectories.csv")
    py = sys.executable
    if a.force or not has_rows(raw):
        cmd = [py, os.path.join(HERE, "track_clip.py"), "--video", a.clip, "--out", a.out, "--device", a.device]
        if a.annotate: cmd.append("--annotate")
        if a.max_seconds: cmd += ["--max-seconds", a.max_seconds]
        run(cmd, "1/3 track")
    if a.force or not has_rows(clean):
        cmd = [py, os.path.join(HERE, "stitch_tracks.py"), "--traj", raw, "--zones", zones, "--names", names,
               "--outdir", a.out, "--name", "clean", "--border", 8, "--cut-vel-samples", 3]
        if SITE_FLAGS.get(a.site, {}).get("mouth_cut_arm"):
            cmd += ["--mouth-cut-arm", SITE_FLAGS[a.site]["mouth_cut_arm"]]
        run(cmd, "2/3 clean and stitch")
    cmd = [py, os.path.join(HERE, "flow_typed.py"), "--traj", clean, "--zones", zones, "--names", names, "--outdir", os.path.join(a.out, "results")]
    if SITE_FLAGS.get(a.site, {}).get("gate_dir_share"):
        cmd += ["--gate-dir-share", SITE_FLAGS[a.site]["gate_dir_share"]]
    run(cmd, "3/3 counts")
    print(f"\nresults in {os.path.join(a.out, 'results')}")
    for f in sorted(os.listdir(os.path.join(a.out, "results"))):
        print("   ", f)


if __name__ == "__main__":
    main()
