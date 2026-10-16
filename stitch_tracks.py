"""
stitch_tracks.py
----------------
Offline cleaning of the per-frame trajectories produced by run_clip_tracker.py.
CPU only. Reads X_trajectories.csv, writes NAME_trajectories.csv with two extra
columns: merged_id (the object's final identity) and maj_class (its class,
decided once over the whole merged track).

The chain, in order:

  trim_bad_tails            a departing object leaves a low-confidence box on
                            empty pavement; those trailing samples corrupt the
                            end velocity, so they are cut
  repair_spikes             a foot point far off the local trend AND against
                            the direction of travel is a glitch, not motion
  split_teleports           cut where the object could not physically have
                            moved that far, or after a long blind gap
  stitch                    rejoin fragments of one journey, subject to
                            prediction error, heading, and per-class limits
  merge_concurrent_duplicates
                            collapse two IDs that sit on ONE object at the
                            same instant (proved by concurrency, not guessed)
  class vote                confidence-weighted, over every sample the final
                            track owns

One principle runs through all of it: **every distance threshold is expressed
in units of the object's own apparent size**, never in raw pixels. This scene
spans a huge depth range — a car is ~700 px wide at the stop bar and ~20 px at
the far end of the street — so a fixed pixel budget means "a fifth of a car"
up close and "seven car lengths" in the distance. Vehicles are measured by box
width, pedestrians by box height (a person's height does not change with the
direction they face, and it was the steadiest measure available).

Usage:
  python stitch_tracks.py --traj X_trajectories.csv --zones Z.json
                          --outdir DIR --name NAME [--border 8]
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import cv2

MIN_DWELL_SECONDS      = 3.0
MIN_VISIBILITY_SECONDS = 5.0

VEHICLE = {"Passenger Vehicle", "Motorcycle", "Bus", "Truck"}
PED     = {"Pedestrian", "Bicycle"}


def class_group(cls):
    return "vehicle" if cls in VEHICLE else "ped"


def load_zones(zones_file):
    with open(zones_file) as f:
        data = json.load(f)
    raw = data["zones"] if "zones" in data and isinstance(data["zones"], dict) else data
    return {k: np.array(v, np.int32) for k, v in raw.items() if isinstance(v, list)}


def majority_class(sub):
    w = defaultdict(float)
    for cls, conf in zip(sub["class"], sub["conf"]):
        w[cls] += conf
    return max(w.items(), key=lambda x: x[1])[0]


def repair_spikes(df, window=5, thresh=150.0, dir_cos=-0.2):
    """Physics-based outlier rejection (user's rule): discard a foot point
    whose displacement is (a) far off the track's local median trend AND
    (b) NOT along the current direction of motion, with neighbors back on
    trend. Replaced by neighbor interpolation. Genuine fast motion is
    directional and monotonic, so it passes untouched."""
    df = df.sort_values(["track_id", "frame"]).copy()
    for col in ("foot_x", "foot_y"):
        df[col] = df[col].astype(float)

    g = df.groupby("track_id")
    medx = g["foot_x"].transform(lambda s: s.rolling(window, center=True,
                                                     min_periods=1).median())
    medy = g["foot_y"].transform(lambda s: s.rolling(window, center=True,
                                                     min_periods=1).median())
    resx = df["foot_x"] - medx
    resy = df["foot_y"] - medy
    res = np.hypot(resx, resy)

    # local motion direction from the median trend (robust to the spike itself)
    vdx = medx.groupby(df["track_id"]).diff()
    vdy = medy.groupby(df["track_id"]).diff()
    vnorm = np.hypot(vdx, vdy)
    # cos between residual (spike offset) and motion direction
    cos = (resx * vdx + resy * vdy) / (res * vnorm + 1e-9)
    moving = vnorm > 30

    prev_ok = res.groupby(df["track_id"]).shift(1).fillna(0) < thresh
    next_ok = res.groupby(df["track_id"]).shift(-1).fillna(0) < thresh
    off_trend = res > thresh
    # while moving: offset must be off-direction to be a glitch;
    # while stopped: any big offset with neighbors on trend is a glitch
    off_direction = np.where(moving, cos < dir_cos, True)
    spike = off_trend & off_direction & prev_ok & next_ok

    n_fixed = 0
    for col in ("foot_x", "foot_y"):
        interp = (df.groupby("track_id")[col].shift(1) +
                  df.groupby("track_id")[col].shift(-1)) / 2
        sel = spike & interp.notna()
        df.loc[sel, col] = interp[sel]
        n_fixed = int(sel.sum())
    return df, n_fixed


def split_teleports(df, max_step_veh=600.0, max_step_ped=300.0,
                    max_time_gap=2.5, vel_samples=3):
    """Break tracks at physically impossible jumps (ID-switch teleports) AND
    at detection gaps longer than max_time_gap seconds — a track that was
    blind for seconds may have re-attached to a different object (platoon
    chimera); the stitcher re-joins only if consistency checks pass.

    Body-scaled cuts (user rule, 2026-08-24), vehicles only. The old fixed
    600 px cap is a dozen body widths at the far end of the scene, so every
    in-track ID hand-off out there survived it: 215 of the 216 outlier
    movement segments at Baldwin carried exactly one such impossible step.
    Two criteria, both measured against the track's OWN recent motion
    (median velocity of the last vel_samples steps — one frame alone is
    dominated by box jitter, a car's median per-frame step is only 0.02 of
    its body width):
      A  prediction error: the point lands farther from where the car's own
         velocity says it should be than 1.5*speed*dt + 2 body widths;
      B  reversal: a genuinely moving car (>0.5 body width/s) steps more
         than one body width AGAINST its own heading (cos < -0.5) — a car
         cannot turn around within one frame.
    A falsely cut same-car pair is re-joined by the stitcher (heading and
    position agree); two different cars fail its scaled-distance and
    opposite-heading guards and stay apart."""
    df = df.sort_values(["track_id", "frame"]).copy()
    tid = df["track_id"]
    dx = df.groupby("track_id")["foot_x"].diff()
    dy = df.groupby("track_id")["foot_y"].diff()
    d = np.hypot(dx, dy)
    dt = df.groupby("track_id")["time_sec"].diff()
    is_veh = df["class"].isin(VEHICLE)
    cap = np.where(is_veh, max_step_veh, max_step_ped)
    new_track = ((d > cap) | (dt > max_time_gap)).fillna(False)
    n_old = int(new_track.sum())
    n_new = 0
    if vel_samples and vel_samples > 0 and {"x1", "x2"} <= set(df.columns):
        dt_s = dt.where(dt > 1e-6)
        vx, vy = dx / dt_s, dy / dt_s
        k = int(vel_samples)
        roll = lambda s_: s_.groupby(tid).transform(
            lambda g: g.shift(1).rolling(k, min_periods=k).median())
        vxp, vyp = roll(vx), roll(vy)
        bwr = (df["x2"] - df["x1"]).groupby(tid).transform(
            lambda g: g.rolling(5, min_periods=1).median())
        sp = np.hypot(vxp, vyp)
        pred_err = np.hypot(dx - vxp * dt_s, dy - vyp * dt_s)
        cut_a = pred_err > (1.5 * sp * dt_s + 2.0 * bwr)
        cosd = (dx * vxp + dy * vyp) / (d * sp).replace(0, np.nan)
        cut_b = (sp > 0.5 * bwr) & (cosd < -0.5) & (d > bwr)
        # C: absolute speed. Hand-offs glide across 2-4 frames, so the
        # windowed velocity chases the glide and criterion A never fires
        # (measured on every one of h17's 18 outliers). But the glide speed
        # itself is impossible: 8-72 body widths/s vs ~7 for the fastest
        # genuine head-on car. A falsely cut fast car is healed by the
        # stitcher; a chimera fails its scaled guards.
        cut_c = (d / dt_s) > 8.0 * bwr
        newcut = ((cut_a | cut_b | cut_c) & is_veh).fillna(False) & ~new_track
        n_new = int(newcut.sum())
        new_track = new_track | newcut
    df["track_id"] = df["track_id"].astype(np.int64) * 10000 + \
        new_track.groupby(tid).cumsum().astype(np.int64)
    print(f"[INFO] teleport cuts: {n_old} fixed-cap/time, "
          f"{n_new} body-scaled (vel_samples={vel_samples})")
    return df, int(new_track.sum())


def trim_bad_tails(df, max_tail=6, conf_thr=0.25, size_ratio=1.5):
    """Drop the garbage at the very end of a track.

    When a vehicle leaves the frame the detector often keeps firing for a
    frame or two on the empty road it left behind: a low-confidence box that
    suddenly changes size. Measured case: a 453-sample school-bus track whose
    final 2 samples were 0.21-confidence boxes twice the usual size, sitting
    on empty pavement. Those samples corrupt the end velocity and therefore
    the exit inference, so the tail is cut back to the last sample that still
    looks like the object.
    """
    df = df.sort_values(["track_id", "frame"]).copy()
    drop = []
    for tid, sub in df.groupby("track_id"):
        if len(sub) < 10:
            continue
        w = (sub["x2"] - sub["x1"]).to_numpy(float)
        conf = sub["conf"].to_numpy(float)
        ref = float(np.median(w[:-max_tail])) if len(w) > max_tail else float(np.median(w))
        idx = sub.index.to_numpy()
        for j in range(len(sub) - 1, max(len(sub) - 1 - max_tail, -1), -1):
            ratio = max(w[j], ref) / max(min(w[j], ref), 1.0)
            if conf[j] < conf_thr and ratio > size_ratio:
                drop.append(idx[j])
            else:
                break                     # stop at the first healthy sample
    if drop:
        df = df.drop(index=drop)
    return df, len(drop)


def _box_overlap(a, b):
    """Returns (IoU, IoM). IoM = intersection over the SMALLER box, so a
    partial detection nested inside the full one scores 1.0 even though its
    IoU is low — measured on this clip: same truck detected as head-only vs
    whole-vehicle gives IoU 0.33-0.42 but IoM 1.0."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + ab - inter
    return (inter / union if union > 0 else 0.0,
            inter / min(aa, ab) if min(aa, ab) > 0 else 0.0)


def merge_concurrent_duplicates(df, id_col="track_id", min_samples=3,
                                foot_sep_frac=0.25, min_iou=0.8, min_iom=0.9,
                                vehicles_only=True, log=None):
    """One vehicle can carry two tracker IDs at the same time — standing cars
    lose the motion model, and moving ones split when the box flickers.
    Two IDs are the SAME object when, on the frames where they coexist:
      * their foot points sit within foot_sep_frac of a box width, and
      * their boxes coincide (IoU >= min_iou) OR one is nested inside the
        other (IoM >= min_iom), and
      * this holds for at least min_samples frames (noise guard).
    Concurrency is what makes this an deduction rather than a guess: one
    object cannot be in two places, and two vehicles cannot share one.
    Measured here: same-object pairs sit at foot separation 0.01-0.07 box
    widths, genuinely different vehicles at 0.31-0.97; overlaps are typically
    0.3-1.5 s, so a 2 s requirement would discard 96% of the real cases."""
    df = df.copy()
    clscol = "maj_class" if "maj_class" in df.columns else "class"
    veh_ids = set(df[id_col]) if not vehicles_only else {
        t for t, s in df.groupby(id_col) if s[clscol].mode()[0] in VEHICLE}

    ev = {}
    for f, fr in df[df[id_col].isin(veh_ids)].groupby("frame"):
        if len(fr) < 2:
            continue
        box = fr[["x1", "y1", "x2", "y2"]].to_numpy(float)
        foot = fr[["foot_x", "foot_y"]].to_numpy(float)
        ids = fr[id_col].to_numpy()
        w = box[:, 2] - box[:, 0]
        for i in range(len(fr)):
            for j in range(i + 1, len(fr)):
                if ids[i] == ids[j]:
                    continue
                sep = float(np.hypot(*(foot[i] - foot[j]))) / max(w[i], w[j], 1)
                if sep >= foot_sep_frac:
                    continue
                iou, iom = _box_overlap(box[i], box[j])
                if iou < min_iou and iom < min_iom:
                    continue
                key = tuple(sorted((ids[i], ids[j])))
                lo, hi = (i, j) if ids[i] < ids[j] else (j, i)
                d = ev.setdefault(key, {"n": 0, "iou": [], "iom": [],
                                        "sep": [], "frames": [], "boxes": []})
                d["n"] += 1
                d["iou"].append(iou)
                d["iom"].append(iom)
                d["sep"].append(sep)
                d["frames"].append(f)
                # keep the geometry so the merge can be reviewed on the video
                d["boxes"].append((box[lo].tolist(), box[hi].tolist()))

    parent = {t: t for t in veh_ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # positions per (group, frame), so a candidate merge can be tested against
    # the physical fact that one object occupies one place at a time
    pos = {}
    for tid, sub in df[df[id_col].isin(veh_ids)].groupby(id_col):
        for f, x, y, w in zip(sub["frame"], sub["foot_x"], sub["foot_y"],
                              sub["x2"] - sub["x1"]):
            pos.setdefault(tid, {}).setdefault(f, []).append((x, y, w))

    members = {t: [t] for t in veh_ids}

    def would_conflict(ra, rb, max_spread=1.0):
        """Reject the merge if the combined track would hold two boxes more
        than max_spread box widths apart within a single frame. Union-find is
        transitive, so A~B and B~C silently implies A~C — a vehicle standing
        still for a minute otherwise becomes a hub that chains together every
        car that passes near it."""
        seen = {}
        for t in members[ra] + members[rb]:
            for f, pts in pos.get(t, {}).items():
                seen.setdefault(f, []).extend(pts)
        for f, pts in seen.items():
            if len(pts) < 2:
                continue
            # smaller box: two objects at different depths must not be
            # excused by the larger one's size
            w = min(p[2] for p in pts)
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                    if d / max(w, 1) > max_spread:
                        return True
        return False

    n_merged, n_rejected, records = 0, 0, []
    # strongest evidence first, so the best-supported merges claim their group
    for (a, b), d in sorted(ev.items(), key=lambda kv: -kv[1]["n"]):
        if d["n"] < min_samples:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if would_conflict(ra, rb):
            n_rejected += 1
            continue
        root, gone = min(ra, rb), max(ra, rb)
        parent[gone] = root
        members[root] = members[root] + members[gone]
        n_merged += 1
        mid = len(d["frames"]) // 2
        ba, bb = d["boxes"][mid]
        records.append({"id_a": a, "id_b": b, "shared_samples": d["n"],
                        "median_iou": round(float(np.median(d["iou"])), 2),
                        "median_iom": round(float(np.median(d["iom"])), 2),
                        "median_foot_sep": round(float(np.median(d["sep"])), 3),
                        "frame_mid": int(d["frames"][mid]),
                        "a_x1": ba[0], "a_y1": ba[1], "a_x2": ba[2], "a_y2": ba[3],
                        "b_x1": bb[0], "b_y1": bb[1], "b_x2": bb[2], "b_y2": bb[3]})
    if n_merged:
        df[id_col] = df[id_col].map(lambda t: find(t) if t in parent else t)
    if log is not None and records:
        pd.DataFrame(records).to_csv(log, index=False, encoding="utf-8-sig")
    if n_rejected:
        print(f"[INFO] duplicate merges rejected as physically impossible: "
              f"{n_rejected}")
    return df, n_merged


def build_tracks(df):
    tracks = {}
    for tid, sub in df.groupby("track_id"):
        sub = sub.sort_values("frame")
        tracks[tid] = {
            "frames": sub["frame"].to_numpy(),
            "times":  sub["time_sec"].to_numpy(),
            "xy":     sub[["foot_x", "foot_y"]].to_numpy(dtype=float),
            "cls":    majority_class(sub),
            # apparent size = the scale reference every distance test uses
            # scale reference, per class: a vehicle is measured by its box
            # WIDTH, a person by their HEIGHT. A standing person's foot point
            # wobbles 0.3-0.4 of their height, which is several shoulder
            # widths — measuring them by width made the queue rule reject
            # obvious continuations of one person standing still.
            "box_w":  float((sub["x2"] - sub["x1"]).median())
                      if {"x1", "x2"} <= set(sub.columns) else 0.0,
            "size":   float((sub["y2"] - sub["y1"]).median())
                      if majority_class(sub) in PED
                      else float((sub["x2"] - sub["x1"]).median()),
        }
    return tracks


def mark_completed(tracks, arms, center):
    """Item 5 - the user's standing rule: once a track has completed a
    movement, it must not be stitched any further.

    Completed means BOTH of:
      A. the fragment's own arm sequence already contains arm X followed by a
         different arm Y (center zones are not arms, so Y is a real exit);
      B. after first entering Y the track moved at least one further body
         width AWAY from the junction center - "really left, at least on the
         trajectory" per the user. This stops a foot point that merely grazes
         an arm's corner (the Zone_16 tip problem) from counting as done.

    Blocking a join never changes a count: the rejected later fragment either
    stays inside one arm (no transition, contributes nothing) or performs a
    genuine second movement, which is then counted separately - exactly the
    user's ruling for Baxter>center>LumpkinS>center>Baxter paths.
    Pedestrians are never marked: destinations there rely on long stitched
    walks, and the movement rule is about vehicle turns."""
    done = {}
    boxes = {n: (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max())
             for n, q in arms.items()}
    for tid, tr in tracks.items():
        if class_group(tr["cls"]) == "ped":
            continue
        xy = tr["xy"]
        seq, cur = [], None
        for i, (x, y) in enumerate(xy):
            hit = None
            for name, poly in arms.items():
                x0, y0, x1, y1 = boxes[name]
                if x < x0 or x > x1 or y < y0 or y > y1:
                    continue
                if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                    hit = name
                    break
            if hit and hit != cur:
                seq.append((hit, i))
            if hit:
                cur = hit
        if not seq:
            done[tid] = False
            continue
        first = seq[0][0]
        y_ent = next(((a, i) for a, i in seq[1:] if a != first), None)
        if y_ent is None or center is None:
            done[tid] = False
            continue
        i0 = y_ent[1]
        w = tr.get("box_w", 0.0) or 0.0
        if w <= 0:
            done[tid] = False
            continue
        d = np.hypot(xy[i0:, 0] - center[0], xy[i0:, 1] - center[1])
        done[tid] = bool(d.max() - d[0] >= 1.0 * w)
    return done


def end_velocity(tr, k=5):
    xy, t = tr["xy"], tr["times"]
    if len(t) < 2:
        return np.zeros(2)
    k = min(k, len(t) - 1)
    dt = t[-1] - t[-1 - k]
    if dt <= 0:
        return np.zeros(2)
    return (xy[-1] - xy[-1 - k]) / dt


def start_velocity(tr, k=5):
    xy, t = tr["xy"], tr["times"]
    if len(t) < 2:
        return np.zeros(2)
    k = min(k, len(t) - 1)
    dt = t[k] - t[0]
    if dt <= 0:
        return np.zeros(2)
    return (xy[k] - xy[0]) / dt


def stitch(tracks, base_dist, speed_dist, max_jump=700.0, min_cos=0.2,
           size_widths=2.0, completed=None,
           max_gap_veh=5.0, max_gap_ped=2.5, stop_gap=45.0, stop_frac=0.5,
           stop_frac_ped=1.0, stop_speed=30.0, stop_speed_ped=30.0,
           min_ped_samples=6, min_ped_seconds=1.0):
    """Greedy best-first merge of tracklets. Returns mapping old_id -> root_id.
    Vehicles get a longer occlusion window (max_gap_veh). Stopped-queue rule:
    if both fragments are near-stationary at the same spot, allow gaps up to
    stop_gap seconds (red-light queues losing ID while stopped).

    "Same spot" is measured in box widths, not pixels. This scene spans a huge
    depth range — a car is ~700 px wide at the stop bar and ~20 px wide at the
    far end of the street — so a fixed pixel threshold means "a fifth of a car"
    up close and "seven car lengths" in the distance. The absolute version of
    this rule was gluing a departing vehicle to an arriving one and reporting
    the pair as a u-turn."""
    # a fragment that never enters a zone is invisible to every count, so it
    # is not allowed to bridge two real trajectories
    # The zone restriction exists to stop tiny far-end vehicle fragments from
    # bridging two unrelated cars. Pedestrians are different: they stand and
    # walk on brick, kerbs and grass that carry no zone at all, so applying it
    # to them discarded 14 obvious continuations of one person standing still.
    ids = sorted((i for i in tracks
                  if class_group(tracks[i]["cls"]) == "ped"
                  or tracks[i].get("in_zone", True)),
                 key=lambda i: tracks[i]["times"][0])
    candidates = []
    for a in ids:
        ta = tracks[a]
        va = end_velocity(ta)
        for b in ids:
            if a == b:
                continue
            tb = tracks[b]
            # a track that already completed a movement takes no further part
            # in stitching, on either side of a join (item 5, user's rule)
            if completed is not None and (completed.get(a) or completed.get(b)):
                continue
            gap = tb["times"][0] - ta["times"][-1]
            if gap <= 0 or gap > stop_gap:
                continue
            if class_group(ta["cls"]) != class_group(tb["cls"]):
                continue
            jump = float(np.linalg.norm(tb["xy"][0] - ta["xy"][-1]))
            vb = start_velocity(tb)
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)

            # scale the "same spot" test by the object's own apparent size
            wa = ta.get("size", ta.get("box_w", 0.0))
            wb = tb.get("size", tb.get("box_w", 0.0))
            is_ped = class_group(ta["cls"]) == "ped"
            if is_ped:
                # Joining two specks, or extending a real journey with one,
                # invents information: measured on this clip, 48% of
                # pedestrian joins had a side of <=5 samples and a quarter had
                # a side of just 2, and those were what glued a car bonnet and
                # a parked bicycle onto a real pedestrian. A person who really
                # did cross while detection dropped out still leaves a
                # substantial fragment on both sides of the gap.
                if (len(ta["times"]) < min_ped_samples
                        or len(tb["times"]) < min_ped_samples
                        or float(ta["times"][-1] - ta["times"][0]) < min_ped_seconds
                        or float(tb["times"][-1] - tb["times"][0]) < min_ped_seconds):
                    continue
            gap_limit = max_gap_ped if is_ped else max_gap_veh
            # A person waiting at a signal is rarely at a dead stop — they
            # shuffle, turn, start to step off — and they are hidden behind
            # poles, trees and each other for longer than a car is. Measured:
            # 20 obvious continuations were rejected only because one end was
            # moving at 30-60 px/s, and most of the rest because the gap
            # exceeded 2.5 s.
            frac = stop_frac_ped if is_ped else stop_frac
            speed_cap = stop_speed_ped if is_ped else stop_speed
            stop_dist = frac * min(wa, wb) if min(wa, wb) > 0 else 0.0
            stopped_pair = (na < speed_cap and nb < speed_cap
                            and jump < stop_dist)
            if gap > gap_limit and not stopped_pair:
                continue
            # jump cap scales with own speed: a fast car blind for `gap`
            # seconds legitimately covers na*gap px (prediction + heading
            # checks below remain the real guards against wrong joins)
            # Item 4: every fixed pixel floor is capped at size_widths of
            # the object's own body. A far-end car images 20-60 px wide, so
            # 700/250/120/200 px there meant several BODY LENGTHS and the
            # guards below were disabled exactly where ID switches happen.
            # min() keeps this tighten-only: near the camera the body exceeds
            # the constants and nothing changes.
            box_ref = min(wa, wb) if min(wa, wb) > 0 else 0.0
            span = size_widths * box_ref if box_ref > 0 else float("inf")
            jump_cap = max(min(max_jump, span),
                           1.6 * na * gap + min(250.0, span))
            if jump > jump_cap:
                continue
            fast = na > 150
            need_cos = 0.5 if fast else min_cos   # looser distance => stricter heading
            # When a fragment ends nearly stationary its instantaneous end
            # velocity is noise, so fall back to where the fragment travelled
            # overall. Skipping the check instead (the old behaviour) let a
            # vehicle that drove off to the far end be joined to a different
            # vehicle driving back from it — reported as a u-turn.
            ha = va if na > 20 else (ta["xy"][-1] - ta["xy"][0])
            hb = vb if nb > 20 else (tb["xy"][-1] - tb["xy"][0])
            nha, nhb = np.linalg.norm(ha), np.linalg.norm(hb)
            if nha > 1e-6 and nhb > 1e-6:
                if float(ha @ hb) / (nha * nhb) < need_cos:
                    continue
            # Instantaneous velocities can agree by coincidence where two
            # vehicles cross paths, so where each fragment actually travelled
            # must agree as well. Verified on video: every join whose overall
            # directions opposed turned out to be two different cars meeting
            # at the intersection, reported as a u-turn.
            da = ta["xy"][-1] - ta["xy"][0]
            db = tb["xy"][-1] - tb["xy"][0]
            nda, ndb = np.linalg.norm(da), np.linalg.norm(db)
            # the old fixed 200 px meant 3.3 body widths at the Z15/Z16
            # corner (61 px cars), so short opposite-direction fragments
            # were never direction-checked and head-on pairs got joined
            dir_len = min(200.0, 0.6 * box_ref) if box_ref > 0 else 200.0
            if (nda > dir_len and ndb > dir_len
                    and float(da @ db) / (nda * ndb) < 0.0):
                continue
            pred = ta["xy"][-1] + va * gap
            dist = float(np.linalg.norm(pred - tb["xy"][0]))
            # prediction-error allowance scales with own speed for vehicles:
            # velocity estimates are coarse, so error grows with na*gap
            # prediction-error allowance, in units of the object's own size:
            # a constant-velocity prediction is wrong by roughly a car length
            # whenever the vehicle is braking or has just stopped, and that is
            # exactly the situation at an intersection. A fixed pixel budget
            # rejects those joins near the camera (box 336 px, error 339 px)
            # while being far too generous at the far end (box 20 px).
            allow = max(min(base_dist, span), box_ref) + speed_dist * gap
            if fast:
                allow = max(allow, 0.5 * na * gap + min(150.0, span))
            if stopped_pair or dist <= allow:
                score = jump if stopped_pair else dist
                candidates.append((score, gap, a, b))
    candidates.sort()

    succ, pred_of = {}, {}
    for dist, gap, a, b in candidates:
        if a in succ or b in pred_of:
            continue
        succ[a] = b
        pred_of[b] = a

    root = {}
    for i in tracks:
        root[i] = i
    for i in ids:
        if i not in pred_of:            # chain head
            r, cur = i, i
            root[cur] = r
            while cur in succ:
                cur = succ[cur]
                root[cur] = r
    merges = [(a, b) for a, b in succ.items()]
    return root, merges


def replay_zone_logic(df, zones, fps, dwell_ped=MIN_DWELL_SECONDS,
                      dwell_veh=None, grace_s=0.0, smooth=1):
    """Visit/dwell/direction replay. Defaults reproduce run_clip_tracker rules.

    dwell_veh : vehicle dwell threshold in seconds (None = same as dwell_ped)
    grace_s   : hysteresis — zone exit only after this long continuously outside
    smooth    : causal rolling-mean window (samples) applied to foot points
    """
    if dwell_veh is None:
        dwell_veh = dwell_ped
    step = df.groupby("merged_id")["frame"].diff().dropna()
    step = int(step[step > 0].median()) if len(step) else 1
    grace_k = max(1, round(grace_s * fps / step)) if grace_s > 0 else 0

    df = df.sort_values(["merged_id", "frame"]).copy()
    if smooth > 1:
        g = df.groupby("merged_id")
        df["px"] = g["foot_x"].transform(lambda s: s.rolling(smooth, min_periods=1).mean())
        df["py"] = g["foot_y"].transform(lambda s: s.rolling(smooth, min_periods=1).mean())
    else:
        df["px"], df["py"] = df["foot_x"], df["foot_y"]

    journeys   = defaultdict(list)
    visits     = defaultdict(dict)      # id -> zone -> visit dict
    events     = []

    df = df.sort_values(["frame"])
    for row in df.itertuples():
        tid, fnum = row.merged_id, row.frame
        dwell_sec = dwell_veh if class_group(row.maj_class) == "vehicle" else dwell_ped
        min_dwell_frames = int(dwell_sec * fps)
        pt = (int(row.px), int(row.py))
        current = {name for name, poly in zones.items()
                   if cv2.pointPolygonTest(poly, pt, False) >= 0}
        for z in current:
            if z not in visits[tid]:
                visits[tid][z] = {"first": fnum, "last": fnum,
                                  "confirmed": False, "miss": 0}
            else:
                v = visits[tid][z]
                v["last"] = fnum
                v["miss"] = 0
                if not v["confirmed"] and v["last"] - v["first"] >= min_dwell_frames:
                    v["confirmed"] = True
                    j = journeys[tid]
                    prev = j[-1]["zone"] if j and j[-1]["zone"] != z else None
                    direction = f"{prev.replace('Zone_','')} -> {z.replace('Zone_','')}" if prev else None
                    if not j or j[-1]["zone"] != z:
                        j.append({"zone": z, "first_frame": v["first"],
                                  "last_frame": fnum,
                                  "dwell_sec": round((fnum - v["first"]) / fps, 2),
                                  "direction": direction})
                    events.append({"id": tid, "zone": z, "direction": direction,
                                   "first_time_sec": round(v["first"] / fps, 2)})
        gone = set(visits[tid].keys()) - current
        for z in gone:
            v = visits[tid][z]
            v["miss"] += 1
            if v["miss"] > grace_k:
                del visits[tid][z]
    return journeys, events


def metrics(df, journeys, label):
    per_id = df.groupby("merged_id").agg(first_t=("time_sec", "min"),
                                         last_t=("time_sec", "max"),
                                         cls=("maj_class", "first"))
    per_id["vis"] = per_id["last_t"] - per_id["first_t"]
    valid = per_id[per_id["vis"] >= MIN_VISIBILITY_SECONDS]
    dir_counts, zone_counts = Counter(), Counter()
    zones_per_journey = []
    for tid in valid.index:
        j = journeys.get(tid, [])
        zones_per_journey.append(len(j))
        for e in j:
            zone_counts[e["zone"]] += 1
            if e["direction"]:
                dir_counts[e["direction"]] += 1
    ids_with_dir = sum(1 for tid in valid.index
                       if any(e["direction"] for e in journeys.get(tid, [])))
    return {
        "label": label,
        "total_ids": len(per_id),
        "valid_ids": len(valid),
        "fragments": len(per_id) - len(valid),
        "class_counts": Counter(valid["cls"]),
        "dir_total": sum(dir_counts.values()),
        "dir_kinds": len(dir_counts),
        "ids_with_dir": ids_with_dir,
        "avg_zones": (sum(zones_per_journey) / len(zones_per_journey))
                     if zones_per_journey else 0,
        "dir_counts": dir_counts,
        "zone_counts": zone_counts,
    }


def fmt(m):
    out = [f"  Total IDs          : {m['total_ids']}",
           f"  Valid IDs (>=5s)   : {m['valid_ids']}",
           f"  Fragments (<5s)    : {m['fragments']} "
           f"({m['fragments']/max(m['total_ids'],1)*100:.0f}%)",
           f"  IDs with direction : {m['ids_with_dir']} "
           f"({m['ids_with_dir']/max(m['valid_ids'],1)*100:.0f}% of valid)",
           f"  Direction events   : {m['dir_total']} ({m['dir_kinds']} kinds)",
           f"  Avg zones/journey  : {m['avg_zones']:.2f}",
           "  Class counts       : " + ", ".join(f"{k}={v}" for k, v in
                                                 m["class_counts"].most_common())]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--zones", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--name", default="stitched")
    ap.add_argument("--base-dist", type=float, default=120.0)
    ap.add_argument("--speed-dist", type=float, default=80.0)
    ap.add_argument("--cut-vel-samples", type=int, default=3,
                    help="velocity window (samples) for body-scaled teleport "
                         "cuts; 0 disables the new criteria")
    ap.add_argument("--size-widths", type=float, default=2.0,
                    help="cap every fixed pixel floor in the stitcher at "
                         "this many of the object own box widths")
    ap.add_argument("--mouth-cut-arm", default=None,
                    help="cut vehicle tracks at this arm's far-end mouth "
                         "(site-validated opt-in; Baldwin uses E)")
    ap.add_argument("--names", default=None,
                    help="zone semantics json; enables the completed-"
                         "movement stitching block (item 5)")
    ap.add_argument("--dwell-veh", type=float, default=None,
                    help="vehicle dwell seconds (improvement 1)")
    ap.add_argument("--grace", type=float, default=0.0,
                    help="zone-exit hysteresis seconds (improvement 2)")
    ap.add_argument("--smooth", type=int, default=1,
                    help="foot-point rolling-mean window (improvement 2)")
    ap.add_argument("--img-w", type=int, default=3840)
    ap.add_argument("--img-h", type=int, default=2160)
    ap.add_argument("--no-stitch", action="store_true",
                    help="skip tracklet stitching entirely; keep only the "
                         "concurrent duplicate merge")
    ap.add_argument("--border", type=float, default=0.0,
                    help="drop samples whose bbox is clipped by the frame "
                         "border (bottom/left/right) within this margin px")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.traj)
    zones = load_zones(args.zones)
    nm = {}
    if args.names:
        with open(args.names, encoding="utf-8") as fh:
            nm = json.load(fh)["zones"]
    fps = df["frame"].iloc[0] / df["time_sec"].iloc[0]

    if args.border > 0 and {"x1", "x2", "y2"}.issubset(df.columns):
        m = args.border
        clipped = ((df["y2"] >= args.img_h - m) | (df["x1"] <= m) |
                   (df["x2"] >= args.img_w - m))
        # vehicles KEEP clipped samples: their exits live at the frame edge
        # and arm-level zone tests tolerate the coarse foot point; dropping
        # them amputates through-movement exits (ped jitter still filtered)
        clipped &= ~df["class"].isin(VEHICLE)
        print(f"[INFO] border-clipped samples dropped (peds only): "
              f"{int(clipped.sum())} of {len(df)} ({clipped.mean()*100:.1f}%)")
        df = df[~clipped].copy()

    df, n_tail = trim_bad_tails(df)
    print(f"[INFO] low-quality tail samples trimmed: {n_tail}")

    df, n_spikes = repair_spikes(df)
    print(f"[INFO] spike outliers repaired (trend-based): {n_spikes}")

    n_raw_ids = df["track_id"].nunique()
    df, n_splits = split_teleports(df, vel_samples=args.cut_vel_samples)

    # User rule (2026-08-24, generalized): cut every vehicle track where it
    # crosses the FAR-END mouth of ANY arm (the spot where vehicles enter or
    # leave the frame). Boxes swap cars exactly there - queue hand-offs,
    # arrivals, slip-lane mergers - and the same disease was visible on
    # every movement panel. Cutting at the mouth boundary lets the stitcher
    # re-join the same car (its guards pass) while different cars stay apart.
    if args.names:
        # gates are screenlines AT the frame edge: cutting there shreds the
        # exit visit itself (through -14% when tried). Cut only real-road
        # arm mouths.
        # Mouth cuts are OPT-IN per site (--mouth-cut-arm): validated only
        # for Baldwin's east arm. Generalizing to the north mouth cost 14%
        # of southbound throughs - never widen without a per-mouth A/B.
        armzones = [k for k, v in nm.items()
                    if args.mouth_cut_arm
                    and v.get("arm") == args.mouth_cut_arm
                    and v.get("role") != "gate" and k in zones]
        if armzones:
            allpts = np.vstack([np.asarray(zones[k], float) for k in armzones])
            cen = allpts.mean(axis=0)
            veh_mask = df["class"].isin(VEHICLE)
            inside_any = pd.Series(False, index=df.index)
            for k in armzones:
                P = np.asarray(zones[k], float)
                far_pt = P[np.argmax(np.hypot(*(P - cen).T))]
                d = np.hypot(df["foot_x"] - far_pt[0],
                             df["foot_y"] - far_pt[1])
                inside_any |= (d <= 200.0)
            inside_any &= veh_mask
            crossed = inside_any.groupby(df["track_id"]).diff().fillna(False)
            crossed = crossed.astype(bool)
            df["track_id"] = df["track_id"].astype(np.int64) * 100 +                 crossed.groupby(df["track_id"]).cumsum().astype(np.int64).clip(upper=99)
            print(f"[INFO] arm-mouth cuts ({len(armzones)} mouths): "
                  f"{int(crossed.sum())}")

    print(f"[INFO] teleport splits: {n_splits} "
          f"({n_raw_ids} raw ids -> {df['track_id'].nunique()})")

    tracks = build_tracks(df)
    print(f"[INFO] {len(tracks)} tracks after split, fps={fps:.2f}")

    # Only fragments that actually enter a zone may take part in merging.
    # A fragment that never touches a zone contributes nothing to any count,
    # so joining it carries risk without benefit — and measured on this clip
    # those fragments are exactly the unreliable ones (median box width 24 px
    # at the far end of the street, vs 144 px for in-zone fragments). They
    # were acting as bridges that glued two unrelated vehicles into fake
    # left turns and fake u-turns.
    zone_polys = list(zones.values())
    for tid, tr in tracks.items():
        xy = tr["xy"]
        probe = xy[::max(1, len(xy) // 40)]
        tr["in_zone"] = any(
            cv2.pointPolygonTest(p, (int(x), int(y)), False) >= 0
            for x, y in probe for p in zone_polys)
    n_fringe = sum(1 for tr in tracks.values() if not tr["in_zone"])
    print(f"[INFO] fragments never entering a zone (excluded from merging): "
          f"{n_fringe} of {len(tracks)}")

    # BEFORE: raw ids, majority class
    maj = {tid: tr["cls"] for tid, tr in tracks.items()}
    df["maj_class"] = df["track_id"].map(maj)
    df["merged_id"] = df["track_id"]
    j_raw, _ = replay_zone_logic(df, zones, fps)
    m_raw = metrics(df, j_raw, "raw")

    # AFTER: stitched
    if args.no_stitch:
        root, merges = {i: i for i in tracks}, 0
        print("[INFO] stitching disabled")
    else:
        completed = None
        if args.names:
            arms = {k: zones[k] for k, v in nm.items()
                    if v.get("arm") and k in zones}
            cen = [zones[k] for k, v in nm.items()
                   if v.get("role") == "center" and k in zones]
            cen_pt = (np.vstack(cen if cen else list(arms.values()))
                      .mean(axis=0) if arms else None)
            completed = mark_completed(tracks, arms, cen_pt)
            print(f"[INFO] completed-movement tracks barred from further "
                  f"stitching: {sum(completed.values())} of {len(completed)} "
                  f"vehicles")
        root, merges = stitch(tracks, args.base_dist, args.speed_dist,
                              size_widths=args.size_widths,
                              completed=completed)
    df["merged_id"] = df["track_id"].map(root)
    root_cls = {}
    for tid, r in root.items():
        root_cls.setdefault(r, Counter())
        root_cls[r][maj[tid]] += len(tracks[tid]["times"])
    df["maj_class"] = df["merged_id"].map(lambda r: root_cls[r].most_common(1)[0][0])

    # a standing vehicle keeps being re-registered under new IDs; after
    # stitching, collapse stitched tracks that are stationary, concurrent and
    # within half a car body of each other (vehicles only — two people can
    # legitimately stand that close)
    df, n_dup = merge_concurrent_duplicates(
        df, id_col="merged_id",
        log=os.path.join(args.outdir, f"{args.name}_dup_merges.csv"))

    # the class must be decided AFTER every merge, over all samples the final
    # track owns: one vehicle labelled Truck on some frames and Passenger
    # Vehicle on others is settled by confidence-weighted vote, not by
    # whichever fragment happened to be the root
    conf_by_class = (df.groupby(["merged_id", "class"])["conf"].sum()
                     .reset_index())
    winner = (conf_by_class.sort_values("conf")
              .drop_duplicates("merged_id", keep="last")
              .set_index("merged_id")["class"])
    df["maj_class"] = df["merged_id"].map(winner)
    print(f"[INFO] duplicate IDs on standing vehicles merged: {n_dup}")

    # When two concurrent boxes on ONE vehicle (cab + trailer, or双重检测)
    # end up under one merged_id, their rows interleave and the foot point
    # zigzags between two positions - the sawtooth the user kept seeing.
    # Within every overlapping time window keep only the source track with
    # more samples; the other's rows are dropped, not interleaved.
    drop_mask = pd.Series(False, index=df.index)
    n_zz = 0
    for mid, g in df.groupby("merged_id"):
        tids = g["track_id"].unique()
        if len(tids) < 2:
            continue
        spans = {t: (g.loc[g.track_id == t, "time_sec"].min(),
                     g.loc[g.track_id == t, "time_sec"].max(),
                     int((g.track_id == t).sum())) for t in tids}
        for i, a in enumerate(tids):
            for b in tids[i + 1:]:
                s0, e0, n0 = spans[a]
                s1, e1, n1 = spans[b]
                lo, hi = max(s0, s1), min(e0, e1)
                if hi > lo + 0.3:
                    loser = a if n0 < n1 else b
                    sel = ((df["merged_id"] == mid)
                           & (df["track_id"] == loser)
                           & (df["time_sec"] >= lo)
                           & (df["time_sec"] <= hi))
                    if sel.any():
                        drop_mask |= sel
                        n_zz += 1
    if n_zz:
        df = df[~drop_mask].copy()
    print(f"[INFO] interleaved duplicate-box windows resolved: {n_zz}")

    j_st, events = replay_zone_logic(df, zones, fps)
    m_st = metrics(df, j_st, "stitched")

    use_rules = (args.dwell_veh is not None or args.grace > 0 or args.smooth > 1)
    if use_rules:
        j_new, _ = replay_zone_logic(df, zones, fps,
                                     dwell_veh=args.dwell_veh,
                                     grace_s=args.grace, smooth=args.smooth)
        m_new = metrics(df, j_new, "stitched+rules")

    df.to_csv(os.path.join(args.outdir, f"{args.name}_trajectories.csv"), index=False)
    pd.DataFrame([{"from_id": a, "to_id": b} for a, b in merges]).to_csv(
        os.path.join(args.outdir, f"{args.name}_merges.csv"), index=False)

    lines = ["=" * 60, f"TRACKLET STITCHING — {args.name}",
             f"params: base_dist={args.base_dist}px "
             f"size_widths={args.size_widths} "
             f"names={'yes' if args.names else 'no'} "
             f"speed_dist={args.speed_dist}px/s",
             f"merges applied: {len(merges)} "
             f"({len(tracks)} tracks -> {df['merged_id'].nunique()})",
             "=" * 60,
             "BEFORE (raw tracker output, majority-vote class):", fmt(m_raw),
             "-" * 60,
             "AFTER (stitched):", fmt(m_st)]
    if use_rules:
        lines += ["-" * 60,
                  f"AFTER (stitched + rules: dwell_veh={args.dwell_veh}s "
                  f"grace={args.grace}s smooth={args.smooth}):", fmt(m_new)]
    final = m_new if use_rules else m_st
    lines += ["-" * 60, "FINAL DIRECTIONS:"]
    for d, c in final["dir_counts"].most_common():
        lines.append(f"  {c:>4}x  {d}")
    lines.append("=" * 60)
    report = "\n".join(lines)
    with open(os.path.join(args.outdir, f"{args.name}_stats.txt"), "w",
              encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
