"""
flow_typed.py
-------------
The semantics layer: turns cleaned trajectories into traffic quantities.
Reads the output of stitch_tracks.py plus the zone polygons and their semantic
names, writes a per-object ledger where every track is accounted for.

VEHICLES
  A movement is the FIRST transition between two different road arms
  (entry arm -> first different arm). Anything the track does afterwards
  cannot change the classification, so a fragment wrongly appended to a
  finished journey can no longer turn a clean turn into a phantom u-turn.
  A track whose exit was never observed gets one chance at inference: the
  line of travel from the last confirmed position is intersected with the
  zone polygons (a LINE, not sampled points — a car at 1900 px/s covers
  475 px between two 0.25 s samples and would skip a whole zone), and the
  first road arm it crosses is recorded, with a confidence.

PEDESTRIANS
  Two independent readings of the same track:
    category         what the person mainly did — mutually exclusive, so the
                     ledger reconciles to the track count
    jaywalk_segment  did any part of the journey run on the roadway away from
                     a crossing? A safety indicator that must NOT be cancelled
                     out because the same person also used a crosswalk.
  Crossing direction comes from projection onto the crosswalk's long axis.
  Zone membership is tolerant (half a body height) because a foot point
  wobbles across a painted edge and consecutive crosswalk polygons do not
  quite touch — strict membership reported legal crossings as jaywalking.

Every pedestrian threshold is in body heights, every vehicle threshold in box
widths. Nothing here is in raw pixels.

Usage:
  python flow_typed.py --traj MERGED.csv --zones Z.json --names names.json
                       --outdir DIR
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd

from track_endpoints import build_adjacency, endpoint_status, extrapolate

VEHICLE = {"Passenger Vehicle", "Motorcycle", "Bus", "Truck"}
MIN_VIS = 5.0
WAIT_BODY_HEIGHTS = 2.0        # moved less than 2 of its own heights = waiting
CW_TOL_BODY_HEIGHTS = 0.5      # counts as on the crosswalk within half a height
JAYWALK_BODY_HEIGHTS = 0.5     # farther than this from every crosswalk
                               # (a person is ~1.7 m, so this is about one
                               #  step outside the stripes; 1.5 would let
                               #  someone walk a whole crosswalk width away)
JAYWALK_MIN_SECONDS = 2.0      # and for at least this long
STATIC_OBJECT_BODY_HEIGHTS = 1.0   # never moved this far ...
STATIC_OBJECT_SECONDS = 30.0       # ... in this long = not a traveller
STATIC_VEH_DISP = 150.0   # vehicle that never moved this far = parked/static

# Both tables below are SITE-SPECIFIC and come from the names JSON when it
# supplies them ("short" per zone, and a "movements" list of
# {from, to, label}). The Lumpkin @ Baxter values stay here as the default so
# the original run keeps working without a config change.
SHORT = {
    "road_lumpkin_south_arm": "LumpkinS(Z16)",
    "road_lumpkin_north_arm": "LumpkinN(Z15)",
    "road_baxter_arm": "Baxter(Z13)",
    "road_intersection_center": "center(Z14)",
}

# (entry arm zone, exit arm zone) -> movement label. Orientation confirmed by
# user: Z15=Lumpkin north side, Z16=Lumpkin south side, Baxter tees from west.
MOVEMENT = {
    ("Zone_13", "Zone_16"): "RIGHT turn  (Baxter -> Lumpkin southbound)",
    ("Zone_16", "Zone_13"): "LEFT turn   (Lumpkin northbound -> Baxter)",
    ("Zone_15", "Zone_16"): "THROUGH     (southbound, N->S)",
    ("Zone_16", "Zone_15"): "THROUGH     (northbound, S->N)",
    ("Zone_13", "Zone_15"): "LEFT turn   (Baxter -> Lumpkin northbound)",
    ("Zone_15", "Zone_13"): "RIGHT turn  (Lumpkin southbound -> Baxter)",
}


def load_site_tables(cfg, zmeta):
    """Replace the movement table with the site's own, if it ships one."""
    movements = cfg.get("movements")
    if not movements:
        return
    MOVEMENT.clear()
    for m in movements:
        MOVEMENT[(m["from"], m["to"])] = m["label"]
    SHORT.clear()
    for z, v in zmeta.items():
        if v.get("type") == "vehicle_road":
            SHORT[v["name"]] = v.get("short", f"{v['name']}({z.replace('Zone_','Z')})")
    CENTER_ARMS.clear()
    CENTER_ARMS.update(z for z, v in zmeta.items()
                       if v.get("role") == "center_and_arm")


def zone_axis(poly, shape=(2160, 3840)):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    mean = pts.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((pts - mean).T))
    axis = evecs[:, 1]
    half = float(np.sqrt(evals[1]) * 1.9)
    return mean, axis, half


# Zones that are an approach arm AND the intersection box at the same time.
# At Baxter @ Hull the east arm of Baxter was drawn as one polygon that also
# covers the middle of the junction, so every turn between the other two arms
# passes through it. Such a zone must not be taken as the exit while a real
# third arm still follows. Filled from the names JSON ("role": "center_and_arm").
CENTER_ARMS = set()


def first_transition(arms):
    """A vehicle's movement is decided by the first arm it entered on and the
    first DIFFERENT arm it reached. Anything after that is travel beyond the
    manoeuvre and must not change the classification: a fragment wrongly
    appended to a finished journey (arms 15-13 followed by 13-15) would
    otherwise turn a clean turn into a phantom u-turn.

    A zone in CENTER_ARMS counts as the exit only when nothing else follows:
    it is the middle of the junction as much as an arm, so [W, centre, N] is a
    turn to N, while [W, centre] really did leave on the centre-arm side."""
    if not arms:
        return None, None
    a0 = arms[0]
    exit_ = None
    for a in arms[1:]:
        if a == a0:
            continue
        if exit_ is None:
            exit_ = a
            if a not in CENTER_ARMS:
                return a0, exit_
            continue
        if a != exit_:              # a real arm beyond the junction box
            return a0, a
    return a0, exit_


def movements_chain(vis, t_end):
    """The user's standing rule (final form, 2026-08-24): once a track has
    completed a movement X->Y, the rest of the trajectory is judged as its own
    track STARTING FROM the exit arm Y - it may hold a genuine second
    movement (Baxter>center>LumpkinS>center>Baxter counts as two), and it must
    never alter the first one. Returns the full list of (entry, exit) pairs;
    almost every track yields exactly one."""
    out, rest = [], list(vis)
    while True:
        arms = [z for z, _t0, _t1 in rest]
        a0, a1 = first_transition(arms)
        if a1 is None:
            break
        try:
            j = arms.index(a1, 1)
        except ValueError:
            break
        # the segment begins at the ENTRY arm, not at track birth: a stolen
        # prefix (box riding another car before it reaches the entry arm)
        # must not be drawn as part of this movement
        # pads must stay under the ~2-3 s an ID hand-off glide takes,
        # or the glide gets drawn as a phantom arc (h12/h16 diagonals)
        seg0 = rest[0][1] - 0.5
        seg1 = min(rest[j][2] + 0.5, t_end)
        out.append((a0, a1, seg0, seg1))
        if j >= len(rest) - 1:
            break
        rest = rest[j:]
    return out


def handoff_evidence(r, par_s1, s1):
    """Forensic on a continuation's hand-off interval (user rule, 2026-08-25):
    a movement minted AFTER the same id already completed one belongs to a
    different car unless the box stayed on the same vehicle. The interval
    runs from the parent's leave-exit to the continuation's end; inside it
    the box never teleports (it glides, like a zombie), so identity is judged
    locally: a 1-second-window class-mode flip or width-median jump >1.4x is
    a different car; a >=8 s standstill is a queue hand-off below the 30 s
    zombie line. 27 of the 30 counted continuations across the four old
    sites carried at least one mark. Returns the evidence string, '' if
    clean. (Labels are documentation: ALL continuations are excluded from
    counts either way - user decision 2026-08-25.)"""
    ts, xy, w, cls = r["ts_arr"], r["xy_arr"], r["w_arr"], r["cls_arr"]
    if w is None:
        return ""
    m = (ts >= par_s1 - 2.5) & (ts <= s1)
    if int(m.sum()) < 4:
        return ""
    ti, pi, wi, ci = ts[m], xy[m], w[m], cls[m]
    wl = pd.Series(wi).rolling(5, center=True, min_periods=1).median()
    wl = np.maximum(wl.to_numpy(), 1e-6)
    step = np.hypot(*np.diff(pi, axis=0).T)
    ev = []
    if len(step) and float((step / wl[1:]).max()) > 1.5:
        ev.append("jump")
    bins = ((ti - ti[0]) // 1.0).astype(int)
    for b in range(int(bins.min()), int(bins.max())):
        a, c = bins == b, bins == b + 1
        if a.sum() >= 2 and c.sum() >= 2:
            if (pd.Series(ci[a]).mode().iloc[0]
                    != pd.Series(ci[c]).mode().iloc[0]):
                ev.append("class")
                break
    for b in range(int(bins.min()), int(bins.max())):
        a, c = bins == b, bins == b + 1
        if a.sum() >= 2 and c.sum() >= 2:
            wa = float(np.median(wi[a]))
            wc = float(np.median(wi[c])) or 1.0
            if max(wa, wc) / max(min(wa, wc), 1.0) > 1.4:
                ev.append("width")
                break
    dt = np.diff(ti)
    slow = step < 0.05 * wl[1:]
    best = cur = 0.0
    for k, s in enumerate(slow):
        cur = cur + float(dt[k]) if s else 0.0
        best = max(best, cur)
    if best >= 8.0:
        ev.append(f"still{best:.0f}s")
    return "+".join(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--zones", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gate-dir-share", type=float, default=0.0,
                    help="terminal gate visits must move this fraction of "
                         "their displacement along the gate outward axis "
                         "(site-validated opt-in; Baldwin uses 0.6)")
    ap.add_argument("--birth-reclass-zone", default=None,
                    help="LEFT segments starting on this zone are re-filed "
                         "as the reversed RIGHT turn (Baldwin: Zone_05)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.zones) as f:
        zpolys = {k: np.array(v, np.float64)
                  for k, v in json.load(f)["zones"].items()}
    with open(args.names, encoding="utf-8") as f:
        site_cfg = json.load(f)
    zmeta = site_cfg["zones"]
    load_site_tables(site_cfg, zmeta)
    adjacency = build_adjacency(zpolys)
    name_of = {k: v["name"] for k, v in zmeta.items()}
    type_of = {k: v["type"] for k, v in zmeta.items()}

    road_arms = [k for k, v in zmeta.items()
                 if v["type"] == "vehicle_road" and "arm" in v["name"]]
    # Screenline gates (role "gate") sit at the frame edge where no roadway
    # is visible. Their only legitimate visits are "entered the frame here"
    # (first arm visit) and "left the frame here" (last arm visit); a track
    # that merely brushes a gate mid-journey is a queue foot-point straddling
    # the compressed corner, and counting it invented turns (user, 2026-08-24).
    gate_arms = {k for k, v in zmeta.items() if v.get("role") == "gate"}
    _cen_zones = [k for k, v in zmeta.items() if v.get("role") == "center"]
    # fallback for center-less sites: mean of ARM CENTROIDS (one vote per
    # arm), never the raw vertex mean - a huge arm with many vertices used to
    # drag the "center" inside itself and the exit-direction gate then killed
    # honest throughs (Physics 2026-08-28: 603 eastbound cars/day)
    _cen_pt = (np.vstack([zpolys[k] for k in _cen_zones]).mean(axis=0)
               if _cen_zones else
               np.vstack([np.asarray(zpolys[k], float).mean(axis=0)
                          for k in road_arms]).mean(axis=0)
               if road_arms else
               np.array([1920.0, 1080.0]))  # ped-only site: unused by any gate
    arm_cent = {k: np.asarray(zpolys[k], float).mean(axis=0)
                for k in road_arms}
    gate_axis = {}
    for gk in gate_arms:
        v = np.asarray(zpolys[gk], float)
        c = v.mean(axis=0)
        d = v - c
        w_, ev = np.linalg.eigh(d.T @ d)
        perp = ev[:, 0]                      # minor axis = across the band
        if float(perp @ (c - _cen_pt)) < 0:
            perp = -perp
        gate_axis[gk] = perp
    road_all = [k for k, v in zmeta.items() if v["type"] == "vehicle_road"]
    ped_zones = [k for k, v in zmeta.items() if v["type"].startswith("ped")]
    crosswalks = [k for k, v in zmeta.items() if v["type"] == "ped_crosswalk"]

    axes = {k: zone_axis(zpolys[k]) for k in crosswalks}
    # Destination corners. A walk that ends on a corner is observed outright;
    # one that ends on paint or roadway still has a heading, and the corner it
    # was walking toward is worth recording as long as it is marked inferred.
    corner_of = {k: v["corner"] for k, v in zmeta.items() if v.get("corner")}
    _pts = {}
    for k, c in corner_of.items():
        _pts.setdefault(c, []).append(zpolys[k].mean(0))
    corner_pt = {c: np.mean(v, axis=0) for c, v in _pts.items()}
    # Inference thresholds, chosen on an exam of 1581 Baxter walks whose true
    # destination was observed, with the last 15% of each trajectory hidden:
    # every angle-based rule stalled at ~65% precision (heading vectors are
    # noise in this fisheye scene), while "which corner did the whole walk
    # close distance on" scored 92.7% while still answering 68% of the exam.
    DEST_MIN_FRAC = 0.45     # must close >=45% of the initial distance
    DEST_MARGIN   = 0.15     # and beat the runner-up corner by >=15 points
    # A crosswalk that only ever lands on one corner can say so directly. The
    # heading test cannot help there: the corner is a step away, so the vector
    # to it is short and its direction is dominated by footstep noise. At
    # Baxter that left 94 of the 100 walks ending on the Tate plaza crossing
    # unresolved, when every one of them was plainly walking onto the plaza.
    leads_to = {k: v["leads_to"] for k, v in zmeta.items() if v.get("leads_to")}
    anchor_zones = [k for k, v in zmeta.items()
                    if v["type"] in ("ped_plaza", "ped_sidewalk")]
    anchors = {k: zpolys[k].mean(0) for k in anchor_zones}

    def zlabel(k):
        """name plus zone id, e.g. sidewalk_tate_side(Z05) — the id is what a
        reader needs to locate it on the zone map."""
        return f"{name_of[k]}({k.replace('Zone_', 'Z')})"

    def nearest_anchor(pt):
        best, bd = "?", 1e18
        for k, c in anchors.items():
            d = float(np.linalg.norm(c - pt))
            if d < bd:
                best, bd = zlabel(k), d
        return best

    end_labels = {}
    for k, (mean, axis, half) in axes.items():
        e1 = nearest_anchor(mean - axis * half)
        e2 = nearest_anchor(mean + axis * half)
        end_labels[k] = (e1, e2)

    df = pd.read_csv(args.traj)
    idcol = "merged_id" if "merged_id" in df.columns else "track_id"
    clscol = "maj_class" if "maj_class" in df.columns else "class"

    poly_i32 = {k: p.astype(np.int32) for k, p in zpolys.items()}

    def zones_at(pt):
        hits = [k for k, p in poly_i32.items()
                if cv2.pointPolygonTest(p, pt, False) >= 0]
        # An off-frame exit band is drawn over whatever roadway or walk already
        # runs to the frame edge, so a point can sit in both. The exit is by
        # definition the last thing the track touches, so sort it to the end
        # and let it close the sequence rather than leaving that to dict order.
        return sorted(hits, key=lambda k: zmeta.get(k, {}).get("role") == "offframe")

    veh_records, ped_records = [], []
    ped_od, cross_counts = Counter(), Counter()
    jaywalkers, waiting, ped_partial, static_objects = 0, 0, 0, 0
    jaywalk_od = Counter()
    clip_t0 = df["time_sec"].min()
    clip_t1 = df["time_sec"].max()

    for tid, sub in df.groupby(idcol):
        sub = sub.sort_values("frame")
        dur = sub["time_sec"].max() - sub["time_sec"].min()
        if dur < MIN_VIS:
            continue
        xy = sub[["foot_x", "foot_y"]].to_numpy(float)
        cls = sub[clscol].iloc[0]
        zhits = [zones_at((int(p[0]), int(p[1]))) for p in xy]

        if cls in VEHICLE:
            # full path over ALL road zones (arms + center), consecutive dedup
            road_seq = []
            for hits in zhits:
                for z in hits:
                    if z in road_all and (not road_seq or road_seq[-1] != z):
                        road_seq.append(z)
            # arm visits with their start times, mirroring road_seq exactly,
            # so each movement can be given the trajectory segment it owns
            tarr = sub["time_sec"].to_numpy()
            arm_vis = []
            _last_road = None
            for _si, hits in enumerate(zhits):
                for z in hits:
                    if z in road_all and z != _last_road:
                        _last_road = z
                        if z in road_arms:
                            arm_vis.append([z, float(tarr[_si]),
                                            float(tarr[_si])])
                if arm_vis and arm_vis[-1][0] in hits:
                    arm_vis[-1][2] = float(tarr[_si])
            arms_raw = [z for z, _t0, _t1 in arm_vis]
            vis_g = []
            for i, (z, t0v, t1v) in enumerate(arm_vis):
                if z in gate_arms and 0 < i < len(arm_vis) - 1:
                    continue          # mid-journey gate brush: invalid
                if z in gate_arms and i == len(arm_vis) - 1 and i > 0:
                    # a terminal gate visit is an EXIT only if the car really
                    # departs through the band - an eastbound tip graze was
                    # being read as a turn to the south leg (user, 2026-08-24)
                    mwin = (tarr >= t0v) & (tarr <= t1v + 2.0)
                    if mwin.sum() >= 2:
                        dsp = xy[mwin][-1] - xy[mwin][0]
                        wloc = float(np.median(
                            (sub["x2"] - sub["x1"]).to_numpy()[mwin])) or 1.0
                        nrm = float(np.hypot(*dsp))
                        # direction share, not pixels: a crosser moves mostly
                        # ACROSS the band, a graze skims ALONG it. SITE
                        # OPT-IN (--gate-dir-share): the band-axis derivation
                        # is calibrated per gate geometry - applied blind to
                        # Tate it killed 40% of genuine southbound throughs.
                        if (args.gate_dir_share > 0 and nrm >= 0.3 * wloc
                                and float(dsp @ gate_axis[z])
                                < args.gate_dir_share * nrm):
                            continue
                if vis_g and vis_g[-1][0] == z:
                    vis_g[-1] = (z, vis_g[-1][1], t1v)
                    continue          # merge, keep first start / last end
                vis_g.append((z, t0v, t1v))
            # (a birth-recovery rule was tried here and REVERTED 2026-08-24:
            # prepending a gate entry for tracks born near the gate mouth
            # also caught Baldwin westbound arrivals born in the crosswalk
            # gap and minted fake S->E right turns - user caught the arrows
            # pointing INTO the junction)
            arms_in_seq = [z for z, _t0, _t1 in vis_g]
            path = " > ".join(SHORT.get(name_of[z], name_of[z]) for z in road_seq)
            start_why = endpoint_status(sub, clip_t0, clip_t1, at_start=True)
            tag = ""
            if start_why == "truncated":
                tag = " [in view at clip start]"
            elif endpoint_status(sub, clip_t0, clip_t1) == "truncated":
                tag = " [cut off by clip end]"

            # full zone path (roads AND crosswalks etc.) for a complete record
            full_seq = []
            for hits in zhits:
                for z in hits:
                    if not full_seq or full_seq[-1] != z:
                        full_seq.append(z)
            full_path = " > ".join(name_of.get(z, z) for z in full_seq)

            # general endpoint handling (same rule for every track, both ends)
            end_why = endpoint_status(sub, clip_t0, clip_t1)
            infer_exit, infer_conf = None, None
            if end_why == "detection_loss":
                want = set(road_arms) - ({road_seq[0]} if road_seq else set())
                pred = extrapolate(sub, zpolys, adjacency, want=want)
                if pred:
                    infer_exit, infer_conf = pred["zone"], pred["confidence"]

            veh_disp = float(np.linalg.norm(xy[-1] - xy[0]))
            veh_records.append({"seq": road_seq, "arms": arms_in_seq,
                                "vis": vis_g,
                                "ts_arr": tarr,
                                "xy_arr": xy,
                                "w_arr": (sub["x2"] - sub["x1"]).to_numpy()
                                if {"x1", "x2"} <= set(sub.columns) else None,
                                "cls_arr": sub["class"].to_numpy(),
                                "path": path, "full_path": full_path,
                                "tag": tag, "id": tid, "cls": cls,
                                "infer_exit": infer_exit,
                                "infer_conf": infer_conf,
                                "end_why": end_why, "disp": veh_disp,
                                "t0": round(sub["time_sec"].min(), 1),
                                "t1": round(sub["time_sec"].max(), 1)})
        else:
            t0s, t1s = sub["time_sec"].min(), sub["time_sec"].max()
            base_rec = {"track_id": tid, "class": cls,
                        "start_sec": round(t0s, 1), "end_sec": round(t1s, 1),
                        "clip_time_hhmmss":
                            f"{int(t0s//60):02d}:{int(t0s%60):02d}"
                            f"-{int(t1s//60):02d}:{int(t1s%60):02d}"}
            # A person's own height is the scale: it is the steadiest size
            # measure available (median wobble 1.10 over 2 s, vs 1.13 for a
            # car's box width, which also swings with heading), and it shrinks
            # with distance exactly as a pixel threshold needs to.
            body_h = np.maximum(
                (sub["y2"] - sub["y1"]).to_numpy(float), 1.0)
            med_h = float(np.median(body_h))
            disp = float(np.linalg.norm(xy[-1] - xy[0]))
            dur = float(t1s - t0s)
            # A parked bicycle, or a person seen through the building glass,
            # sits perfectly still for minutes. Those are not traffic
            # participants and must not land in the waiting count: verified on
            # video, the long motionless tracks are bicycles at the railing
            # and people inside the lobby.
            if disp < STATIC_OBJECT_BODY_HEIGHTS * med_h and                     dur > STATIC_OBJECT_SECONDS:
                static_objects += 1
                ped_records.append({**base_rec, "category": "static_object", "jaywalk_segment": "no",
                                    "jaywalk_seconds": 0.0,
                                    "first_zone": "", "last_zone": "",
                                    "crossings": "", "zone_path": ""})
                continue
            if disp < WAIT_BODY_HEIGHTS * med_h:
                waiting += 1
                ped_records.append({**base_rec, "category": "waiting", "jaywalk_segment": "no",
                                    "jaywalk_seconds": 0.0,
                                    "first_zone": "", "last_zone": "",
                                    "crossings": "", "zone_path": ""})
                continue
            # peds may legitimately be on road zones too — track ALL zones
            pseq = []
            for hits in zhits:
                for z in hits:
                    if not pseq or pseq[-1] != z:
                        pseq.append(z)
            if len(pseq) >= 2:
                ped_od[f"{name_of[pseq[0]]} -> {name_of[pseq[-1]]}"] += 1
            else:
                ped_partial += 1

            # distance from each sample to each crosswalk, in body heights.
            # Membership by polygon alone is too brittle: a foot point wobbles
            # across the painted edge, and consecutive crosswalk polygons do
            # not quite touch, so a legal crossing repeatedly reads as "on the
            # road, outside every crosswalk".
            cw_dist = {}
            for cw in crosswalks:
                poly = poly_i32[cw]
                cw_dist[cw] = np.array(
                    [-cv2.pointPolygonTest(poly, (int(p[0]), int(p[1])), True)
                     for p in xy]) / body_h        # <=0 inside, in body heights
            near_any_cw = np.min(np.stack(list(cw_dist.values())), axis=0)                 if cw_dist else np.full(len(xy), 1e9)

            # crosswalk axis projection, using the tolerant membership
            my_crossings = []
            for cw in crosswalks:
                mean, axis, half = axes[cw]
                on = np.nonzero(cw_dist[cw] <= CW_TOL_BODY_HEIGHTS)[0]
                if len(on) < 2:
                    continue
                s0 = float((xy[on[0]] - mean) @ axis)
                s1 = float((xy[on[-1]] - mean) @ axis)
                if abs(s1 - s0) < 0.4 * half:
                    continue
                target = end_labels[cw][1] if s1 > s0 else end_labels[cw][0]
                cross_counts[f"{zlabel(cw)}  toward {target}"] += 1
                my_crossings.append(f"{zlabel(cw)} toward {target}")

            # Jaywalking means walking on the roadway AWAY from any crossing —
            # not merely "on a road zone", because every legal crossing puts a
            # pedestrian on the road by definition. Require distance from every
            # crosswalk and a sustained duration, and only consider it when the
            # person never completed a crossing.
            on_road = np.array([any(z in road_all for z in h) for h in zhits])
            far = near_any_cw > JAYWALK_BODY_HEIGHTS
            jw_samples = int(np.sum(on_road & far))
            jw_seconds = jw_samples * float(
                np.median(np.diff(sub["time_sec"].to_numpy())) or 0.15)

            # Two independent readings of the same track:
            #   category      — what the person mainly did (mutually exclusive,
            #                   so the ledger reconciles)
            #   jaywalk_segment — did any part of this journey run on the
            #                   roadway away from a crossing? A safety
            #                   indicator, and it must not be cancelled out by
            #                   the fact that the same person also used a
            #                   crosswalk somewhere else.
            jaywalk_segment = jw_seconds >= JAYWALK_MIN_SECONDS
            if jaywalk_segment:
                anc_j = [z for z in pseq
                         if type_of.get(z, "").startswith("ped")
                         and type_of.get(z) != "ped_crosswalk"]
                if len(anc_j) >= 2:
                    jaywalk_od[f"{name_of[anc_j[0]]} -> {name_of[anc_j[-1]]}"] += 1
                elif len(pseq) >= 2:
                    jaywalk_od[f"{name_of[pseq[0]]} -> {name_of[pseq[-1]]}"] += 1
                elif pseq:
                    jaywalk_od[f"seen at {name_of[pseq[0]]}, path broken"] += 1
                jaywalkers += 1

            if my_crossings:
                cat = "crosswalk_crossing"
            elif jw_seconds >= JAYWALK_MIN_SECONDS:
                cat = "jaywalking"
            elif len(pseq) >= 2:
                cat = "moving_no_crossing"
            else:
                cat = "partial"
            last_z = pseq[-1] if pseq else None
            dest, basis = corner_of.get(last_z), ""
            if dest:
                basis = "observed"
            elif leads_to.get(last_z):
                dest, basis = leads_to[last_z], "inferred"
            elif len(xy) >= 4 and corner_pt:
                # Judge by the WHOLE journey, per the user (2026-08-18): the
                # destination is the corner the walk closed distance on, not
                # the one its last few steps happened to point at. The walk
                # must have closed at least DEST_MIN_FRAC of its starting
                # distance to the corner, at least one body height of it in
                # absolute terms, and beaten the runner-up by DEST_MARGIN.
                scores = []
                for c, pt in corner_pt.items():
                    d0 = float(np.linalg.norm(pt - xy[0]))
                    d1 = float(np.linalg.norm(pt - xy[-1]))
                    scores.append(((d0 - d1) / max(d0, 1e-6), d0 - d1, c))
                scores.sort(reverse=True)
                f1, a1, c1 = scores[0]
                f2 = scores[1][0] if len(scores) > 1 else -1.0
                if f1 >= DEST_MIN_FRAC and a1 >= med_h and f1 - f2 >= DEST_MARGIN:
                    dest, basis = c1, "inferred"
            ped_records.append({**base_rec, "category": cat,
                "dest_corner": dest or "", "dest_basis": basis,
                "jaywalk_segment": "yes" if jaywalk_segment else "no",
                "jaywalk_seconds": round(jw_seconds, 1),
                "first_zone": zlabel(pseq[0]) if pseq else "",
                "last_zone": zlabel(pseq[-1]) if pseq else "",
                "crossings": "; ".join(my_crossings),
                "zone_path": " > ".join(zlabel(z) for z in pseq)})

    # ── Vehicle ledger: every track accounted for, integers only ──────
    def snm(z):
        return SHORT.get(name_of[z], name_of[z])

    lines = ["=" * 66,
             f"VEHICLE LEDGER — every track accounted for "
             f"(total {len(veh_records)})", "=" * 66]
    by_entry = defaultdict(list)
    for r in veh_records:
        if not r["seq"]:
            by_entry["none"].append(r)
        elif r["seq"][0] in road_arms:
            by_entry[r["seq"][0]].append(r)
        else:
            by_entry["center_born"].append(r)

    died_center_total = 0
    for arm in [a for a in road_arms if a in by_entry]:
        recs = by_entry[arm]
        n_static = sum(1 for r in recs
                       if len(r["seq"]) == 1 and r["disp"] < STATIC_VEH_DISP)
        lines.append(f"\nENTERED FROM {snm(arm)} : {len(recs)} vehicles "
                     f"({n_static} of them queued and never crossed)")
        groups = Counter()
        for r in recs:
            if len(set(r["arms"])) >= 2:
                a0, a1 = first_transition(r["arms"])
                mv = MOVEMENT.get((a0, a1), "U-TURN/anomaly")
                groups[f"{mv}   path {r['path']}{r['tag']}"] += 1
            elif r["seq"][-1] not in road_arms:
                if r["infer_exit"]:
                    mv = MOVEMENT.get((r["arms"][0], r["infer_exit"]),
                                      "U-TURN/anomaly")
                    groups[f"{mv} [EXIT INFERRED by extrapolation]   "
                           f"path {r['path']} ~> {snm(r['infer_exit'])}"
                           f"{r['tag']}"] += 1
                else:
                    groups[f"VANISHED in center   path {r['path']}{r['tag']}"] += 1
                    died_center_total += 1
            elif len(r["seq"]) == 1:
                if r["disp"] < STATIC_VEH_DISP:
                    groups["queued at signal, did not cross during clip"] += 1
                else:
                    groups[f"approached but track ended on arm   "
                           f"path {r['path']}{r['tag']}"] += 1
            else:
                groups[f"returned to same arm   path {r['path']}{r['tag']}"] += 1
        for k, c in groups.most_common():
            lines.append(f"  {c:>4}  {k}")

    born = by_entry.get("center_born", [])
    lines.append(f"\nAPPEARED IN CENTER (ID broken; came from another arm) : "
                 f"{len(born)} vehicles")
    groups = Counter()
    for r in born:
        if r["seq"][-1] in road_arms:
            mv_exit = snm(r["seq"][-1])
            groups[f"exited to {mv_exit}   path {r['path']}{r['tag']}"] += 1
        else:
            groups[f"also vanished in center   path {r['path']}{r['tag']}"] += 1
    for k, c in groups.most_common():
        lines.append(f"  {c:>4}  {k}")
    if by_entry.get("none"):
        lines.append(f"\nNO ROAD ZONE CONTACT : {len(by_entry['none'])}")

    lines += ["", "-" * 66,
              f"RECONCILIATION: {died_center_total} vanished in center vs "
              f"{len(born)} appeared in center.",
              f"  => at least {min(died_center_total, len(born))} of the "
              f"vanished are the same vehicles as the appeared ones",
              f"  (cannot be paired individually - multiple vehicles occupy "
              f"the center simultaneously)."]
    lines += ["", "=" * 66, "PEDESTRIAN CROSSINGS (crosswalk axis projection)", "=" * 66]
    for k, c in cross_counts.most_common():
        lines.append(f"  {c:>4}x  {k}")
    lines += ["", "=" * 66, "PEDESTRIAN OD (first ped zone -> last ped zone, top 25)", "=" * 66]
    for k, c in ped_od.most_common(25):
        lines.append(f"  {c:>4}x  {k}")
    lines += ["", "=" * 66, "JAYWALKING DIRECTIONS (first zone -> last zone)",
              "=" * 66]
    for k, c in jaywalk_od.most_common():
        lines.append(f"  {c:>4}x  {k}")
    lines += ["", "-" * 66,
              f"waiting (moved < {WAIT_BODY_HEIGHTS:.0f} body heights) : {waiting}",
              f"jaywalking (in road, outside crosswalk): {jaywalkers}",
              f"ped partial (<2 ped zones)             : {ped_partial}",
              "=" * 66]
    report = "\n".join(lines)
    with open(os.path.join(args.outdir, "flow_typed_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(report)
    print(report)

    # ── CSV outputs ───────────────────────────────────────────────────
    rows = []
    for r in veh_records:
        seg0, seg1 = r["t0"], r["t1"]
        if not r["seq"]:
            cat, mv = "no_road_contact", ""
        elif r["seq"][0] in road_arms:
            if len(set(r["arms"])) >= 2:
                cat = "complete"
                chain = movements_chain(r.get("vis", []), r["t1"])
                r["_chain"] = chain
                a0, a1 = chain[0][:2] if chain else first_transition(r["arms"])
                mv = MOVEMENT.get((a0, a1), "u-turn/anomaly")
                seg0 = max(r["t0"], chain[0][2]) if chain else r["t0"]
                seg1 = chain[0][3] if chain else r["t1"]
                # User rule (2026-08-24): a movement SEGMENT that starts on
                # Zone_05 (the crosswalk the slip-lane merger crosses) is a
                # south-entry right-turner - it must never be read as E->S.
                # Uses the SEGMENT start (the white dot the user sees), not
                # the track birth: the first version used the wrong point and
                # caught 1 of ~19.
                def _seg_first_pt(_r, _s0):
                    _ts = _r["ts_arr"]
                    _i = int(np.searchsorted(_ts, _s0 - 0.05))
                    _i = min(max(_i, 0), len(_ts) - 1)
                    return _r["xy_arr"][_i]
                def _on_z05(_pt):
                    _bz = args.birth_reclass_zone
                    return (_bz and _bz in zpolys and cv2.pointPolygonTest(
                        np.asarray(zpolys[_bz], np.int32),
                        (float(_pt[0]), float(_pt[1])), False) >= 0)
                if (chain and mv.startswith("LEFT")
                        and _on_z05(_seg_first_pt(r, seg0))
                        and MOVEMENT.get((chain[0][1], chain[0][0]))):
                    a0, a1 = chain[0][1], chain[0][0]
                    mv = MOVEMENT[(a0, a1)] + " [entry corrected: born on Z05]"
                # ── Zombie filter (user rule, 2026-08-24, video-verified) ──
                # A tracker box that camps on queued/parked cars and then
                # rides off on a passer-by produces a phantom movement. The
                # genuine-queue case (red light) is the SAME car before and
                # after the wait; the zombie is a DIFFERENT one. So: a
                # segment holding >=30 s of standstill is a zombie when the
                # standing phase and the moving phase disagree on identity
                # (majority class, or median box width off by >1.4x); a
                # gate-entry segment additionally dies at >=60 s standstill
                # outright (a genuine gate entrant rolls in; p99 of real
                # right turns stands 46 s).
                zts, zxy, zw = r["ts_arr"], r["xy_arr"], r["w_arr"]
                zcls = r["cls_arr"]
                m_ = (zts >= seg0 - 0.05) & (zts <= seg1 + 0.05)
                if zw is not None and int(m_.sum()) >= 5:
                    ts_ = zts[m_]; xy_ = zxy[m_]; w_ = zw[m_]; c_ = zcls[m_]
                    bwm = float(np.median(w_)) or 1.0
                    dt_ = np.diff(ts_)
                    sp_ = (np.hypot(*np.diff(xy_, axis=0).T)
                           / np.maximum(dt_, 1e-6) / bwm)
                    still_t = float(dt_[sp_ < 0.1].sum())
                    if still_t >= 30.0:
                        # identity is only comparable at the SAME depth, so
                        # test the 2 s just before vs just after each long
                        # standstill ends (phase-median widths are wrecked by
                        # perspective: a genuine queue waits far and drives
                        # near, 10x width change)
                        zomb = False
                        stand = sp_ < 0.1
                        i = 0
                        while i < len(stand):
                            if not stand[i]:
                                i += 1
                                continue
                            j = i
                            while j < len(stand) and stand[j]:
                                j += 1
                            run_t = float(ts_[min(j, len(ts_) - 1)] - ts_[i])
                            if run_t >= 20.0 and j < len(stand) - 3:
                                tb = ts_[j]
                                mA = (ts_ >= tb - 2.0) & (ts_ <= tb)
                                mB = (ts_ > tb) & (ts_ <= tb + 2.0)
                                if mA.sum() >= 3 and mB.sum() >= 3:
                                    cA = pd.Series(c_[mA]).mode().iloc[0]
                                    cB = pd.Series(c_[mB]).mode().iloc[0]
                                    wA = float(np.median(w_[mA]))
                                    wB = float(np.median(w_[mB])) or 1.0
                                    ratio = (max(wA, wB)
                                             / max(min(wA, wB), 1.0))
                                    if cA != cB or ratio > 1.4:
                                        zomb = True
                                        break
                            i = j
                        if (not zomb and chain
                                and chain[0][0] in gate_arms
                                and still_t >= 30.0):
                            zomb = True
                        if zomb:
                            cat, mv = "zombie_id_handoff", ""
                            r["_chain"] = []
                # ── Exit-direction gate (user, 2026-08-24): the segment must
                # END moving roughly TOWARD its exit arm. A row filed as S->E
                # whose tail drives back into the junction is a different
                # car's leftover and must not stay in the class.
                if cat == "complete" and chain:
                    _ts, _xy = r["ts_arr"], r["xy_arr"]
                    _m = (_ts >= seg1 - 2.0) & (_ts <= seg1 + 0.05)
                    if int(_m.sum()) >= 3:
                        _d = _xy[_m][-1] - _xy[_m][0]
                        _n = float(np.hypot(*_d))
                        _w = float(np.median(r["w_arr"][_m])) or 1.0
                        if _cen_zones:
                            # junction sites with a drawn center: the tail
                            # must move roughly TOWARD the exit arm as seen
                            # from the center (validated on the four 2026-08
                            # sites - do not touch)
                            _u = arm_cent[chain[0][1]] - _cen_pt
                            _u = _u / max(float(np.hypot(*_u)), 1e-9)
                            _bad = _n > 0.5 * _w and float(_d @ _u) < -0.1 * _n
                        else:
                            # center-less sites (road segments, long curved
                            # arms): "toward the exit centroid" is meaningless
                            # once the car is deep inside a long arm - the
                            # Physics east arm curves and 603 honest cars/day
                            # died. Kill only a tail clearly driving BACK
                            # toward the middle of the site.
                            _v = _cen_pt - _xy[_m][-1]
                            _v = _v / max(float(np.hypot(*_v)), 1e-9)
                            _bad = _n > 0.5 * _w and float(_d @ _v) > 0.5 * _n
                        if _bad:
                            cat, mv = "exit_direction_mismatch", ""
                            r["_chain"] = []
            elif r["seq"][-1] not in road_arms:
                if r.get("infer_exit"):
                    cat = "exit_inferred"
                    mv = MOVEMENT.get((r["arms"][0], r["infer_exit"]),
                                      "u-turn/anomaly") + " [inferred]"
                else:
                    cat, mv = "vanished_in_center", ""
            elif len(r["seq"]) == 1:
                cat = ("queued_never_crossed" if r["disp"] < STATIC_VEH_DISP
                       else "approached_track_ended")
                mv = ""
            else:
                cat, mv = "returned_to_same_arm", ""
        else:
            cat = ("appeared_center_exited" if r["seq"][-1] in road_arms
                   else "appeared_center_vanished")
            mv = ""
        rows.append({
            "track_id": r["id"], "class": r["cls"],
            "start_sec": r["t0"], "end_sec": r["t1"],
            "clip_time_hhmmss": f"{int(r['t0']//60):02d}:{int(r['t0']%60):02d}"
                                f"-{int(r['t1']//60):02d}:{int(r['t1']%60):02d}",
            "entry_arm": (SHORT.get(name_of[r["seq"][0]], "")
                          if r["seq"] and r["seq"][0] in road_arms else ""),
            "exit_arm": (SHORT.get(name_of[r["seq"][-1]], "")
                         if r["seq"] and r["seq"][-1] in road_arms
                         and len(r["seq"]) >= 2 else ""),
            "category": cat, "movement": mv, "zone_path": r["path"],
            "zone_path_full": r.get("full_path", ""),
            "clip_note": r["tag"].strip(" []"),
            "seg_start_sec": round(seg0, 1), "seg_end_sec": round(seg1, 1),
        })
        # continuation movements (user rule): one extra row per further
        # completed movement, id suffixed so nothing double-joins on track_id
        for kk, (b0, b1, s0, s1) in enumerate(r.get("_chain", [])[1:],
                                              start=2):
            _mvc = MOVEMENT.get((b0, b1), "u-turn/anomaly")
            if (_mvc.startswith("LEFT") and _on_z05(_seg_first_pt(r, s0))
                    and MOVEMENT.get((b1, b0))):
                b0, b1 = b1, b0
                _mvc = MOVEMENT[(b0, b1)] + " [entry corrected: born on Z05]"
            _ev = handoff_evidence(r, r["_chain"][kk - 2][3], s1)
            rows.append({**rows[-1],
                "track_id": f"{r['id']}#{kk}",
                "entry_arm": SHORT.get(name_of[b0], ""),
                "exit_arm": SHORT.get(name_of[b1], ""),
                "category": ("handoff_continuation" if _ev
                             else "complete_continued"),
                "movement": _mvc,
                "clip_note": (f"handoff evidence: {_ev}" if _ev
                              else rows[-1]["clip_note"]),
                "seg_start_sec": round(s0, 1),
                "seg_end_sec": round(s1, 1)})
    # a pedestrian-only pass produces no vehicle rows at all
    COLS = ["track_id", "class", "start_sec", "end_sec", "clip_time_hhmmss",
            "entry_arm", "exit_arm", "category", "movement", "zone_path",
            "zone_path_full", "seg_start_sec", "seg_end_sec", "clip_note"]
    detail_df = pd.DataFrame(rows, columns=COLS)
    if len(detail_df):
        detail_df = detail_df.sort_values(["category", "entry_arm",
                                           "start_sec"])
    p1 = os.path.join(args.outdir, "vehicle_ledger_per_track.csv")
    detail_df.to_csv(p1, index=False, encoding="utf-8-sig")

    summary = (detail_df.groupby(["entry_arm", "category", "movement",
                                  "zone_path"], dropna=False)
               .size().reset_index(name="count")
               .sort_values(["entry_arm", "count"], ascending=[True, False]))
    p2 = os.path.join(args.outdir, "vehicle_ledger_summary.csv")
    summary.to_csv(p2, index=False, encoding="utf-8-sig")

    p3 = os.path.join(args.outdir, "pedestrian_ledger_per_track.csv")
    pd.DataFrame(ped_records).sort_values(["category", "start_sec"]).to_csv(
        p3, index=False, encoding="utf-8-sig")
    print(f"\n[SAVED] {p1}\n[SAVED] {p2}\n[SAVED] {p3}")


if __name__ == "__main__":
    main()
