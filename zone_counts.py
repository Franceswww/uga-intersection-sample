"""
zone_counts.py - turn trajectories into zone sequences and simple counts.

    python zone_counts.py --traj results/11_SanfordCedar/trajectories.csv \
        --zones zones/Sanford_at_Cedar.json --names zones/Sanford_at_Cedar_names.json --out results/11_SanfordCedar

Each zone is a polygon in pixel coordinates (zones json) with a type and a readable
name (names json): vehicle_road = a road arm or the junction box, ped_crosswalk,
ped_sidewalk. A track's zone sequence is the order of zones its foot point entered
(a zone counts once it has been occupied for `--dwell` seconds, to ignore grazing).

Outputs:
    tracks.csv      one row per track: class, seconds visible, zone sequence
    vehicle_movements.csv   entry arm -> exit arm counts (vehicles whose first and last
                            road zone are two different arms)
    pedestrian_crossings.csv  crossings per crosswalk zone, and the sidewalk zone the
                            walk ended on
Tracks visible for less than `--min-visible` seconds are dropped as fragments.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib.path import Path

VEHICLES = {"Passenger Vehicle", "Bus", "Truck", "Motorcycle"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True); ap.add_argument("--zones", required=True); ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dwell", type=float, default=0.5, help="seconds inside a zone before it counts")
    ap.add_argument("--min-visible", type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    polys = {z: Path(np.array(p, float)) for z, p in json.load(open(a.zones))["zones"].items()}
    meta = json.load(open(a.names))["zones"]
    df = pd.read_csv(a.traj).sort_values(["track_id", "frame"])
    pts = df[["foot_x", "foot_y"]].to_numpy(float)
    inside = {z: poly.contains_points(pts) for z, poly in polys.items()}
    rows, movements, crossings = [], Counter(), []
    for tid, g in df.groupby("track_id", sort=False):
        vis = g.time_sec.iloc[-1] - g.time_sec.iloc[0]
        if vis < a.min_visible:
            continue
        cls = g["class"].mode().iloc[0]
        idx = g.index.to_numpy()
        seq, cur, t_in = [], None, None
        for k, i in enumerate(idx):
            here = [z for z in polys if inside[z][i]]
            z = here[0] if here else None
            t = g.time_sec.iloc[k]
            if z != cur:
                cur, t_in = z, t
            elif z is not None and (not seq or seq[-1] != z) and t - t_in >= a.dwell:
                seq.append(z)
        rows.append(dict(track_id=tid, cls=cls, visible_s=round(vis, 1), zones=" > ".join(f"{z}:{meta.get(z, {}).get('name', z)}" for z in seq)))
        if cls in VEHICLES:
            roads = [z for z in seq if meta.get(z, {}).get("type") == "vehicle_road" and meta[z].get("arm")]
            if len(roads) >= 2 and roads[0] != roads[-1]:
                movements[(meta[roads[0]]["arm"], meta[roads[-1]]["arm"])] += 1
        elif cls == "Pedestrian":
            cw = [z for z in seq if meta.get(z, {}).get("type") == "ped_crosswalk"]
            end = [z for z in seq if meta.get(z, {}).get("type") == "ped_sidewalk"]
            if cw:
                crossings.append(dict(track_id=tid, crosswalk=meta[cw[0]]["name"], ended_on=meta[end[-1]]["name"] if end else ""))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "tracks.csv"), index=False)
    pd.DataFrame([dict(entry_arm=k[0], exit_arm=k[1], vehicles=v) for k, v in sorted(movements.items())]).to_csv(os.path.join(a.out, "vehicle_movements.csv"), index=False)
    C = pd.DataFrame(crossings)
    (C.groupby(["crosswalk", "ended_on"]).size().rename("pedestrians").reset_index() if len(C) else C).to_csv(os.path.join(a.out, "pedestrian_crossings.csv"), index=False)
    print(f"{len(rows)} tracks kept; {sum(movements.values())} vehicle movements; {len(C)} pedestrian crossings -> {a.out}")


if __name__ == "__main__":
    main()
