"""
track_endpoints.py
------------------
One general rule for every track that ends (or begins) prematurely —
vehicles and pedestrians, any zone type, both ends of the track.

Why a track can stop being observed:
  left_view   : its detection box was clipped by the frame border -> the
                object physically left the camera's field of view. Real end.
  truncated   : its first/last sample sits within `clip_edge_s` of the clip
                boundary -> the recording started/stopped mid-journey.
  detection_loss : neither of the above -> the object was still inside the
                scene and still being recorded, so the track ended because
                detection failed. This is the only case worth extrapolating.

For detection_loss ends we march the smoothed end velocity forward (or the
start velocity backward) in small steps and report the first zone entered
that differs from the current one. The prediction is accepted only if that
zone is reachable in the zone-adjacency graph, which is derived from the
zone polygons themselves — this is what stops a projection from "walking
through a building".

Confidence combines: how long the projection had to run, how steady the
motion was, and whether the predicted zone is directly adjacent.
"""
import numpy as np
import cv2

BORDER_PX = 8


def build_adjacency(zone_polys, shape=(2160, 3840), gap_px=60,
                    min_overlap=200):
    """Zones whose painted areas lie within gap_px of each other touch.
    Keep this tight and let `reachable()` allow one intermediate hop: at this
    site the road zones are separated by crosswalk bands (centre->north arm
    is 374 px of painted crosswalk), which a vehicle legitimately drives
    over. A loose gap_px instead would make every zone adjacent to every
    other and the check would stop rejecting anything."""
    masks, dil = {}, {}
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (gap_px * 2 + 1,) * 2)
    for k, poly in zone_polys.items():
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 1)
        masks[k] = m
        dil[k] = cv2.dilate(m, kern)
    keys = sorted(zone_polys)
    adj = {k: set() for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if int(np.count_nonzero(dil[a] & masks[b])) > min_overlap:
                adj[a].add(b)
                adj[b].add(a)
    return adj


def reachable(adjacency, here, target, max_hops=2):
    """Can `target` be reached from any zone in `here` within max_hops?
    One intermediate hop covers the case where the object crossed a zone
    (typically a crosswalk band) without leaving a sample in it."""
    if not here:
        return True                      # endpoint outside every zone
    frontier = set(here)
    seen = set(here)
    for _ in range(max_hops):
        nxt = set()
        for z in frontier:
            nxt |= adjacency.get(z, set())
        if target in nxt:
            return True
        nxt -= seen
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    return False


def endpoint_status(sub, clip_t0, clip_t1, img_w=3840, img_h=2160,
                    clip_edge_s=0.5, at_start=False):
    """Why did this track begin/end? Uses the ORIGINAL box columns, so it is
    unaffected by any border filtering applied upstream."""
    row = sub.iloc[0] if at_start else sub.iloc[-1]
    t = row["time_sec"]
    if at_start and t - clip_t0 <= clip_edge_s:
        return "truncated"
    if not at_start and clip_t1 - t <= clip_edge_s:
        return "truncated"
    if ({"x1", "x2", "y2"} <= set(sub.columns)
            and (row["x1"] <= BORDER_PX or row["x2"] >= img_w - BORDER_PX
                 or row["y2"] >= img_h - BORDER_PX)):
        return "left_view"
    return "detection_loss"


def _seg_enters(p0, p1, poly):
    """Fraction along the segment p0->p1 at which it first enters `poly`,
    or None if it never does.

    The test is on the SEGMENT, not on points sampled along it. Sampling
    skips zones: a car crossing the far arm at 1900 px/s covers 475 px
    between two 0.25 s samples, so a zone narrower than that can fall
    entirely between consecutive samples and never be detected — which is
    why fast through-movements were reported as vanishing inside the
    intersection while the projected line clearly passed through the exit
    arm."""
    poly = np.asarray(poly, dtype=np.float64)
    if cv2.pointPolygonTest(poly.astype(np.int32),
                            (int(p0[0]), int(p0[1])), False) >= 0:
        return 0.0
    d = np.asarray(p1, float) - np.asarray(p0, float)
    best = None
    n = len(poly)
    for i in range(n):
        a = poly[i]
        e = poly[(i + 1) % n] - a
        den = d[0] * (-e[1]) + d[1] * e[0]
        if abs(den) < 1e-9:
            continue                              # parallel to this edge
        rhs = a - np.asarray(p0, float)
        t = (rhs[0] * (-e[1]) + rhs[1] * e[0]) / den      # along the segment
        u = (d[0] * rhs[1] - d[1] * rhs[0]) / den         # along the edge
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 and (best is None or t < best):
            best = t
    return best


def extrapolate(sub, zone_polys, adjacency, at_start=False, k=5,
                min_speed=100.0, max_horizon_s=3.0, want=None):
    """Project the endpoint velocity forward and return every zone the
    projected path crosses, in the order it crosses them:
        [{"zone", "seconds", "confidence", "speed"}, ...]
    `want` is an optional set of zone keys the caller cares about; the first
    match is returned as the single best guess. A vehicle leaving the
    intersection box crosses the painted crosswalk bands before reaching the
    far road arm, so callers must be able to look past the first zone."""
    xy = sub[["foot_x", "foot_y"]].to_numpy(float)
    t = sub["time_sec"].to_numpy(float)
    if len(xy) < 3:
        return None
    kk = min(k, len(xy) - 1)
    if at_start:
        p0, dt, v = xy[0], t[kk] - t[0], xy[0] - xy[kk]
    else:
        p0, dt, v = xy[-1], t[-1] - t[-1 - kk], xy[-1] - xy[-1 - kk]
    if dt <= 0:
        return None
    v = v / dt
    speed = float(np.linalg.norm(v))
    if speed < min_speed:
        return None

    here = [z for z, poly in zone_polys.items()
            if cv2.pointPolygonTest(poly.astype(np.int32),
                                    (int(p0[0]), int(p0[1])), False) >= 0]
    # steadiness of the motion the projection is based on
    seg = xy[:kk + 1] if at_start else xy[-kk - 1:]
    mag = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    steady = float(np.clip(1.0 - mag.std() / (mag.mean() + 1e-9), 0, 1))

    # one straight segment covering the whole horizon; every zone it crosses
    # is found exactly, in crossing order, at any speed
    p1 = p0 + v * max_horizon_s
    path = []
    for z, poly in zone_polys.items():
        if z in here:
            continue
        frac = _seg_enters(p0, p1, poly)
        if frac is None:
            continue
        if not reachable(adjacency, here, z):
            continue              # cannot get there from where it vanished
        secs = float(frac) * max_horizon_s
        conf = float(np.clip((1.0 - secs / max_horizon_s) * 0.6
                             + steady * 0.4, 0, 1))
        path.append({"zone": z, "seconds": round(secs, 2),
                     "confidence": round(conf, 2), "speed": round(speed)})
    path.sort(key=lambda e: e["seconds"])
    if not path:
        return None
    if want is not None:
        hit = next((e for e in path if e["zone"] in want), None)
        return hit
    return path[0]
