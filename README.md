# UGA campus intersections - 15-minute sample and tracking code

Prepared for the AI Team, from the UGA roadside-camera study (15 intersections, March 2026).

## What is here

| item | content |
|---|---|
| `clips/NN_<Site>.mp4` (**not in this repository**: 4.8 GB, in the OneDrive package, see below) | the same 15 minutes, **Monday 2026-03-23, 13:00-13:15**, at every intersection (4.4 GB in all); 4K HEVC, ~14 fps, no audio. Clips are cut without re-encoding, so each one starts at the last keyframe before 13:00, up to about one minute early; `clips/manifest.csv` gives each clip's true start time and the camera it comes from |
| `zones/<Site>.json` | zone polygons in pixel coordinates of that clip's camera (road arms, junction box, crosswalks, sidewalks) |
| `zones/<Site>_names.json` | what each zone is: `type` (vehicle_road / ped_crosswalk / ped_sidewalk), a readable `name`, the road `arm` letter for vehicle zones, the `corner` a sidewalk leads to |
| `run_site.py` | **the study's full chain on one clip**: track -> clean/stitch -> counts (use this to reproduce the study's numbers) |
| `track_clip.py` | stage 1: detection + tracking (YOLOv8m + BoT-SORT) -> `trajectories.csv` |
| `stitch_tracks.py` | stage 2: the study's trajectory cleaner, unchanged - spike repair, teleport and time-gap splits, duplicate merge, frame-border filter, fragment re-identification -> `clean_trajectories.csv` |
| `flow_typed.py`, `track_endpoints.py` | stage 3: the study's counting stage, unchanged - vehicle movements (entry arm -> exit arm, with inferred exits for tracks lost at the edge) and pedestrian trips with destinations -> `results/` |
| `zone_counts.py` | a short, readable alternative to stage 3 (zone sequences and simple counts) for understanding the data; its numbers are first-pass and differ from the study's |
| `botsort.yaml` | tracker settings used in the study |
| `cut_clips.py` | how the clips were cut (needs the NAS; included for reference) |
| `maps/camera_map.png`, `maps/intersections.csv` | where the 15 cameras are (numbers match the clip names), with coordinates and Google Maps links |

Why this window: summed over all 15 intersections it is the busiest quarter hour of the day
(about 8,600 pedestrian trips and 1,600 vehicle movements in 15 minutes), the first period
after the 12:55 class dismissal; it is the busiest or second-busiest window at 14 of the 15 sites.

## Getting the clips

The videos are too large for GitHub. The complete package, including the 15 clips (4.8 GB),
is shared as a OneDrive folder (link provided by email); this repository mirrors its code and
zone files. Put the clips in `clips/` next to `manifest.csv` and the commands below work as written.

## Which intersections to start with

Volumes in this 15-minute window on 2026-03-23 (the study's counts; vehicles = movements
through the junction, pedestrians = walks with a destination):

| no | intersection | vehicles / 15 min | pedestrians / 15 min | note |
|---|---|---|---|---|
| 6 | Lumpkin at Cedar | 249 | 470 | signalised, 4 arms |
| 5 | Lumpkin at Baxter | 246 | 1,001 | busiest mixed site: heavy on both |
| 8 | Lumpkin at Tate | 243 | 373 | |
| 7 | Lumpkin at No Name Park | 190 | 276 | |
| 4 | Lumpkin at Baldwin | 180 | 500 | |
| 1 | Baxter at Hull | 157 | 458 | |
| 11 | Sanford at Cedar | 105 | 509 | four-camera site; this clip is camera 02 |
| 10 | Sanford at Baldwin | 96 | 776 | one-way Sanford arm |
| 14 | Sanford Dr at Physics lot | 41 | 327 | bus stop |
| 12 | Sanford at Hooper | 38 | 570 | 7.5 fps |
| 13 | Sanford Dr northbound | 34 | 198 | |
| 15 | Stadium West | 30 | 1,395 | almost all pedestrians |
| 2 | Bocock at Founders | 4 | 133 | quiet service street |
| 3 | Bookstore Southwest | 0 | 1,048 | plaza, pedestrians only |
| 9 | MLC / Georgia Quad | 0 | 558 | plaza, pedestrians only |

For vehicle-pedestrian interaction work, start with the Lumpkin Street corridor (sites 4,
5, 6, 7, 8) and Baxter at Hull (1): they carry real vehicle traffic and many crossings in
the same frame. Sites 3, 9 and 15 are pedestrian-only or nearly so (good for crowd
counting, useless for vehicles); site 2 has too little traffic of either kind to be
informative in 15 minutes.

## Running

```
pip install -r requirements.txt            # ultralytics pulls torch; a CUDA GPU is strongly advised
python run_site.py --clip clips/11_SanfordCedar.mp4 --site Sanford_at_Cedar --out results/11_SanfordCedar
```
`--site` is the zone file stem (see `clips/manifest.csv`, column `zones`). Add `--annotate` for a
boxed video, `--max-seconds 120` for a quick test. Outputs in `results/11_SanfordCedar/results/`:
`vehicle_ledger_per_track.csv` (one row per vehicle movement: entry arm, exit arm, category
complete / exit_inferred / approached_track_ended, times), `vehicle_ledger_summary.csv` (the
turning-movement table), `pedestrian_ledger_per_track.csv` (one row per walk: destination
corner, crosswalk used or jaywalk, waiting), `flow_typed_report.txt` (readable summary).
The study's report counts vehicles with category complete or exit_inferred and pedestrians
that have a destination. Lumpkin at Baldwin needs two site-specific options; `run_site.py`
applies them itself.

The stages can also be run one at a time:
```
python track_clip.py --video clips/11_SanfordCedar.mp4 --out results/11_SanfordCedar --annotate
python stitch_tracks.py --traj results/11_SanfordCedar/trajectories.csv --zones zones/Sanford_at_Cedar.json --names zones/Sanford_at_Cedar_names.json --outdir results/11_SanfordCedar --name clean --border 8 --cut-vel-samples 3
python flow_typed.py --traj results/11_SanfordCedar/clean_trajectories.csv --zones zones/Sanford_at_Cedar.json --names zones/Sanford_at_Cedar_names.json --outdir results/11_SanfordCedar/results
```

`track_clip.py` processes every 2nd frame at 1920 px with confidence 0.10 and takes about
10 minutes per clip on an RTX 4090 (`--max-seconds 60` for a quick test, `--device cpu`
works but is slow). The weights `yolov8m.pt` are downloaded automatically on first use.

## Notes and known limits

* Coordinates are pixels. Foot point = bottom-centre of the box; for buses and trucks it
  lies under the front/rear face, not the centre of the vehicle.
* Tracks break in occlusion (behind buses, in crowds). `stitch_tracks.py` repairs what it
  can (it is the study's own code); what remains is reported honestly by `flow_typed.py` as
  `approached_track_ended` rather than guessed.
* Through movements are the most occluded and therefore most under-counted; turns are the
  cleanest.
* The Bookstore and MLC cameras look at plazas with no vehicle traffic (vehicle counts are 0).
* Sanford Dr at Hooper (site 12) records at 7.5 fps, half the others, and its NVR export has a malformed sample-description table that standard decoders reject; the clip here was cut from a repaired copy (`patch_hooper_mp4.py` in the study's scripts), which is why it is listed separately in the manifest.
* The cameras are fixed; the zones were drawn on frames from the same day and camera.
* Privacy: the footage shows identifiable people. Keep it within the team and do not
  publish frames with faces.

Contact: Qianyi Wu, qw28137@uga.edu
