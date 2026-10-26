"""Passes with StatsBomb 360 freeze frames, as extra training data for SoccerMap.

    python tools/statsbomb_passes.py --cache open_data

StatsBomb open data (https://github.com/statsbomb/open-data; free for research and analysis, credit
StatsBomb): for every pass of the men's competitions with 360 data, the positions of the players the
broadcast camera saw at that moment (like soccercal's own data, but without velocities), the pass
origin, its end location and whether it was completed. Saved in the sample format of
``tools/train_soccermap.py`` (``statsbomb_samples.pkl``), attacking towards +x, metres, centred.
"""
from __future__ import annotations

import argparse
import json
import pickle
import urllib.request
from pathlib import Path

import numpy as np

RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
FAILED = {"Incomplete", "Out"}  # completed passes have no outcome; offside / unknown / injury are skipped


def get(path: str):
    with urllib.request.urlopen(RAW + path, timeout=120) as r:
        return json.loads(r.read())


def to_m(p) -> np.ndarray:
    """StatsBomb 120 x 80 yards (origin top-left, attacking +x) -> metres, pitch-centred, y up."""
    return np.array([p[0] / 120.0 * 105.0 - 52.5, (40.0 - p[1]) / 80.0 * 68.0], np.float32)


def match_samples(mid: int) -> list:
    ev = get(f"events/{mid}.json")
    frames = {f["event_uuid"]: f for f in get(f"three-sixty/{mid}.json")}
    out = []
    for e in ev:
        if e.get("type", {}).get("name") != "Pass" or e["id"] not in frames or "location" not in e:
            continue
        p = e["pass"]
        outcome = p.get("outcome", {}).get("name")
        if outcome is not None and outcome not in FAILED:
            continue
        ff = frames[e["id"]].get("freeze_frame") or []
        if len(ff) < 6:
            continue
        pos = np.array([to_m(f["location"]) for f in ff], np.float32)
        team = np.array([0 if f["teammate"] else 1 for f in ff])
        actor = [i for i, f in enumerate(ff) if f.get("actor")]
        pid = np.arange(len(ff))
        out.append({"pos": pos, "vel": np.zeros_like(pos), "team": team, "gk": np.array([bool(f.get("keeper")) for f in ff]),
                    "pid": pid, "ball": to_m(e["location"]), "dest": to_m(p["end_location"]),
                    "passer": int(actor[0]) if actor else -1, "att": 0, "right": (True, False), "pitch": (105.0, 68.0),
                    "y": int(outcome is None), "source": "statsbomb", "match": str(mid)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="open_data")
    ap.add_argument("--women", action="store_true", help="also the women's competitions")
    a = ap.parse_args()
    out_path = Path(a.cache) / "statsbomb_samples.pkl"
    done_path = Path(a.cache) / "statsbomb_done.json"
    samples = pickle.load(open(out_path, "rb")) if out_path.exists() else []
    done = set(json.load(open(done_path))) if done_path.exists() else set()
    comps = [c for c in get("competitions.json") if c.get("match_available_360")
             and (a.women or c["competition_gender"] == "male")]
    for c in comps:
        for m in get(f"matches/{c['competition_id']}/{c['season_id']}.json"):
            mid = m["match_id"]
            if mid in done:
                continue
            try:
                s = match_samples(mid)
            except Exception as e:  # noqa: BLE001 - a match without 360 file
                print(mid, "skipped:", e, flush=True)
                s = []
            samples += s
            done.add(mid)
            if len(done) % 20 == 0:
                pickle.dump(samples, open(out_path, "wb"))
                json.dump(sorted(done), open(done_path, "w"))
                print(len(done), "matches,", len(samples), "passes", flush=True)
    pickle.dump(samples, open(out_path, "wb"))
    json.dump(sorted(done), open(done_path, "w"))
    print("done:", len(done), "matches,", len(samples), "passes,", round(float(np.mean([s["y"] for s in samples])), 3), "completed")


if __name__ == "__main__":
    main()
