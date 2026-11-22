"""Evaluation helpers in tools/."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def test_game_state_reconstruction_labels(tmp_path):
    from soccernet_eval import ground_truth

    ann = []
    for f in (1, 2):
        ann.append({"image_id": f"9{f}", "track_id": 3, "category_id": 1, "bbox_image": {"x": 10, "y": 20, "w": 30, "h": 60},
                    "attributes": {"role": "player", "team": "left", "jersey": "7"}})
        ann.append({"image_id": f"9{f}", "track_id": 4, "category_id": 2, "bbox_image": {"x": 100, "y": 20, "w": 30, "h": 60},
                    "attributes": {"role": "goalkeeper", "team": "right", "jersey": None}})
        ann.append({"image_id": f"9{f}", "track_id": 5, "category_id": 3, "bbox_image": {"x": 200, "y": 20, "w": 30, "h": 60},
                    "attributes": {"role": "referee", "team": None, "jersey": None}})
    ann.append({"image_id": "91", "category_id": 5, "lines": {}})  # pitch annotation: no box
    lab = {"images": [{"image_id": "91", "file_name": "000001.jpg"}, {"image_id": "92", "file_name": "000002.jpg"}],
           "annotations": ann}
    (tmp_path / "Labels-GameState.json").write_text(json.dumps(lab))
    gt, info = ground_truth(tmp_path)
    assert info == {3: ("left", 7, False), 4: ("right", None, True)}
    assert gt.shape == (4, 6) and set(gt[:, 0]) == {1, 2} and np.allclose(gt[0, 2:], [10, 20, 30, 60])
    assert ground_truth(tmp_path, all_people=True)[0].shape == (6, 6)  # the referee's boxes too
