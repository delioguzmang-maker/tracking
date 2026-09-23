"""Network architecture matches the public checkpoints (shapes / key names)."""
import pytest

torch = pytest.importorskip("torch")

from soccercal.field_model import decode_keypoints  # noqa: E402
from soccercal.hrnet import build_keypoint_net, build_line_net  # noqa: E402


def test_shapes_and_state_dict_layout():
    net = build_keypoint_net().eval()
    sd = net.state_dict()
    assert len(sd) == 1839  # same tensor count as the NBJW/PnLCalib checkpoints
    assert "head.0.3.weight" in sd and sd["head.0.3.weight"].shape[0] == 58
    assert build_line_net().state_dict()["head.0.3.weight"].shape[0] == 24
    with torch.inference_mode():
        y = net(torch.rand(1, 3, 128, 224))
    assert y.shape == (1, 58, 64, 112)
    assert torch.allclose(y.sum(1), torch.ones(1, 64, 112), atol=1e-4)  # softmax over channels


def test_subpixel_decoding():
    import numpy as np

    h, w = 60, 80
    yy, xx = np.mgrid[0:h, 0:w]
    hm = np.zeros((57, h, w), np.float32)
    hm[4] = 0.9 * np.exp(-((xx - 30.3) ** 2 + (yy - 20.7) ** 2) / (2 * 2.0 ** 2))
    kp = decode_keypoints(hm, threshold=0.1)
    x, y, s = kp[5]
    assert abs(x - 30.3) < 0.05 and abs(y - 20.7) < 0.05
    assert set(kp) == {5}
