"""All tunable settings in one place (defaults work for standard broadcast footage)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # ---------------------------------------------------------------- hardware
    device: str | None = None  # None/"auto": cuda > mps (Apple) > cpu
    batch: int = 8  # frames per network batch (lower it if you run out of memory)

    # ------------------------------------------------------- what to process
    start_s: float = 0.0  # start time in the video (seconds)
    max_seconds: float | None = None  # process at most this many seconds (None = all)
    stride: int | None = None  # analyse every n-th frame; None = automatic (~25 analysed fps,
    # i.e. 1 for 25/30 fps video, 2 for 50/60 fps)
    keyframe_every: int = 5  # run the pitch network every n analysed frames
    # ------------------------------------------------------------- detection
    det_model: str = "yolo11m.pt"  # any Ultralytics model; COCO person/ball classes
    det_imgsz: int = 1280
    det_conf: float = 0.10  # people
    ball_conf: float = 0.03  # the ball is tiny: keep weak candidates, the ball tracker filters them
    jersey_ocr: bool = True  # read shirt numbers (bundled OCR model; needs rapidocr_onnxruntime)
    jersey_crops: int = 3  # players read per keyframe (the largest ones)
    # ----------------------------------------------------------- calibration
    cut_threshold: float = 0.30  # frame-signature distance that means a shot cut
    min_align: float = 0.35  # minimum line alignment to accept a calibration
    max_keyframe_gap_s: float = 3.0  # trust registration at most this far from a calibration
    main_camera_only: bool = True  # only track shots from the main (most used) camera position:
    # other angles are mostly replays / close-ups, which SkillCorner does not track either
    # --------------------------------------------------------------- players
    pitch_margin: float = 3.0  # keep people within the pitch + margin (m)
    staff_outside_share: float = 0.6  # a person outside the lines this often = coach / assistant referee
    closeup_person_frac: float = 0.4  # a person taller than this fraction of the image = close-up shot
    min_grass: float = 0.25  # less grass than this in the image = crowd / bench / close-up shot
    height_ratio: tuple = (0.45, 1.9)  # box height / expected height of a 1.8 m person
    # -------------------------------------------------------------- tracking
    track_high: float = 0.45
    track_low: float = 0.15
    max_lost_s: float = 1.0
    stitch_max_gap_s: float = 60.0
    players_per_team: int | None = 11  # at most this many identities per team (10 outfield + GK):
    # fragments of the same player are joined until the count fits; None = no limit
    # ---------------------------------------------------------------- output
    out_fps: float = 10.0  # SkillCorner delivers 10 fps
    extrapolate: bool = True  # fill off-screen players (is_detected = False)
    max_extrapolate_s: float = 15.0
    period: int = 1
    time_offset_s: float = 0.0  # match clock at the first processed frame
    home_name: str = "Team A"
    away_name: str = "Team B"
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
