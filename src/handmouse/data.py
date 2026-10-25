"""Load raw landmark clips written by negative_logger.save_clip. Features are computed on load, never stored."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .negative_logger import DATA_DIR


@dataclass
class Clip:
    subject_id: str
    label: str
    lighting: str
    handedness: str
    timestamps: np.ndarray  # (T,)
    landmarks: np.ndarray  # (T, 21, 3)


def load_clips(root: Path = DATA_DIR) -> list[Clip]:
    clips = []
    for p in sorted(root.glob("*/*/*.npz")):
        d = np.load(p)
        clips.append(Clip(str(d["subject_id"]), str(d["label"]), str(d["lighting"]),
                          str(d["handedness"]), d["timestamps"], d["landmarks"]))
    return clips
