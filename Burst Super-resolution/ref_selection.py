"""Reference-frame selection rule shared by the sweep, robustness and analysis scripts (pure Python, no GPU).

The "guarded" rule: use a fixed default frame unless that frame looks abnormal next to its neighbours.
    - Only the first `window` frames are candidates: on BurstSR the best reference was always one of the first five,
      and later frames drift away from the area the ground truth covers.
    - The default is frame 2.
    - The default is kept while its sharpness is within `margin` (0.2 = 20%) of the median sharpness of the other
      candidates. Much lower means it is blurred; much higher means it is noisy (noise inflates sharpness scores).
    - When the default is abnormal, the candidate closest to that median is used: the most typical frame, not the
      sharpest one.

Why not simply take the sharpest candidate: the first version did (switch when another frame is sharper by a
margin). It avoided a blurred default, but it also switched TO a frame that had been made noisy, losing up to 23 dB
(docs/PLAN.md, "Guarded selector"). On clean bursts chasing sharpness gained nothing.
"""

from statistics import median
from typing import Mapping

REF_POLICIES = ['frame0', 'frame2', 'guarded']
GUARD_DEFAULT = 2
GUARD_WINDOW = 5
GUARD_MARGIN = 0.2


def guarded_choice(sharpness: Mapping[int, float], default: int = GUARD_DEFAULT, window: int = GUARD_WINDOW,
                   margin: float = GUARD_MARGIN) -> int:
    """sharpness: frame index -> sharpness score (higher = sharper). Returns the frame index to use as reference."""
    candidates = [i for i in sorted(sharpness) if i < window]
    if default not in candidates:
        raise ValueError(f'Default frame {default} is not among the candidates {candidates}')
    others = [i for i in candidates if i != default]
    if not others:
        return default
    typical = median(sharpness[i] for i in others)
    if abs(sharpness[default] - typical) <= margin * typical:
        return default
    return min(others, key=lambda i: abs(sharpness[i] - typical))
