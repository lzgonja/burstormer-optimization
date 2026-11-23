"""Common reference-selection interface used by the Burstormer pipeline."""

import csv
import importlib
from numbers import Integral


NUM_FRAMES = 14
AVAILABLE_METHODS = ("frame0", "gbrs", "algorithm2")
EXPERIMENT_FIELDS = (
    "burst_id",
    "selection_method",
    "selected_reference_index",
    "corrected_psnr_gain",
)


def validate_reference_index(reference_index, num_frames=NUM_FRAMES):
    if isinstance(reference_index, bool) or not isinstance(reference_index, Integral):
        raise TypeError("Reference index must be an integer.")
    if not 0 <= reference_index < num_frames:
        raise ValueError(f"Reference index must be between 0 and {num_frames - 1}.")
    return int(reference_index)


def reorder_burst(burst, reference_index=0):
    """Move the selected frame first; keep every other frame in capture order."""
    frame_dim = 1 if burst.ndim == 5 else 0
    if burst.ndim not in (4, 5) or burst.shape[frame_dim] != NUM_FRAMES:
        raise ValueError("Expected burst shape (14, 4, H, W) or (B, 14, 4, H, W).")
    reference_index = validate_reference_index(reference_index, burst.shape[frame_dim])
    order = [reference_index] + [i for i in range(NUM_FRAMES) if i != reference_index]
    return burst[:, order] if frame_dim == 1 else burst[order]


def select_reference_frame(burst, method="frame0", reference_index=0):
    """Select from the input burst only, using ``fixed`` or an algorithm module.

    Algorithm modules implement ``select_reference_frame(burst) -> int``. Thus
    algorithm2 and algorithm3 require no pipeline changes when they are added.
    """
    if method in ("fixed", "frame0"):
        return validate_reference_index(reference_index)
    module_name = "algorithm1" if method == "gbrs" else method
    module = importlib.import_module(f"reference_selection.{module_name}")
    selector = getattr(module, "select_reference_frame", None)
    if not callable(selector):
        raise ValueError(f"Selection module {module_name!r} has no select_reference_frame callable.")
    unbatched = burst[0] if burst.ndim == 5 else burst
    return validate_reference_index(selector(unbatched))


def select_available_references(burst):
    """Run every selector currently integrated, in stable experiment order."""
    return {method: select_reference_frame(burst, method) for method in AVAILABLE_METHODS}


def reconstruct_selected_references(burst, model, selections):
    """Run one reconstruction per unique selected index."""
    return {reference: model(reorder_burst(burst, reference))
            for reference in set(selections.values())}


def write_experiment_csv(path, rows):
    """Write the Issue #10 per-burst result schema."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
