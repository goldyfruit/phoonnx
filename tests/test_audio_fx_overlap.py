"""The overlap search of the WSOLA time stretch."""
import numpy as np
import pytest

from phoonnx import audio_fx


@pytest.mark.parametrize(("size", "frame", "step"), [
    (2646, 1764, 4),   # 44.1 kHz, the whole search range
    (1322, 882, 2),    # 22.05 kHz
    (1000, 882, 2),    # cut short at an end of the clip
    (883, 882, 4),     # fewer offsets than phases
    (882, 882, 1),     # one offset
])
def test_each_offset_is_scored_as_its_window_times_the_template(size, frame, step):
    rng = np.random.default_rng(size + step)
    within = rng.standard_normal(size).astype(np.float32)
    template = rng.standard_normal(frame).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(within, frame)[:, ::step]
    expected = windows.astype(np.float64) @ template[::step].astype(np.float64)
    scores = audio_fx._overlaps(within, template, step)
    assert scores.shape == expected.shape and scores.dtype == np.float32
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-3)
