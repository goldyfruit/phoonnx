import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.vits import VitsPriorSplitAdapter


def _req(phoneme_ids=None, **params):
    ids = phoneme_ids if phoneme_ids is not None else np.array([[1, 0, 2, 0, 3]], np.int64)
    return AdapterSynthesisRequest(
        phoneme_ids=ids,
        phoneme_lengths=np.array([ids.shape[1]], np.int64),
        speaker_id=0, language_id=0, params=params,
    )


class _Out:
    def __init__(self, name):
        self.name = name


class _FakeDurationSession:
    """Stands in for Inflect's duration.onnx: records the feed dict and returns
    a fixed-shape (m_p_exp, logs_p_exp, y_mask) triple."""

    def __init__(self, frames=8, channels=4):
        self.last_feed = None
        self.frames = frames
        self.channels = channels

    def get_outputs(self):
        return [_Out("m_p_exp"), _Out("logs_p_exp"), _Out("y_mask")]

    def run(self, output_names, feed):
        assert output_names == ["m_p_exp", "logs_p_exp", "y_mask"]
        self.last_feed = feed
        shape = (1, self.channels, self.frames)
        m_p_exp = np.zeros(shape, np.float32)
        logs_p_exp = np.zeros(shape, np.float32)
        y_mask = np.ones((1, 1, self.frames), np.float32)
        return [m_p_exp, logs_p_exp, y_mask]


class _FakeDecodeSession:
    """Stands in for Inflect's decode.onnx."""

    def __init__(self):
        self.last_feed = None

    def run(self, output_names, feed):
        assert output_names == ["waveform"]
        self.last_feed = feed
        frames = feed["m_p_exp"].shape[-1]
        wav = (feed["zp_noise"].reshape(-1)[:frames] * 0.5).astype(np.float32)
        return [wav.reshape(1, 1, -1)]


def _configured_adapter(decode_session=None):
    adapter = VitsPriorSplitAdapter()
    adapter.decode = decode_session or _FakeDecodeSession()
    return adapter


# ---------------------------------------------------------------------------
# registration / detect
# ---------------------------------------------------------------------------

def test_vits_prior_split_registered():
    from phoonnx.engines import list_engines
    assert "vits_prior_split" in list_engines()


def test_vits_prior_split_detect_by_output_signature_and_decode_path():
    config = {"engine_params": {"decode_path": "decode.onnx"}}
    assert VitsPriorSplitAdapter.detect(config, _FakeDurationSession())


def test_vits_prior_split_detect_requires_decode_path():
    assert not VitsPriorSplitAdapter.detect({"engine_params": {}}, _FakeDurationSession())
    assert not VitsPriorSplitAdapter.detect(None, _FakeDurationSession())


def test_vits_prior_split_detect_requires_matching_output_signature():
    class _PlainVitsSession:
        def get_outputs(self):
            return [_Out("output")]

    config = {"engine_params": {"decode_path": "decode.onnx"}}
    assert not VitsPriorSplitAdapter.detect(config, _PlainVitsSession())


def test_vits_prior_split_detect_requires_a_session():
    config = {"engine_params": {"decode_path": "decode.onnx"}}
    assert not VitsPriorSplitAdapter.detect(config, None)


def test_vits_prior_split_does_not_steal_plain_vits():
    # A plain single-graph VITS config has no decode_path at all.
    assert not VitsPriorSplitAdapter.detect({"engine": "vits"}, _FakeDurationSession())


# ---------------------------------------------------------------------------
# configure()
# ---------------------------------------------------------------------------

def test_vits_prior_split_configure_loads_decode_graph(monkeypatch):
    calls = {}

    def _fake_make_session(path, providers=None):
        calls["path"] = path
        calls["providers"] = providers
        return _FakeDecodeSession()

    monkeypatch.setattr("phoonnx.engines.vits.make_session", _fake_make_session)

    class _VoiceConfig:
        engine_params = {"decode_path": "/tmp/decode.onnx", "providers": ["CPUExecutionProvider"]}

    adapter = VitsPriorSplitAdapter()
    adapter.configure(_VoiceConfig())
    assert adapter.decode is not None
    assert calls["path"] == "/tmp/decode.onnx"

    # a second configure() call must not clobber an already-loaded graph
    adapter.configure(_VoiceConfig())
    assert calls["path"] == "/tmp/decode.onnx"


def test_vits_prior_split_configure_without_decode_path_is_noop():
    class _VoiceConfig:
        engine_params = {}

    adapter = VitsPriorSplitAdapter()
    adapter.configure(_VoiceConfig())
    assert adapter.decode is None


# ---------------------------------------------------------------------------
# synthesize()
# ---------------------------------------------------------------------------

def test_vits_prior_split_synthesize_missing_decode_graph_raises():
    adapter = VitsPriorSplitAdapter()
    with pytest.raises(RuntimeError):
        adapter.synthesize(_req(), _FakeDurationSession())


def test_vits_prior_split_synthesize_basic():
    adapter = _configured_adapter()
    duration = _FakeDurationSession()
    result = adapter.synthesize(_req(), duration)

    assert result.audio.ndim == 1
    assert result.audio.dtype == np.float32
    assert result.audio.size > 0
    assert "tokens" in duration.last_feed
    assert "lengths" in duration.last_feed
    assert "length_scale" in duration.last_feed
    np.testing.assert_array_equal(duration.last_feed["tokens"], _req().phoneme_ids)


def test_vits_prior_split_length_scale_follows_vits_convention():
    # Same convention as the single-graph VitsAdapter: length_scale > 1 = slower.
    adapter = _configured_adapter()
    duration = _FakeDurationSession()
    adapter.synthesize(_req(length_scale=2.0), duration)
    assert float(duration.last_feed["length_scale"]) == pytest.approx(2.0)


def test_vits_prior_split_noise_scale_maps_to_decode_noise_scale():
    decode = _FakeDecodeSession()
    adapter = _configured_adapter(decode)
    adapter.synthesize(_req(noise_scale=0.1), _FakeDurationSession())
    assert float(decode.last_feed["noise_scale"]) == pytest.approx(0.1)


def test_vits_prior_split_seed_is_deterministic():
    decode1 = _FakeDecodeSession()
    adapter1 = _configured_adapter(decode1)
    audio1 = adapter1.synthesize(_req(seed=42), _FakeDurationSession()).audio

    decode2 = _FakeDecodeSession()
    adapter2 = _configured_adapter(decode2)
    audio2 = adapter2.synthesize(_req(seed=42), _FakeDurationSession()).audio

    np.testing.assert_array_equal(audio1, audio2)


def test_vits_prior_split_different_seeds_differ():
    decode1 = _FakeDecodeSession()
    adapter1 = _configured_adapter(decode1)
    audio1 = adapter1.synthesize(_req(seed=1), _FakeDurationSession()).audio

    decode2 = _FakeDecodeSession()
    adapter2 = _configured_adapter(decode2)
    audio2 = adapter2.synthesize(_req(seed=2), _FakeDurationSession()).audio

    assert not np.array_equal(audio1, audio2)


def test_vits_prior_split_default_params_include_seed():
    adapter = VitsPriorSplitAdapter()
    defaults = adapter.default_params()
    assert defaults["seed"] == 0.0
    assert "noise_scale" in defaults
    assert "length_scale" in defaults


# ---------------------------------------------------------------------------
# adversarial: empty and long token sequences
# ---------------------------------------------------------------------------

def test_vits_prior_split_empty_phoneme_sequence_still_synthesizes():
    adapter = _configured_adapter()
    empty_ids = np.zeros((1, 0), np.int64)
    result = adapter.synthesize(_req(phoneme_ids=empty_ids), _FakeDurationSession(frames=0))
    assert result.audio.dtype == np.float32
    assert result.audio.size == 0


def test_vits_prior_split_long_phoneme_sequence():
    adapter = _configured_adapter()
    long_ids = np.arange(1, 2001, dtype=np.int64).reshape(1, -1)
    duration = _FakeDurationSession(frames=4000)
    result = adapter.synthesize(_req(phoneme_ids=long_ids), duration)
    assert duration.last_feed["tokens"].shape[1] == 2000
    assert result.audio.size > 0
