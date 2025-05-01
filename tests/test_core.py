import numpy as np
import pytest

from silero_vad import SileroVad
from silero_vad.model_types import SileroVadType


@pytest.fixture
def vad():
    """Create a basic VAD instance for testing"""
    return SileroVad(
        model_type=SileroVadType.silero_vad,
        sample_rate=16000
    )

def test_vad_initialization(vad):
    """Test if VAD is initialized with correct default parameters"""
    assert vad.sample_rate == 16000
    assert vad.window_ms == 32
    assert vad.threshold == 0.5
    assert vad.min_silence_ms == 100
    assert vad.speech_pad_ms == 30
    assert vad.min_speech_ms == 250
    assert np.isinf(vad.max_speech_s)

def test_vad_setters(vad):
    """Test parameter setters"""
    vad.set_window_ms(64)
    assert vad.window_ms == 64
    assert vad.window_size_samples == 1024  # 64ms * 16000Hz / 1000

    vad.set_threshold(0.7)
    assert vad.threshold == 0.7

    vad.set_min_silence_ms(200)
    assert vad.min_silence_ms == 200
    assert vad.min_silence_samples == 3200  # 200ms * 16000Hz / 1000

def test_state_management(vad):
    """Test state initialization and reset"""
    assert vad._state.shape == (2, 1, 128)
    assert vad._context.shape == (64,)

    # Test reset states
    vad._state.fill(1.0)
    vad._context.fill(1.0)
    vad.reset_states()
    assert np.all(vad._state == 0)
    assert np.all(vad._context == 0)

def test_process_empty_audio(vad):
    """Test processing empty audio"""
    empty_audio = np.array([], dtype=np.float32)
    result = vad.process(empty_audio)
    assert len(result) == 0

@pytest.mark.parametrize("duration_sec", [0.5, 1.0, 2.0])
def test_process_silent_audio(vad, duration_sec):
    """Test processing silent audio of different durations"""
    samples = int(duration_sec * vad.sample_rate)
    silent_audio = np.zeros(samples, dtype=np.float32)
    result = vad.process(silent_audio)
    assert len(result) == 0

def test_process_file_mono(vad, tmp_path):
    """Test processing a mono audio file"""
    import soundfile as sf

    # Create a temporary mono audio file
    audio_path = tmp_path / "test_mono.wav"
    samples = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    sf.write(audio_path, samples, 16000)

    result = vad.process_file(audio_path)
    assert isinstance(result, list)

def test_process_file_stereo(vad, tmp_path):
    """Test processing a stereo audio file"""
    import soundfile as sf

    # Create a temporary stereo audio file
    audio_path = tmp_path / "test_stereo.wav"
    samples = np.zeros((16000, 2), dtype=np.float32)  # 1 second of stereo silence
    sf.write(audio_path, samples, 16000)

    result = vad.process_file(audio_path)
    assert isinstance(result, list)

def test_invalid_sample_rate(tmp_path):
    """Test handling of audio with mismatched sample rate"""
    import soundfile as sf

    # Create audio file with different sample rate
    audio_path = tmp_path / "test_44100.wav"
    samples = np.zeros(44100, dtype=np.float32)
    sf.write(audio_path, samples, 44100)

    vad = SileroVad(
        model_type=SileroVadType.silero_vad,
        sample_rate=16000
    )

    with pytest.raises(ValueError):
        vad.process_file(audio_path)
