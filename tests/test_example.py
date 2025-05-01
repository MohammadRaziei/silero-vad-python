import logging
from pathlib import Path

import pytest

from silero_vad import SileroVad, SileroVadType
from silero_vad.utils.download import download_file


@pytest.fixture
def audio_url():
    """Fixture for the test audio URL"""
    return "https://github.com/MohammadRaziei/advanced-python-course/raw/master/2-python-libraries/assets/en.wav"

@pytest.fixture
def audio_path(tmp_path):
    """Fixture for the test audio file path"""
    return tmp_path / "en.wav"

@pytest.fixture
def vad_instance():
    """Fixture for configured VAD instance"""
    return (
        SileroVad(SileroVadType.silero_vad, 16000)
        .set_threshold(0.6)
        .set_min_silence_ms(150)
    )

@pytest.fixture
def caplog(caplog):
    """Fixture to capture log messages with proper format"""
    caplog.set_level(logging.INFO)
    return caplog


def test_vad_initialization(vad_instance):
    """Test VAD initialization and configuration"""
    assert vad_instance.sample_rate == 16000
    assert vad_instance.threshold == 0.6
    assert vad_instance.min_silence_ms == 150
