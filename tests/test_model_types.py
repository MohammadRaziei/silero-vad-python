from pathlib import Path

import pytest

from silero_vad.model_types import SILERO_VAD_PATH, SILERO_VAD_URL_BASE, SileroVadType


def test_silero_vad_url():
    """Test if URLs are correctly formed for each model type"""
    assert SileroVadType.silero_vad.url() == SILERO_VAD_URL_BASE + "silero_vad.onnx"
    assert SileroVadType.silero_vad_16k_op15.url() == SILERO_VAD_URL_BASE + "silero_vad_16k_op15.onnx"
    assert SileroVadType.silero_vad_half.url() == SILERO_VAD_URL_BASE + "silero_vad_half.onnx"

def test_silero_vad_path():
    """Test if local paths are correctly formed for each model type"""
    for model_type in SileroVadType:
        expected_path = SILERO_VAD_PATH / (model_type.value + ".onnx")
        assert model_type.path() == expected_path

def test_vad_path_exists():
    """Test if the VAD resources directory exists"""
    assert SILERO_VAD_PATH.exists()
    assert SILERO_VAD_PATH.is_dir()

def test_model_type_values():
    """Test if all expected model types are present"""
    model_types = {model.value for model in SileroVadType}
    expected_types = {
        "silero_vad",
        "silero_vad_16k_op15",
        "silero_vad_half"
    }
    assert model_types == expected_types
