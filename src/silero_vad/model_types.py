import enum
from pathlib import Path

SILERO_VAD_URL_BASE = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/"
SILERO_VAD_PATH = Path(__file__).parent / "resources"

class SileroVadType(enum.Enum):
    silero_vad: str = "silero_vad"
    silero_vad_16k_op15: str = "silero_vad_16k_op15"
    silero_vad_half: str = "silero_vad_half"

    def url(self):
        return SILERO_VAD_URL_BASE + self.value + ".onnx"

    def path(self):
        return SILERO_VAD_PATH / (self.value + ".onnx")
