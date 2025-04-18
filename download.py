import urllib.request
from pathlib import Path
from urllib.parse import urlparse
import enum
import logging

logger = logging.getLogger(__name__)

def download_file(url, output_path="", force=False):
    output_path = Path(output_path)
    # If output_path is a directory, extract filename from URL
    if output_path.is_dir() or str(output_path).endswith(("\\", "/")):
        filename = Path(urlparse(url).path).name
        output_path = output_path / filename

    if output_path.exists() and not force:
        logger.info(f"File {output_path} already exists. Skipping download.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(output_path, "wb") as out_file:
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            out_file.write(chunk)
    logger.info(f"Downloaded {url} to {output_path}")


SILERO_VAD_URL_BASE = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/"
SILERO_VAD_PATH = Path(__file__).parent / "resources"

class SileroVadType(enum.StrEnum):
    silero_vad: str = "silero_vad.onnx"
    silero_vad_16k_op15: str = "silero_vad_16k_op15.onnx"
    silero_vad_half: str = "silero_vad_half.onnx"

    def url(self):
        return SILERO_VAD_URL_BASE + self.value

    def path(self):
        return SILERO_VAD_PATH / self.value

def silero_download(type: SileroVadType = SileroVadType.silero_vad, output_path: Path = SILERO_VAD_PATH, force: bool = False):
    download_file(url=type.url(), output_path=output_path, force=force) 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    silero_download()
