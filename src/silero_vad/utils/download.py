import logging
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

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



