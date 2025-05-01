import logging

from silero_vad import SileroVad, SileroVadType
from silero_vad.utils.download import download_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


logger = logging.getLogger(__name__)

logger.info("Starting VAD processing...")

# Ensure the download function also uses logging if possible, or wrap its call
logger.info("Downloading audio file if needed...")
download_file(url="https://github.com/MohammadRaziei/advanced-python-course/raw/master/2-python-libraries/assets/en.wav")
logger.info("Download check complete.")


# Usage with custom settings (chaining setters)
logger.info("Initializing SileroVad...")
vad = (
    SileroVad(SileroVadType.silero_vad, 16000)
    .set_threshold(0.6)
    .set_min_silence_ms(150)
)
logger.info("Processing audio file...")
segments = vad.process_file("en.wav")

logger.info(f"Found {len(segments)} speech segments:")
for seg in segments:
    # Format the time values with rich markup for color
    logger.info(f"Speech from {seg['start']/vad.sample_rate:.2f} to {seg['end']/vad.sample_rate:.2f}.")

logger.info("VAD processing finished.")
