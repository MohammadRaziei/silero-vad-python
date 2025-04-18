from silero_vad import SileroVad, SileroVadType
from download import download_file

download_file(url="https://github.com/MohammadRaziei/advanced-python-course/raw/master/2-python-libraries/assets/en.wav")

vad = SileroVad(model_type=SileroVadType.silero_vad)
segments = vad.process_file("en.wav")
for seg in segments:
    print(f"Speech from {seg['start']/vad.sample_rate:.2f}s to {seg['end']/vad.sample_rate:.2f}s")