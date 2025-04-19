import numpy as np
import onnxruntime as ort
import soundfile as sf
from pathlib import Path
from urllib.parse import urlparse
import enum

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

class SileroVad:
    def __init__(
        self,
        model_type: SileroVadType = SileroVadType.silero_vad,
        sample_rate: int = 16000,
        window_ms: int = 32,
        threshold: float = 0.5,
        min_silence_ms: int = 100,
        speech_pad_ms: int = 30,
        min_speech_ms: int = 250,
        max_speech_s: float = float("inf"),
    ):
        self.model_path = str(model_type.path())
        self.sample_rate = sample_rate
        self.window_size_samples = window_ms * sample_rate // 1000
        self.context_samples = 64
        self.effective_window_size = self.window_size_samples + self.context_samples
        self.threshold = threshold
        self.min_silence_samples = min_silence_ms * sample_rate // 1000
        self.min_silence_samples_at_max_speech = 98 * sample_rate // 1000
        self.min_speech_samples = min_speech_ms * sample_rate // 1000
        # --- fix for infinity ---
        if np.isinf(max_speech_s):
            self.max_speech_samples = np.iinfo(np.int64).max
        else:
            self.max_speech_samples = int(sample_rate * max_speech_s - self.window_size_samples - 2 * (speech_pad_ms * sample_rate // 1000))
        # --- end fix ---
        self.speech_pad_samples = speech_pad_ms * sample_rate // 1000

        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.state_name = self.session.get_inputs()[1].name
        self.sr_name = self.session.get_inputs()[2].name
        self.output_name = self.session.get_outputs()[0].name
        self.state_out_name = self.session.get_outputs()[1].name

        self.size_state = 2 * 1 * 128
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.context_samples, dtype=np.float32)

    def reset_states(self):
        self._state.fill(0)
        self._context.fill(0)

    def predict(self, data_chunk):
        new_data = np.concatenate([self._context, data_chunk])
        input_tensor = new_data.astype(np.float32)[None, :]
        state_tensor = self._state.astype(np.float32)
        sr_tensor = np.array([self.sample_rate], dtype=np.int64)

        ort_inputs = {
            self.input_name: input_tensor,
            self.state_name: state_tensor,
            self.sr_name: sr_tensor,
        }
        ort_outs = self.session.run([self.output_name, self.state_out_name], ort_inputs)
        speech_prob = ort_outs[0].item()
        self._state = ort_outs[1]
        self._context = new_data[-self.context_samples:]
        return speech_prob

    def process(self, wav: np.ndarray):
        self.reset_states()
        speeches = []
        triggered = False
        temp_end = 0
        current_sample = 0
        prev_end = 0
        next_start = 0
        current_speech = {"start": -1, "end": -1}
        audio_length_samples = len(wav)

        for j in range(0, audio_length_samples - self.window_size_samples + 1, self.window_size_samples):
            chunk = wav[j:j + self.window_size_samples]
            speech_prob = self.predict(chunk)
            current_sample = j + self.window_size_samples

            if speech_prob >= self.threshold:
                if temp_end != 0:
                    temp_end = 0
                    if next_start < prev_end:
                        next_start = current_sample - self.window_size_samples
                if not triggered:
                    triggered = True
                    current_speech["start"] = current_sample - self.window_size_samples
                continue

            if triggered and ((current_sample - current_speech["start"]) > self.max_speech_samples):
                if prev_end > 0:
                    current_speech["end"] = prev_end
                    speeches.append(current_speech.copy())
                    current_speech = {"start": -1, "end": -1}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech["start"] = next_start
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                else:
                    current_speech["end"] = current_sample
                    speeches.append(current_speech.copy())
                    current_speech = {"start": -1, "end": -1}
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                    triggered = False
                continue

            if (speech_prob >= (self.threshold - 0.15)) and (speech_prob < self.threshold):
                continue

            if speech_prob < (self.threshold - 0.15):
                if triggered:
                    if temp_end == 0:
                        temp_end = current_sample
                    if (current_sample - temp_end) > self.min_silence_samples_at_max_speech:
                        prev_end = temp_end
                    if (current_sample - temp_end) >= self.min_silence_samples:
                        current_speech["end"] = temp_end
                        if current_speech["end"] - current_speech["start"] > self.min_speech_samples:
                            speeches.append(current_speech.copy())
                            current_speech = {"start": -1, "end": -1}
                            prev_end = 0
                            next_start = 0
                            temp_end = 0
                            triggered = False
        # Handle last segment
        if current_speech["start"] >= 0:
            current_speech["end"] = audio_length_samples
            speeches.append(current_speech.copy())
        return speeches

    def process_file(self, wav_path):
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav[:, 0]
        if sr != self.sample_rate:
            raise ValueError(f"Sample rate must be {self.sample_rate}, got {sr}")
        return self.process(wav)