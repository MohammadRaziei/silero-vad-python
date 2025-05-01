import numpy as np
import onnxruntime as ort
import soundfile as sf

from silero_vad.model_types import SileroVadType
from silero_vad.utils.download import download_file


class SileroVad:
    def __init__(
        self,
        model_type: SileroVadType,
        sample_rate: int,
    ):
        # --- Core parameters ---
        self.model_type = model_type
        self.sample_rate = sample_rate
        self.model_path = str(model_type.path())
        download_file(self.model_type.url(), output_path=self.model_type.path())


        # --- Fixed internal parameters ---
        self.context_samples = 64 # Specific to this model architecture
        self.size_state = 2 * 1 * 128 # Specific to this model architecture
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.context_samples, dtype=np.float32)

        # --- Initialize configurable parameters via setters with defaults ---
        # Initialize base values first
        self.window_ms: float = 32
        self.threshold: float = 0.5
        self.min_silence_ms = 100
        self.speech_pad_ms = 30
        self.min_speech_ms = 250
        self.max_speech_s = float("inf")

        # Call setters to calculate derived values and ensure consistency
        self.set_window_ms(self.window_ms)
        self.set_threshold(self.threshold)
        self.set_min_silence_ms(self.min_silence_ms)
        self.set_speech_pad_ms(self.speech_pad_ms)
        self.set_min_speech_ms(self.min_speech_ms)
        self.set_max_speech_s(self.max_speech_s) # This depends on window and pad samples

        # --- Initialize ONNX session and states ---
        self._initialize_session()
        self.reset_states()

    def _initialize_session(self):
        """Initializes the ONNX runtime session and state variables."""
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.state_name = self.session.get_inputs()[1].name
        self.sr_name = self.session.get_inputs()[2].name
        self.output_name = self.session.get_outputs()[0].name
        self.state_out_name = self.session.get_outputs()[1].name


    def _update_max_speech_samples(self):
        """Helper to recalculate max_speech_samples based on current settings."""
        if np.isinf(self.max_speech_s):
            self.max_speech_samples = np.iinfo(np.int64).max
        else:
            # Ensure dependent values exist (they should due to init order)
            pad_term = 2 * self.speech_pad_samples
            self.max_speech_samples = int(
                self.sample_rate * self.max_speech_s
                - self.window_size_samples
                - pad_term
            )
            # Ensure it's not negative if parameters are set aggressively
            self.max_speech_samples = max(0, self.max_speech_samples)


    # --- Configuration Setters ---
    def set_window_ms(self, window_ms: int):
        """Sets the window size in milliseconds and updates related sample counts."""
        self.window_ms = window_ms
        self.window_size_samples = window_ms * self.sample_rate // 1000
        self.effective_window_size = self.window_size_samples + self.context_samples
        self._update_max_speech_samples() # max_speech_samples depends on window_size_samples
        return self # Allow chaining

    def set_threshold(self, threshold: float):
        """Sets the speech probability threshold."""
        self.threshold = threshold
        return self

    def set_min_silence_ms(self, min_silence_ms: int):
        """Sets the minimum silence duration in milliseconds."""
        self.min_silence_ms = min_silence_ms
        self.min_silence_samples = min_silence_ms * self.sample_rate // 1000
        # This seems fixed relative to sample rate in the original code
        self.min_silence_samples_at_max_speech = 98 * self.sample_rate // 1000
        return self

    def set_speech_pad_ms(self, speech_pad_ms: int):
        """Sets the speech padding in milliseconds."""
        self.speech_pad_ms = speech_pad_ms
        self.speech_pad_samples = speech_pad_ms * self.sample_rate // 1000
        self._update_max_speech_samples() # max_speech_samples depends on speech_pad_samples
        return self

    def set_min_speech_ms(self, min_speech_ms: int):
        """Sets the minimum speech duration in milliseconds."""
        self.min_speech_ms = min_speech_ms
        self.min_speech_samples = min_speech_ms * self.sample_rate // 1000
        return self

    def set_max_speech_s(self, max_speech_s: float):
        """Sets the maximum speech duration in seconds."""
        self.max_speech_s = max_speech_s
        self._update_max_speech_samples() # Update the derived sample count
        return self
    # --- End Configuration Setters ---

    def reset_states(self):
        """Resets the VAD internal states."""
        self._state.fill(0)
        self._context.fill(0)

    def predict(self, data_chunk):
        # Ensure chunk is the correct size (window_size_samples)
        if len(data_chunk) != self.window_size_samples:
             # This case should ideally not happen if process() feeds correctly
             # Handle potential padding or error if necessary, or assume correct input
             # For now, assume input is correct as per original logic
             pass

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
        """Processes the entire audio waveform to detect speech segments."""
        self.reset_states()
        speeches = []
        triggered = False
        temp_end = 0        # Potential end timestamp if silence continues
        current_sample = 0  # Tracks current position in audio samples
        prev_end = 0        # End of the previous segment, used for max_speech logic
        next_start = 0      # Potential start of the next segment if max_speech triggered
        current_speech = {"start": -1, "end": -1}
        audio_length_samples = len(wav)

        # Use effective_window_size for iteration step? No, original uses window_size_samples
        # The predict function handles the context internally.
        window_step = self.window_size_samples

        for j in range(0, audio_length_samples - window_step + 1, window_step):
            chunk = wav[j : j + window_step]
            speech_prob = self.predict(chunk)
            current_sample = j + window_step # Point to the end of the current window

            # --- Speech detected ---
            if speech_prob >= self.threshold:
                if temp_end != 0: # If we were in temporary silence, reset it
                    temp_end = 0
                    # Check if the silence was long enough to be a potential start for next segment
                    # This logic seems complex and potentially related to max_speech handling
                    if next_start < prev_end: # Original logic, purpose unclear without deeper analysis
                         next_start = current_sample - window_step

                if not triggered: # Start of a new speech segment
                    triggered = True
                    # Apply start padding (ensure not negative)
                    current_speech["start"] = max(0, current_sample - window_step - self.speech_pad_samples)
                # Continue processing chunks while speech is detected
                continue

            # --- Silence or low probability detected ---
            if triggered: # Only process silence if we are currently in a speech segment
                # Handle potential segment end due to max speech duration
                # Check duration from the *padded* start
                current_duration = current_sample - current_speech["start"]
                if current_duration > self.max_speech_samples:
                    # Segment exceeds max length, end it.
                    # Use prev_end if available (related to specific silence condition)
                    # Apply end padding (ensure not beyond audio length)
                    end_sample = prev_end if prev_end > 0 else current_sample # Original logic unclear here
                    current_speech["end"] = min(audio_length_samples, end_sample + self.speech_pad_samples)

                    # Check minimum duration before adding
                    if current_speech["end"] - current_speech["start"] >= self.min_speech_samples:
                        speeches.append(current_speech.copy())

                    # Reset for next potential segment
                    current_speech = {"start": -1, "end": -1}
                    if next_start < prev_end: # Original logic
                        triggered = False
                    else:
                        # Start next segment immediately? Apply padding.
                        current_speech["start"] = max(0, next_start - self.speech_pad_samples)
                        triggered = True # Remain triggered for the new segment

                    # Reset max_speech related state
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                    continue # Continue to next chunk

                # Handle potential segment end due to sufficient silence
                # Check for silence confirmation threshold (slightly lower)
                if speech_prob < (self.threshold - 0.15):
                    if temp_end == 0: # Start counting silence
                        temp_end = current_sample

                    # Check if silence duration meets the minimum required
                    silence_duration = current_sample - temp_end
                    if silence_duration >= self.min_silence_samples:
                        # Silence is long enough, end the speech segment
                        # Apply end padding
                        current_speech["end"] = min(audio_length_samples, temp_end + self.speech_pad_samples)

                        # Check minimum duration before adding
                        if current_speech["end"] - current_speech["start"] >= self.min_speech_samples:
                            speeches.append(current_speech.copy())

                        # Reset state for next segment
                        current_speech = {"start": -1, "end": -1}
                        triggered = False
                        temp_end = 0
                        prev_end = 0 # Reset max_speech state as well
                        next_start = 0
                        continue # Continue to next chunk

                    # Check for the specific silence condition related to max_speech
                    # This condition seems to store a potential earlier end point
                    if silence_duration > self.min_silence_samples_at_max_speech:
                         prev_end = temp_end

                # If speech_prob is between (threshold - 0.15) and threshold, just continue (ambiguous region)
                # No action needed here, loop continues

            # If not triggered and silence/low prob, do nothing, just continue scanning

        # --- Handle the last segment ---
        if current_speech["start"] != -1: # If speech was active at the end
            # End the segment at the end of the audio
            current_speech["end"] = audio_length_samples
            # Check minimum duration
            if current_speech["end"] - current_speech["start"] >= self.min_speech_samples:
                speeches.append(current_speech.copy())

        return speeches


    def process_file(self, wav_path):
        """Reads an audio file and processes it."""
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1: # Convert to mono if necessary
            wav = wav[:, 0]
        if sr != self.sample_rate:
            # Consider resampling here instead of raising an error for more robustness
            # import librosa
            # wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
            # logger.warning(f"Resampling audio from {sr} Hz to {self.sample_rate} Hz")
            raise ValueError(f"Input audio sample rate ({sr}) must match VAD sample rate")
        return self.process(wav)
