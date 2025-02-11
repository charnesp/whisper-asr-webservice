from typing import BinaryIO, Union
from io import StringIO
from threading import Thread
import whisperx
import whisper
import time
from whisperx.utils import SubtitlesWriter, ResultWriter

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG
from app.utils import WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON


class WhisperXASR(ASRModel):

    def load_model(self):
        asr_options = {"without_timestamps": False, "multilingual": True, "word_timestamps": False}
        self.model = whisperx.load_model(
            CONFIG.MODEL_NAME, device=CONFIG.DEVICE, compute_type=CONFIG.MODEL_QUANTIZATION, asr_options=asr_options
        )

        Thread(target=self.monitor_idleness, daemon=True).start()

    def load_diarize_model(self):
        if CONFIG.HF_TOKEN != "":
            self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=CONFIG.HF_TOKEN, device=CONFIG.DEVICE)
        else:
            raise Exception("HF_TOKEN is not set. Diarization will not work")

    def transcribe(
        self,
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        options: Union[dict, None],
        output,
    ):
        self.last_activity_time = time.time()

        with self.model_lock:
            if self.diarize_model is None:
                self.load_diarize_model()

        options_dict = {"task": task}
        if language:
            options_dict["language"] = language

        ### Not handled by whisperx yet
        # if initial_prompt:
        #    options_dict["initial_prompt"] = initial_prompt
        # if word_timestamps:
        #    options_dict["word_timestamps"] = word_timestamps

        with self.model_lock:
            if self.model is None:
                self.load_model()
            result = self.model.transcribe(audio, **options_dict)

        # Load the required model and cache it
        # If we transcribe models in many different languages, this may lead to OOM propblems
        if result["language"] in self.x_models:
            model_x, metadata = self.x_models[result["language"]]
        else:
            with self.model_lock:
                self.x_models[result["language"]] = whisperx.load_align_model(
                    language_code=result["language"], device=CONFIG.DEVICE
                )
                model_x, metadata = self.x_models[result["language"]]

        # Align whisper output
        with self.model_lock:
            result = whisperx.align(
                result["segments"], model_x, metadata, audio, CONFIG.DEVICE, return_char_alignments=False
            )

        with self.model_lock:
            if options.get("diarize", False):
                if CONFIG.HF_TOKEN == "":
                    print("Warning! HF_TOKEN is not set. Diarization may not work as expected.")
                min_speakers = options.get("min_speakers", None)
                max_speakers = options.get("max_speakers", None)
                diarize_segments, embeddings = self.diarize_model(
                    audio, min_speakers, max_speakers, return_embeddings=True
                )

                embeddings["embeddings"] = embeddings["embeddings"].apply(lambda x: x.tolist())
                result["embeddings"] = embeddings.to_dict('records')

                result = whisperx.assign_word_speakers(diarize_segments, result)

        output_file = StringIO()
        # If word_timestamps is 0, then we don't write the timestamps:
        if not options_dict.get("word_timestamps", False):
            for segment in result["segments"]:
                segment.pop("words", None)
            result.pop("word_segments", None)
        self.write_result(result, output_file, output)
        output_file.seek(0)

        return output_file

    def language_detection(self, audio):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # detect the spoken language
        with self.model_lock:
            if self.model is None:
                self.load_model()
            _, probs = self.model.detect_language(mel)
        detected_lang_code = max(probs, key=probs.get)

        return detected_lang_code

    def write_result(self, result: dict, file: BinaryIO, output: Union[str, None]):
        if output == "srt":
            if CONFIG.HF_TOKEN != "":
                WriteSRT(SubtitlesWriter).write_result(result, file=file, options={})
            else:
                WriteSRT(ResultWriter).write_result(result, file=file, options={})
        elif output == "vtt":
            if CONFIG.HF_TOKEN != "":
                WriteVTT(SubtitlesWriter).write_result(result, file=file, options={})
            else:
                WriteVTT(ResultWriter).write_result(result, file=file, options={})
        elif output == "tsv":
            WriteTSV(ResultWriter).write_result(result, file=file, options={})
        elif "json" in str(output):
            WriteJSON(ResultWriter).write_result(result, file=file, options={"output": output})
        elif output == "txt":
            WriteTXT(ResultWriter).write_result(result, file=file, options={})
        else:
            return 'Please select an output method!'
