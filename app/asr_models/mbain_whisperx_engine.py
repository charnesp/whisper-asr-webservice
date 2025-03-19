import time
from io import StringIO
from threading import Thread
from typing import BinaryIO, Union

import whisper
import whisperx
from whisperx.utils import ResultWriter, SubtitlesWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG


class WhisperXASR(ASRModel):
    def __init__(self):
        super().__init__()
        self.model = {
            'whisperx': None,
            'diarize_model': None,
            'align_model': {}
        }

    def load_model(self):
        asr_options = {"without_timestamps": False}
        self.model['whisperx'] = whisperx.load_model(
            CONFIG.MODEL_NAME,
            device=CONFIG.DEVICE,
            compute_type=CONFIG.MODEL_QUANTIZATION,
            asr_options=asr_options
        )

        if CONFIG.HF_TOKEN != "":
            self.model['diarize_model'] = whisperx.DiarizationPipeline(
                use_auth_token=CONFIG.HF_TOKEN,
                device=CONFIG.DEVICE
            )

        Thread(target=self.monitor_idleness, daemon=True).start()

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
            if self.model is None:
                self.load_model()

        options_dict = {"task": task}
        if language:
            options_dict["language"] = language
        if initial_prompt:
            options_dict["initial_prompt"] = initial_prompt
        with self.model_lock:
            result = self.model['whisperx'].transcribe(audio, **options_dict)
            language = result["language"]

        # Load the required model and cache it
        # If we transcribe models in many different languages, this may lead to OOM propblems
        if result["language"] in self.model['align_model']:
            model_x, metadata = self.model['align_model'][result["language"]]
        else:
            self.model['align_model'][result["language"]] = whisperx.load_align_model(
                language_code=result["language"], device=CONFIG.DEVICE
            )
            model_x, metadata = self.model['align_model'][result["language"]]

        # Align whisper output
        result = whisperx.align(
            result["segments"], model_x, metadata, audio, CONFIG.DEVICE, return_char_alignments=False
        )

        if options.get("diarize", False) and CONFIG.HF_TOKEN != "":
            min_speakers = options.get("min_speakers", None)
            max_speakers = options.get("max_speakers", None)
            # add min/max number of speakers if known
            diarize_segments, embeddings = self.model['diarize_model'](
                    audio, min_speakers, max_speakers, return_embeddings=True
                )
            result = whisperx.assign_word_speakers(diarize_segments, result)
            embeddings["embeddings"] = embeddings["embeddings"].apply(lambda x: x.tolist())
            result["embeddings"] = embeddings.to_dict('records')
        result["language"] = language

        output_file = StringIO()
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
        default_options = {
            "max_line_width": CONFIG.SUBTITLE_MAX_LINE_WIDTH,
            "max_line_count": CONFIG.SUBTITLE_MAX_LINE_COUNT,
            "highlight_words": CONFIG.SUBTITLE_HIGHLIGHT_WORDS
        }

        if output == "srt":
            WriteSRT(SubtitlesWriter).write_result(result, file=file, options=default_options)
        elif output == "vtt":
            WriteVTT(SubtitlesWriter).write_result(result, file=file, options=default_options)
        elif output == "tsv":
            WriteTSV(ResultWriter).write_result(result, file=file, options=default_options)
        elif output == "json":
            WriteJSON(ResultWriter).write_result(result, file=file, options=default_options)
        else:
            WriteTXT(ResultWriter).write_result(result, file=file, options=default_options)
