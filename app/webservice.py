import importlib.metadata
from io import BytesIO
import os
from os import path
from typing import Annotated, Optional, Union
from urllib.parse import quote
import httpx

import click
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException

from whisper import tokenizer

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio

asr_model = ASRModelFactory.create_asr_model()
asr_model.load_model()

LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())

projectMetadata = importlib.metadata.metadata("whisper-asr-webservice")
app = FastAPI(
    title=projectMetadata["Name"].title().replace("-", " "),
    description=projectMetadata["Summary"],
    version=projectMetadata["Version"],
    contact={"url": projectMetadata["Home-page"]},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={"name": "MIT License", "url": projectMetadata["License"]},
)

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")

    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )

    applications.get_swagger_ui_html = swagger_monkey_patch


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"

@app.post("/audio/transcriptions", tags=["Endpoints"])
async def transcriptions(
    file: UploadFile = File(None),  # noqa: B008
    model: str = Query(default=None, description="Model (not used)")
):
    return await asr(
        audio_file=file,
        audio_url=None,
        encode=True,
        task="transcribe",
        language=None,
        initial_prompt=None,
        vad_filter=True if CONFIG.ASR_ENGINE in ["faster_whisper", "whisperx"] else False,
        word_timestamps=False,
        diarize=True,
        min_speakers=None,
        max_speakers=None,
        output="json"
    )
@app.post("/asr", tags=["Endpoints"])
async def asr(
    audio_file: UploadFile = File(None),  # noqa: B008
    audio_url: str = Query(default=None, description="(optional) URL of the audio file to transcribe"),
    encode: bool = Query(default=True, description="Encode audio first through pydub/ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if CONFIG.ASR_ENGINE in ["faster_whisper", "whisperx"] else False),
        ),
    ] = False,
    word_timestamps: bool = Query(
        default=False,
        description="Word level timestamps",
        include_in_schema=(True if CONFIG.ASR_ENGINE in ["faster_whisper", "whisperx"] else False),
    ),
    diarize: bool = Query(
        default=False,
        description="Diarize the input",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN != "" else False),
    ),
    min_speakers: Union[int, None] = Query(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    max_speakers: Union[int, None] = Query(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
):
    if audio_file is None and audio_url is None:
        raise HTTPException(status_code=400, detail="Either audio_file or audio_url must be provided.")

    if type(audio_url) == str:
        async with httpx.AsyncClient() as client:
            response = await client.get(audio_url)
            audio_content = response.content
        audio_file = BytesIO(audio_content)
        audio_file.content_type = response.headers["content-type"]
        audio_file.seek(0)
        audio_data = audio_file      
    else:
        audio_data = audio_file.file


    #if not audio_file.content_type.startswith(("audio/", "video/")):
    #    raise HTTPException(status_code=400, detail="File must be of audio or video type.")


    result = asr_model.transcribe(
        load_audio(audio_data, audio_file.content_type, encode),
        task,
        language,
        initial_prompt,
        vad_filter,
        word_timestamps,
        {"diarize": diarize, "min_speakers": min_speakers, "max_speakers": max_speakers},
        output,
    )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="output.{output}"',
        },
    )

@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    detected_lang_code, confidence = asr_model.language_detection(load_audio(audio_file.file, encode))
    return {
        "detected_language": tokenizer.LANGUAGES[detected_lang_code],
        "language_code": detected_lang_code,
        "confidence": confidence,
    }


@click.command()
@click.option(
    "-h",
    "--host",
    metavar="HOST",
    default="0.0.0.0",
    help="Host for the webservice (default: 0.0.0.0)",
)
@click.option(
    "-p",
    "--port",
    metavar="PORT",
    default=9000,
    help="Port for the webservice (default: 9000)",
)
@click.version_option(version=projectMetadata["Version"])
def start(host: str, port: Optional[int] = None):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
