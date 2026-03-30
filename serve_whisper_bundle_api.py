# SPDX-License-Identifier: Apache-2.0
import argparse
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from whisper_bundle_runner import (
    WhisperBundleRunner,
    load_audio,
    render_result,
    resolve_language,
)


def make_app(args):
    runner = WhisperBundleRunner(args.artifacts_dir, args.tokenizer_json, args.device)
    lock = threading.Lock()
    app = FastAPI()

    def err(msg: str, status: int = 400, typ: str = "invalid_request_error"):
        return JSONResponse(
            status_code=status, content={"error": {"message": msg, "type": typ}}
        )

    async def handle(request: Request, task: str):
        try:
            form = await request.form()
        except Exception as e:
            return err(f"Failed to parse multipart/form-data: {e}")
        allowed = {
            "file",
            "model",
            "response_format",
            "temperature",
            "temperature_inc",
            "beam_size",
            "best_of",
            "length_penalty",
            "compression_ratio_threshold",
            "logprob_threshold",
            "no_speech_threshold",
            "stream",
        }
        if task == "transcribe":
            allowed |= {
                "language",
                "timestamp_granularities",
                "timestamp_granularities[]",
            }
        extra = sorted(set(form.keys()) - allowed)
        if extra:
            return err(f"Unsupported parameter(s): {', '.join(extra)}")
        file = form.get("file")
        if file is None or not hasattr(file, "read"):
            return err("Missing required form field: file")
        model = str(form.get("model") or "")
        if not model:
            return err("Missing required form field: model")
        if model != args.served_model:
            return err(
                f"This server only serves model={args.served_model!r}, got {model!r}."
            )
        fmt = str(form.get("response_format") or "json").lower()
        if fmt not in {"json", "text", "verbose_json", "srt", "vtt"}:
            return err(f"Unsupported response_format: {fmt}")

        def parse_optional_float(name: str, default=None):
            value = form.get(name)
            if value in {None, ""}:
                return default
            try:
                return float(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be a number.") from exc

        def parse_optional_int(name: str, default=None):
            value = form.get(name)
            if value in {None, ""}:
                return default
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer.") from exc

        try:
            temperature = parse_optional_float("temperature", 0.0)
            temperature_inc = parse_optional_float("temperature_inc", 0.0)
            beam_size = parse_optional_int("beam_size", 1)
            best_of = parse_optional_int("best_of", 1)
            length_penalty = parse_optional_float("length_penalty", None)
            compression_ratio_threshold = parse_optional_float(
                "compression_ratio_threshold", 2.4
            )
            logprob_threshold = parse_optional_float("logprob_threshold", -1.0)
            no_speech_threshold = parse_optional_float("no_speech_threshold", 0.6)
        except ValueError as e:
            return err(str(e))
        if str(form.get("stream") or "false").lower() in {"1", "true", "yes", "on"}:
            return err("stream is not supported.")
        gran = form.getlist("timestamp_granularities[]") + form.getlist(
            "timestamp_granularities"
        )
        if gran and (task != "transcribe" or fmt != "verbose_json"):
            return err(
                "timestamp_granularities is only supported on /v1/audio/transcriptions with response_format=verbose_json."
            )
        if any(str(x).lower() != "segment" for x in gran):
            return err("Only timestamp_granularities[]=segment is supported.")
        try:
            audio = load_audio(await file.read(), int(runner.meta["sample_rate"]))
            language = (
                None
                if task == "translate"
                else resolve_language(form.get("language"), runner.meta)
            )
        except ValueError as e:
            return err(str(e))
        except Exception as e:
            return err(f"Failed to decode audio with soundfile: {e}")
        try:
            with lock:
                result = runner.run(
                    audio=audio,
                    language=language,
                    task=task,
                    timestamps=fmt in {"verbose_json", "srt", "vtt"} or bool(gran),
                    max_new_tokens=None,
                    condition_on_previous_text=True,
                    temperature=temperature,
                    temperature_inc=temperature_inc,
                    beam_size=beam_size,
                    best_of=best_of,
                    length_penalty=length_penalty,
                    compression_ratio_threshold=compression_ratio_threshold,
                    logprob_threshold=logprob_threshold,
                    no_speech_threshold=no_speech_threshold,
                )
        except ValueError as e:
            return err(str(e))
        except Exception as e:
            return err(str(e), 500, "internal_server_error")
        body = render_result(result, fmt)
        return (
            PlainTextResponse(body, media_type="text/plain; charset=utf-8")
            if isinstance(body, str)
            else JSONResponse(body)
        )

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(request: Request):
        return await handle(request, "transcribe")

    @app.post("/v1/audio/translations")
    async def translations(request: Request):
        return await handle(request, "translate")

    return app


p = argparse.ArgumentParser(
    description="Serve a Whisper TVM bundle over an OpenAI-style audio API."
)
p.add_argument("--artifacts-dir", type=Path, required=True)
p.add_argument("--tokenizer-json", type=Path)
p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
p.add_argument("--served-model", default="whisper-1")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=8000)
args = p.parse_args()

if __name__ == "__main__":
    uvicorn.run(make_app(args), host=args.host, port=args.port)
