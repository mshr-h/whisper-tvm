# Whisper TVM Bundle

A small repository for compiling a Whisper model into a TVM bundle, running it from the command line, and serving it over OpenAI-style REST endpoints.

## Files

- `compile_whisper_bundle.py`: compiles a Hugging Face Whisper checkpoint into a TVM shared library, TVM params, and metadata.
- `whisper_bundle_runner.py`: shared runtime used by both the CLI and the API server.
- `run_whisper_bundle.py`: thin CLI wrapper around `WhisperBundleRunner`.
- `serve_whisper_bundle_api.py`: FastAPI server exposing `/v1/audio/transcriptions` and `/v1/audio/translations`.
- `requirements.txt`: Python packages used by the scripts. Apache TVM and `tokenizers_tvm_ffi` must be installed separately.

## What the bundle contains

`compile_whisper_bundle.py` writes these files into `--output-dir`:

- `whisper_bundle.so`
- `whisper_bundle.params`
- `whisper_bundle_metadata.json`
- `tokenizer.json`

## Requirements

### Compile-time

- uv
- LLVM 17+
- Python 3.10+
- Apache TVM
- [`tokenizers_tvm_ffi`](https://github.com/mshr-h/tokenizers-tvm-ffi)
- `transformers`
- `torch`
- `numpy`

### Runtime

- uv
- Python 3.10+
- Apache TVM
- [`tokenizers_tvm_ffi`](https://github.com/mshr-h/tokenizers-tvm-ffi)
- `numpy`
- `scipy`
- `soundfile`

### API server

- Everything in [Runtime](#runtime)
- `fastapi`
- `uvicorn`
- `python-multipart`

## Install Python packages

```bash
uv venv
uv python -m pip install -r requirements.txt
```

Install Apache TVM separately in the same environment. You can installe it via: `./build-tvm.sh --cuda --clean --llvm llvm-config`.

## Compile a bundle

```bash
python compile_whisper_bundle.py \
  --model-id openai/whisper-tiny \
  --output-dir ./artifacts/whisper-tiny \
  --target cpu
```

Options:

- `--model-id`: Hugging Face model ID.
- `--output-dir`: directory for the compiled bundle.
- `--target`: `cpu` or `cuda`.
- `--max-new-tokens`: runtime default stored in metadata. The compiled decoder still uses `config.max_target_positions`.

## Download sample audio

```bash
curl -o jfk.flac https://raw.githubusercontent.com/openai/whisper/refs/heads/main/tests/jfk.flac
```

## Run from the CLI

```bash
python run_whisper_bundle.py \
  --artifacts-dir ./artifacts/whisper-tiny \
  --audio ./jfk.flac \
  --response-format verbose_json
```

Useful options:

- `--device auto|cpu|cuda`
- `--language auto|en|ja|...`
- `--task transcribe|translate`
- `--timestamps`
- `--response-format json|text|verbose_json|srt|vtt`
- `--max-new-tokens N`
- `--no-prev-text`
- `--output PATH`

Notes:

- `verbose_json`, `srt`, and `vtt` automatically enable timestamps.
- For `--task translate`, the CLI ignores `--language` and lets the runtime detect the source language.
- Audio decoding is done with `soundfile`. Only formats supported by `soundfile` / `libsndfile` are expected to work.

## Start the REST server

```bash
python serve_whisper_bundle_api.py \
  --artifacts-dir ./artifacts/whisper-tiny \
  --served-model whisper-tiny \
  --host 127.0.0.1 \
  --port 8000
```

## REST API

### `POST /v1/audio/transcriptions`

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@jfk.flac \
  -F model=whisper-tiny \
  -F language=ja \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=segment'
```

### `POST /v1/audio/translations`

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/translations \
  -F file=@jfk.flac \
  -F model=whisper-1 \
  -F response_format=text
```

### Supported request fields

The server intentionally supports a small subset of the OpenAI Audio API:

- `file`
- `model`
- `language` on `/v1/audio/transcriptions`
- `response_format`: `json`, `text`, `verbose_json`, `srt`, `vtt`
- `temperature`: only `0`
- `timestamp_granularities[]=segment` only on `/v1/audio/transcriptions` with `response_format=verbose_json`

### Unsupported request fields

These are rejected with `400 invalid_request_error`:

- `prompt`
- `stream=true`
- `temperature != 0`
- word timestamps
- diarization-related fields
- any extra OpenAI Audio API fields not listed above

### Response behavior

- `json`: `{ "text": "..." }`
- `text`: plain text
- `verbose_json`: text + duration + segments
- `srt` / `vtt`: subtitle text generated from segment timestamps
- For `/v1/audio/translations`, `verbose_json.language` is always `"english"`.

## Design notes

- `WhisperBundleRunner` is the single shared runtime.
- The CLI and the API server are intentionally thin wrappers.
- Long-form audio is handled by 30-second windowing plus timestamp-based advancement.
- The current server serializes inference with a lock to keep the TVM runtime usage simple.

## Typical workflow

1. Compile a model with `compile_whisper_bundle.py`.
2. Download or copy `tokenizer.json` into the same artifact directory.
3. Run local inference with `run_whisper_bundle.py`.
4. Start the API server with `serve_whisper_bundle_api.py` if you want OpenAI-style HTTP access.
