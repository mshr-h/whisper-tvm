# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import sys
from pathlib import Path

from whisper_bundle_runner import (
    WhisperBundleRunner,
    load_audio,
    render_result,
    resolve_language,
    resolve_task,
)


def _processing_line(
    audio_path: Path,
    sample_rate: int,
    num_samples: int,
    language: str | None,
    task: str,
    timestamps: bool,
    beam_size: int,
    best_of: int,
) -> str:
    return (
        f"main: processing '{audio_path}' ({num_samples} samples, "
        f"{num_samples / float(sample_rate):.1f} sec), "
        f"beam_size = {beam_size}, best_of = {best_of}, "
        f"lang = {language or 'auto'}, task = {task}, "
        f"timestamps = {1 if timestamps else 0} ..."
    )


p = argparse.ArgumentParser(description="Run a Whisper TVM bundle.")
p.add_argument("--artifacts-dir", type=Path, required=True)
p.add_argument("--audio", type=Path, required=True)
p.add_argument("--tokenizer-json", type=Path)
p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
p.add_argument("--task", choices=["transcribe", "translate"])
p.add_argument("--language", default="auto")
p.add_argument("--timestamps", action="store_true")
p.add_argument(
    "--response-format",
    choices=["json", "text", "verbose_json", "srt", "vtt"],
    default="text",
)
p.add_argument("--max-new-tokens", type=int)
p.add_argument("--temperature", type=float, default=0.0)
p.add_argument("--temperature-inc", type=float, default=0.0)
p.add_argument("--beam-size", type=int, default=1)
p.add_argument("--best-of", type=int, default=1)
p.add_argument("--length-penalty", type=float)
p.add_argument("--compression-ratio-threshold", type=float, default=2.4)
p.add_argument("--logprob-threshold", type=float, default=-1.0)
p.add_argument("--no-speech-threshold", type=float, default=0.6)
p.add_argument("--eos-timestamp-margin", type=float, default=0.0)
p.add_argument("--skip-final-tail-s", type=float, default=0.0)
p.add_argument("--show-model-info", action="store_true")
p.add_argument("--show-system-info", action="store_true")
p.add_argument("--show-perf", "--show-timings", action="store_true")
p.add_argument("--show-decode-debug", action="store_true")
p.add_argument("--no-prev-text", action="store_true")
p.add_argument("--out", type=Path)
args = p.parse_args()

runner = WhisperBundleRunner(args.artifacts_dir, args.tokenizer_json, args.device)
if args.show_model_info:
    for line in runner.whisper_cpp_model_lines():
        print(line, file=sys.stderr)
if args.show_system_info or args.show_perf:
    for line in runner.whisper_cpp_system_info_lines():
        print(line, file=sys.stderr)

fmt = args.response_format.lower()
sample_rate = int(runner.meta["sample_rate"])
audio = load_audio(args.audio, sample_rate)
resolved_task = resolve_task(args.task, runner.meta)
resolved_language = resolve_language(
    args.language if resolved_task != "translate" else None,
    runner.meta,
)
use_timestamps = args.timestamps or fmt in {"verbose_json", "srt", "vtt"}

if args.show_perf:
    print(
        _processing_line(
            args.audio,
            sample_rate,
            int(audio.shape[0]),
            resolved_language,
            resolved_task,
            use_timestamps,
            args.beam_size,
            args.best_of,
        ),
        file=sys.stderr,
    )

result = runner.run(
    audio=audio,
    language=resolved_language,
    task=resolved_task,
    timestamps=use_timestamps,
    max_new_tokens=args.max_new_tokens,
    condition_on_previous_text=not args.no_prev_text,
    temperature=args.temperature,
    temperature_inc=args.temperature_inc,
    beam_size=args.beam_size,
    best_of=args.best_of,
    length_penalty=args.length_penalty,
    compression_ratio_threshold=args.compression_ratio_threshold,
    logprob_threshold=args.logprob_threshold,
    no_speech_threshold=args.no_speech_threshold,
    eos_timestamp_margin_s=args.eos_timestamp_margin,
    skip_final_tail_s=args.skip_final_tail_s,
    collect_perf=args.show_perf or args.show_decode_debug,
)

if args.show_perf:
    for line in runner.whisper_cpp_timing_lines():
        print(line, file=sys.stderr)
if args.show_decode_debug:
    for line in runner.whisper_cpp_decode_debug_lines():
        print(line, file=sys.stderr)

out = render_result(result, fmt)
out = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, indent=2)
if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out, encoding="utf-8")
else:
    print(out)
