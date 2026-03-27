# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path

from whisper_bundle_runner import (
    WhisperBundleRunner,
    load_audio,
    render_result,
    resolve_language,
    resolve_task,
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
p.add_argument("--no-prev-text", action="store_true")
p.add_argument("--out", type=Path)
args = p.parse_args()

runner = WhisperBundleRunner(args.artifacts_dir, args.tokenizer_json, args.device)
fmt = args.response_format.lower()
result = runner.run(
    audio=load_audio(args.audio, int(runner.meta["sample_rate"])),
    language=resolve_language(
        args.language if args.task != "translate" else None, runner.meta
    ),
    task=resolve_task(args.task, runner.meta),
    timestamps=args.timestamps or fmt in {"verbose_json", "srt", "vtt"},
    max_new_tokens=args.max_new_tokens,
    condition_on_previous_text=not args.no_prev_text,
)
out = render_result(result, fmt)
out = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, indent=2)
if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out, encoding="utf-8")
else:
    print(out)
