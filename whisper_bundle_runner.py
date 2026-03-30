import io
import json
import math
import os
import platform
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import tokenizers_tvm_ffi
from scipy.signal import resample_poly

import tvm
from tvm import relax


@dataclass
class DecodeConfig:
    temperature: float = 0.0
    temperature_inc: float = 0.0
    beam_size: int = 1
    best_of: int = 1
    length_penalty: float | None = None
    compression_ratio_threshold: float | None = 2.4
    logprob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6


@dataclass
class DecodeHypothesis:
    tokens: list[int] = field(default_factory=list)
    sum_logprob: float = 0.0
    next_logits: np.ndarray | None = None
    position: int = 0


@dataclass
class DecodeAttempt:
    tokens: list[int]
    sum_logprob: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    temperature: float
    strategy: str
    stop_reason: str = "unknown"
    generated_len: int = 0
    returned_len: int = 0
    last_token_id: int | None = None
    last_is_timestamp: bool = False
    ended_with_eos: bool = False
    fallback_compression_ratio: bool = False
    fallback_logprob: bool = False
    silence: bool = False
    finished_hypotheses: int = 0
    active_hypotheses: int = 0


@dataclass
class DecodeWindowTrace:
    index: int
    seek_samples: int
    valid_samples: int
    prompt_len: int
    max_new_tokens: int
    generated_len: int
    returned_len: int
    stop_reason: str
    strategy: str
    temperature: float
    last_token_id: int | None
    last_is_timestamp: bool
    ended_with_eos: bool
    fallback_compression_ratio: bool
    fallback_logprob: bool
    silence: bool
    finished_hypotheses: int
    active_hypotheses: int
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    timestamps: bool
    segment_count: int
    advance_samples: int
    text_chars: int


@dataclass
class PerfMetric:
    total_ms: float = 0.0
    runs: int = 0

    def add(self, elapsed_ms: float, runs: int = 1):
        self.total_ms += float(elapsed_ms)
        self.runs += int(runs)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.runs if self.runs > 0 else 0.0


@dataclass
class BundlePerfStats:
    load_ms: float = 0.0
    total_ms: float = 0.0
    mel: PerfMetric = field(default_factory=PerfMetric)
    encode: PerfMetric = field(default_factory=PerfMetric)
    cross_kv: PerfMetric = field(default_factory=PerfMetric)
    prompt: PerfMetric = field(default_factory=PerfMetric)
    lang_detect: PerfMetric = field(default_factory=PerfMetric)
    sample: PerfMetric = field(default_factory=PerfMetric)
    decode: PerfMetric = field(default_factory=PerfMetric)
    batchd: PerfMetric = field(default_factory=PerfMetric)
    fallbacks_compression_ratio: int = 0
    fallbacks_logprob: int = 0
    silence_skips: int = 0
    stop_reason_counts: dict[str, int] = field(default_factory=dict)
    windows: list[DecodeWindowTrace] = field(default_factory=list)


def choose_device(name: str = "auto"):
    if name == "cuda":
        dev = tvm.cuda(0)
        if not dev.exist:
            raise RuntimeError("CUDA was requested, but tvm.cuda(0).exist is False.")
        return dev
    return tvm.cpu(0) if name == "cpu" or not tvm.cuda(0).exist else tvm.cuda(0)


def to_tvm(x, dev):
    if isinstance(x, np.ndarray):
        return tvm.runtime.tensor(x, dev)
    if hasattr(x, "copyto") and hasattr(x, "numpy"):
        return x.copyto(target=dev)
    raise TypeError(f"Unsupported tensor input type: {type(x)}")


def unwrap(x):
    while not hasattr(x, "numpy"):
        x = x[0]
    return x


def unwrap_many(x):
    return x if hasattr(x, "numpy") else [unwrap_many(x[i]) for i in range(len(x))]


def load_audio(src, sr: int) -> np.ndarray:
    audio, src_sr = sf.read(
        io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src,
        dtype="float32",
        always_2d=True,
    )
    audio = audio.mean(axis=1)
    if src_sr != sr:
        g = math.gcd(src_sr, sr)
        audio = resample_poly(audio, sr // g, src_sr // g)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def resolve_task(task: str | None, meta: dict[str, Any]) -> str:
    task = str(task or meta.get("default_task", "transcribe")).lower()
    if task not in {"transcribe", "translate"}:
        raise ValueError(f"Unsupported task: {task}")
    if task == "translate" and not bool(meta.get("supports_translate", True)):
        raise ValueError("This bundle does not support translate.")
    return task


def resolve_language(language: str | None, meta: dict[str, Any]) -> str | None:
    if not bool(meta.get("is_multilingual", False)):
        return str(meta.get("default_language_code", "en"))
    language = str(language or "auto").strip().lower()
    if language in {"", "auto"}:
        return None
    aliases = {
        str(k).lower(): str(v)
        for k, v in meta.get("language_alias_to_code", {}).items()
    }
    if language in aliases:
        return aliases[language]
    if language in meta.get("language_token_ids", {}):
        return language
    raise ValueError(f"Unsupported language: {language}")


def _stamp(seconds: float, sep: str) -> str:
    ms = int(round(max(0.0, float(seconds)) * 1000.0))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def render_result(result: dict[str, Any], fmt: str):
    fmt = str(fmt or "json").lower()
    if fmt == "text":
        return result["text"]
    if fmt == "json":
        return {"text": result["text"]}
    if fmt == "verbose_json":
        return {
            "task": result["task"],
            "language": "english"
            if result["task"] == "translate"
            else str(result.get("language_name") or result["language"]).lower(),
            "duration": float(result["duration"]),
            "text": result["text"],
            "segments": [
                {
                    "id": int(s["id"]),
                    "start": float(s["start"]),
                    "end": float(s["end"]),
                    "text": s["text"],
                    "tokens": [int(x) for x in s["tokens"]],
                }
                for s in result["segments"]
            ],
        }
    if fmt in {"srt", "vtt"}:
        lines = ["WEBVTT", ""] if fmt == "vtt" else []
        for i, s in enumerate(result["segments"], 1):
            if fmt == "srt":
                lines.append(str(i))
            lines.append(
                f"{_stamp(s['start'], ',' if fmt == 'srt' else '.')} --> {_stamp(s['end'], ',' if fmt == 'srt' else '.')}"
            )
            lines.append(str(s["text"]).strip())
            lines.append("")
        return "\n".join(lines).rstrip() + ("\n" if lines else "")
    raise ValueError(f"Unsupported response_format: {fmt}")


def _logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("-inf")
    m = float(np.max(x))
    return float("-inf") if np.isneginf(m) else float(m + np.log(np.exp(x - m).sum()))


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    ln = _logsumexp(x)
    if not np.isfinite(ln):
        return np.full_like(x, -np.inf, dtype=np.float64)
    return x - ln


def _format_bytes(size: int | None) -> str:
    if size is None:
        return "unknown"
    n = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{int(size)} B"


def _default_thread_count() -> int:
    for name in ("WHISPER_N_THREADS", "OMP_NUM_THREADS", "TVM_NUM_THREADS"):
        value = os.getenv(name)
        if value and value.isdigit() and int(value) > 0:
            return int(value)
    return max(1, int(os.cpu_count() or 1))


class WhisperBundleRunner:
    def __init__(
        self,
        artifacts_dir: str | Path,
        tokenizer_json: str | Path | None = None,
        device: str = "auto",
    ):
        init_started = time.perf_counter()
        artifacts_dir = Path(artifacts_dir)
        tokenizer_json = Path(tokenizer_json or artifacts_dir / "tokenizer.json")
        self.meta = json.loads(
            (artifacts_dir / "whisper_bundle_metadata.json").read_text(encoding="utf-8")
        )
        self.requested_device = str(device)
        self.device_kind = (
            "cpu" if self.requested_device == "cpu" or not tvm.cuda(0).exist else "cuda"
        )
        self.device_id = 0
        self.dev = choose_device(device)
        self.vm = relax.VirtualMachine(
            tvm.runtime.load_module(str(artifacts_dir / self.meta["lib_name"])),
            self.dev,
        )
        by_name = {
            str(k): v
            for k, v in tvm.runtime.load_param_dict_from_file(
                str(artifacts_dir / self.meta["params_name"])
            ).items()
        }
        names = [str(x) for x in self.meta["param_names"]]
        self.params = [to_tvm(by_name[name], self.dev) for name in names]
        self.tok = tokenizers_tvm_ffi.Tokenizer.from_json_bytes(
            tokenizer_json.read_bytes()
        )
        self.rng = np.random.default_rng()

        self.special_ids = {int(x) for x in self.meta["special_ids"]}
        self.timestamp_begin = int(self.meta["timestamp_begin"])
        self.eos = int(self.meta["eos_token_id"])
        self.vocab_size = int(self.meta["vocab_size"])
        self.no_speech_token_id = (
            int(self.meta["no_speech_token_id"])
            if self.meta.get("no_speech_token_id") is not None
            and 0 <= int(self.meta["no_speech_token_id"]) < self.vocab_size
            else None
        )
        self.suppress = np.asarray(
            [
                int(x)
                for x in self.meta.get("suppress_tokens", [])
                if 0 <= int(x) < self.vocab_size
            ],
            dtype=np.int64,
        )
        self.begin_suppress = np.asarray(
            [
                int(x)
                for x in self.meta.get("begin_suppress_tokens", [])
                if 0 <= int(x) < self.vocab_size
            ],
            dtype=np.int64,
        )
        self.blank_suppress = np.asarray(
            [
                int(x)
                for x in self.meta.get("blank_suppress_tokens", [])
                if 0 <= int(x) < self.vocab_size
            ],
            dtype=np.int64,
        )
        self.no_timestamps = (
            int(self.meta["no_timestamps_token_id"])
            if self.meta.get("no_timestamps_token_id") is not None
            and 0 <= int(self.meta["no_timestamps_token_id"]) < self.vocab_size
            else None
        )
        self.compute_dtype = str(self.meta.get("compute_dtype", "float32"))
        self.cache_dtype = str(self.meta.get("cache_dtype", "float32"))
        self.mask_dtype = str(self.meta.get("mask_dtype", "float32"))
        self.logits_dtype = str(self.meta.get("logits_dtype", "float32"))
        self.cache_np_dtype = np.dtype(self.cache_dtype)
        self.mask_np_dtype = np.dtype(self.mask_dtype)
        self.mask_fill_value = (
            -1e9
            if self.mask_np_dtype.itemsize >= np.dtype("float32").itemsize
            else float(np.finfo(self.mask_np_dtype).min)
        )
        self.max_past_len = int(self.meta["max_dec_len_compiled"]) - 1
        self.max_decode_batch = int(self.meta.get("max_decode_batch", 1))
        self.self_shape = (
            int(self.meta["decoder_layers"]),
            self.max_decode_batch,
            int(self.meta["decoder_attention_heads"]),
            self.max_past_len,
            int(self.meta["head_dim"]),
        )
        self.load_time_ms = (time.perf_counter() - init_started) * 1000.0
        self.last_perf: BundlePerfStats | None = None
        self.collect_perf = False

    def _reset_perf(self, enabled: bool = False):
        self.collect_perf = bool(enabled)
        self.last_perf = (
            BundlePerfStats(load_ms=float(self.load_time_ms))
            if self.collect_perf
            else None
        )
        return self.last_perf

    def _sync_device(self):
        sync = getattr(self.dev, "sync", None)
        if callable(sync):
            sync()

    def _time_call(self, fn):
        if self.last_perf is None:
            return fn(), 0.0
        self._sync_device()
        started = time.perf_counter()
        out = fn()
        self._sync_device()
        return out, (time.perf_counter() - started) * 1000.0

    def _record_metric(self, name: str, elapsed_ms: float, runs: int = 1):
        if self.last_perf is None:
            return
        metric = getattr(self.last_perf, name)
        metric.add(elapsed_ms, runs=runs)

    def _record_stop_reason(self, stop_reason: str):
        if self.last_perf is None:
            return
        key = str(stop_reason or "unknown")
        self.last_perf.stop_reason_counts[key] = (
            int(self.last_perf.stop_reason_counts.get(key, 0)) + 1
        )

    def _record_window_trace(
        self,
        *,
        seek_samples: int,
        valid_samples: int,
        prompt_len: int,
        max_new_tokens: int,
        attempt: DecodeAttempt,
        timestamps: bool,
        segment_count: int,
        advance_samples: int,
        text_chars: int,
    ):
        if self.last_perf is None:
            return
        trace = DecodeWindowTrace(
            index=len(self.last_perf.windows),
            seek_samples=int(seek_samples),
            valid_samples=int(valid_samples),
            prompt_len=int(prompt_len),
            max_new_tokens=int(max_new_tokens),
            generated_len=int(attempt.generated_len),
            returned_len=int(attempt.returned_len),
            stop_reason=str(attempt.stop_reason or "unknown"),
            strategy=str(attempt.strategy),
            temperature=float(attempt.temperature),
            last_token_id=(
                None if attempt.last_token_id is None else int(attempt.last_token_id)
            ),
            last_is_timestamp=bool(attempt.last_is_timestamp),
            ended_with_eos=bool(attempt.ended_with_eos),
            fallback_compression_ratio=bool(attempt.fallback_compression_ratio),
            fallback_logprob=bool(attempt.fallback_logprob),
            silence=bool(attempt.silence),
            finished_hypotheses=int(attempt.finished_hypotheses),
            active_hypotheses=int(attempt.active_hypotheses),
            avg_logprob=float(attempt.avg_logprob),
            compression_ratio=float(attempt.compression_ratio),
            no_speech_prob=float(attempt.no_speech_prob),
            timestamps=bool(timestamps),
            segment_count=int(segment_count),
            advance_samples=int(advance_samples),
            text_chars=int(text_chars),
        )
        self.last_perf.windows.append(trace)
        self._record_stop_reason(trace.stop_reason)

    def _cpu_capability(self) -> str | None:
        try:
            import torch

            backend = getattr(getattr(torch, "backends", None), "cpu", None)
            getter = getattr(backend, "get_cpu_capability", None)
            if callable(getter):
                value = getter()
                return str(value) if value is not None else None
        except Exception:
            return None
        return None

    def _gpu_system_info_parts(self) -> list[str]:
        if self.device_kind != "cuda":
            return ["CUDA = 0"]
        parts = ["CUDA = 1"]
        try:
            import torch

            if not torch.cuda.is_available():
                return parts
            props = torch.cuda.get_device_properties(self.device_id)
            cap = torch.cuda.get_device_capability(self.device_id)
            parts.extend(
                [
                    f"GPU = {props.name}",
                    f"CC = {cap[0]}.{cap[1]}",
                    f"VRAM = {int(props.total_memory // (1024 * 1024))} MiB",
                ]
            )
            return parts
        except Exception:
            return parts

    def whisper_cpp_system_info_lines(self) -> list[str]:
        active_threads = _default_thread_count()
        cpu_count = max(1, int(os.cpu_count() or 1))
        thread_part = (
            f"{active_threads} / {cpu_count}"
            if active_threads != cpu_count
            else f"{active_threads}"
        )
        parts = [
            f"system_info: n_threads = {thread_part}",
            "BACKEND = TVM_RELAX_VM",
            f"device = {self.device_kind}:{self.device_id}",
        ]
        parts.extend(self._gpu_system_info_parts())
        cpu_cap = self._cpu_capability()
        if cpu_cap:
            parts.append(f"CPU_CAP = {cpu_cap}")
        parts.append(f"CPU = {platform.machine() or 'unknown'}")
        return [" | ".join(parts)]

    def _format_total_only_perf_line(self, label: str, total_ms: float) -> str:
        return f"whisper_print_timings: {label:<11} = {total_ms:8.2f} ms"

    def _format_perf_line(
        self,
        label: str,
        metric: PerfMetric,
        unit_label: str = "runs",
    ) -> str:
        unit_singular = unit_label[:-1] if unit_label.endswith("s") else unit_label
        return (
            f"whisper_print_timings: {label:<11} = {metric.total_ms:8.2f} ms / "
            f"{metric.runs} {unit_label} ({metric.avg_ms:6.2f} ms per {unit_singular})"
        )

    def whisper_cpp_timing_lines(self) -> list[str]:
        perf = self.last_perf
        if perf is None:
            return []
        encode_total = PerfMetric(
            total_ms=perf.encode.total_ms + perf.cross_kv.total_ms,
            runs=max(perf.encode.runs, perf.cross_kv.runs),
        )
        lines = [
            self._format_total_only_perf_line("load time", perf.load_ms),
            (
                "whisper_print_timings: fallbacks   = "
                f"{perf.fallbacks_compression_ratio} cr / {perf.fallbacks_logprob} lp"
            ),
            self._format_total_only_perf_line("mel time", perf.mel.total_ms),
            self._format_perf_line("sample time", perf.sample, unit_label="decisions"),
            self._format_perf_line("encode time", encode_total),
            self._format_perf_line("decode time", perf.decode),
            self._format_perf_line("batchd time", perf.batchd),
            self._format_perf_line("prompt time", perf.prompt),
            self._format_total_only_perf_line("total time", perf.total_ms),
        ]
        if perf.lang_detect.runs > 0:
            lines.insert(7, self._format_perf_line("lang detect", perf.lang_detect))
        if perf.silence_skips:
            lines.append(f"whisper_print_timings: silence skips = {perf.silence_skips}")
        if perf.stop_reason_counts:
            summary = " | ".join(
                f"{key} = {perf.stop_reason_counts[key]}"
                for key in sorted(perf.stop_reason_counts)
            )
            lines.append(f"whisper_print_timings: stop reasons = {summary}")
        return lines

    def whisper_cpp_decode_debug_lines(self) -> list[str]:
        perf = self.last_perf
        if perf is None:
            return []
        sample_rate = max(1, int(self.meta.get("sample_rate", 16000)))
        lines: list[str] = []
        for trace in perf.windows:
            last_token_id = (
                -1 if trace.last_token_id is None else int(trace.last_token_id)
            )
            lines.append(
                "whisper_decode_debug: "
                f"window = {trace.index} | "
                f"seek = {trace.seek_samples} ({trace.seek_samples / float(sample_rate):.2f} s) | "
                f"valid = {trace.valid_samples} | "
                f"prompt = {trace.prompt_len} | "
                f"max_new = {trace.max_new_tokens} | "
                f"generated = {trace.generated_len} | "
                f"returned = {trace.returned_len} | "
                f"stop = {trace.stop_reason} | "
                f"strategy = {trace.strategy} | "
                f"temp = {trace.temperature:.2f} | "
                f"last = {last_token_id} | "
                f"last_is_ts = {1 if trace.last_is_timestamp else 0} | "
                f"ended_with_eos = {1 if trace.ended_with_eos else 0} | "
                f"fallback_cr = {1 if trace.fallback_compression_ratio else 0} | "
                f"fallback_lp = {1 if trace.fallback_logprob else 0} | "
                f"silence = {1 if trace.silence else 0} | "
                f"finished = {trace.finished_hypotheses} | "
                f"active = {trace.active_hypotheses} | "
                f"avg_logprob = {trace.avg_logprob:.3f} | "
                f"compression_ratio = {trace.compression_ratio:.3f} | "
                f"no_speech = {trace.no_speech_prob:.3f} | "
                f"segments = {trace.segment_count} | "
                f"advance = {trace.advance_samples} | "
                f"text_chars = {trace.text_chars}"
            )
        return lines

    def whisper_cpp_model_lines(self) -> list[str]:
        model_size = int(self.meta.get("model_size_bytes", 0) or 0)
        params_size = int(self.meta.get("params_size_bytes", 0) or 0)
        lib_size = int(self.meta.get("lib_size_bytes", 0) or 0)
        tokenizer_size = int(self.meta.get("tokenizer_size_bytes", 0) or 0)
        bundle_size = int(self.meta.get("bundle_size_bytes", 0) or 0)
        model_id = str(self.meta.get("model_id", "unknown"))
        model_type = model_id.rsplit("/", 1)[-1]
        if model_type.startswith("whisper-"):
            model_type = model_type[len("whisper-") :]
        return [
            f"whisper_model_load: model_id      = {model_id}",
            f"whisper_model_load: n_vocab       = {int(self.meta['vocab_size'])}",
            f"whisper_model_load: n_audio_ctx   = {int(self.meta.get('n_audio_ctx', self.meta['max_source_positions']))}",
            f"whisper_model_load: n_audio_state = {int(self.meta.get('n_audio_state', self.meta.get('d_model', 0)))}",
            f"whisper_model_load: n_audio_head  = {int(self.meta.get('n_audio_head', self.meta.get('encoder_attention_heads', 0)))}",
            f"whisper_model_load: n_audio_layer = {int(self.meta.get('n_audio_layer', self.meta.get('encoder_layers', 0)))}",
            f"whisper_model_load: n_text_ctx    = {int(self.meta.get('n_text_ctx', self.meta['max_target_positions']))}",
            f"whisper_model_load: n_text_state  = {int(self.meta.get('n_text_state', self.meta.get('d_model', 0)))}",
            f"whisper_model_load: n_text_head   = {int(self.meta.get('n_text_head', self.meta.get('decoder_attention_heads', 0)))}",
            f"whisper_model_load: n_text_layer  = {int(self.meta.get('n_text_layer', self.meta.get('decoder_layers', 0)))}",
            f"whisper_model_load: n_mels        = {int(self.meta.get('n_mels', self.meta.get('num_mel_bins', 0)))}",
            f"whisper_model_load: type          = {model_type or 'unknown'}",
            f"whisper_model_load: ftype         = {self.meta.get('ftype', 'unknown')}",
            f"whisper_model_load: compute dtype = {self.meta.get('compute_dtype', 'float32')}",
            f"whisper_model_load: cache dtype   = {self.meta.get('cache_dtype', 'float32')}",
            f"whisper_model_load: logits dtype  = {self.meta.get('logits_dtype', 'float32')}",
            f"whisper_model_load: qntvr         = {int(self.meta.get('qntvr', 0))}",
            f"whisper_model_load: n_langs       = {len(self.meta.get('language_token_ids', {}))}",
            f"whisper_model_load: model size    = {_format_bytes(model_size)}",
            f"whisper_model_load: lib size      = {_format_bytes(lib_size)}",
            f"whisper_model_load: params size   = {_format_bytes(params_size)}",
            f"whisper_model_load: tokenizer size= {_format_bytes(tokenizer_size)}",
            f"whisper_model_load: bundle size   = {_format_bytes(bundle_size)}",
        ]

    def decode_text(self, ids: Sequence[int]) -> str:
        ids = [
            int(x)
            for x in ids
            if int(x) < self.timestamp_begin and int(x) not in self.special_ids
        ]
        return "" if not ids else str(self.tok.decode(ids))

    def build_prompt(
        self,
        language_code: str | None,
        task: str,
        timestamps: bool,
        history: Sequence[int],
    ):
        ids = [int(self.meta["decoder_start_token_id"])]
        lang_ids = {
            str(k): int(v) for k, v in self.meta.get("language_token_ids", {}).items()
        }
        if language_code is not None and language_code in lang_ids:
            ids.append(lang_ids[language_code])
        ids.append(
            int(
                self.meta["translate_token_id"]
                if task == "translate"
                else self.meta["transcribe_token_id"]
            )
        )
        if not timestamps and self.meta.get("no_timestamps_token_id") is not None:
            ids.append(int(self.meta["no_timestamps_token_id"]))
        if history and self.meta.get("startofprev_token_id") is not None:
            limit = max(0, int(self.meta["max_target_positions"]) // 2 - len(ids) - 1)
            ids = (
                [int(self.meta["startofprev_token_id"]), *list(history)[-limit:], *ids]
                if limit > 0
                else ids
            )
        return ids

    def max_new_tokens(self, requested: int | None, prompt_len: int) -> int:
        available = int(self.meta["max_target_positions"]) - int(prompt_len)
        if available <= 0:
            raise ValueError("Prompt leaves no room for new tokens.")
        if requested is None:
            return available
        if int(requested) <= 0:
            raise ValueError("max_new_tokens must be positive.")
        return min(int(requested), available)

    def _slice_window(self, audio: np.ndarray, start: int):
        n = int(self.meta["n_samples"])
        chunk = np.asarray(audio[start : start + n], dtype=np.float32)
        valid = int(chunk.shape[0])
        if valid < n:
            chunk = np.pad(chunk, (0, n - valid))
        return chunk[None, :], np.asarray([valid], dtype=np.int32)

    def load_window(self, audio: np.ndarray, start: int):
        wave, valid = self._slice_window(audio, start)
        pp, elapsed_ms = self._time_call(
            lambda: unwrap_many(
                self.vm["preprocess"](to_tvm(wave, self.dev), to_tvm(valid, self.dev))
            )
        )
        self._record_metric("mel", elapsed_ms)
        feats, _ = pp
        enc, elapsed_ms = self._time_call(
            lambda: unwrap(self.vm["encode"](feats, self.params))
        )
        self._record_metric("encode", elapsed_ms)
        cross, elapsed_ms = self._time_call(
            lambda: unwrap_many(self.vm["cross_kv"](enc, self.params))
        )
        self._record_metric("cross_kv", elapsed_ms)
        cross_k, cross_v = cross
        return unwrap(cross_k), unwrap(cross_v), int(valid[0])

    def _zero_self_cache(self):
        self_k = to_tvm(np.zeros(self.self_shape, dtype=self.cache_np_dtype), self.dev)
        self_v = to_tvm(np.zeros(self.self_shape, dtype=self.cache_np_dtype), self.dev)
        return self_k, self_v

    def _make_past_keep_mask(
        self, positions: Sequence[int], active_count: int
    ) -> np.ndarray:
        mask = np.full(
            (self.max_decode_batch, 1, 1, self.max_past_len),
            self.mask_fill_value,
            dtype=self.mask_np_dtype,
        )
        for i in range(min(active_count, len(positions))):
            keep = max(0, min(int(positions[i]), self.max_past_len))
            if keep > 0:
                mask[i, 0, 0, self.max_past_len - keep :] = 0.0
        return mask

    def _decode_step_batch(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
        self_k,
        self_v,
        cross_k,
        cross_v,
        *,
        record_perf: bool = True,
        time_op: bool = True,
    ):
        active_count = len(token_ids)
        if active_count <= 0:
            raise ValueError("decode batch must contain at least one token")
        if active_count > self.max_decode_batch:
            raise ValueError(
                f"decode batch {active_count} exceeds compiled max_decode_batch={self.max_decode_batch}"
            )
        token_pad = np.full((self.max_decode_batch, 1), int(self.eos), dtype=np.int32)
        position_pad = np.zeros((self.max_decode_batch, 1), dtype=np.int32)
        token_pad[:active_count, 0] = np.asarray(token_ids, dtype=np.int32)
        position_pad[:active_count, 0] = np.asarray(positions, dtype=np.int32)
        mask = self._make_past_keep_mask(positions, active_count)

        def _run_decode_step():
            out = self.vm["decode_step"](
                to_tvm(token_pad, self.dev),
                to_tvm(position_pad, self.dev),
                self_k,
                self_v,
                to_tvm(mask, self.dev),
                cross_k,
                cross_v,
                self.params,
            )
            logits, next_k, next_v = unwrap_many(out)
            return (
                np.asarray(unwrap(logits).numpy(), dtype=np.float32)[:active_count],
                unwrap(next_k),
                unwrap(next_v),
            )

        if time_op:
            result, elapsed_ms = self._time_call(_run_decode_step)
        else:
            result = _run_decode_step()
            elapsed_ms = 0.0
        if record_perf:
            self._record_metric("decode" if active_count == 1 else "batchd", elapsed_ms)
        return result

    def _gather_self_kv(
        self,
        self_k,
        self_v,
        parent_indices: Sequence[int],
        *,
        metric_name: str | None = None,
    ):
        active_count = len(parent_indices)
        if active_count <= 0:
            raise ValueError("parent_indices must be non-empty")
        if active_count > self.max_decode_batch:
            raise ValueError(
                f"gather batch {active_count} exceeds compiled max_decode_batch={self.max_decode_batch}"
            )
        gather = np.zeros((self.max_decode_batch,), dtype=np.int32)
        gather[:active_count] = np.asarray(parent_indices, dtype=np.int32)
        out, elapsed_ms = self._time_call(
            lambda: unwrap_many(
                self.vm["gather_self_kv"](
                    self_k,
                    self_v,
                    to_tvm(gather, self.dev),
                )
            )
        )
        if metric_name is not None:
            self._record_metric(metric_name, elapsed_ms, runs=0)
        next_k, next_v = out
        return unwrap(next_k), unwrap(next_v)

    def _prime_batch(
        self,
        prompt_ids: Sequence[int],
        cross_k,
        cross_v,
        active_count: int,
        *,
        metric_name: str = "prompt",
    ):
        if active_count <= 0:
            raise ValueError("active_count must be positive")

        def _run_prompt():
            self_k, self_v = self._zero_self_cache()
            logits = None
            for pos, token_id in enumerate(prompt_ids):
                logits, self_k, self_v = self._decode_step_batch(
                    [int(token_id)] * active_count,
                    [pos] * active_count,
                    self_k,
                    self_v,
                    cross_k,
                    cross_v,
                    record_perf=False,
                    time_op=False,
                )
            if logits is None:
                raise RuntimeError("Failed to prime decoder.")
            return logits, self_k, self_v, len(prompt_ids)

        result, elapsed_ms = self._time_call(_run_prompt)
        self._record_metric(metric_name, elapsed_ms)
        return result

    def detect_language(self, cross_k, cross_v) -> str:
        if not bool(self.meta.get("is_multilingual", False)):
            return str(self.meta.get("default_language_code", "en"))
        lang_ids = {
            str(k): int(v) for k, v in self.meta.get("language_token_ids", {}).items()
        }
        if not lang_ids:
            return str(self.meta.get("default_language_code", "en"))
        logits, *_ = self._prime_batch(
            [int(self.meta["decoder_start_token_id"])],
            cross_k,
            cross_v,
            1,
            metric_name="lang_detect",
        )
        cand = np.asarray(sorted(lang_ids.values()), dtype=np.int64)
        return {v: k for k, v in lang_ids.items()}.get(
            int(cand[int(np.argmax(logits[0][cand]))]),
            str(self.meta.get("default_language_code", "en")),
        )

    def filter_logits(
        self, logits: np.ndarray, generated: Sequence[int], timestamps: bool
    ) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32).copy()
        if self.suppress.size:
            logits[self.suppress] = -np.inf
        if not generated:
            if self.begin_suppress.size:
                logits[self.begin_suppress] = -np.inf
            if self.blank_suppress.size:
                logits[self.blank_suppress] = -np.inf
        if not timestamps:
            return logits
        if self.no_timestamps is not None:
            logits[self.no_timestamps] = -np.inf
        seq = [int(x) for x in generated]
        last_is_ts = len(seq) >= 1 and seq[-1] >= self.timestamp_begin
        prev_is_ts = len(seq) < 2 or seq[-2] >= self.timestamp_begin
        if last_is_ts:
            logits[
                self.timestamp_begin if prev_is_ts else 0 : None
                if prev_is_ts
                else self.eos
            ] = -np.inf
        ts_tokens = [x for x in seq if x >= self.timestamp_begin]
        if ts_tokens:
            logits[
                self.timestamp_begin : (
                    ts_tokens[-1]
                    if last_is_ts and not prev_is_ts
                    else ts_tokens[-1] + 1
                )
            ] = -np.inf
        if not seq:
            logits[: self.timestamp_begin] = -np.inf
            limit = self.timestamp_begin + int(
                round(1.0 / float(self.meta["time_precision"]))
            )
            if limit + 1 < logits.shape[0]:
                logits[limit + 1 :] = -np.inf
        ln = _logsumexp(logits)
        if np.isfinite(ln):
            lp = logits.astype(np.float64) - ln
            if _logsumexp(lp[self.timestamp_begin :]) > (
                float(np.max(lp[: self.timestamp_begin]))
                if self.timestamp_begin > 0
                else float("-inf")
            ):
                logits[: self.timestamp_begin] = -np.inf
        return logits

    def _normalize_decode_config(
        self,
        temperature: float,
        temperature_inc: float,
        beam_size: int,
        best_of: int,
        length_penalty: float | None,
        compression_ratio_threshold: float | None,
        logprob_threshold: float | None,
        no_speech_threshold: float | None,
    ) -> DecodeConfig:
        cfg = DecodeConfig(
            temperature=float(temperature),
            temperature_inc=float(temperature_inc),
            beam_size=int(beam_size),
            best_of=int(best_of),
            length_penalty=(None if length_penalty is None else float(length_penalty)),
            compression_ratio_threshold=(
                None
                if compression_ratio_threshold is None
                else float(compression_ratio_threshold)
            ),
            logprob_threshold=(
                None if logprob_threshold is None else float(logprob_threshold)
            ),
            no_speech_threshold=(
                None if no_speech_threshold is None else float(no_speech_threshold)
            ),
        )
        if cfg.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if cfg.temperature_inc < 0:
            raise ValueError("temperature_inc must be >= 0")
        if cfg.beam_size <= 0:
            raise ValueError("beam_size must be >= 1")
        if cfg.best_of <= 0:
            raise ValueError("best_of must be >= 1")
        if max(cfg.beam_size, cfg.best_of) > self.max_decode_batch:
            raise ValueError(
                "beam_size/best_of exceed compiled max_decode_batch="
                f"{self.max_decode_batch}"
            )
        return cfg

    def _temperature_schedule(
        self, temperature: float, temperature_inc: float
    ) -> list[float]:
        base = max(0.0, float(temperature))
        inc = max(0.0, float(temperature_inc))
        temps = [base]
        if inc <= 0:
            return temps
        seen = {round(base, 6)}
        cur = base
        while cur < 1.0 - 1e-6:
            cur = min(1.0, cur + inc)
            key = round(cur, 6)
            if key in seen:
                break
            seen.add(key)
            temps.append(cur)
        return temps

    def _token_probability(self, logits: np.ndarray, token_id: int | None) -> float:
        if token_id is None or token_id < 0 or token_id >= logits.shape[0]:
            return 0.0
        ln = _logsumexp(logits)
        if not np.isfinite(ln):
            return 0.0
        return float(np.exp(float(logits[token_id]) - ln))

    def _pick_next_token(
        self, scores: np.ndarray, temperature: float
    ) -> tuple[int, float]:
        scaled = np.asarray(scores, dtype=np.float64)
        if temperature > 0:
            scaled = scaled / float(temperature)
        logprobs = _log_softmax(scaled)
        if temperature > 0:
            probs = np.exp(logprobs)
            next_id = int(self.rng.choice(probs.shape[0], p=probs))
        else:
            next_id = int(np.argmax(logprobs))
        return next_id, float(logprobs[next_id])

    def _top_indices(self, scores: np.ndarray, k: int) -> np.ndarray:
        k = max(1, min(int(k), int(scores.shape[0])))
        if k >= scores.shape[0]:
            return np.argsort(scores)[::-1]
        top = np.argpartition(scores, -k)[-k:]
        return top[np.argsort(scores[top])[::-1]]

    def _rank_score(
        self, sum_logprob: float, length: int, length_penalty: float | None
    ) -> float:
        if length_penalty is None:
            return float(sum_logprob)
        norm = ((5.0 + max(1, int(length))) / 6.0) ** float(length_penalty)
        return float(sum_logprob / norm)

    def _compression_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        raw = text.encode("utf-8")
        if not raw:
            return 0.0
        return float(len(raw) / max(len(zlib.compress(raw)), 1))

    def _finalize_attempt(
        self,
        candidates: Sequence[DecodeHypothesis],
        no_speech_prob: float,
        temperature: float,
        strategy: str,
        length_penalty: float | None,
    ) -> DecodeAttempt:
        if not candidates:
            return DecodeAttempt(
                tokens=[],
                sum_logprob=0.0,
                avg_logprob=0.0,
                compression_ratio=0.0,
                no_speech_prob=float(no_speech_prob),
                temperature=float(temperature),
                strategy=strategy,
                generated_len=0,
                returned_len=0,
                last_token_id=None,
                last_is_timestamp=False,
                ended_with_eos=False,
            )
        best = max(
            candidates,
            key=lambda h: self._rank_score(
                h.sum_logprob,
                len(h.tokens),
                length_penalty,
            ),
        )
        raw_tokens = [int(x) for x in best.tokens]
        last_token_id = raw_tokens[-1] if raw_tokens else None
        ended_with_eos = bool(last_token_id == self.eos)
        tokens = raw_tokens[:-1] if ended_with_eos else raw_tokens
        text = self.decode_text(tokens)
        return DecodeAttempt(
            tokens=tokens,
            sum_logprob=float(best.sum_logprob),
            avg_logprob=float(best.sum_logprob / max(len(best.tokens), 1)),
            compression_ratio=self._compression_ratio(text),
            no_speech_prob=float(no_speech_prob),
            temperature=float(temperature),
            strategy=strategy,
            generated_len=len(raw_tokens),
            returned_len=len(tokens),
            last_token_id=last_token_id,
            last_is_timestamp=bool(
                last_token_id is not None and last_token_id >= self.timestamp_begin
            ),
            ended_with_eos=ended_with_eos,
        )

    def _decode_sampling(
        self,
        cross_k,
        cross_v,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        timestamps: bool,
        temperature: float,
        n_samples: int,
        length_penalty: float | None,
    ) -> DecodeAttempt:
        logits, self_k, self_v, pos = self._prime_batch(
            prompt_ids,
            cross_k,
            cross_v,
            n_samples,
        )
        no_speech_prob = self._token_probability(logits[0], self.no_speech_token_id)
        stop_reason = "max_new_tokens"
        eos_seen = False
        active = [
            DecodeHypothesis(
                tokens=[],
                sum_logprob=0.0,
                next_logits=np.asarray(logits[i], dtype=np.float32),
                position=int(pos),
            )
            for i in range(n_samples)
        ]
        finished: list[DecodeHypothesis] = []
        for _ in range(max_new_tokens):
            if not active:
                stop_reason = "eos" if eos_seen else "no_active_hypotheses"
                break
            parent_indices: list[int] = []
            next_tokens: list[int] = []
            positions: list[int] = []
            next_active: list[DecodeHypothesis] = []
            sample_started = time.perf_counter()
            decisions = 0
            for parent_idx, hyp in enumerate(active):
                scores = self.filter_logits(hyp.next_logits, hyp.tokens, timestamps)
                if np.all(np.isneginf(scores)):
                    finished.append(hyp)
                    continue
                next_id, token_logprob = self._pick_next_token(scores, temperature)
                decisions += 1
                child = DecodeHypothesis(
                    tokens=[*hyp.tokens, int(next_id)],
                    sum_logprob=float(hyp.sum_logprob + token_logprob),
                    next_logits=None,
                    position=int(hyp.position + 1),
                )
                if next_id == self.eos:
                    eos_seen = True
                    finished.append(child)
                    continue
                parent_indices.append(parent_idx)
                next_tokens.append(int(next_id))
                positions.append(int(hyp.position))
                next_active.append(child)
            self._record_metric(
                "sample",
                (time.perf_counter() - sample_started) * 1000.0,
                runs=decisions,
            )
            if not next_tokens:
                stop_reason = "eos" if eos_seen else "all_logits_filtered"
                active = []
                break
            metric_name = "decode" if len(next_tokens) == 1 else "batchd"
            parent_k, parent_v = self._gather_self_kv(
                self_k,
                self_v,
                parent_indices,
                metric_name=metric_name,
            )
            logits, self_k, self_v = self._decode_step_batch(
                next_tokens,
                positions,
                parent_k,
                parent_v,
                cross_k,
                cross_v,
            )
            for i, hyp in enumerate(next_active):
                hyp.next_logits = np.asarray(logits[i], dtype=np.float32)
            active = next_active
        strategy = (
            "best_of"
            if temperature > 0 and n_samples > 1
            else "sampling"
            if temperature > 0
            else "greedy"
        )
        attempt = self._finalize_attempt(
            [*finished, *active],
            no_speech_prob,
            temperature,
            strategy,
            length_penalty,
        )
        attempt.stop_reason = stop_reason
        attempt.finished_hypotheses = len(finished)
        attempt.active_hypotheses = len(active)
        return attempt

    def _decode_beam_search(
        self,
        cross_k,
        cross_v,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        timestamps: bool,
        beam_size: int,
        length_penalty: float | None,
    ) -> DecodeAttempt:
        logits, self_k, self_v, pos = self._prime_batch(prompt_ids, cross_k, cross_v, 1)
        no_speech_prob = self._token_probability(logits[0], self.no_speech_token_id)
        stop_reason = "max_new_tokens"
        eos_seen = False
        active = [
            DecodeHypothesis(
                tokens=[],
                sum_logprob=0.0,
                next_logits=np.asarray(logits[0], dtype=np.float32),
                position=int(pos),
            )
        ]
        finished: list[DecodeHypothesis] = []
        for _ in range(max_new_tokens):
            if not active:
                stop_reason = "eos" if eos_seen else "no_active_hypotheses"
                break
            finished_candidates: list[DecodeHypothesis] = []
            active_candidates: list[tuple[float, int, int, int, DecodeHypothesis]] = []
            per_hyp_topk = max(beam_size + 1, 2)
            sample_started = time.perf_counter()
            decisions = 0
            for parent_idx, hyp in enumerate(active):
                scores = self.filter_logits(hyp.next_logits, hyp.tokens, timestamps)
                if np.all(np.isneginf(scores)):
                    finished.append(hyp)
                    continue
                logprobs = _log_softmax(scores)
                top_indices = self._top_indices(logprobs, per_hyp_topk).tolist()
                decisions += 1
                for next_id in top_indices:
                    token_logprob = float(logprobs[int(next_id)])
                    if not np.isfinite(token_logprob):
                        continue
                    child = DecodeHypothesis(
                        tokens=[*hyp.tokens, int(next_id)],
                        sum_logprob=float(hyp.sum_logprob + token_logprob),
                        next_logits=None,
                        position=int(hyp.position + 1),
                    )
                    if int(next_id) == self.eos:
                        eos_seen = True
                        finished_candidates.append(child)
                    else:
                        active_candidates.append(
                            (
                                self._rank_score(
                                    child.sum_logprob,
                                    len(child.tokens),
                                    length_penalty,
                                ),
                                parent_idx,
                                int(next_id),
                                int(hyp.position),
                                child,
                            )
                        )
            self._record_metric(
                "sample",
                (time.perf_counter() - sample_started) * 1000.0,
                runs=decisions,
            )
            if finished_candidates:
                finished.extend(finished_candidates)
                if len(finished) >= beam_size:
                    stop_reason = "beam_finished"
                    active = []
                    break
            if not active_candidates:
                stop_reason = "eos" if eos_seen else "all_logits_filtered"
                active = []
                break
            active_candidates.sort(key=lambda x: x[0], reverse=True)
            survivors = active_candidates[:beam_size]
            parent_indices = [cand[1] for cand in survivors]
            next_tokens = [cand[2] for cand in survivors]
            positions = [cand[3] for cand in survivors]
            next_active = [cand[4] for cand in survivors]
            metric_name = "decode" if len(next_tokens) == 1 else "batchd"
            parent_k, parent_v = self._gather_self_kv(
                self_k,
                self_v,
                parent_indices,
                metric_name=metric_name,
            )
            logits, self_k, self_v = self._decode_step_batch(
                next_tokens,
                positions,
                parent_k,
                parent_v,
                cross_k,
                cross_v,
            )
            for i, hyp in enumerate(next_active):
                hyp.next_logits = np.asarray(logits[i], dtype=np.float32)
            active = next_active
        attempt = self._finalize_attempt(
            [*finished, *active],
            no_speech_prob,
            0.0,
            "beam_search",
            length_penalty,
        )
        attempt.stop_reason = stop_reason
        attempt.finished_hypotheses = len(finished)
        attempt.active_hypotheses = len(active)
        return attempt

    def _decode_once(
        self,
        cross_k,
        cross_v,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        timestamps: bool,
        cfg: DecodeConfig,
        temperature: float,
    ) -> DecodeAttempt:
        if temperature <= 0.0:
            if cfg.beam_size > 1:
                return self._decode_beam_search(
                    cross_k,
                    cross_v,
                    prompt_ids,
                    max_new_tokens,
                    timestamps,
                    cfg.beam_size,
                    cfg.length_penalty,
                )
            return self._decode_sampling(
                cross_k,
                cross_v,
                prompt_ids,
                max_new_tokens,
                timestamps,
                0.0,
                1,
                cfg.length_penalty,
            )
        return self._decode_sampling(
            cross_k,
            cross_v,
            prompt_ids,
            max_new_tokens,
            timestamps,
            float(temperature),
            max(1, cfg.best_of),
            cfg.length_penalty,
        )

    def _should_skip_silence(self, attempt: DecodeAttempt, cfg: DecodeConfig) -> bool:
        if cfg.no_speech_threshold is None:
            return False
        if attempt.no_speech_prob <= float(cfg.no_speech_threshold):
            return False
        if cfg.logprob_threshold is None:
            return len(attempt.tokens) == 0
        return attempt.avg_logprob < float(cfg.logprob_threshold)

    def _fallback_reasons(
        self, attempt: DecodeAttempt, cfg: DecodeConfig
    ) -> tuple[bool, bool]:
        compression_ratio = bool(
            cfg.compression_ratio_threshold is not None
            and attempt.compression_ratio > float(cfg.compression_ratio_threshold)
        )
        logprob = bool(
            cfg.logprob_threshold is not None
            and attempt.avg_logprob < float(cfg.logprob_threshold)
        )
        return compression_ratio, logprob

    def _should_fallback(self, attempt: DecodeAttempt, cfg: DecodeConfig) -> bool:
        return any(self._fallback_reasons(attempt, cfg))

    def decode_window(
        self,
        cross_k,
        cross_v,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        timestamps: bool,
        cfg: DecodeConfig,
    ) -> DecodeAttempt:
        temperatures = self._temperature_schedule(cfg.temperature, cfg.temperature_inc)
        last_attempt: DecodeAttempt | None = None
        fallback_cr_seen = False
        fallback_lp_seen = False
        for temperature in temperatures:
            attempt = self._decode_once(
                cross_k,
                cross_v,
                prompt_ids,
                max_new_tokens,
                timestamps,
                cfg,
                temperature,
            )
            last_attempt = attempt
            compression_ratio, logprob = self._fallback_reasons(attempt, cfg)
            fallback_cr_seen = bool(fallback_cr_seen or compression_ratio)
            fallback_lp_seen = bool(fallback_lp_seen or logprob)
            attempt.fallback_compression_ratio = bool(fallback_cr_seen)
            attempt.fallback_logprob = bool(fallback_lp_seen)
            if self._should_skip_silence(attempt, cfg):
                if self.last_perf is not None:
                    self.last_perf.silence_skips += 1
                attempt.stop_reason = "silence"
                attempt.silence = True
                return attempt
            if temperature != temperatures[-1] and (compression_ratio or logprob):
                if self.last_perf is not None:
                    self.last_perf.fallbacks_compression_ratio += int(compression_ratio)
                    self.last_perf.fallbacks_logprob += int(logprob)
                continue
            return attempt
        if last_attempt is None:
            return DecodeAttempt(
                tokens=[],
                sum_logprob=0.0,
                avg_logprob=0.0,
                compression_ratio=0.0,
                no_speech_prob=0.0,
                temperature=float(cfg.temperature),
                strategy="unknown",
                stop_reason="empty",
                generated_len=0,
                returned_len=0,
                last_token_id=None,
                last_is_timestamp=False,
                ended_with_eos=False,
                fallback_compression_ratio=bool(fallback_cr_seen),
                fallback_logprob=bool(fallback_lp_seen),
                silence=False,
            )
        last_attempt.fallback_compression_ratio = bool(fallback_cr_seen)
        last_attempt.fallback_logprob = bool(fallback_lp_seen)
        return last_attempt

    def build_segments(
        self, ids: Sequence[int], offset_s: float, valid_samples: int, timestamps: bool
    ):
        ids = [int(x) for x in ids]
        if not ids:
            return [], valid_samples
        if not timestamps:
            text = self.decode_text(ids)
            return (
                [
                    {
                        "start": float(offset_s),
                        "end": float(
                            offset_s + valid_samples / float(self.meta["sample_rate"])
                        ),
                        "text": text,
                        "tokens": ids,
                    }
                ]
                if text.strip()
                else []
            ), valid_samples
        arr = np.asarray(ids, dtype=np.int64)
        ts_mask = arr >= self.timestamp_begin
        pair_ends = np.where(ts_mask[:-1] & ts_mask[1:])[0] + 1
        tail_is_single_ts = arr.size >= 2 and ts_mask[-2:].tolist() == [False, True]
        segments, advance = [], valid_samples
        if len(pair_ends) > 0:
            last = 0
            for cur in pair_ends.tolist() + ([len(arr)] if tail_is_single_ts else []):
                part = arr[last:cur]
                last = cur
                if len(part) <= 1:
                    continue
                a = float(
                    offset_s
                    + (int(part[0]) - self.timestamp_begin)
                    * float(self.meta["time_precision"])
                )
                b = float(
                    offset_s
                    + (int(part[-1]) - self.timestamp_begin)
                    * float(self.meta["time_precision"])
                )
                text = self.decode_text(part.tolist())
                if b > a and text.strip():
                    segments.append(
                        {
                            "start": a,
                            "end": b,
                            "text": text,
                            "tokens": [int(x) for x in part],
                        }
                    )
            if not tail_is_single_ts and last > 0:
                advance = int(
                    (int(arr[last - 1]) - self.timestamp_begin)
                    * int(self.meta["samples_per_timestamp"])
                )
        else:
            end_s = float(offset_s + valid_samples / float(self.meta["sample_rate"]))
            ts_only = arr[ts_mask]
            if len(ts_only) > 0 and int(ts_only[-1]) != self.timestamp_begin:
                end_s = float(
                    offset_s
                    + (int(ts_only[-1]) - self.timestamp_begin)
                    * float(self.meta["time_precision"])
                )
            text = self.decode_text(ids)
            if end_s > offset_s and text.strip():
                segments.append(
                    {
                        "start": float(offset_s),
                        "end": end_s,
                        "text": text,
                        "tokens": ids,
                    }
                )
        if advance <= 0:
            advance = min(
                valid_samples, max(1, int(self.meta["samples_per_timestamp"]))
            )
        return segments, int(advance)

    def run(
        self,
        audio: np.ndarray,
        language: str | None = None,
        task: str = "transcribe",
        timestamps: bool = False,
        max_new_tokens: int | None = None,
        condition_on_previous_text: bool = True,
        temperature: float = 0.0,
        temperature_inc: float = 0.0,
        beam_size: int = 1,
        best_of: int = 1,
        length_penalty: float | None = None,
        compression_ratio_threshold: float | None = 2.4,
        logprob_threshold: float | None = -1.0,
        no_speech_threshold: float | None = 0.6,
        collect_perf: bool = False,
    ):
        self._reset_perf(collect_perf)
        run_started = time.perf_counter()
        try:
            cfg = self._normalize_decode_config(
                temperature=temperature,
                temperature_inc=temperature_inc,
                beam_size=beam_size,
                best_of=best_of,
                length_penalty=length_penalty,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
            )
            sr, seek, history, segments, detected = (
                int(self.meta["sample_rate"]),
                0,
                [],
                [],
                language,
            )
            while seek < len(audio) or (len(audio) == 0 and not segments):
                cross_k, cross_v, valid = self.load_window(audio, seek)
                if valid <= 0 and len(audio) > 0:
                    break
                if detected is None:
                    detected = self.detect_language(cross_k, cross_v)
                prompt = self.build_prompt(
                    detected,
                    task,
                    timestamps,
                    history if condition_on_previous_text else [],
                )
                window_max_new_tokens = self.max_new_tokens(max_new_tokens, len(prompt))
                attempt = self.decode_window(
                    cross_k,
                    cross_v,
                    prompt,
                    window_max_new_tokens,
                    timestamps,
                    cfg,
                )
                ids = attempt.tokens
                window_segments, advance = self.build_segments(
                    ids, seek / float(sr), valid, timestamps
                )
                seek_advance = int(
                    (advance if timestamps else valid)
                    or min(valid, int(self.meta["samples_per_timestamp"]))
                )
                self._record_window_trace(
                    seek_samples=seek,
                    valid_samples=valid,
                    prompt_len=len(prompt),
                    max_new_tokens=window_max_new_tokens,
                    attempt=attempt,
                    timestamps=timestamps,
                    segment_count=len(window_segments),
                    advance_samples=seek_advance,
                    text_chars=sum(
                        len(str(s.get("text", ""))) for s in window_segments
                    ),
                )
                for s in window_segments:
                    s["id"] = len(segments)
                    segments.append(s)
                    if condition_on_previous_text:
                        history.extend(int(x) for x in s["tokens"])
                seek += seek_advance
                if len(audio) == 0:
                    break
            detected = detected or str(self.meta.get("default_language_code", "en"))
            return {
                "text": "".join(str(s["text"]) for s in segments).strip(),
                "language": detected,
                "language_name": self.meta.get("language_code_to_name", {}).get(
                    detected, detected
                ),
                "task": task,
                "timestamps": bool(timestamps),
                "duration": float(len(audio) / float(sr)),
                "segments": segments,
            }
        finally:
            if self.last_perf is not None:
                self.last_perf.total_ms = (time.perf_counter() - run_started) * 1000.0
