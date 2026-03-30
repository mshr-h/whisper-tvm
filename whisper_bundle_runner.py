import io
import json
import math
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


class WhisperBundleRunner:
    def __init__(
        self,
        artifacts_dir: str | Path,
        tokenizer_json: str | Path | None = None,
        device: str = "auto",
    ):
        artifacts_dir = Path(artifacts_dir)
        tokenizer_json = Path(tokenizer_json or artifacts_dir / "tokenizer.json")
        self.meta = json.loads(
            (artifacts_dir / "whisper_bundle_metadata.json").read_text(encoding="utf-8")
        )
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
        self.max_past_len = int(self.meta["max_dec_len_compiled"]) - 1
        self.max_decode_batch = int(self.meta.get("max_decode_batch", 1))
        self.self_shape = (
            int(self.meta["decoder_layers"]),
            self.max_decode_batch,
            int(self.meta["decoder_attention_heads"]),
            self.max_past_len,
            int(self.meta["head_dim"]),
        )

    def whisper_cpp_model_lines(self) -> list[str]:
        model_size = int(self.meta.get("model_size_bytes", 0) or 0)
        return [
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
            f"whisper_model_load: ftype         = {self.meta.get('ftype', 'unknown')}",
            f"whisper_model_load: qntvr         = {int(self.meta.get('qntvr', 0))}",
            f"whisper_model_load: model size    = {_format_bytes(model_size)}",
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
        pp = self.vm["preprocess"](to_tvm(wave, self.dev), to_tvm(valid, self.dev))
        feats, _ = unwrap_many(pp)
        enc = unwrap(self.vm["encode"](feats, self.params))
        cross_k, cross_v = unwrap_many(self.vm["cross_kv"](enc, self.params))
        return unwrap(cross_k), unwrap(cross_v), int(valid[0])

    def _zero_self_cache(self):
        self_k = to_tvm(np.zeros(self.self_shape, dtype=np.float32), self.dev)
        self_v = to_tvm(np.zeros(self.self_shape, dtype=np.float32), self.dev)
        return self_k, self_v

    def _make_past_keep_mask(
        self, positions: Sequence[int], active_count: int
    ) -> np.ndarray:
        mask = np.full(
            (self.max_decode_batch, 1, 1, self.max_past_len),
            -1e9,
            dtype=np.float32,
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

    def _gather_self_kv(self, self_k, self_v, parent_indices: Sequence[int]):
        active_count = len(parent_indices)
        if active_count <= 0:
            raise ValueError("parent_indices must be non-empty")
        if active_count > self.max_decode_batch:
            raise ValueError(
                f"gather batch {active_count} exceeds compiled max_decode_batch={self.max_decode_batch}"
            )
        gather = np.zeros((self.max_decode_batch,), dtype=np.int32)
        gather[:active_count] = np.asarray(parent_indices, dtype=np.int32)
        out = self.vm["gather_self_kv"](
            self_k,
            self_v,
            to_tvm(gather, self.dev),
        )
        next_k, next_v = unwrap_many(out)
        return unwrap(next_k), unwrap(next_v)

    def _prime_batch(
        self, prompt_ids: Sequence[int], cross_k, cross_v, active_count: int
    ):
        if active_count <= 0:
            raise ValueError("active_count must be positive")
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
            )
        if logits is None:
            raise RuntimeError("Failed to prime decoder.")
        return logits, self_k, self_v, len(prompt_ids)

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
            )
        best = max(
            candidates,
            key=lambda h: self._rank_score(
                h.sum_logprob,
                len(h.tokens),
                length_penalty,
            ),
        )
        tokens = [int(x) for x in best.tokens]
        if tokens and tokens[-1] == self.eos:
            tokens = tokens[:-1]
        text = self.decode_text(tokens)
        return DecodeAttempt(
            tokens=tokens,
            sum_logprob=float(best.sum_logprob),
            avg_logprob=float(best.sum_logprob / max(len(best.tokens), 1)),
            compression_ratio=self._compression_ratio(text),
            no_speech_prob=float(no_speech_prob),
            temperature=float(temperature),
            strategy=strategy,
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
                break
            parent_indices: list[int] = []
            next_tokens: list[int] = []
            positions: list[int] = []
            next_active: list[DecodeHypothesis] = []
            for parent_idx, hyp in enumerate(active):
                scores = self.filter_logits(hyp.next_logits, hyp.tokens, timestamps)
                if np.all(np.isneginf(scores)):
                    finished.append(hyp)
                    continue
                next_id, token_logprob = self._pick_next_token(scores, temperature)
                child = DecodeHypothesis(
                    tokens=[*hyp.tokens, int(next_id)],
                    sum_logprob=float(hyp.sum_logprob + token_logprob),
                    next_logits=None,
                    position=int(hyp.position + 1),
                )
                if next_id == self.eos:
                    finished.append(child)
                    continue
                parent_indices.append(parent_idx)
                next_tokens.append(int(next_id))
                positions.append(int(hyp.position))
                next_active.append(child)
            if not next_tokens:
                active = []
                break
            parent_k, parent_v = self._gather_self_kv(self_k, self_v, parent_indices)
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
        return self._finalize_attempt(
            [*finished, *active],
            no_speech_prob,
            temperature,
            strategy,
            length_penalty,
        )

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
                break
            finished_candidates: list[DecodeHypothesis] = []
            active_candidates: list[tuple[float, int, int, int, DecodeHypothesis]] = []
            per_hyp_topk = max(beam_size + 1, 2)
            for parent_idx, hyp in enumerate(active):
                scores = self.filter_logits(hyp.next_logits, hyp.tokens, timestamps)
                if np.all(np.isneginf(scores)):
                    finished.append(hyp)
                    continue
                logprobs = _log_softmax(scores)
                for next_id in self._top_indices(logprobs, per_hyp_topk).tolist():
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
            if finished_candidates:
                finished.extend(finished_candidates)
            if not active_candidates:
                active = []
                break
            active_candidates.sort(key=lambda x: x[0], reverse=True)
            survivors = active_candidates[:beam_size]
            parent_indices = [cand[1] for cand in survivors]
            next_tokens = [cand[2] for cand in survivors]
            positions = [cand[3] for cand in survivors]
            next_active = [cand[4] for cand in survivors]
            parent_k, parent_v = self._gather_self_kv(self_k, self_v, parent_indices)
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
        return self._finalize_attempt(
            [*finished, *active],
            no_speech_prob,
            0.0,
            "beam_search",
            length_penalty,
        )

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

    def _should_fallback(self, attempt: DecodeAttempt, cfg: DecodeConfig) -> bool:
        if (
            cfg.compression_ratio_threshold is not None
            and attempt.compression_ratio > float(cfg.compression_ratio_threshold)
        ):
            return True
        if cfg.logprob_threshold is not None and attempt.avg_logprob < float(
            cfg.logprob_threshold
        ):
            return True
        return False

    def decode_window(
        self,
        cross_k,
        cross_v,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        timestamps: bool,
        cfg: DecodeConfig,
    ) -> list[int]:
        temperatures = self._temperature_schedule(cfg.temperature, cfg.temperature_inc)
        last_attempt: DecodeAttempt | None = None
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
            if self._should_skip_silence(attempt, cfg):
                return []
            if temperature != temperatures[-1] and self._should_fallback(attempt, cfg):
                continue
            return attempt.tokens
        return [] if last_attempt is None else list(last_attempt.tokens)

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
    ):
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
            ids = self.decode_window(
                cross_k,
                cross_v,
                prompt,
                self.max_new_tokens(max_new_tokens, len(prompt)),
                timestamps,
                cfg,
            )
            window_segments, advance = self.build_segments(
                ids, seek / float(sr), valid, timestamps
            )
            for s in window_segments:
                s["id"] = len(segments)
                segments.append(s)
                if condition_on_previous_text:
                    history.extend(int(x) for x in s["tokens"])
            seek += int(
                (advance if timestamps else valid)
                or min(valid, int(self.meta["samples_per_timestamp"]))
            )
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
