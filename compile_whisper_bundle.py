# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import math
from pathlib import Path

import numpy as np
import tokenizers_tvm_ffi
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.tokenization_whisper import LANGUAGES, TO_LANGUAGE_CODE
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

import tvm

SAMPLE_RATE = 16000
CHUNK_LENGTH = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH  # 480000
N_FFT = 400
HOP_LENGTH = 160
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000
N_FREQ = 1 + N_FFT // 2  # 201
REFLECT_PAD = N_FFT // 2  # 200

PREPROCESS_DTYPE = "float32"
MASK_DTYPE = "float32"
LOGITS_DTYPE = "float32"
ATTN_SOFTMAX_DTYPE = "float32"


def normalize_dtype_name(dtype: str) -> str:
    dtype_name = str(dtype).strip().lower()
    if dtype_name not in {"float32", "float16"}:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_name


def numpy_dtype(dtype: str):
    return np.dtype(normalize_dtype_name(dtype))


def tensor_const(data, dtype: str | None = None) -> Tensor:
    if dtype is None:
        return Tensor.from_const(np.asarray(data))
    return Tensor.from_const(np.asarray(data, dtype=numpy_dtype(dtype)))


def tensor_scalar(value: float, dtype: str) -> Tensor:
    return Tensor.from_scalar(float(value), normalize_dtype_name(dtype))


def maybe_cast(x: Tensor, dtype: str) -> Tensor:
    dtype = normalize_dtype_name(dtype)
    return x if x.dtype == dtype else op.astype(x, dtype)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compile WhisperBundle into a single Relax executable (.so) plus native TVM params (.params), "
            "metadata.json, and tokenizer.json for tokenizers_tvm_ffi. "
            "This step requires transformers/torch."
        )
    )
    parser.add_argument("--model-id", default="openai/whisper-tiny")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
        help=(
            "Compile dtype for encoder/decoder weights and KV cache. "
            "float16 is currently supported only with --target cuda. "
            f"Preprocess, mask, and logits remain {PREPROCESS_DTYPE}/{MASK_DTYPE}/{LOGITS_DTYPE}."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Suggested runtime default only. The compiled decoder context uses config.max_target_positions.",
    )
    parser.add_argument(
        "--max-decode-batch",
        type=int,
        default=8,
        help=(
            "Maximum decode batch baked into decode_step. Runtime beam_size/best_of "
            "must be <= this value."
        ),
    )
    return parser.parse_args()


def build_target(name: str):
    if name == "cuda":
        return tvm.target.Target.from_device("cuda")
    return tvm.target.Target.from_device("cpu")


def get_s_tir_pipeline():
    return tvm.transform.Sequential(
        [
            tvm.s_tir.transform.DefaultGPUSchedule(),
            tvm.s_tir.pipeline.default_s_tir_pipeline(),
        ]
    )


def token_id_or_none(tokenizer, token_text: str):
    if hasattr(tokenizer, "token_to_id"):
        token_id = int(tokenizer.token_to_id(token_text))
        return token_id if token_id >= 0 else None

    token_id = tokenizer.convert_tokens_to_ids(token_text)
    if token_id is None:
        return None
    token_id = int(token_id)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is not None and token_id == int(unk_token_id):
        unk_token = getattr(tokenizer, "unk_token", None)
        if token_text != unk_token:
            return None
    return token_id


def build_language_token_ids(tokenizer):
    token_ids = {}
    for code in LANGUAGES.keys():
        token_id = token_id_or_none(tokenizer, f"<|{code}|>")
        if token_id is not None:
            token_ids[str(code)] = int(token_id)
    return dict(sorted(token_ids.items()))


def build_language_alias_to_code(language_token_ids: dict[str, int]):
    aliases = {}
    for alias, code in TO_LANGUAGE_CODE.items():
        if code in language_token_ids:
            aliases[str(alias).lower()] = str(code)
    for code in language_token_ids:
        aliases[str(code).lower()] = str(code)
        language_name = LANGUAGES.get(code)
        if language_name is not None:
            aliases[str(language_name).lower()] = str(code)
    return dict(sorted(aliases.items()))


def export_hf_tokenizer_json(tokenizer, out_dir: Path) -> tuple[Path, bytes]:
    out_path = out_dir / "tokenizer.json"

    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is not None and hasattr(backend_tokenizer, "to_str"):
        payload = backend_tokenizer.to_str()
        out_path.write_text(payload, encoding="utf-8")
        return out_path, payload.encode("utf-8")

    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(out_dir)
        if out_path.exists():
            return out_path, out_path.read_bytes()

    tokenizer_file = getattr(tokenizer, "init_kwargs", {}).get("tokenizer_file")
    if tokenizer_file is not None:
        tokenizer_file = Path(tokenizer_file)
        if tokenizer_file.exists():
            payload = tokenizer_file.read_bytes()
            out_path.write_bytes(payload)
            return out_path, payload

    raise RuntimeError(
        "Failed to export tokenizer.json from the Hugging Face tokenizer. "
        "Please use a fast tokenizer with tokenizer.json available."
    )


def build_metadata_tokenizer(hf_tokenizer, tokenizer_json_bytes: bytes):
    if tokenizers_tvm_ffi is not None:
        return tokenizers_tvm_ffi.Tokenizer.from_json_bytes(tokenizer_json_bytes)
    return hf_tokenizer


def extract_special_ids(tokenizer, tokenizer_json_bytes: bytes) -> list[int]:
    all_special_ids = getattr(tokenizer, "all_special_ids", None)
    if all_special_ids:
        return unique_sorted_token_ids(all_special_ids)

    payload = json.loads(tokenizer_json_bytes.decode("utf-8"))
    ids = []
    for item in payload.get("added_tokens", []):
        if bool(item.get("special")) and "id" in item:
            ids.append(int(item["id"]))
    return unique_sorted_token_ids(ids)


def add_axis0(x: Tensor):
    shape = [1] + list(x.shape)
    return op.reshape(x, shape)


# -----------------------------------------------------------------------------
# TVM-native preprocess
# -----------------------------------------------------------------------------
def make_reflect_indices(
    n_samples: int = N_SAMPLES, pad: int = REFLECT_PAD
) -> np.ndarray:
    left = np.arange(pad, 0, -1, dtype=np.int32)
    center = np.arange(n_samples, dtype=np.int32)
    right = np.arange(n_samples - 2, n_samples - pad - 2, -1, dtype=np.int32)
    return np.concatenate([left, center, right], axis=0)


def make_periodic_hann(n_fft: int = N_FFT) -> np.ndarray:
    n = np.arange(n_fft, dtype=np.float32)
    return 0.5 - 0.5 * np.cos((2.0 * np.pi * n) / float(n_fft))


def make_stft_conv_kernels(n_fft: int = N_FFT, n_freq: int = N_FREQ):
    n = np.arange(n_fft, dtype=np.float32)
    k = np.arange(n_freq, dtype=np.float32)[:, None]
    window = make_periodic_hann(n_fft)[None, :]
    phase = (2.0 * np.pi / float(n_fft)) * (k * n[None, :])
    real = np.cos(phase) * window
    imag = -np.sin(phase) * window
    return real.astype(np.float32)[:, None, :], imag.astype(np.float32)[:, None, :]


class WhisperPreprocessTVM(nn.Module):
    def __init__(self, mel_filters: np.ndarray):
        super().__init__()
        mel_filters = np.asarray(mel_filters, dtype=numpy_dtype(PREPROCESS_DTYPE))
        if mel_filters.ndim != 2:
            raise ValueError(f"Unexpected mel filter rank: {mel_filters.shape}")
        if mel_filters.shape[0] != N_FREQ and mel_filters.shape[1] == N_FREQ:
            mel_filters = mel_filters.T
        if mel_filters.shape[0] != N_FREQ:
            raise ValueError(f"Unexpected mel filter shape: {mel_filters.shape}")
        self.num_mel_bins = int(mel_filters.shape[1])

        real_kernel, imag_kernel = make_stft_conv_kernels()
        self.reflect_indices = tensor_const(make_reflect_indices())
        self.keep_frame_indices = tensor_const(np.arange(N_FRAMES, dtype=np.int32))
        self.frame_starts = tensor_const(
            np.arange(0, N_SAMPLES, HOP_LENGTH, dtype=np.int32).reshape(1, N_FRAMES)
        )
        self.real_kernel = tensor_const(real_kernel, PREPROCESS_DTYPE)
        self.imag_kernel = tensor_const(imag_kernel, PREPROCESS_DTYPE)
        self.mel_filters = tensor_const(mel_filters, PREPROCESS_DTYPE)
        self.log_eps = tensor_scalar(1e-10, PREPROCESS_DTYPE)
        self.inv_ln10 = tensor_scalar(1.0 / math.log(10.0), PREPROCESS_DTYPE)
        self.eight = tensor_scalar(8.0, PREPROCESS_DTYPE)
        self.four = tensor_scalar(4.0, PREPROCESS_DTYPE)

    def forward(self, waveform: Tensor, valid_samples: Tensor):
        x = op.take(waveform, self.reflect_indices, axis=1)
        x = op.reshape(x, [1, 1, N_SAMPLES + 2 * REFLECT_PAD])
        real = op.conv1d(x, self.real_kernel, stride=HOP_LENGTH, padding=0)
        imag = op.conv1d(x, self.imag_kernel, stride=HOP_LENGTH, padding=0)
        power = op.add(op.square(real), op.square(imag))
        power = op.take(power, self.keep_frame_indices, axis=2)
        power_t = op.permute_dims(power, axes=[0, 2, 1])
        mel = op.matmul(power_t, self.mel_filters)
        mel = op.permute_dims(mel, axes=[0, 2, 1])
        log_spec = op.log(op.maximum(mel, self.log_eps))
        log_spec = op.multiply(log_spec, self.inv_ln10)
        max_val = op.max(log_spec, axis=[1, 2], keepdims=True)
        log_spec = op.maximum(log_spec, op.subtract(max_val, self.eight))
        input_features = op.divide(op.add(log_spec, self.four), self.four)
        valid_samples_2d = op.broadcast_to(
            op.reshape(valid_samples, [1, 1]), [1, N_FRAMES]
        )
        feature_attention_mask = op.astype(
            op.less(self.frame_starts, valid_samples_2d), "int32"
        )
        return input_features, feature_attention_mask


# -----------------------------------------------------------------------------
# TVM-native encoder
# -----------------------------------------------------------------------------
class WhisperAttentionTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dtype: str = "float32"):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.compute_dtype = normalize_dtype_name(dtype)
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=self.compute_dtype
        )
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=False, dtype=self.compute_dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=self.compute_dtype
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=self.compute_dtype
        )
        self.scale = tensor_scalar(self.head_dim**-0.5, self.compute_dtype)

    def _reshape_qkv(self, x: Tensor):
        bsz, seq_len, _ = x.shape
        x = op.reshape(x, [bsz, seq_len, self.num_heads, self.head_dim])
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        return x

    def _merge_heads(self, x: Tensor):
        bsz, num_heads, seq_len, head_dim = x.shape
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        x = op.reshape(x, [bsz, seq_len, num_heads * head_dim])
        return x

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None):
        query_states = op.multiply(self.q_proj(hidden_states), self.scale)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = self._reshape_qkv(query_states)
        key_states = self._reshape_qkv(key_states)
        value_states = self._reshape_qkv(value_states)
        attn_weights = op.matmul(
            query_states, op.permute_dims(key_states, axes=[0, 1, 3, 2])
        )
        attn_weights = maybe_cast(attn_weights, ATTN_SOFTMAX_DTYPE)
        if attention_mask is not None:
            attn_weights = op.add(
                attn_weights, maybe_cast(attention_mask, ATTN_SOFTMAX_DTYPE)
            )
        attn_weights = op.softmax(attn_weights, axis=-1)
        attn_weights = maybe_cast(attn_weights, self.compute_dtype)
        attn_output = op.matmul(attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        return self.out_proj(attn_output)


class WhisperEncoderLayerTVM(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, ffn_dim: int, dtype: str = "float32"
    ):
        super().__init__()
        self.compute_dtype = normalize_dtype_name(dtype)
        self.self_attn = WhisperAttentionTVM(
            embed_dim, num_heads, dtype=self.compute_dtype
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            embed_dim, eps=1e-5, dtype=self.compute_dtype
        )
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=True, dtype=self.compute_dtype)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=True, dtype=self.compute_dtype)
        self.final_layer_norm = nn.LayerNorm(
            embed_dim, eps=1e-5, dtype=self.compute_dtype
        )

    def forward(self, hidden_states: Tensor):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=None)
        hidden_states = op.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = op.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = op.add(residual, hidden_states)
        return hidden_states


class WhisperEncoderTVM(nn.Module):
    def __init__(self, config, dtype: str = "float32"):
        super().__init__()
        self.compute_dtype = normalize_dtype_name(dtype)
        self.num_mel_bins = int(config.num_mel_bins)
        self.d_model = int(config.d_model)
        self.max_source_positions = int(config.max_source_positions)
        self.encoder_layers = int(config.encoder_layers)
        self.encoder_attention_heads = int(config.encoder_attention_heads)
        self.encoder_ffn_dim = int(config.encoder_ffn_dim)
        self.conv1 = nn.Conv1D(
            self.num_mel_bins,
            self.d_model,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            dtype=self.compute_dtype,
        )
        self.conv2 = nn.Conv1D(
            self.d_model,
            self.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
            dtype=self.compute_dtype,
        )
        self.embed_positions = nn.Embedding(
            self.max_source_positions,
            self.d_model,
            dtype=self.compute_dtype,
        )
        self.layers = []
        for i in range(self.encoder_layers):
            layer = WhisperEncoderLayerTVM(
                self.d_model,
                self.encoder_attention_heads,
                self.encoder_ffn_dim,
                dtype=self.compute_dtype,
            )
            setattr(self, f"layer_{i}", layer)
            self.layers.append(layer)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-5, dtype=self.compute_dtype)
        self.position_ids = tensor_const(
            np.arange(self.max_source_positions, dtype=np.int32)[None, :]
        )

    def forward(self, input_features: Tensor):
        input_features = maybe_cast(input_features, self.compute_dtype)
        hidden_states = self.conv1(input_features)
        hidden_states = op.gelu(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = op.gelu(hidden_states)
        hidden_states = op.permute_dims(hidden_states, axes=[0, 2, 1])
        hidden_states = op.add(hidden_states, self.embed_positions(self.position_ids))
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.layer_norm(hidden_states)


# -----------------------------------------------------------------------------
# TVM-native cross-KV and cached decoder-step
# -----------------------------------------------------------------------------
class WhisperCrossKVLayerTVM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dtype: str = "float32"):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.compute_dtype = normalize_dtype_name(dtype)
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=False, dtype=self.compute_dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=self.compute_dtype
        )

    def _reshape_qkv(self, x: Tensor):
        bsz, seq_len, _ = x.shape
        x = op.reshape(x, [bsz, seq_len, self.num_heads, self.head_dim])
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        return x

    def forward(self, encoder_hidden_states: Tensor):
        k = self._reshape_qkv(self.k_proj(encoder_hidden_states))
        v = self._reshape_qkv(self.v_proj(encoder_hidden_states))
        return k, v


class WhisperCrossKVCachedTVM(nn.Module):
    def __init__(self, config, dtype: str = "float32"):
        super().__init__()
        self.compute_dtype = normalize_dtype_name(dtype)
        self.d_model = int(config.d_model)
        self.decoder_layers = int(config.decoder_layers)
        self.decoder_attention_heads = int(config.decoder_attention_heads)
        self.head_dim = self.d_model // self.decoder_attention_heads
        self.max_source_positions = int(config.max_source_positions)
        self.layers = []
        for i in range(self.decoder_layers):
            layer = WhisperCrossKVLayerTVM(
                self.d_model,
                self.decoder_attention_heads,
                dtype=self.compute_dtype,
            )
            setattr(self, f"layer_{i}", layer)
            self.layers.append(layer)

    def forward(self, encoder_hidden_states: Tensor):
        all_k = []
        all_v = []
        for layer in self.layers:
            k, v = layer(encoder_hidden_states)
            all_k.append(add_axis0(k))
            all_v.append(add_axis0(v))
        return op.concat(all_k, dim=0), op.concat(all_v, dim=0)


class WhisperSelfAttentionStepTVM(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_past_len: int,
        dtype: str = "float32",
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_past_len = max_past_len
        self.compute_dtype = normalize_dtype_name(dtype)
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=self.compute_dtype
        )
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=False, dtype=self.compute_dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=self.compute_dtype
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=self.compute_dtype
        )
        self.scale = tensor_scalar(self.head_dim**-0.5, self.compute_dtype)
        self.zero_mask = tensor_const(
            np.zeros((1, 1, 1, 1), dtype=np.float32), MASK_DTYPE
        )
        self.shift_indices = tensor_const(np.arange(1, max_past_len, dtype=np.int32))

    def _reshape_qkv(self, x: Tensor):
        bsz, seq_len, _ = x.shape
        x = op.reshape(x, [bsz, seq_len, self.num_heads, self.head_dim])
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        return x

    def _merge_heads(self, x: Tensor):
        bsz, num_heads, seq_len, head_dim = x.shape
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        x = op.reshape(x, [bsz, seq_len, num_heads * head_dim])
        return x

    def forward(
        self,
        hidden_states: Tensor,
        past_k: Tensor,
        past_v: Tensor,
        past_keep_mask: Tensor,
    ):
        bsz = hidden_states.shape[0]
        query_states = op.multiply(self.q_proj(hidden_states), self.scale)
        new_k = self.k_proj(hidden_states)
        new_v = self.v_proj(hidden_states)
        query_states = self._reshape_qkv(query_states)
        new_k = self._reshape_qkv(new_k)
        new_v = self._reshape_qkv(new_v)
        all_k = op.concat([past_k, new_k], dim=2)
        all_v = op.concat([past_v, new_v], dim=2)
        attn_mask = op.concat(
            [past_keep_mask, op.broadcast_to(self.zero_mask, [bsz, 1, 1, 1])],
            dim=-1,
        )
        attn_weights = op.matmul(
            query_states, op.permute_dims(all_k, axes=[0, 1, 3, 2])
        )
        attn_weights = maybe_cast(attn_weights, ATTN_SOFTMAX_DTYPE)
        attn_weights = op.add(attn_weights, attn_mask)
        attn_weights = op.softmax(attn_weights, axis=-1)
        attn_weights = maybe_cast(attn_weights, self.compute_dtype)
        attn_output = op.matmul(attn_weights, all_v)
        attn_output = self._merge_heads(attn_output)
        updated_k = op.concat(
            [op.take(past_k, self.shift_indices, axis=2), new_k], dim=2
        )
        updated_v = op.concat(
            [op.take(past_v, self.shift_indices, axis=2), new_v], dim=2
        )
        return self.out_proj(attn_output), updated_k, updated_v


class WhisperCrossAttentionStepTVM(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_source_positions: int,
        dtype: str = "float32",
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_source_positions = max_source_positions
        self.compute_dtype = normalize_dtype_name(dtype)
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=self.compute_dtype
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=self.compute_dtype
        )
        self.scale = tensor_scalar(self.head_dim**-0.5, self.compute_dtype)

    def _reshape_qkv(self, x: Tensor):
        bsz, seq_len, _ = x.shape
        x = op.reshape(x, [bsz, seq_len, self.num_heads, self.head_dim])
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        return x

    def _merge_heads(self, x: Tensor):
        bsz, num_heads, seq_len, head_dim = x.shape
        x = op.permute_dims(x, axes=[0, 2, 1, 3])
        x = op.reshape(x, [bsz, seq_len, num_heads * head_dim])
        return x

    def forward(self, hidden_states: Tensor, cross_k: Tensor, cross_v: Tensor):
        bsz = hidden_states.shape[0]
        query_states = op.multiply(self.q_proj(hidden_states), self.scale)
        query_states = self._reshape_qkv(query_states)
        cross_k = op.broadcast_to(
            cross_k,
            [bsz, self.num_heads, self.max_source_positions, self.head_dim],
        )
        cross_v = op.broadcast_to(
            cross_v,
            [bsz, self.num_heads, self.max_source_positions, self.head_dim],
        )
        attn_weights = op.matmul(
            query_states, op.permute_dims(cross_k, axes=[0, 1, 3, 2])
        )
        attn_weights = maybe_cast(attn_weights, ATTN_SOFTMAX_DTYPE)
        attn_weights = op.softmax(attn_weights, axis=-1)
        attn_weights = maybe_cast(attn_weights, self.compute_dtype)
        attn_output = op.matmul(attn_weights, cross_v)
        attn_output = self._merge_heads(attn_output)
        return self.out_proj(attn_output)


class WhisperDecoderLayerStepTVM(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        max_past_len: int,
        max_source_positions: int,
        dtype: str = "float32",
    ):
        super().__init__()
        self.compute_dtype = normalize_dtype_name(dtype)
        self.self_attn = WhisperSelfAttentionStepTVM(
            embed_dim,
            num_heads,
            max_past_len,
            dtype=self.compute_dtype,
        )
        self.encoder_attn = WhisperCrossAttentionStepTVM(
            embed_dim,
            num_heads,
            max_source_positions=max_source_positions,
            dtype=self.compute_dtype,
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            embed_dim, eps=1e-5, dtype=self.compute_dtype
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(
            embed_dim, eps=1e-5, dtype=self.compute_dtype
        )
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=True, dtype=self.compute_dtype)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=True, dtype=self.compute_dtype)
        self.final_layer_norm = nn.LayerNorm(
            embed_dim, eps=1e-5, dtype=self.compute_dtype
        )

    def forward(
        self,
        hidden_states: Tensor,
        past_k: Tensor,
        past_v: Tensor,
        past_keep_mask: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, new_k, new_v = self.self_attn(
            hidden_states, past_k, past_v, past_keep_mask
        )
        hidden_states = op.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(hidden_states, cross_k, cross_v)
        hidden_states = op.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = op.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = op.add(residual, hidden_states)
        return hidden_states, new_k, new_v


class WhisperDecoderCachedStepTVM(nn.Module):
    def __init__(
        self,
        config,
        max_dec_len: int,
        max_decode_batch: int,
        dtype: str = "float32",
    ):
        super().__init__()
        self.compute_dtype = normalize_dtype_name(dtype)
        self.vocab_size = int(config.vocab_size)
        self.d_model = int(config.d_model)
        self.max_target_positions = int(config.max_target_positions)
        self.decoder_layers = int(config.decoder_layers)
        self.decoder_attention_heads = int(config.decoder_attention_heads)
        self.decoder_ffn_dim = int(config.decoder_ffn_dim)
        self.max_source_positions = int(config.max_source_positions)
        self.max_dec_len = max_dec_len
        self.max_past_len = max_dec_len - 1
        self.max_decode_batch = max_decode_batch
        self.head_dim = self.d_model // self.decoder_attention_heads
        if max_dec_len > self.max_target_positions:
            raise ValueError("max_dec_len exceeds Whisper config max_target_positions")
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.d_model,
            dtype=self.compute_dtype,
        )
        self.embed_positions = nn.Embedding(
            self.max_target_positions,
            self.d_model,
            dtype=self.compute_dtype,
        )
        self.layers = []
        self.layer_take_indices = []
        for i in range(self.decoder_layers):
            layer = WhisperDecoderLayerStepTVM(
                self.d_model,
                self.decoder_attention_heads,
                self.decoder_ffn_dim,
                self.max_past_len,
                self.max_source_positions,
                dtype=self.compute_dtype,
            )
            setattr(self, f"layer_{i}", layer)
            self.layers.append(layer)
            self.layer_take_indices.append(
                tensor_const(np.asarray([i], dtype=np.int32))
            )
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-5, dtype=self.compute_dtype)
        self.proj_out = nn.Linear(
            self.d_model,
            self.vocab_size,
            bias=False,
            dtype=self.compute_dtype,
            out_dtype=LOGITS_DTYPE,
        )

    def _take_layer(self, stacked: Tensor, layer_index: Tensor, out_shape):
        x = op.take(stacked, layer_index, axis=0)
        return op.reshape(x, out_shape)

    def forward(
        self,
        token_id: Tensor,
        position_id: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        past_keep_mask: Tensor,
        cross_k_cache: Tensor,
        cross_v_cache: Tensor,
    ):
        hidden_states = self.embed_tokens(token_id)
        hidden_states = op.add(hidden_states, self.embed_positions(position_id))
        new_k_list = []
        new_v_list = []
        for i, layer in enumerate(self.layers):
            idx = self.layer_take_indices[i]
            past_k = self._take_layer(
                self_k_cache,
                idx,
                [
                    self.max_decode_batch,
                    self.decoder_attention_heads,
                    self.max_past_len,
                    self.head_dim,
                ],
            )
            past_v = self._take_layer(
                self_v_cache,
                idx,
                [
                    self.max_decode_batch,
                    self.decoder_attention_heads,
                    self.max_past_len,
                    self.head_dim,
                ],
            )
            cross_k = self._take_layer(
                cross_k_cache,
                idx,
                [
                    1,
                    self.decoder_attention_heads,
                    self.max_source_positions,
                    self.head_dim,
                ],
            )
            cross_v = self._take_layer(
                cross_v_cache,
                idx,
                [
                    1,
                    self.decoder_attention_heads,
                    self.max_source_positions,
                    self.head_dim,
                ],
            )
            hidden_states, new_k, new_v = layer(
                hidden_states, past_k, past_v, past_keep_mask, cross_k, cross_v
            )
            new_k_list.append(add_axis0(new_k))
            new_v_list.append(add_axis0(new_v))
        hidden_states = self.layer_norm(hidden_states)
        logits = self.proj_out(hidden_states)
        logits = op.reshape(logits, [self.max_decode_batch, self.vocab_size])
        return logits, op.concat(new_k_list, dim=0), op.concat(new_v_list, dim=0)


# -----------------------------------------------------------------------------
# WhisperBundle
# -----------------------------------------------------------------------------
class WhisperBundle(nn.Module):
    def __init__(
        self,
        config,
        mel_filters: np.ndarray,
        max_dec_len: int,
        max_decode_batch: int,
        dtype: str = "float32",
    ):
        super().__init__()
        self.compute_dtype = normalize_dtype_name(dtype)
        self.cache_dtype = self.compute_dtype
        self.preprocess_dtype = PREPROCESS_DTYPE
        self.mask_dtype = MASK_DTYPE
        self.logits_dtype = LOGITS_DTYPE
        self.preprocess_mod = WhisperPreprocessTVM(mel_filters)
        self.encoder_mod = WhisperEncoderTVM(config, dtype=self.compute_dtype)
        self.cross_kv_mod = WhisperCrossKVCachedTVM(config, dtype=self.compute_dtype)
        self.decoder_step_mod = WhisperDecoderCachedStepTVM(
            config,
            max_dec_len=max_dec_len,
            max_decode_batch=max_decode_batch,
            dtype=self.compute_dtype,
        )

        self.num_mel_bins = int(config.num_mel_bins)
        self.decoder_layers = int(config.decoder_layers)
        self.decoder_attention_heads = int(config.decoder_attention_heads)
        self.head_dim = int(config.d_model) // int(config.decoder_attention_heads)
        self.max_source_positions = int(config.max_source_positions)
        self.max_dec_len = int(max_dec_len)
        self.max_decode_batch = int(max_decode_batch)
        self.max_past_len = self.max_dec_len - 1

    def preprocess(self, waveform: Tensor, valid_samples: Tensor):
        return self.preprocess_mod(waveform, valid_samples)

    def encode(self, input_features: Tensor):
        return self.encoder_mod(input_features)

    def cross_kv(self, encoder_hidden_states: Tensor):
        return self.cross_kv_mod(encoder_hidden_states)

    def decode_step(
        self,
        token_id: Tensor,
        position_id: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        past_keep_mask: Tensor,
        cross_k_cache: Tensor,
        cross_v_cache: Tensor,
    ):
        return self.decoder_step_mod(
            token_id,
            position_id,
            self_k_cache,
            self_v_cache,
            past_keep_mask,
            cross_k_cache,
            cross_v_cache,
        )

    def gather_self_kv(
        self,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        beam_indices: Tensor,
    ):
        return (
            op.take(self_k_cache, beam_indices, axis=1),
            op.take(self_v_cache, beam_indices, axis=1),
        )

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "preprocess": {
                    "waveform": nn.spec.Tensor([1, N_SAMPLES], self.preprocess_dtype),
                    "valid_samples": nn.spec.Tensor([1], "int32"),
                    "$": {"param_mode": "none", "effect_mode": "none"},
                },
                "encode": {
                    "input_features": nn.spec.Tensor(
                        [1, self.num_mel_bins, N_FRAMES], self.preprocess_dtype
                    ),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "cross_kv": {
                    "encoder_hidden_states": nn.spec.Tensor(
                        [
                            1,
                            self.max_source_positions,
                            self.decoder_attention_heads * self.head_dim,
                        ],
                        self.compute_dtype,
                    ),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "decode_step": {
                    "token_id": nn.spec.Tensor([self.max_decode_batch, 1], "int32"),
                    "position_id": nn.spec.Tensor([self.max_decode_batch, 1], "int32"),
                    "self_k_cache": nn.spec.Tensor(
                        [
                            self.decoder_layers,
                            self.max_decode_batch,
                            self.decoder_attention_heads,
                            self.max_past_len,
                            self.head_dim,
                        ],
                        self.cache_dtype,
                    ),
                    "self_v_cache": nn.spec.Tensor(
                        [
                            self.decoder_layers,
                            self.max_decode_batch,
                            self.decoder_attention_heads,
                            self.max_past_len,
                            self.head_dim,
                        ],
                        self.cache_dtype,
                    ),
                    "past_keep_mask": nn.spec.Tensor(
                        [self.max_decode_batch, 1, 1, self.max_past_len],
                        self.mask_dtype,
                    ),
                    "cross_k_cache": nn.spec.Tensor(
                        [
                            self.decoder_layers,
                            1,
                            self.decoder_attention_heads,
                            self.max_source_positions,
                            self.head_dim,
                        ],
                        self.cache_dtype,
                    ),
                    "cross_v_cache": nn.spec.Tensor(
                        [
                            self.decoder_layers,
                            1,
                            self.decoder_attention_heads,
                            self.max_source_positions,
                            self.head_dim,
                        ],
                        self.cache_dtype,
                    ),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "gather_self_kv": {
                    "self_k_cache": nn.spec.Tensor(
                        [
                            self.decoder_layers,
                            self.max_decode_batch,
                            self.decoder_attention_heads,
                            self.max_past_len,
                            self.head_dim,
                        ],
                        self.cache_dtype,
                    ),
                    "self_v_cache": nn.spec.Tensor(
                        [
                            self.decoder_layers,
                            self.max_decode_batch,
                            self.decoder_attention_heads,
                            self.max_past_len,
                            self.head_dim,
                        ],
                        self.cache_dtype,
                    ),
                    "beam_indices": nn.spec.Tensor([self.max_decode_batch], "int32"),
                    "$": {"param_mode": "none", "effect_mode": "none"},
                },
            },  # ty:ignore[invalid-argument-type]
            self,
        )


# -----------------------------------------------------------------------------
# Weight copy
# -----------------------------------------------------------------------------
def bind_param_from_torch(param: nn.Parameter, tensor: torch.Tensor):
    arr = tensor.detach().cpu().numpy()
    if str(arr.dtype) != param.dtype:
        arr = arr.astype(param.dtype)
    param.data = tvm.runtime.tensor(arr)


def copy_encoder_weights_from_hf(
    tvm_encoder: WhisperEncoderTVM, hf_model: WhisperForConditionalGeneration
):
    hf_state = hf_model.state_dict()
    bind_param_from_torch(
        tvm_encoder.conv1.weight, hf_state["model.encoder.conv1.weight"]
    )
    bind_param_from_torch(tvm_encoder.conv1.bias, hf_state["model.encoder.conv1.bias"])
    bind_param_from_torch(
        tvm_encoder.conv2.weight, hf_state["model.encoder.conv2.weight"]
    )
    bind_param_from_torch(tvm_encoder.conv2.bias, hf_state["model.encoder.conv2.bias"])
    bind_param_from_torch(
        tvm_encoder.embed_positions.weight,
        hf_state["model.encoder.embed_positions.weight"],
    )
    bind_param_from_torch(
        tvm_encoder.layer_norm.weight, hf_state["model.encoder.layer_norm.weight"]
    )
    bind_param_from_torch(
        tvm_encoder.layer_norm.bias, hf_state["model.encoder.layer_norm.bias"]
    )
    for i, layer in enumerate(tvm_encoder.layers):
        prefix = f"model.encoder.layers.{i}."
        bind_param_from_torch(
            layer.self_attn.q_proj.weight, hf_state[prefix + "self_attn.q_proj.weight"]
        )
        bind_param_from_torch(
            layer.self_attn.q_proj.bias, hf_state[prefix + "self_attn.q_proj.bias"]
        )
        bind_param_from_torch(
            layer.self_attn.k_proj.weight, hf_state[prefix + "self_attn.k_proj.weight"]
        )
        bind_param_from_torch(
            layer.self_attn.v_proj.weight, hf_state[prefix + "self_attn.v_proj.weight"]
        )
        bind_param_from_torch(
            layer.self_attn.v_proj.bias, hf_state[prefix + "self_attn.v_proj.bias"]
        )
        bind_param_from_torch(
            layer.self_attn.out_proj.weight,
            hf_state[prefix + "self_attn.out_proj.weight"],
        )
        bind_param_from_torch(
            layer.self_attn.out_proj.bias, hf_state[prefix + "self_attn.out_proj.bias"]
        )
        bind_param_from_torch(
            layer.self_attn_layer_norm.weight,
            hf_state[prefix + "self_attn_layer_norm.weight"],
        )
        bind_param_from_torch(
            layer.self_attn_layer_norm.bias,
            hf_state[prefix + "self_attn_layer_norm.bias"],
        )
        bind_param_from_torch(layer.fc1.weight, hf_state[prefix + "fc1.weight"])
        bind_param_from_torch(layer.fc1.bias, hf_state[prefix + "fc1.bias"])
        bind_param_from_torch(layer.fc2.weight, hf_state[prefix + "fc2.weight"])
        bind_param_from_torch(layer.fc2.bias, hf_state[prefix + "fc2.bias"])
        bind_param_from_torch(
            layer.final_layer_norm.weight, hf_state[prefix + "final_layer_norm.weight"]
        )
        bind_param_from_torch(
            layer.final_layer_norm.bias, hf_state[prefix + "final_layer_norm.bias"]
        )


def copy_cross_kv_weights_from_hf(
    tvm_cross: WhisperCrossKVCachedTVM, hf_model: WhisperForConditionalGeneration
):
    hf_state = hf_model.state_dict()
    for i, layer in enumerate(tvm_cross.layers):
        prefix = f"model.decoder.layers.{i}.encoder_attn."
        bind_param_from_torch(layer.k_proj.weight, hf_state[prefix + "k_proj.weight"])
        bind_param_from_torch(layer.v_proj.weight, hf_state[prefix + "v_proj.weight"])
        bind_param_from_torch(layer.v_proj.bias, hf_state[prefix + "v_proj.bias"])


def copy_decoder_step_weights_from_hf(
    tvm_decoder: WhisperDecoderCachedStepTVM, hf_model: WhisperForConditionalGeneration
):
    hf_state = hf_model.state_dict()
    bind_param_from_torch(
        tvm_decoder.embed_tokens.weight, hf_state["model.decoder.embed_tokens.weight"]
    )
    bind_param_from_torch(
        tvm_decoder.embed_positions.weight,
        hf_state["model.decoder.embed_positions.weight"],
    )
    bind_param_from_torch(
        tvm_decoder.layer_norm.weight, hf_state["model.decoder.layer_norm.weight"]
    )
    bind_param_from_torch(
        tvm_decoder.layer_norm.bias, hf_state["model.decoder.layer_norm.bias"]
    )
    bind_param_from_torch(tvm_decoder.proj_out.weight, hf_state["proj_out.weight"])
    for i, layer in enumerate(tvm_decoder.layers):
        prefix = f"model.decoder.layers.{i}."
        bind_param_from_torch(
            layer.self_attn.q_proj.weight, hf_state[prefix + "self_attn.q_proj.weight"]
        )
        bind_param_from_torch(
            layer.self_attn.q_proj.bias, hf_state[prefix + "self_attn.q_proj.bias"]
        )
        bind_param_from_torch(
            layer.self_attn.k_proj.weight, hf_state[prefix + "self_attn.k_proj.weight"]
        )
        bind_param_from_torch(
            layer.self_attn.v_proj.weight, hf_state[prefix + "self_attn.v_proj.weight"]
        )
        bind_param_from_torch(
            layer.self_attn.v_proj.bias, hf_state[prefix + "self_attn.v_proj.bias"]
        )
        bind_param_from_torch(
            layer.self_attn.out_proj.weight,
            hf_state[prefix + "self_attn.out_proj.weight"],
        )
        bind_param_from_torch(
            layer.self_attn.out_proj.bias, hf_state[prefix + "self_attn.out_proj.bias"]
        )
        bind_param_from_torch(
            layer.encoder_attn.q_proj.weight,
            hf_state[prefix + "encoder_attn.q_proj.weight"],
        )
        bind_param_from_torch(
            layer.encoder_attn.q_proj.bias,
            hf_state[prefix + "encoder_attn.q_proj.bias"],
        )
        bind_param_from_torch(
            layer.encoder_attn.out_proj.weight,
            hf_state[prefix + "encoder_attn.out_proj.weight"],
        )
        bind_param_from_torch(
            layer.encoder_attn.out_proj.bias,
            hf_state[prefix + "encoder_attn.out_proj.bias"],
        )
        bind_param_from_torch(
            layer.self_attn_layer_norm.weight,
            hf_state[prefix + "self_attn_layer_norm.weight"],
        )
        bind_param_from_torch(
            layer.self_attn_layer_norm.bias,
            hf_state[prefix + "self_attn_layer_norm.bias"],
        )
        bind_param_from_torch(
            layer.encoder_attn_layer_norm.weight,
            hf_state[prefix + "encoder_attn_layer_norm.weight"],
        )
        bind_param_from_torch(
            layer.encoder_attn_layer_norm.bias,
            hf_state[prefix + "encoder_attn_layer_norm.bias"],
        )
        bind_param_from_torch(layer.fc1.weight, hf_state[prefix + "fc1.weight"])
        bind_param_from_torch(layer.fc1.bias, hf_state[prefix + "fc1.bias"])
        bind_param_from_torch(layer.fc2.weight, hf_state[prefix + "fc2.weight"])
        bind_param_from_torch(layer.fc2.bias, hf_state[prefix + "fc2.bias"])
        bind_param_from_torch(
            layer.final_layer_norm.weight, hf_state[prefix + "final_layer_norm.weight"]
        )
        bind_param_from_torch(
            layer.final_layer_norm.bias, hf_state[prefix + "final_layer_norm.bias"]
        )


def save_params_tvm(named_params, out_path: Path):
    params = {}
    names = []
    for name, param in named_params:
        if param.data is None:
            raise ValueError(f"Parameter {name} is not bound")
        params[name] = param.data
        names.append(name)
    tvm.runtime.save_param_dict_to_file(params, str(out_path))
    return names


def unique_sorted_token_ids(values):
    token_ids = {int(v) for v in values if v is not None}
    return sorted(token_ids)


def summarize_parameter_dtypes(named_params):
    dtype_histogram = {}
    total_elements = 0
    total_bytes = 0
    for name, param in named_params:
        del name
        if param.data is None:
            raise ValueError("Found unbound parameter while summarizing dtypes")
        tensor = param.data
        dtype_name = str(tensor.dtype)
        numel = 1
        for dim in tensor.shape:
            numel *= int(dim)
        dtype_histogram[dtype_name] = int(dtype_histogram.get(dtype_name, 0) + numel)
        total_elements += numel
        total_bytes += int(numel * np.dtype(dtype_name).itemsize)
    return dtype_histogram, int(total_elements), int(total_bytes)


def choose_ftype_name(dtype_histogram: dict[str, int]) -> str:
    if not dtype_histogram:
        return "unknown"
    if len(dtype_histogram) == 1:
        return next(iter(dtype_histogram))
    return "mixed"


def main():
    args = parse_args()
    compile_dtype = normalize_dtype_name(args.dtype)
    if compile_dtype == "float16" and args.target != "cuda":
        raise ValueError(
            "--dtype float16 is currently supported only with --target cuda"
        )
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_id)
    hf_model = WhisperForConditionalGeneration.from_pretrained(args.model_id).eval()
    hf_tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer_json_path, tokenizer_json_bytes = export_hf_tokenizer_json(
        hf_tokenizer, out_dir
    )
    tokenizer = tokenizers_tvm_ffi.Tokenizer.from_json_bytes(tokenizer_json_bytes)
    special_ids = extract_special_ids(hf_tokenizer, tokenizer_json_bytes)

    decoder_start_token_id = int(hf_model.config.decoder_start_token_id)
    max_dec_len = int(hf_model.config.max_target_positions)
    max_new_tokens_default = max(1, min(int(args.max_new_tokens), max_dec_len - 1))

    bundle = WhisperBundle(
        hf_model.config,
        mel_filters=np.asarray(
            processor.feature_extractor.mel_filters,
            dtype=numpy_dtype(PREPROCESS_DTYPE),
        ),
        max_dec_len=max_dec_len,
        max_decode_batch=int(args.max_decode_batch),
        dtype=compile_dtype,
    )
    copy_encoder_weights_from_hf(bundle.encoder_mod, hf_model)
    copy_cross_kv_weights_from_hf(bundle.cross_kv_mod, hf_model)
    copy_decoder_step_weights_from_hf(bundle.decoder_step_mod, hf_model)

    target = build_target(args.target)
    mod, named_params = bundle.export_tvm(spec=bundle.get_default_spec())
    param_dtype_histogram, parameter_elements, parameter_bytes = (
        summarize_parameter_dtypes(named_params)
    )
    compile_kwargs = {"target": target}
    if target.kind.name == "cuda":
        compile_kwargs["tir_pipeline"] = get_s_tir_pipeline()
    executable = tvm.compile(mod, **compile_kwargs)

    lib_path = out_dir / "whisper_bundle.so"
    params_path = out_dir / "whisper_bundle.params"
    metadata_path = out_dir / "whisper_bundle_metadata.json"

    executable.export_library(lib_path)
    param_names = save_params_tvm(named_params, params_path)

    language_token_ids = build_language_token_ids(tokenizer)
    is_multilingual = getattr(hf_model.generation_config, "is_multilingual", None)
    if is_multilingual is None:
        is_multilingual = getattr(hf_model.config, "is_multilingual", None)
    if is_multilingual is None:
        is_multilingual = not str(args.model_id).endswith(".en")
    is_multilingual = bool(is_multilingual)
    if not is_multilingual:
        language_token_ids = {
            code: token_id
            for code, token_id in language_token_ids.items()
            if code == "en"
        }

    language_alias_to_code = build_language_alias_to_code(language_token_ids)
    language_code_to_name = {
        code: LANGUAGES[code] for code in language_token_ids if code in LANGUAGES
    }

    try:
        blank_suppress_tokens = tokenizer.encode(" ", add_special_tokens=False)
    except TypeError:
        blank_suppress_tokens = tokenizer.encode(" ")
    blank_suppress_tokens = unique_sorted_token_ids(
        list(blank_suppress_tokens) + [int(hf_model.config.eos_token_id)]
    )

    no_timestamps_token_id = token_id_or_none(tokenizer, "<|notimestamps|>")
    transcribe_token_id = token_id_or_none(tokenizer, "<|transcribe|>")
    translate_token_id = token_id_or_none(tokenizer, "<|translate|>")
    startofprev_token_id = token_id_or_none(tokenizer, "<|startofprev|>")
    no_speech_token_id = token_id_or_none(tokenizer, "<|nospeech|>")
    timestamp_begin = max(int(x) for x in special_ids) + 1
    ftype_name = choose_ftype_name(param_dtype_histogram)
    lib_size_bytes = int(lib_path.stat().st_size)
    params_size_bytes = int(params_path.stat().st_size)
    tokenizer_size_bytes = int(tokenizer_json_path.stat().st_size)
    bundle_size_bytes = int(lib_size_bytes + params_size_bytes + tokenizer_size_bytes)

    metadata = {
        "model_id": str(args.model_id),
        "tokenizer_json_name": tokenizer_json_path.name,
        "lib_name": lib_path.name,
        "params_name": params_path.name,
        "sample_rate": SAMPLE_RATE,
        "chunk_length_seconds": CHUNK_LENGTH,
        "n_samples": N_SAMPLES,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_frames": N_FRAMES,
        "n_freq": N_FREQ,
        "num_mel_bins": int(hf_model.config.num_mel_bins),
        "n_mels": int(hf_model.config.num_mel_bins),
        "max_source_positions": int(hf_model.config.max_source_positions),
        "max_target_positions": int(hf_model.config.max_target_positions),
        "n_audio_ctx": int(hf_model.config.max_source_positions),
        "n_text_ctx": int(hf_model.config.max_target_positions),
        "d_model": int(hf_model.config.d_model),
        "n_audio_state": int(hf_model.config.d_model),
        "n_text_state": int(hf_model.config.d_model),
        "time_precision": float(
            CHUNK_LENGTH / int(hf_model.config.max_source_positions)
        ),
        "samples_per_timestamp": int(
            round(
                SAMPLE_RATE * (CHUNK_LENGTH / int(hf_model.config.max_source_positions))
            )
        ),
        "max_new_tokens_default": int(max_new_tokens_default),
        "max_dec_len_compiled": int(max_dec_len),
        "max_decode_batch": int(args.max_decode_batch),
        "compile_dtype": compile_dtype,
        "compute_dtype": bundle.compute_dtype,
        "cache_dtype": bundle.cache_dtype,
        "mask_dtype": bundle.mask_dtype,
        "logits_dtype": bundle.logits_dtype,
        "preprocess_dtype": bundle.preprocess_dtype,
        "decoder_start_token_id": int(decoder_start_token_id),
        "startofprev_token_id": int(startofprev_token_id)
        if startofprev_token_id is not None
        else None,
        "no_timestamps_token_id": int(no_timestamps_token_id)
        if no_timestamps_token_id is not None
        else None,
        "no_speech_token_id": int(no_speech_token_id)
        if no_speech_token_id is not None
        else None,
        "transcribe_token_id": int(transcribe_token_id)
        if transcribe_token_id is not None
        else None,
        "translate_token_id": int(translate_token_id)
        if translate_token_id is not None
        else None,
        "timestamp_begin": int(timestamp_begin),
        "eos_token_id": int(hf_model.config.eos_token_id),
        "vocab_size": int(hf_model.config.vocab_size),
        "pad_token_id": (
            int(hf_model.config.pad_token_id)
            if hf_model.config.pad_token_id is not None
            else None
        ),
        "special_ids": special_ids,
        "suppress_tokens": unique_sorted_token_ids(
            getattr(hf_model.generation_config, "suppress_tokens", []) or []
        ),
        "begin_suppress_tokens": unique_sorted_token_ids(
            getattr(hf_model.generation_config, "begin_suppress_tokens", []) or []
        ),
        "blank_suppress_tokens": blank_suppress_tokens,
        "is_multilingual": bool(is_multilingual),
        "language_token_ids": language_token_ids,
        "language_alias_to_code": language_alias_to_code,
        "language_code_to_name": language_code_to_name,
        "default_language_code": "en",
        "default_task": "transcribe",
        "supports_translate": bool(is_multilingual),
        "encoder_layers": int(hf_model.config.encoder_layers),
        "encoder_attention_heads": int(hf_model.config.encoder_attention_heads),
        "n_audio_layer": int(hf_model.config.encoder_layers),
        "n_audio_head": int(hf_model.config.encoder_attention_heads),
        "decoder_layers": int(hf_model.config.decoder_layers),
        "decoder_attention_heads": int(hf_model.config.decoder_attention_heads),
        "n_text_layer": int(hf_model.config.decoder_layers),
        "n_text_head": int(hf_model.config.decoder_attention_heads),
        "head_dim": int(hf_model.config.d_model)
        // int(hf_model.config.decoder_attention_heads),
        "ftype": ftype_name,
        "weight_dtype": ftype_name,
        "qntvr": 0,
        "quant_scheme": "none",
        "quant_version": 0,
        "model_size_bytes": params_size_bytes,
        "lib_size_bytes": lib_size_bytes,
        "params_size_bytes": params_size_bytes,
        "tokenizer_size_bytes": tokenizer_size_bytes,
        "bundle_size_bytes": bundle_size_bytes,
        "parameter_elements": parameter_elements,
        "parameter_bytes": parameter_bytes,
        "param_dtype_histogram": param_dtype_histogram,
        "param_count": len(param_names),
        "param_names": param_names,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"wrote: {lib_path}")
    print(f"wrote: {params_path}")
    print(f"wrote: {metadata_path}")
    print(f"wrote: {tokenizer_json_path}")
    print(f"param_count: {len(param_names)}")


if __name__ == "__main__":
    main()
