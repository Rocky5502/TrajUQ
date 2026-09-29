from __future__ import annotations

import os
from importlib import metadata as importlib_metadata
from typing import Any


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
NORMALIZATION_BACKEND = "torch_float32_rmsnorm_v1"
ATTENTION_BACKEND = "sdar_flash_attn_decode_torch_sdpa_prefill"


def deterministic_rms_norm(
    hidden_states,
    weight=None,
    bias=None,
    eps: float = 1e-6,
    *args: object,
    **kwargs: object,
):
    """Deterministic equivalent of SDAR's inference-only Triton RMSNorm."""

    del args, kwargs
    input_dtype = hidden_states.dtype
    values = hidden_states.float()
    values = values * values.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    values = values.to(input_dtype)
    if weight is not None:
        values = values * weight
    if bias is not None:
        values = values + bias
    return values


def configure_sdar_runtime(torch) -> dict[str, Any]:
    """Pin and verify the reproducible runtime required by frozen SDAR runs."""

    observed_workspace = os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG
    )
    if observed_workspace != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "SDAR requires CUBLAS_WORKSPACE_CONFIG="
            f"{CUBLAS_WORKSPACE_CONFIG}; observed {observed_workspace!r}"
        )

    try:
        import flash_attn.ops.triton.layer_norm as flash_layer_norm
    except ImportError as exc:
        raise RuntimeError(
            "SDAR reproducible runtime requires flash-attn so its RMSNorm "
            "entrypoint can be replaced before loading remote model code"
        ) from exc

    flash_layer_norm.rms_norm_fn = deterministic_rms_norm
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    payload = sdar_runtime_provenance(torch)
    failures = validate_sdar_runtime(torch, flash_layer_norm=flash_layer_norm)
    if failures:
        raise RuntimeError("SDAR deterministic runtime enforcement failed: " + "; ".join(failures))
    return payload


def validate_sdar_runtime(torch, *, flash_layer_norm=None) -> list[str]:
    failures: list[str] = []
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        failures.append("CUBLAS_WORKSPACE_CONFIG is not pinned")
    if not torch.are_deterministic_algorithms_enabled():
        failures.append("torch deterministic algorithms are disabled")
    if torch.backends.cudnn.benchmark:
        failures.append("cuDNN benchmark is enabled")
    if not torch.backends.cudnn.deterministic:
        failures.append("cuDNN deterministic mode is disabled")
    if torch.backends.cuda.matmul.allow_tf32:
        failures.append("CUDA matmul TF32 is enabled")
    if torch.backends.cudnn.allow_tf32:
        failures.append("cuDNN TF32 is enabled")
    if flash_layer_norm is None:
        try:
            import flash_attn.ops.triton.layer_norm as flash_layer_norm
        except ImportError:
            failures.append("flash-attn RMSNorm entrypoint is unavailable")
            return failures
    if flash_layer_norm.rms_norm_fn is not deterministic_rms_norm:
        failures.append("deterministic Torch RMSNorm is not installed")
    return failures


def sdar_runtime_provenance(torch) -> dict[str, Any]:
    try:
        transformers_version = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError:
        transformers_version = None
    cuda_available = bool(torch.cuda.is_available())
    return {
        "normalization_backend": NORMALIZATION_BACKEND,
        "attention_backend": ATTENTION_BACKEND,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers_version,
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }
