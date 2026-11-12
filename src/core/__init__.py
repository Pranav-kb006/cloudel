"""Async progressive layer-streaming engine for safetensors checkpoints.

Public surface::

    SafetensorsManifest(path_or_bytes)
    S3ManifestWriter(manifest, object_key).write(path) -> dict
    AsyncPrefetcher(url, manifest_or_layers, buffer=..., lookahead=2)
    MemoryBuffer(max_depth=..., rss_limit_mb=...)
    PipelineDriver(prefetcher, compute_engine=..., buffer=...)

STATUS:
  Built: package exports for the five core modules; stdlib-only import path.
  TBD:   Nothing required for the EL artifact.
  Packages: Python standard library only.  aiohttp/psutil/boto3 are optional;
            torch/numpy/safetensors are NOT required.
"""

from __future__ import annotations

from .AsyncPrefetcher import AsyncPrefetcher, LayerPayload, RangeFetchError
from .MemoryBuffer import BufferEntry, MemoryBuffer, get_rss_mb
from .PipelineDriver import (
    ComputeResult,
    PipelineDriver,
    SyntheticComputeEngine,
    decode_tensor_bytes,
)
from .S3ManifestWriter import ManifestWriterError, S3ManifestWriter
from .SafetensorsManifest import (
    LayerRange,
    SafetensorsError,
    SafetensorsManifest,
    TensorInfo,
    group_layer_name,
)

__version__ = "0.1.0"

__all__ = [
    "AsyncPrefetcher",
    "BufferEntry",
    "ComputeResult",
    "LayerPayload",
    "LayerRange",
    "ManifestWriterError",
    "MemoryBuffer",
    "PipelineDriver",
    "RangeFetchError",
    "S3ManifestWriter",
    "SafetensorsError",
    "SafetensorsManifest",
    "SyntheticComputeEngine",
    "TensorInfo",
    "decode_tensor_bytes",
    "get_rss_mb",
    "group_layer_name",
]
