"""Build and optionally upload a JSON chunk manifest for a safetensors file.

The manifest is intentionally storage-agnostic: every layer entry describes an
absolute byte range in the original object.  A client can therefore use one S3
object plus HTTP Range requests (the default), or use ``key_template`` to
describe one object per layer.

STATUS:
  Built: S3ManifestWriter with to_dict()/write()/upload(), per-layer absolute
         ranges, tensor metadata, atomic JSON writes, optional boto3 upload.
  TBD:   No MinIO integration test is run here; the local RangeServer in
         src/storage is the loopback stand-in for MinIO (WS-3).
  Packages: Python standard library only.  boto3 is optional and only imported
            inside upload(); torch/numpy are NOT required.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .SafetensorsManifest import LayerRange, SafetensorsManifest

__all__ = ["S3ManifestWriter", "ManifestWriterError"]

_FORMAT = "safetensors-layer-manifest/v1"


class ManifestWriterError(RuntimeError):
    """Raised for invalid writer configuration or upload failures."""


class S3ManifestWriter:
    """Serialize a :class:`SafetensorsManifest` into a chunk manifest.

    Parameters
    ----------
    manifest:
        A parsed manifest, or a path/bytes accepted by ``SafetensorsManifest``.
    object_key:
        Key of the safetensors object in the bucket (used as the default
        ``key`` for every layer because layers are fetched with Range).
    bucket:
        Optional bucket name recorded as metadata.
    model_name:
        Optional model identifier; defaults to the source file stem.
    prefix:
        Optional prefix used when ``key_template`` is supplied.
    key_template:
        Optional format string such as ``"{prefix}{layer}.safetensors"``.
        Available fields: ``object_key``, ``prefix``, ``layer``, ``bucket``.
        When omitted, every layer uses ``object_key`` (Range mode).
    include_tensors:
        Include per-tensor descriptors inside each layer entry.
    extra:
        Optional extra top-level metadata merged into the manifest.
    """

    def __init__(
        self,
        manifest: SafetensorsManifest | str | os.PathLike[str] | bytes,
        object_key: str,
        *,
        bucket: str | None = None,
        model_name: str | None = None,
        prefix: str = "",
        key_template: str | None = None,
        include_tensors: bool = True,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if not object_key:
            raise ManifestWriterError("object_key must be non-empty")
        if not isinstance(manifest, SafetensorsManifest):
            manifest = SafetensorsManifest(manifest)
        self.manifest = manifest
        self.object_key = object_key
        self.bucket = bucket
        self.prefix = prefix
        self.key_template = key_template
        self.include_tensors = bool(include_tensors)
        self.extra = dict(extra or {})
        if model_name is None:
            model_name = (
                Path(manifest.source).stem
                if manifest.source
                else Path(object_key).stem
            )
        self.model_name = model_name

    # ------------------------------------------------------------------ build
    def _layer_key(self, layer_name: str) -> str:
        if self.key_template is None:
            return self.object_key
        try:
            key = self.key_template.format(
                object_key=self.object_key,
                prefix=self.prefix,
                layer=layer_name,
                bucket=self.bucket or "",
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise ManifestWriterError(
                f"invalid key_template {self.key_template!r}: {exc}"
            ) from exc
        if not key:
            raise ManifestWriterError(
                f"key_template produced an empty key for {layer_name!r}"
            )
        return key

    def _layer_dict(self, layer: LayerRange) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self._layer_key(layer.name),
            "start": layer.start,
            "end": layer.end,
            "length": layer.length,
            "nbytes": layer.nbytes,
            "dtypes": list(layer.dtypes),
            "tensor_names": list(layer.tensor_names),
        }
        if self.include_tensors:
            payload["tensors"] = [tensor.to_dict() for tensor in layer.tensors]
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Return the chunk manifest as a JSON-serializable dictionary."""
        dtypes = self.manifest.dtypes
        dtype = dtypes[0] if len(dtypes) == 1 else ("mixed" if dtypes else None)
        manifest: dict[str, Any] = {
            "format": _FORMAT,
            "model": self.model_name,
            "file": (
                Path(self.manifest.source).name
                if self.manifest.source
                else Path(self.object_key).name
            ),
            "object_key": self.object_key,
            "bucket": self.bucket,
            "size": self.manifest.file_size,
            "header_len": self.manifest.header_len,
            "data_offset": self.manifest.data_offset,
            "data_size": self.manifest.data_size,
            "dtype": dtype,
            "dtypes": dtypes,
            "tensor_count": len(self.manifest.tensors),
            "layer_count": len(self.manifest.layers),
            "metadata": dict(self.manifest.metadata),
            "layers": {
                name: self._layer_dict(layer)
                for name, layer in self.manifest.layers.items()
            },
        }
        manifest.update(self.extra)
        return manifest

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def write(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        """Atomically write the manifest to ``path`` and return the dict."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        text = json.dumps(data, indent=2, sort_keys=False) + "\n"
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)
        return data

    # ----------------------------------------------------------------- upload
    def upload(
        self,
        *,
        bucket: str | None = None,
        key: str | None = None,
        s3_client: Any = None,
        extra_args: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Upload the manifest JSON with boto3 if it is installed.

        boto3 is deliberately not a hard dependency.  Raises
        :class:`ManifestWriterError` with an install hint when it is absent.
        """
        bucket = bucket or self.bucket
        if not bucket:
            raise ManifestWriterError("bucket is required for upload()")
        key = key or f"{self.prefix}{Path(self.object_key).name}.manifest.json"
        if s3_client is None:
            try:
                import boto3  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ManifestWriterError(
                    "boto3 is required for upload(); install with "
                    "`pip install boto3` or use write() locally"
                ) from exc
            s3_client = boto3.client("s3")
        body = self.to_json().encode("utf-8")
        arguments: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "Body": body,
            "ContentType": "application/json",
        }
        arguments.update(extra_args or {})
        response = s3_client.put_object(**arguments)
        return {
            "bucket": bucket,
            "key": key,
            "bytes": len(body),
            "etag": response.get("ETag"),
        }
