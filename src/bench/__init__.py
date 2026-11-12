"""Benchmark harness package for the progressive layer-streaming engine.

STATUS:
  BUILT: package marker.  The runnable entry points are
         ``python3 -m src.bench.BenchmarkHarness`` and
         ``python3 -m src.bench.make_synthetic_model``.  The harness has been
         run end-to-end against src/core; see results/benchmark.csv.
  TBD:   production MinIO/S3 + ``tc`` validation and a real transformer
         checkpoint (current numbers are synthetic, loopback-only).
  PACKAGES: aiohttp>=3.9, psutil>=5.9, requests>=2.31 (see requirements.txt).

SUBSTITUTION NOTE:
  The harness runs against the local ``src.storage.RangeServer`` over HTTP
  loopback.  That is a stand-in for MinIO/S3 + Linux ``tc`` shaping; the
  production deployment path is described in the report.
"""
