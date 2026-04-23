"""
shared/grpc_stubs
==================
This package is populated at **build time** by grpcio-tools from
``shared/proto/inference.proto``.

For local development (outside Docker), run::

    python scripts/generate_protos.py

which generates ``inference_pb2.py`` and ``inference_pb2_grpc.py`` here.
Those files are NOT committed to version control (listed in .gitignore).

See the proto source at: shared/proto/inference.proto
"""
# grpcio-tools generates `import inference_pb2` (bare import) inside
# inference_pb2_grpc.py.  When the stubs live inside this package
# (shared/grpc_stubs/) the bare import would fail because Python looks
# for a top-level `inference_pb2` module.
#
# Adding this directory to sys.path makes the bare import resolve to
# shared/grpc_stubs/inference_pb2.py as intended.
import os as _os
import sys as _sys

_pkg_dir = _os.path.dirname(__file__)
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)
