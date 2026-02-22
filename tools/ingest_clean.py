"""
Compatibility wrapper for importing ingest utilities.

Allows:
    from tools.ingest_clean import load_source
while implementation lives in tools/1_ingest_clean.py
"""

from importlib import util
from pathlib import Path


_impl_path = Path(__file__).with_name("1_ingest_clean.py")
_spec = util.spec_from_file_location("tools._ingest_clean_impl", _impl_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load ingest implementation from {_impl_path}")
_mod = util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SUPPORTED_EXTENSIONS = _mod.SUPPORTED_EXTENSIONS
load_source = _mod.load_source
main = _mod.main

