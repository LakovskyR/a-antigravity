"""
Compatibility wrapper for importing schema parser utilities.

Allows:
    from tools.schema_parser import find_all_schema_files
while implementation lives in tools/0_schema_parser.py
"""

from importlib import util
from pathlib import Path


_impl_path = Path(__file__).with_name("0_schema_parser.py")
_spec = util.spec_from_file_location("tools._schema_parser_impl", _impl_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load schema parser implementation from {_impl_path}")
_mod = util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

find_all_schema_files = _mod.find_all_schema_files
parse_schema = _mod.parse_schema
extract_variable_ids = _mod.extract_variable_ids
main = _mod.main

