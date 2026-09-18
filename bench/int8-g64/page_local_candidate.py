"""Compatibility shim for the component tests.

The page-local ABI, the writer and the prefill kernel all live in
``int8_g64.py`` (the module that is deployed as
``vllm/v1/attention/ops/int8_g64.py``).  This shim exists only so the tests in
this directory can import the historical names; it deliberately holds no
implementation of its own.
"""
from int8_g64 import g64_views, reshape_and_cache_int8_g64  # noqa: F401

__all__ = ["g64_views", "reshape_and_cache_int8_g64"]
