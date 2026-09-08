"""Compatibility shim for the shared Modal/wrapper model registry."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_shared_registry():
    try:
        import acs_model_registry
    except ModuleNotFoundError:
        registry_path = (
            Path(__file__).resolve().parents[1] / "acs_model_registry" / "__init__.py"
        )
        spec = importlib.util.spec_from_file_location(
            "acs_model_registry", registry_path
        )
        if spec is None or spec.loader is None:
            raise
        acs_model_registry = importlib.util.module_from_spec(spec)
        sys.modules["acs_model_registry"] = acs_model_registry
        spec.loader.exec_module(acs_model_registry)
    return acs_model_registry


_registry = _load_shared_registry()

MODELS = _registry.MODELS
default_model_id = _registry.default_model_id
get_model_config = _registry.get_model_config
get_model_spec = _registry.get_model_spec
