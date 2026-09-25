from __future__ import annotations

import types
import yaml
import torch


def _to_namespace(d):
    if isinstance(d, dict):
        ns = types.SimpleNamespace()
        for k, v in d.items():
            setattr(ns, k, _to_namespace(v))
        return ns
    if isinstance(d, list):
        return [_to_namespace(v) for v in d]
    return d


def load_config(path: str):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[config] '{requested}' requested but CUDA is not available -> falling back to CPU.")
        return "cpu"
    return requested
