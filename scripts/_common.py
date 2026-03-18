from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def external_scldm_root() -> Path:
    return repo_root() / "external" / "scldm"


def local_src_root() -> Path:
    return repo_root() / "src"


def upstream_scldm_src_root() -> Path:
    return external_scldm_root() / "src"


def _pythonpath_entries() -> list[Path]:
    return [path.resolve() for path in (local_src_root(), upstream_scldm_src_root()) if path.exists()]


def prepend_project_pythonpath() -> None:
    for path in reversed(_pythonpath_entries()):
        path_str = path.as_posix()
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def build_pythonpath_env(existing: str | None = None) -> str:
    parts = [path.as_posix() for path in _pythonpath_entries()]
    if existing:
        parts.extend(part for part in existing.split(os.pathsep) if part)
    return os.pathsep.join(parts)


def load_local_sit_flow_model():
    sit_path = repo_root() / "external" / "SiT.py"
    if not sit_path.exists():
        raise FileNotFoundError(f"Local SiT implementation not found: {sit_path}")

    spec = importlib.util.spec_from_file_location("genprot_external_sit", sit_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {sit_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "SiTFlowModel"):
        raise ImportError(f"SiTFlowModel was not found in: {sit_path}")
    return module.SiTFlowModel
