"""
统一读取 configs/pipeline_config.yaml。

所有阈值、数据路径和学习率以 YAML 为唯一来源；
Python 流水线与 run_all.sh 都通过本模块取值，避免硬编码漂移。
"""
from __future__ import annotations

import os
from typing import Any

_CACHE = None
_CACHE_PATH = None


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def config_path() -> str:
    env_path = os.environ.get("AUTOIF_CONFIG")
    if env_path:
        return os.path.abspath(env_path)
    return os.path.join(project_root(), "configs", "pipeline_config.yaml")


def load_config(force: bool = False) -> dict:
    global _CACHE, _CACHE_PATH
    path = config_path()
    if not force and _CACHE is not None and _CACHE_PATH == path:
        return _CACHE

    data = {}
    if os.path.isfile(path):
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
    _CACHE = data
    _CACHE_PATH = path
    return data


def get(dotted: str, default: Any = None) -> Any:
    cur: Any = load_config()
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return default if cur is None else cur


def resolve_path(path: str | None, default: str | None = None) -> str:
    value = path or default or ""
    if not value:
        return ""
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(project_root(), value))


if __name__ == "__main__":
    import json
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else None
    if key:
        print(get(key, ""))
    else:
        print(json.dumps(load_config(), ensure_ascii=False, indent=2))
