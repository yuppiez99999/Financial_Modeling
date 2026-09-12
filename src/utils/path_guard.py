"""路径收口助手：把「路径来自变量」的文件读写钉死在预期范围内。

背景：安全扫描把所有 ``open(<变量路径>)`` 写入位标记为潜在路径穿越
（CWE-22）。这些写入位的路径虽来自内部配置而非外部输入，但统一加
包含校验是廉价的纵深防御：一旦配置被污染（如相对路径携带 ``..``），
读写会显式失败，而不是落到预期目录之外。

两个口径：
  - :func:`ensure_path_under`：有明确基目录时用（文件必须落在基目录内）；
  - :func:`ensure_no_escape`：无固定基目录（路径本身是配置项）时用，
    拒绝任何 ``..`` 成分并返回解析后的绝对路径。
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, "Path"]


def ensure_path_under(path: PathLike, base: PathLike) -> Path:
    """校验 ``path`` 解析后位于 ``base`` 之内（允许等于基目录本身）。

    Python 3.8 兼容：不用 ``Path.is_relative_to``（3.9+），用 parents 判定。

    Raises:
        ValueError: ``path`` 逃逸出 ``base`` 时抛出（fail-close，不静默改写）。
    """
    p = Path(path).resolve()
    b = Path(base).resolve()
    if p != b and b not in p.parents:
        raise ValueError(f"path escapes base: {p} not under {b}")
    return p


def ensure_no_escape(path: PathLike) -> Path:
    """无固定基目录时的收口：拒绝任何 ``..`` 成分，返回解析后的绝对路径。"""
    p = Path(path)
    if ".." in p.parts:
        raise ValueError(f"path contains '..': {path}")
    return p.resolve()
