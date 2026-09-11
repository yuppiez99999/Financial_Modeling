"""按资产类别分层训练（S9）：给每个分池训一套独立模型。

为什么需要它（承接 S7 实测证据）：
  S7 的 `--stratify` 已实测证明：整池 IC 被"木桶短板"绑架 ——
  个股与宽基 ETF 的波动结构完全不同，混在一起训一个模型，
  等于用一个决策边界去拟合两种分布，双方都拟合不好。

  但"按标的分层建模"当时只是**建议**，未落地。本模块把它落地。

设计取舍（重要，避免造出一个"看起来很强"的东西）：
  - **不做池化回退**：某个分池样本不足时，**如实拒答**（`status=insufficient_data`），
    不静默回退到整池模型。原因：静默回退会让"分层"变成一句空话 ——
    用户以为拿到的是 ETF 专用模型，其实是个股池的模型，这比没有更糟。
    想要回退必须显式配置 `pool_train.fallback_to_pooled: true`，且结果里标注 `fallback=true`。
  - **产物自带元数据**：每个分池模型存 `pool_<class>_<model_type>_<horizon>.pkl`，
    并落盘 `pool_manifest.json`（分池 → 标的 → 文件 → 样本数 → 训练时间），
    推理侧据此**只对属于该分池的标的**用对应模型，绝不跨池串用。
  - **零网络假设**：只吃传入的行情字典；行情缺失 → 该分池标记 `no_data`，不触网补数据。
  - **不改变门禁**：产出的模型评分口径与整池一致（概率 - 0.5），
    是否放行仍由分池门禁（`pool_gate`）判定。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.eval.asset_class import CLASS_ORDER, group_symbols, label

logger = logging.getLogger(__name__)

MANIFEST_NAME = "pool_manifest.json"
DEFAULT_MIN_SAMPLES = 200


def _cfg(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return (config or {}).get("pool_train", {}) or {}


class StratifiedTrainer:
    """按资产类别逐池训练模型（每个分池一套独立产物）。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        cfg = _cfg(config)
        self.enabled = bool(cfg.get("enabled", True))
        self.min_samples = int(cfg.get("min_samples", DEFAULT_MIN_SAMPLES))
        self.fallback_to_pooled = bool(cfg.get("fallback_to_pooled", False))
        self.model_type = str(
            cfg.get("model_type") or (config.get("model", {}) or {}).get("type", "lightgbm")
        )
        save_dir = (config.get("training", {}) or {}).get("save_dir", "models")
        self.save_dir = Path(cfg.get("save_dir") or save_dir)
        self.pool_dir = self.save_dir / "pools"

    # ------------------------------------------------------------------
    def plan(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """训练计划（只读）：每个分池有多少标的、够不够训。

        先出计划再训练 —— 让人在花 20 分钟训练前就知道会得到什么、哪些池会被跳过。
        """
        groups = group_symbols(list((data or {}).keys()))
        pools: Dict[str, Any] = {}
        for cls, symbols in groups.items():
            present = [s for s in symbols if data.get(s) is not None and len(data[s]) > 0]
            rows = sum(len(data[s]) for s in present)
            pools[cls] = {
                "asset_class": cls,
                "label": label(cls),
                "symbols": symbols,
                "available_symbols": present,
                "rows": rows,
                "trainable": bool(present) and rows >= self.min_samples,
                "reason": "" if (present and rows >= self.min_samples) else (
                    "无可用行情" if not present else f"样本行数 {rows} < {self.min_samples}"
                ),
            }
        return {
            "model_type": self.model_type,
            "min_samples": self.min_samples,
            "fallback_to_pooled": self.fallback_to_pooled,
            "pools": pools,
            "trainable_pools": [c for c, p in pools.items() if p["trainable"]],
        }

    # ------------------------------------------------------------------
    def train_pools(
        self,
        data: Dict[str, Any],
        horizons: Dict[str, int],
        splits: int = 3,
    ) -> Dict[str, Any]:
        """逐池 × 逐周期训练并落盘。

        Args:
            data    : ``{symbol: DataFrame}``（只读，不修改）；
            horizons: ``{horizon_name: horizon_days}``；
            splits  : 训练/验证切分折数（时序，不打乱）。

        Returns:
            `pool_manifest.json` 的内容（可直接进报表/API）。
        """
        plan = self.plan(data)
        manifest: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model_type": self.model_type,
            "min_samples": self.min_samples,
            "fallback_to_pooled": self.fallback_to_pooled,
            "pool_dir": str(self.pool_dir),
            "plan": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                     for k, v in plan["pools"].items()},
            "pools": {},
        }
        if not self.enabled:
            manifest["skipped"] = "pool_train.enabled=false，跳过分池训练"
            return manifest

        for cls, info in plan["pools"].items():
            entry: Dict[str, Any] = {
                "asset_class": cls,
                "label": label(cls),
                "symbols": info["symbols"],
                "horizons": {},
                "status": "skipped",
                "reason": info["reason"],
            }
            if not info["trainable"]:
                if self.fallback_to_pooled:
                    entry["status"] = "fallback"
                    entry["reason"] = (
                        f"{info['reason']}；已按 pool_train.fallback_to_pooled=true "
                        "显式回退到整池模型（推理侧会标注 fallback=true）"
                    )
                manifest["pools"][cls] = entry
                continue

            pool_data = {s: data[s] for s in info["available_symbols"]}
            files: List[str] = []
            for hname, hdays in (horizons or {}).items():
                res = self._train_one_pool(cls, pool_data, hname, int(hdays), splits)
                entry["horizons"][hname] = res
                if res.get("file"):
                    files.append(res["file"])
            entry["files"] = files
            entry["status"] = "trained" if files else "failed"
            manifest["pools"][cls] = entry

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        path = self.pool_dir / MANIFEST_NAME
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(path)
        return manifest

    # ------------------------------------------------------------------
    def _train_one_pool(self, cls: str, pool_data: Dict[str, Any],
                        horizon_name: str, horizon_days: int,
                        splits: int) -> Dict[str, Any]:
        """单池单周期训练（时序切分，逐池构造目标，防跨池 shift 污染）。"""
        from src.data.preprocessor import FeatureEngineer

        fe = FeatureEngineer(self.config)
        parts: List[pd.DataFrame] = []
        for symbol, df in pool_data.items():
            try:
                feats = fe.transform(df.copy(), horizon_days)
                feats = fe.create_target(feats, horizon_days)
                feats["_symbol"] = symbol
                feats = feats.dropna(subset=[f"target_{horizon_days}d"])
                if not feats.empty:
                    parts.append(feats)
            except Exception as e:  # noqa: BLE001 - 单标的失败不拖垮分池
                logger.warning(f"[pool-train] {cls}/{symbol} 特征构建失败: {e}")
        if not parts:
            return {"status": "no_data", "reason": "分池无可用的特征/目标样本"}

        combined = pd.concat(parts, ignore_index=True)
        if "date" in combined.columns:
            combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
            combined = combined.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if len(combined) < self.min_samples:
            return {
                "status": "insufficient_data",
                "reason": f"样本 {len(combined)} < {self.min_samples}",
                "samples": len(combined),
            }

        cols = [c for c in fe.get_feature_columns(combined, horizon_days)
                if not str(c).startswith("_")]
        target_col = f"target_{horizon_days}d"
        X = combined[cols].to_numpy(dtype=float)
        y = combined[target_col].to_numpy(dtype=float)

        cut = max(int(len(X) * (splits / (splits + 1))), 1)
        X_tr, y_tr = X[:cut], y[:cut]
        if len(np.unique(y_tr)) < 2:
            return {"status": "single_class", "reason": "训练段标签单一，无法训练",
                    "samples": len(combined)}

        try:
            from src.train.models.lightgbm_model import LightGBMModel

            model = LightGBMModel(self.config)
            model.train(X_tr, y_tr)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[pool-train] {cls}/{horizon_name} 训练失败: {e}")
            return {"status": "failed", "reason": f"训练异常: {e}"}

        self.pool_dir.mkdir(parents=True, exist_ok=True)
        filename = f"pool_{cls}_{self.model_type}_{horizon_name}_{horizon_days}d.pkl"
        target = self.pool_dir / filename
        try:
            model.save(str(target))
        except Exception as e:  # noqa: BLE001
            return {"status": "save_failed", "reason": f"落盘失败: {e}"}

        return {
            "status": "ok",
            "horizon_days": horizon_days,
            "samples": int(len(combined)),
            "train_samples": int(len(X_tr)),
            "feature_count": len(cols),
            "symbols": sorted(pool_data.keys()),
            "file": str(target),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }

    # ------------------------------------------------------------------
    def load_manifest(self) -> Dict[str, Any]:
        """读取分池模型清单（缺失返回 available=False，不臆测）。"""
        path = self.pool_dir / MANIFEST_NAME
        if not path.exists():
            return {"available": False, "reason": "no_pool_manifest"}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            return {"available": False, "error": str(e)}
        payload["available"] = True
        payload["manifest_path"] = str(path)
        return payload

    def resolve_pool_model(self, symbol: str) -> Optional[Dict[str, Any]]:
        """查该标的应该用哪个分池模型（供推理侧路由）。

        返回 ``None`` = 该标的没有可用的分池模型（**不静默回退**，由调用方决定是否回退）。
        返回 ``{"asset_class":..., "file":..., "fallback": bool}``。
        """
        from src.eval.asset_class import classify

        manifest = self.load_manifest()
        if not manifest.get("available"):
            return None
        cls = classify(symbol)["asset_class"]
        pool = (manifest.get("pools") or {}).get(cls) or {}
        if pool.get("status") == "trained":
            return {"asset_class": cls, "file": pool.get("files", [None])[0],
                    "fallback": False, "status": pool.get("status")}
        if pool.get("status") == "fallback":
            return {"asset_class": cls, "file": None, "fallback": True,
                    "status": pool.get("status")}
        return {"asset_class": cls, "file": None, "fallback": False,
                "status": pool.get("status", "unknown")}


def pool_label_for(symbol: str) -> str:
    """标的 → 分池显示名（一行工具函数，供报表复用）。"""
    from src.eval.asset_class import classify

    return label(classify(symbol)["asset_class"])


def summarize_manifest(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 manifest 压成报表行。"""
    rows: List[Dict[str, Any]] = []
    pools = manifest.get("pools") or {}
    for cls in CLASS_ORDER:
        if cls not in pools:
            continue
        p = pools[cls] or {}
        horizons = p.get("horizons") or {}
        rows.append({
            "asset_class": cls,
            "label": p.get("label") or label(cls),
            "status": p.get("status", "unknown"),
            "symbol_count": len(p.get("symbols") or []),
            "files": len(p.get("files") or []),
            "horizons": {
                h: {"status": v.get("status"), "samples": v.get("samples"),
                    "reason": v.get("reason", "")}
                for h, v in horizons.items()
            },
            "reason": p.get("reason", ""),
        })
    return rows
