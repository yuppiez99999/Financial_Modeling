#!/usr/bin/env python
"""TrendCast Pro · 预测评估脚本

面向 Q1 排期：为训练产物提供**一键可复现的预测评估**，产出评估报告，
用于监控模型质量、判断是否满足二期「命中率/IC 门禁」的准入条件。

评估口径（重要，避免误读）：
  1. **walk-forward 时序切分**：按时间顺序滚动切分，绝不打乱（打乱会造成
     未来数据泄漏，指标虚高）。默认 3 折、每折测试窗口约 20% 数据。
  2. **特征口径与训练一致**：复用 `FeatureEngineer.transform` +
     `get_feature_columns`，强制排除 `target_*`（防目标泄漏）。
  3. **逐标的构造目标**：`create_target` 逐标的调用，避免跨标的 shift 污染。
  4. **交易成本可选**：`--fee` 计入单边手续费，用于得到更保守的金融指标。
  5. 指标含 **IC（信息系数）**：预测概率与未来收益的 Spearman 秩相关，
     衡量信号的单调区分度（二期门禁关注项）。

用法：
  python scripts/evaluate_models.py                          # 用 config_pro 全标的池
  python scripts/evaluate_models.py --symbols 600519.SH 000858.SZ
  python scripts/evaluate_models.py --horizon all --folds 3 --fee 0.0005
  python scripts/evaluate_models.py --model-type lightgbm --output logs/eval.json
  python scripts/evaluate_models.py --offline                # 仅用本地缓存，不触网
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("evaluate_models")


# ----------------------------------------------------------------------
# 配置与数据
# ----------------------------------------------------------------------
def load_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """加载配置（默认优先 config_pro.yaml）。"""
    if config_path:
        path = Path(config_path)
    else:
        pro = PROJECT_ROOT / "configs" / "config_pro.yaml"
        path = pro if pro.exists() else PROJECT_ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"已加载配置: {path}")
    return cfg


def resolve_symbols(config: dict[str, Any], cli_symbols: Optional[List[str]] = None) -> List[str]:
    """确定评估标的池：CLI 优先，其次 config 中启用的 markets。"""
    if cli_symbols:
        return list(dict.fromkeys(cli_symbols))
    markets = (config.get("data", {}) or {}).get("markets", {}) or {}
    symbols: List[str] = []
    for _name, mcfg in markets.items():
        if (mcfg or {}).get("enabled"):
            for s in (mcfg or {}).get("symbols", []) or []:
                if s not in symbols:
                    symbols.append(s)
    return symbols


def resolve_horizons(config: dict[str, Any], horizon_arg: str) -> Dict[str, int]:
    """确定评估周期：all → 配置全部；否则仅指定周期。"""
    horizons = (config.get("data", {}) or {}).get("prediction_horizons", {}) or {}
    if horizon_arg == "all":
        return dict(horizons)
    if horizon_arg in horizons:
        return {horizon_arg: horizons[horizon_arg]}
    raise SystemExit(f"未知周期: {horizon_arg}（可选 all / {list(horizons)}）")


def load_market_data(
    config: dict[str, Any], symbols: List[str], offline: bool = False
) -> Dict[str, pd.DataFrame]:
    """加载行情数据：优先本地缓存，缓存缺失且非 offline 时联网拉取。"""
    from src.data.collector import DataCollector

    collector = DataCollector(config)
    data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = collector.load_cached(symbol)
        if (df is None or len(df) == 0) and not offline:
            try:
                df = collector.fetch_realtime(symbol)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[eval] {symbol} 拉取失败: {e}")
                df = None
        if df is None or len(df) == 0:
            logger.warning(f"[eval] {symbol} 无可用数据，跳过")
            continue
        data[symbol] = df.sort_values("date").reset_index(drop=True)
    logger.info(f"[eval] 可用标的: {len(data)}/{len(symbols)}")
    return data


# ----------------------------------------------------------------------
# 数据集构建
# ----------------------------------------------------------------------
def build_supervised(
    data: Dict[str, pd.DataFrame], config: dict[str, Any], horizon_days: int
) -> pd.DataFrame:
    """构建监督学习数据集：逐标的特征工程 + 逐标的构造目标（防泄漏）。"""
    from src.data.preprocessor import FeatureEngineer

    fe = FeatureEngineer(config)
    parts: List[pd.DataFrame] = []
    for symbol, df in data.items():
        try:
            feats = fe.transform(df.copy(), horizon_days)
            feats = fe.create_target(feats, horizon_days)
            feats = feats.copy()
            feats["_symbol"] = symbol
            feats["_fwd_ret"] = feats["close"].shift(-int(horizon_days)) / feats["close"] - 1
            feats = feats.dropna(subset=[f"target_{horizon_days}d"])
            if feats.empty:
                continue
            parts.append(feats)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[eval] {symbol} 特征/目标构建失败: {e}")
    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return combined


def walk_forward_splits(n: int, folds: int, min_train_ratio: float = 0.4
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """生成 walk-forward 时序切分索引（严格按时间顺序，测试集在训练集之后）。

    返回 [(train_idx, test_idx), ...]。数据过少时退化为单折 70/30。
    """
    if n <= 0:
        return []
    if folds <= 1 or n < 100:
        cut = max(int(n * 0.7), 1)
        return [(np.arange(0, cut), np.arange(cut, n))]

    first_train_end = max(int(n * min_train_ratio), 1)
    remaining = n - first_train_end
    step = max(remaining // folds, 1)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(folds):
        train_end = first_train_end + k * step
        test_end = n if k == folds - 1 else min(train_end + step, n)
        if train_end >= test_end:
            continue
        splits.append((np.arange(0, train_end), np.arange(train_end, test_end)))
    return splits


# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------
def information_coefficient(proba: np.ndarray, fwd_ret: np.ndarray) -> float:
    """IC：预测概率与未来收益的 Spearman 秩相关（衡量信号单调性）。"""
    if len(proba) < 3 or len(fwd_ret) < 3:
        return 0.0
    if np.std(proba) == 0 or np.std(fwd_ret) == 0:
        return 0.0
    try:
        import warnings

        from scipy.stats import spearmanr  # type: ignore

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic, _ = spearmanr(proba, fwd_ret)
        return float(ic) if ic is not None and not np.isnan(ic) else 0.0
    except ImportError:
        # 无 scipy 时用秩相关的手工实现（Spearman = 秩的 Pearson）
        p_rank = pd.Series(proba).rank().to_numpy()
        r_rank = pd.Series(fwd_ret).rank().to_numpy()
        if p_rank.std() == 0 or r_rank.std() == 0:
            return 0.0
        return float(np.corrcoef(p_rank, r_rank)[0, 1])


def compute_metrics(
    y_true: np.ndarray, proba: np.ndarray, fwd_ret: np.ndarray,
    horizon_days: int, fee: float = 0.0,
) -> Dict[str, Any]:
    """计算分类 + 金融 + IC 指标。

    金融指标口径：预测涨做多 / 预测跌做空，收益按未来实际收益率符号计入，
    可选扣除双边手续费 `fee`（单边 fee → 双边 2*fee）。
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y_pred = (proba > 0.5).astype(int)

    metrics: Dict[str, Any] = {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else 0.5
    except ValueError:
        metrics["auc"] = 0.5

    # 方向命中率（与 accuracy 同口径，显式命名便于业务阅读）
    metrics["hit_rate"] = metrics["accuracy"]
    metrics["ic"] = information_coefficient(proba, fwd_ret)

    # 金融指标：按实际收益率符号计盈亏，含手续费
    position = np.where(y_pred == 1, 1.0, -1.0)
    ret = np.nan_to_num(fwd_ret, nan=0.0)
    gross = position * ret
    net = gross - 2.0 * fee

    wins = net[net > 0]
    losses = net[net < 0]
    total_win = float(wins.sum()) if len(wins) else 0.0
    total_loss = float(abs(losses.sum())) if len(losses) else 0.0

    ann_factor = float(np.sqrt(252.0 / max(int(horizon_days), 1)))
    std = float(net.std())
    metrics["sharpe_ratio"] = float(net.mean() / std * ann_factor) if std > 0 else 0.0

    cumulative = np.cumsum(net)
    running_max = np.maximum.accumulate(cumulative) if len(cumulative) else np.array([0.0])
    drawdowns = cumulative - running_max
    # 回撤单位是"累计净收益"（已按约当年化口径使用），不是百分比 —— 字段名保留
    # 历史兼容，另给 *_unit 显式标注，避免下游把它读成"回撤 5%"。
    metrics["max_drawdown_pct"] = float(drawdowns.min()) if len(drawdowns) else 0.0
    metrics["max_drawdown_unit"] = "cum_return"
    metrics["profit_factor"] = float(total_win / total_loss) if total_loss > 0 else 999.0
    # 全对时盈亏比数学上发散，999 只是哨兵值：显式标记，防被读作"盈亏比 999 倍"
    metrics["profit_factor_is_capped"] = bool(total_loss <= 0)
    metrics["total_return"] = float(net.sum())
    metrics["win_rate"] = float(np.mean(net > 0)) if len(net) else 0.0
    metrics["trades"] = int(len(net))
    metrics["fee_per_side"] = float(fee)
    # 保本胜率：等权对赌 + 双边费用下的盈亏平衡点，供"胜率是否够覆盖成本"对照
    metrics["breakeven_win_rate"] = round(min(max((1.0 + 2.0 * float(fee)) / 2.0, 0.0), 1.0), 4)
    return metrics


# ----------------------------------------------------------------------
# 模型加载与评估
# ----------------------------------------------------------------------
def _extract_model_and_scaler(entry: Any) -> Tuple[Any, Any]:
    """从 pickle 加载结果中提取 (model, scaler)。"""
    if isinstance(entry, dict) and "model" in entry:
        return entry["model"], entry.get("scaler")
    return entry, None


def load_trained_model(config: dict[str, Any], model_type: str, horizon_key: str) -> Any:
    """加载训练好的模型文件（不存在返回 None）。"""
    import joblib

    save_dir = Path((config.get("training", {}) or {}).get("save_dir", "models"))
    path = save_dir / f"{model_type}_{horizon_key}.pkl"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[eval] 加载模型失败 {path}: {e}")
        return None


def evaluate_horizon(
    config: dict[str, Any], data: Dict[str, pd.DataFrame], horizon_name: str,
    horizon_days: int, model_type: str, folds: int, fee: float,
) -> Dict[str, Any]:
    """评估单个预测周期（walk-forward 多折）。"""
    from src.data.preprocessor import FeatureEngineer
    from src.train.models.lightgbm_model import LightGBMModel

    combined = build_supervised(data, config, horizon_days)
    if combined.empty:
        return {"horizon": horizon_name, "horizon_days": horizon_days,
                "status": "no_data", "folds": []}

    fe = FeatureEngineer(config)
    # 排除评估辅助列（_symbol / _fwd_ret 为字符串或未来收益，绝不能进特征）
    feature_cols = [
        c for c in fe.get_feature_columns(combined, horizon_days)
        if not str(c).startswith("_")
    ]
    target_col = f"target_{horizon_days}d"

    X = combined[feature_cols].to_numpy(dtype=float)
    y = combined[target_col].to_numpy(dtype=float)
    fwd = combined["_fwd_ret"].to_numpy(dtype=float)

    splits = walk_forward_splits(len(X), folds)
    if not splits:
        return {"horizon": horizon_name, "horizon_days": horizon_days,
                "status": "insufficient_data", "folds": []}

    fold_results: List[Dict[str, Any]] = []
    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        if len(np.unique(y[train_idx])) < 2:
            logger.warning(f"[eval] {horizon_name} 第 {i} 折训练集标签单一，跳过")
            continue
        try:
            model = LightGBMModel(config)
            model.train(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[eval] {horizon_name} 第 {i} 折训练失败: {e}")
            continue
        m = compute_metrics(y[test_idx], proba, fwd[test_idx], horizon_days, fee=fee)
        m["fold"] = i
        m["test_start"] = str(combined["date"].iloc[test_idx[0]].date())
        m["test_end"] = str(combined["date"].iloc[test_idx[-1]].date())
        fold_results.append(m)

    if not fold_results:
        return {"horizon": horizon_name, "horizon_days": horizon_days,
                "status": "failed", "folds": []}

    # 汇总（简单平均，样本数加权另给一份）
    numeric_keys = [
        k for k, v in fold_results[0].items()
        if isinstance(v, (int, float)) and k != "fold"
    ]
    agg = {k: float(np.mean([f[k] for f in fold_results])) for k in numeric_keys}
    total_samples = int(sum(f["samples"] for f in fold_results))
    weighted = {
        k: float(sum(f[k] * f["samples"] for f in fold_results) / total_samples)
        for k in numeric_keys
    } if total_samples else {}

    return {
        "horizon": horizon_name,
        "horizon_days": horizon_days,
        "status": "ok",
        "model_type": model_type,
        "feature_count": len(feature_cols),
        "total_samples": total_samples,
        "aggregate": agg,
        "weighted": weighted,
        "folds": fold_results,
    }


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def _grade(accuracy: float, auc: float) -> str:
    """按准确率/AUC 给出粗略评级（与 README 口径一致）。"""
    if auc >= 0.60:
        return "较强"
    if auc >= 0.55:
        return "弱（略优于随机）"
    if auc >= 0.52:
        return "接近随机"
    return "接近随机（无区分度）"


def render_markdown(results: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    """生成 Markdown 评估报告。"""
    lines = [
        "# TrendCast Pro · 预测评估报告",
        "",
        f"- 生成时间：{meta['generated_at']}",
        f"- 模型类型：`{meta['model_type']}`",
        f"- 标的数：{meta['symbol_count']}",
        f"- walk-forward 折数：{meta['folds']}",
        f"- 手续费（单边）：{meta['fee']}",
        f"- 数据口径：{meta['data_source']}",
        "",
        "> 评估为历史回测口径，未计滑点与冲击成本；指标不代表未来收益。",
        "",
        "**指标口径（防误读，务必先读）**",
        "",
        "- 收益按**预测方向做多/做空**的实际收益率符号计入，单边手续费 "
        f"`{meta['fee']}`，扣费口径为双边 `2×{meta['fee']}`；",
        "- 胜率/盈亏比是**逐笔对赌**口径；夏普为按 `sqrt(252/horizon_days)` 近似的年化值，",
        "  量级高于真实资金夏普，仅用于跨模型/跨周期横向比较；",
        "- 盈亏比带 `*` 表示**已达解析上限**（无亏损笔时数学发散，取哨兵值），不可读作真实倍数；",
        "- 最大回撤单位为**累计净收益**（非百分比），不可与资金回撤率混用；",
        f"- 保本胜率（覆盖双边费用所需的最低胜率）约 "
        f"{min(max((1.0 + 2.0 * float(meta['fee'])) / 2.0, 0.0), 1.0):.2%}。",
        "",
        "## 汇总",
        "",
        "| 周期 | 样本 | 准确率 | AUC | IC | 胜率 | 盈亏比 | 夏普(近似) | 最大回撤 | 评级 |",
        "|------|------|--------|-----|----|------|--------|-----------|---------|------|",
    ]
    ok_results = [r for r in results if r.get("status") == "ok"]
    for r in ok_results:
        a = r["weighted"]
        lines.append(
            f"| {r['horizon']} ({r['horizon_days']}d) | {r['total_samples']} | "
            f"{a['accuracy']:.2%} | {a['auc']:.4f} | {a['ic']:.4f} | "
            f"{a['win_rate']:.2%} | {a['profit_factor']:.2f}"
            f"{'*' if a.get('profit_factor_is_capped') else ''} | "
            f"{a['sharpe_ratio']:.2f} | {a['max_drawdown_pct']:.2%} | "
            f"{_grade(a['accuracy'], a['auc'])} |"
        )
    if not ok_results:
        lines.append("| — | — | — | — | — | — | — | — | — | 无可用评估结果 |")

    lines.extend(["", "## 分折明细", ""])
    for r in ok_results:
        lines.append(f"### {r['horizon']} ({r['horizon_days']}d) · 特征 {r['feature_count']} 维")
        lines.append("")
        lines.append("| 折 | 测试区间 | 样本 | 准确率 | AUC | IC | 夏普 | 最大回撤 |")
        lines.append("|----|---------|------|--------|-----|----|------|---------|")
        for f in r["folds"]:
            lines.append(
                f"| {f['fold']} | {f['test_start']} ~ {f['test_end']} | {f['samples']} | "
                f"{f['accuracy']:.2%} | {f['auc']:.4f} | {f['ic']:.4f} | "
                f"{f['sharpe_ratio']:.2f} | {f['max_drawdown_pct']:.2%} |"
            )
        lines.append("")

    skipped = [r for r in results if r.get("status") != "ok"]
    if skipped:
        lines.append("## 跳过项")
        lines.append("")
        for r in skipped:
            lines.append(f"- {r['horizon']} ({r['horizon_days']}d)：{r.get('status')}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*本报告仅供研究参考，不构成投资建议。*")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="TrendCast Pro 预测评估（walk-forward 时序回测）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbols", nargs="*", default=None, help="标的列表（缺省用配置启用标的）")
    parser.add_argument("--horizon", default="all",
                        help="评估周期：all / short_term / mid_term / long_term")
    parser.add_argument("--folds", type=int, default=3, help="walk-forward 折数（默认 3）")
    parser.add_argument("--fee", type=float, default=0.0, help="单边手续费率（默认 0）")
    parser.add_argument("--model-type", default="lightgbm", help="模型类型（用于报告标注）")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--output", default=None, help="JSON 结果输出路径")
    parser.add_argument("--report", default=None, help="Markdown 报告输出路径")
    parser.add_argument("--offline", action="store_true", help="仅使用本地缓存，不联网拉取")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    symbols = resolve_symbols(config, args.symbols)
    if not symbols:
        logger.error("无评估标的：请用 --symbols 指定，或在配置中启用 markets")
        return 2
    horizons = resolve_horizons(config, args.horizon)

    data = load_market_data(config, symbols, offline=args.offline)
    if not data:
        logger.error("无可用行情数据（离线模式请先准备 data/raw/<symbol>.csv）")
        return 2

    logger.info(f"开始评估: {len(data)} 标的 × {len(horizons)} 周期 × {args.folds} 折")
    results: List[Dict[str, Any]] = []
    for horizon_name, horizon_days in horizons.items():
        logger.info(f"--- 评估 {horizon_name} ({horizon_days}d) ---")
        results.append(
            evaluate_horizon(config, data, horizon_name, int(horizon_days),
                             args.model_type, args.folds, args.fee)
        )

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": args.model_type,
        "symbol_count": len(data),
        "symbols": sorted(data.keys()),
        "folds": args.folds,
        "fee": args.fee,
        "data_source": "local_cache_only" if args.offline else "local_cache+realtime_fallback",
    }
    payload = {"meta": meta, "results": results}

    report = render_markdown(results, meta)
    print("\n" + report)

    report_path = Path(args.report) if args.report else PROJECT_ROOT / "logs" / "evaluation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Markdown 报告已保存: {report_path}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"JSON 结果已保存: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
