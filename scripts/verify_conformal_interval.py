"""S16/T16.1+T16.2 结论复算脚本（离线合成数据，不触网）。

用途：`00_kickoff/conformal_interval_conclusion.md` 里的每一个数字都由本脚本产出，
换机器/过期后可一键复算，避免"结论没有可复现来源"（T16.3 踩过的坑）。

用法：
    python scripts/verify_conformal_interval.py
    python scripts/verify_conformal_interval.py --json 报告落盘路径

退出码：0 = 正常出数；2 = 缺 mapie 可选依赖（明确失败，不静默）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 7            # 与结论文档完全一致，改这里文档数字就作废
N_SAMPLES = 3000
N_FEATURES = 6


def synthetic_dataset(seed: int = SEED, n: int = N_SAMPLES):
    """与结论文档 §二/§三 完全同口径的合成数据（离线可复现）。"""
    import pandas as pd

    rng = np.random.RandomState(seed)
    X = rng.randn(n, N_FEATURES)
    signal = 0.9 * X[:, 0] + 0.5 * X[:, 1] - 0.4 * X[:, 2]
    proba_true = 1.0 / (1.0 + np.exp(-signal))
    y = (rng.rand(n) < proba_true).astype(int)
    cols = [f"f{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=cols)
    df["target_5d"] = y
    df["_fwd_ret"] = np.where(y == 1, 0.008, -0.008) + rng.randn(n) * 0.01
    return df, cols


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="S16 保形区间结论复算")
    parser.add_argument("--json", default=None, help="把复算结果另存为 JSON")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    try:
        from sklearn.preprocessing import StandardScaler
        import lightgbm as lgb
    except ImportError as e:  # pragma: no cover
        print(f"缺少必需依赖（lightgbm/scikit-learn）: {e}")
        return 2

    from src.eval import conformal_probability as cp

    if not cp.mapie_available():
        print("缺少可选依赖 mapie：pip install mapie（不降级、不静默）")
        return 2

    df, cols = synthetic_dataset(args.seed)
    split = cp.three_way_split(len(df))
    fitted = cp.train_and_conformal(
        df, cols, "target_5d", split, config={},
        lgb_module=lgb, scaler_cls=StandardScaler)

    result: dict = {"split": split.as_dict(),
                    "seed": args.seed, "samples": int(len(df)),
                    "levels": {}}
    print(f"[split] proper-train/calibration/holdout = "
          f"{len(split.proper_train)}/{len(split.calibration)}/{len(split.holdout)}")

    for level, sets in fitted["sets"].items():
        cov = cp.coverage_report(fitted["y_true"], sets, level)
        conf = cp.interval_confidence(sets)
        result["levels"][f"{level:g}"] = {
            "coverage": cov,
            "confidence_values": sorted({round(float(v), 6) for v in conf}),
        }
        print(f"[coverage] 档位 {level:g}: 实测 {cov['empirical_coverage']} "
              f"CI95 {cov['ci95']} 判定 {cov['verdict']} "
              f"平均集合大小 {cov['set_size_mean']} 空集 {cov['empty_set_ratio']}")

    rel = cp.reliability_curve(fitted["proba"], fitted["y_true"])
    result["reliability"] = {"brier": rel["brier"], "ece": rel["ece"]}
    print(f"[reliability] Brier={rel['brier']} ECE={rel['ece']}")

    main_level = float(sorted(fitted["sets"])[0])
    cmp_res = cp.compare_interval_vs_proba(
        fitted["proba"], fitted["returns"],
        cp.interval_confidence(fitted["sets"][main_level]))
    result["comparison"] = {"level": main_level, "verdict": cmp_res["verdict"],
                            "reason": cmp_res["reason"],
                            "usable_thresholds": cmp_res["usable_thresholds"],
                            "pairs": cmp_res["pairs"]}
    print(f"[compare] 档位 {main_level:g} 判定 = {cmp_res['verdict']} "
          f"（{cmp_res['reason']}）")
    for pr in cmp_res["pairs"]:
        if not pr.get("available"):
            continue
        i, d = pr["interval"], pr["proba_distance"]
        dd = pr["delta_interval_minus_proba"]
        print(f"  thr={pr['threshold']:<4} 区间 hit={i['hit_rate']} ic={i['ic']} "
              f"cov={i['coverage']} | 概率距离 hit={d['hit_rate']} ic={d['ic']} "
              f"cov={d['coverage']} | Δ hit={dd['hit_rate']} Δic={dd['ic']}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\n复算结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
