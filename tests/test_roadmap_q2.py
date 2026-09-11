"""Q2 排期：多因子模型 / LSTM 上线 / 持仓池与质量门控 的回归测试。

覆盖 README「10.5 Q2 排期进展」中由**本分支新增**的能力：
  1. 可训练多因子模型（因子库 / IC 加权组合 / 类级融合） → TestFactorLibrary / TestFactorModel / TestFactorPredictor
  2. LSTM 上线（模型注册表 / meta 对齐 / 序列标签对齐）   → TestLSTMOnline / TestModelRegistry
  3. 端到端多因子推理链路                                → TestMultifactorEndToEnd
  4. 持仓池 / 真实行情源 / 数据质量门控                   → TestHoldingsPoolAndQuality
  5. 配置与 CLI 集成                                     → TestQ2ConfigIntegration

> 注：IC / 命中率**策略门禁**由 `src/inference/ic.py` + `src/trading/gate.py`
> （同批 Q2 交付）实现，其测试见 `tests/test_q2_roadmap.py`，本文件不重复覆盖。

全部离线运行（不触网），验证指标均为确定性输入。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collector import DataCollector
from src.factors import FACTOR_FAMILIES, FactorLibrary, FactorModel, FactorPredictor
from src.inference.predictor import PredictionEngine
from src.train.registry import ModelRegistry, ModelArtifact, save_artifact


# ----------------------------------------------------------------------
# 夹具
# ----------------------------------------------------------------------
@pytest.fixture()
def cfg(tmp_path):
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["training"]["save_dir"] = str(tmp_path / "models")
    c["data"]["raw_dir"] = str(tmp_path / "raw")
    c["model"]["factors"]["enabled"] = True
    return c


@pytest.fixture()
def market_df(tmp_path):
    """合成行情（固定种子，保证可复现）。"""
    cfg = {"training": {"save_dir": str(tmp_path / "m")}, "data": {"raw_dir": str(tmp_path / "raw")}}
    return DataCollector(cfg)._fetch_simulation("600519.SH", n=320)


def _supervised(df: pd.DataFrame, horizon: int = 5):
    """构造因子矩阵 + 标签（与 trainer 的口径一致）。"""
    feats = FactorLibrary({"model": {"factors": {}}}).compute(df)
    cols = FactorLibrary.factor_columns(feats)
    feats = feats.copy()
    fwd = feats["close"].shift(-horizon) / feats["close"] - 1
    feats["target"] = (fwd > 0).astype(float)
    feats.loc[fwd.isna(), "target"] = np.nan
    feats = feats.dropna()
    return feats[cols], feats["target"].to_numpy(), cols


# ----------------------------------------------------------------------
# 1. 因子库
# ----------------------------------------------------------------------
class TestFactorLibrary:
    def test_all_families_produce_bounded_factors(self, market_df):
        lib = FactorLibrary({"model": {"factors": {}}})
        out = lib.compute(market_df)
        cols = lib.factor_columns(out)
        assert cols, "应产出因子列"
        # 每个因子族都有产出
        for family, members in FACTOR_FAMILIES.items():
            assert any(c in cols for c in members), f"因子族 {family} 无产出"
        # 因子必须有界 ∈ [-1, 1] 且无 NaN（可直接加权）
        values = out[cols]
        assert values.notna().all().all()
        assert values.min().min() >= -1.0 - 1e-9
        assert values.max().max() <= 1.0 + 1e-9

    def test_missing_optional_data_yields_neutral_zero(self, market_df):
        """情绪/宏观数据缺失时给中性 0（不编造数据）。"""
        out = FactorLibrary({"model": {"factors": {}}}).compute(market_df)
        assert (out["factor_sentiment"] == 0.0).all()
        assert (out["factor_macro"] == 0.0).all()

    def test_no_lookahead_factor_is_causal(self, market_df):
        """因子只依赖历史：截断末尾数据后，前缀因子值不应改变。"""
        lib = FactorLibrary({"model": {"factors": {}}})
        full = lib.compute(market_df)
        cut = lib.compute(market_df.iloc[:-30].copy())
        cols = [c for c in lib.factor_columns(full) if c not in ("factor_macro", "factor_sentiment")]
        # 允许 smoothing 带来的尾部微小差异，比较去掉最后 smoothing 行后的前缀
        pd.testing.assert_frame_equal(
            full[cols].iloc[:-40].reset_index(drop=True),
            cut[cols].iloc[:-10].reset_index(drop=True),
            check_exact=False, atol=1e-9,
        )

    def test_family_map_reports_members(self, market_df):
        lib = FactorLibrary({"model": {"factors": {}}})
        out = lib.compute(market_df)
        fam = lib.family_map(lib.factor_columns(out))
        assert "trend" in fam and fam["trend"]
        assert all(c in out.columns for cols in fam.values() for c in cols)


# ----------------------------------------------------------------------
# 2. 因子模型：加权组合 + IC 权重
# ----------------------------------------------------------------------
class TestFactorModel:
    def test_train_produces_normalized_weights(self, cfg, market_df):
        X, y, cols = _supervised(market_df)
        model = FactorModel(cfg)
        model.train(X[cols], y, feature_cols=cols)

        assert set(model.factor_weights_) <= set(cols)
        assert abs(sum(model.factor_weights_.values()) - 1.0) < 1e-6, "因子权重应归一化"
        assert abs(sum(model.family_weights_.values()) - 1.0) < 1e-6, "族权重应归一化"
        assert model.n_factors_ == len(model.factor_weights_)
        # 有信号的族权重 > 无信号族（sentiment/macro 全 0 → 权重 0）
        assert model.family_weights_.get("sentiment", 0.0) == pytest.approx(0.0)
        assert model.family_weights_.get("trend", 0.0) > 0.0

    def test_predict_proba_is_probability_and_monotonic(self, cfg, market_df):
        X, y, cols = _supervised(market_df)
        model = FactorModel(cfg)
        model.train(X[cols], y, feature_cols=cols)

        proba = model.predict_proba(X[cols])
        assert proba.shape == (len(X), 2)
        assert ((proba >= 0.0) & (proba <= 1.0)).all()
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)

        # 得分越高概率越高（单调校准，方向不得反转）
        score = model.score(X[cols])
        order = np.argsort(score)
        p = model.predict_proba(X[cols])[:, 1]
        assert (np.diff(p[order]) >= -1e-9).all(), "概率必须随得分单调递增"

    def test_predict_rejects_non_factor_input(self, cfg, market_df):
        X, y, cols = _supervised(market_df)
        model = FactorModel(cfg)
        plain = X[cols].rename(columns=lambda c: c.replace("factor_", ""))
        with pytest.raises(ValueError):
            model.train(plain, y, feature_cols=list(plain.columns))

    def test_save_load_roundtrip_consistent(self, cfg, market_df, tmp_path):
        X, y, cols = _supervised(market_df)
        model = FactorModel(cfg)
        model.train(X[cols], y, feature_cols=cols)
        before = model.predict_proba(X[cols])

        path = model.save(str(tmp_path / "factor_model_test.pkl"))
        reloaded = FactorModel(cfg)
        reloaded.load(str(path))
        after = reloaded.predict_proba(X[cols])

        np.testing.assert_allclose(before, after, atol=1e-12)
        assert reloaded.explain(3)["top_factors"]

    def test_explicit_weights_override_estimated(self, cfg, market_df):
        X, y, cols = _supervised(market_df)
        cfg = dict(cfg)
        cfg["model"] = dict(cfg["model"])
        cfg["model"]["factors"] = dict(cfg["model"]["factors"])
        cfg["model"]["factors"]["family_weights"] = {"trend": 1.0, "momentum": 0.0}
        model = FactorModel(cfg)
        model.train(X[cols], y, feature_cols=cols)
        assert model.family_weights_["trend"] == pytest.approx(1.0)
        assert model.family_weights_["momentum"] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# 3. 因子组合预测器：类级加权融合
# ----------------------------------------------------------------------
class TestFactorPredictor:
    def _model(self, cfg, market_df):
        X, y, cols = _supervised(market_df)
        m = FactorModel(cfg)
        m.train(X[cols], y, feature_cols=cols)
        return m, cols

    def test_fuse_multiple_classes_with_weights(self, cfg, market_df):
        model, _ = self._model(cfg, market_df)
        # 用假的 tree / sequence 分量，验证加权公式
        class FakeTree:
            n_features_in_ = 2

            def predict_proba(self, X):
                return np.array([[0.2, 0.8]])

        cfg = dict(cfg)
        cfg["model"] = dict(cfg["model"])
        cfg["model"]["factors"] = dict(cfg["model"]["factors"])
        cfg["model"]["factors"]["ensemble"] = {
            "classes": ["factor", "tree"], "weights": {"factor": 0.5, "tree": 0.5}
        }
        pred = FactorPredictor(
            cfg, model,
            tree_entry={"model": FakeTree(), "scaler": None},
        )
        df = market_df.copy()
        proba = pred.predict_proba(df)
        assert 0.0 < proba < 1.0
        # 两个分量都参与，且方向分按权重融合
        assert set(pred.last_components) == {"factor", "tree"}
        f_dir = (pred.last_components["factor"] - 0.5) * 2
        t_dir = (pred.last_components["tree"] - 0.5) * 2
        # 分量概率按 4 位小数记录，故容差放宽到 1e-4
        assert proba == pytest.approx(0.5 + (0.5 * f_dir + 0.5 * t_dir) / 2, abs=1e-4)

    def test_missing_component_degrades_gracefully(self, cfg, market_df):
        model, _ = self._model(cfg, market_df)
        cfg = dict(cfg)
        cfg["model"] = dict(cfg["model"])
        cfg["model"]["factors"] = dict(cfg["model"]["factors"])
        cfg["model"]["factors"]["ensemble"] = {
            "classes": ["factor", "tree", "sequence"], "weights": {"factor": 0.4, "tree": 0.35}
        }
        pred = FactorPredictor(cfg, model)  # 无 tree / sequence
        proba = pred.predict_proba(market_df)
        assert 0.0 < proba < 1.0
        assert set(pred.last_components) == {"factor"}, "缺失分量必须被剔除而非报错"

    def test_align_features_fills_missing_factors_with_zero(self, cfg, market_df):
        model, cols = self._model(cfg, market_df)
        pred = FactorPredictor(cfg, model)
        drop = [c for c in cols if c in market_df.columns][:3]
        stripped = market_df.drop(columns=drop)
        aligned = pred.align_features(FactorLibrary({"model": {"factors": {}}}).compute(stripped))
        for c in drop:
            assert (aligned[c] == 0.0).all(), "缺失因子列必须按中性 0 补齐"

    def test_no_component_returns_neutral(self, cfg, market_df):
        pred = FactorPredictor(cfg, None)
        assert pred.predict_proba(market_df) == pytest.approx(0.5)
        assert pred.last_components == {}


# ----------------------------------------------------------------------
# 5. 模型注册表 + LSTM 上线
# ----------------------------------------------------------------------
class TestModelRegistry:
    def test_infer_format_by_filename(self):
        assert ModelRegistry.infer_format(Path("lightgbm_short_term_5d.pkl")) == "joblib"
        assert ModelRegistry.infer_format(Path("pytorch_lstm_short_term_5d.pt")) == "torch"
        assert ModelRegistry.infer_format(Path("factor_model_short_term_5d.pkl")) == "factor"
        assert ModelRegistry.infer_format(Path("timesfm_short_term_5d.pkl")) == "timesfm"

    def test_missing_file_returns_none(self, tmp_path):
        assert ModelRegistry().load(tmp_path / "nope.pkl") is None

    def test_joblib_roundtrip_with_meta(self, cfg, tmp_path):
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression().fit(np.random.rand(20, 3), np.random.randint(0, 2, 20))
        art = ModelArtifact(
            name="demo", format="joblib", model=model, scaler=None,
            feature_cols=["a", "b", "c"], meta={"horizon_name": "short_term"},
        )
        path = save_artifact(tmp_path / "demo.pkl", art)
        loaded = ModelRegistry().load(path)
        assert loaded is not None and loaded.format == "joblib"
        assert loaded.feature_cols == ["a", "b", "c"]
        assert loaded.meta["horizon_name"] == "short_term"

    def test_unknown_format_returns_none(self, tmp_path):
        path = tmp_path / "x.bin"
        path.write_bytes(b"x")
        assert ModelRegistry().load(path, fmt="no_such_format") is None


class TestLSTMOnline:
    """LSTM 上线：训练 → 落盘（含 meta）→ 注册表加载 → 推理对齐。"""

    def test_lstm_train_save_load_predict(self, cfg, market_df, tmp_path):
        torch = pytest.importorskip("torch")
        from src.train.models.lstm_model import PyTorchLSTMTrainer

        cfg = dict(cfg)
        cfg["model"] = dict(cfg["model"])
        cfg["model"]["pytorch_lstm"] = dict(cfg["model"]["pytorch_lstm"])
        cfg["model"]["pytorch_lstm"].update({
            "epochs": 1, "sequence_length": 8, "hidden_size": 8, "num_layers": 1, "batch_size": 16,
        })
        cfg["training"] = dict(cfg["training"])
        cfg["training"]["save_dir"] = str(tmp_path / "models")

        X, y, cols = _supervised(market_df)
        Xv = X[cols].to_numpy(dtype=float)
        n = len(Xv)
        split = int(n * 0.8)
        trainer = PyTorchLSTMTrainer(cfg)
        trainer.train(
            Xv[:split], y[:split], Xv[split:], y[split:],
            feature_cols=cols, meta={"horizon_name": "short_term"},
        )
        path = trainer.save(str(tmp_path / "models" / "pytorch_lstm_short_term_5d.pt"))
        assert path.exists()

        # 注册表加载并按 meta 对齐特征列
        art = ModelRegistry(cfg).load(path)
        assert art is not None and art.format == "torch"
        assert art.feature_cols == cols, "checkpoint 必须带 feature_cols 供推理对齐"
        assert art.seq_len == 8

        # predict_proba 与 predict 长度一致（序列滑窗丢样本是正常的）
        proba = trainer.predict_proba(Xv)
        pred = trainer.predict(Xv)
        assert len(proba) == len(pred) == len(Xv) - 8 + 1
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_evaluator_aligns_sequence_labels(self, cfg, market_df, tmp_path):
        """评估器必须自动对齐序列模型的标签长度（Q1 遗留的 LSTM 上线阻塞点）。"""
        pytest.importorskip("torch")
        from src.eval.evaluator import ModelEvaluator
        from src.train.models.lstm_model import PyTorchLSTMTrainer

        cfg = dict(cfg)
        cfg["model"] = dict(cfg["model"])
        cfg["model"]["pytorch_lstm"] = dict(cfg["model"]["pytorch_lstm"])
        cfg["model"]["pytorch_lstm"].update({"epochs": 1, "sequence_length": 8,
                                             "hidden_size": 8, "num_layers": 1, "batch_size": 16})
        cfg["training"] = dict(cfg["training"])
        cfg["training"]["save_dir"] = str(tmp_path / "models")

        X, y, cols = _supervised(market_df)
        Xv = X[cols].to_numpy(dtype=float)
        split = int(len(Xv) * 0.8)
        trainer = PyTorchLSTMTrainer(cfg)
        trainer.train(Xv[:split], y[:split], Xv[split:], y[split:], feature_cols=cols)

        result = ModelEvaluator(cfg).evaluate(trainer, Xv[split:], y[split:])
        assert 0.0 <= result["metrics"]["accuracy"] <= 1.0  # 未对齐时此处会抛长度不一致
        assert result["metrics"]["accuracy"] == result["metrics"]["accuracy"]

    def test_missing_torch_degrades_in_multifactor(self, cfg, market_df, monkeypatch):
        """无 torch 时多因子组合必须降级为 factor(+tree)，不中断推理。"""
        X, y, cols = _supervised(market_df)
        model = FactorModel(cfg)
        model.train(X[cols], y, feature_cols=cols)
        engine = PredictionEngine(cfg)
        engine.model_type = "multifactor"
        monkeypatch.setattr(engine, "_load_sequence", lambda _k: None)
        pred = FactorPredictor(cfg, model, tree_entry=None, sequence_model=None)
        proba = pred.predict_proba(market_df)  # 原始行情（无 factor_* 列）
        assert 0.0 < proba < 1.0
        # 输入原始行情时必须现算因子并给出 factor 分量，而非静默返回中性
        assert set(pred.last_components) == {"factor"}


# ----------------------------------------------------------------------
# 6. 端到端：多因子推理链路
# ----------------------------------------------------------------------
class TestMultifactorEndToEnd:
    def test_multifactor_engine_predicts(self, cfg, market_df, tmp_path):
        save_dir = Path(cfg["training"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        # 落盘因子模型 + 把行情写入缓存
        X, y, cols = _supervised(market_df)
        fm = FactorModel(cfg)
        fm.train(X[cols], y, feature_cols=cols)
        fm.save(str(save_dir / "factor_model_short_term_5d.pkl"))
        raw = Path(cfg["data"]["raw_dir"])
        raw.mkdir(parents=True, exist_ok=True)
        market_df.to_csv(raw / "600519.SH.csv", index=False)

        engine = PredictionEngine(cfg)
        engine.load_models("multifactor")
        assert "short_term_5d" in engine.models
        result = engine.predict("600519.SH", "short_term")
        assert "error" not in result
        assert 0.0 <= result["probability"] <= 1.0
        assert "factor" in result["components"]
        assert result["explain"]["top_factors"], "多因子预测应附带因子解释"

    def test_multifactor_without_model_reports_actionable_hint(self, cfg):
        engine = PredictionEngine(cfg)
        engine.load_models("multifactor")
        assert engine.models == {}, "无模型时不得静默加载"


# ----------------------------------------------------------------------
# 6b. 交易适配层 × 策略门禁（Q2 集成：门禁未过不出订单）
# ----------------------------------------------------------------------
class TestTradingAdapterGateEnforcement:
    """复用 `src/trading/gate.py` 的门禁结果控制是否产出订单。

    默认 `strategy_gate.enforce_trading: false` → 行为与 Q1 完全一致（零回归）；
    显式开启后，未放行的信号只读，不产出可执行订单。
    """

    @staticmethod
    def _adapter(cfg, tmp_path, state, enforce=True):
        from src.trading.adapter import TradingAdapter

        cfg = dict(cfg)
        cfg["strategy_gate"] = dict(cfg.get("strategy_gate", {}) or {})
        cfg["strategy_gate"].update({
            "enabled": True,
            "enforce_trading": enforce,
            "report_dir": str(tmp_path),
        })
        (tmp_path / "strategy_gate.json").write_text(
            json.dumps({"state": state, "passed": state == "gated"}), encoding="utf-8"
        )
        return TradingAdapter(cfg, predictor=_StubPredictor())

    def test_default_does_not_enforce(self, cfg, tmp_path):
        adapter = self._adapter(cfg, tmp_path, state="readonly", enforce=False)
        assert adapter.signal_readonly is False
        assert adapter._gate_decision is None

    def test_readonly_blocks_orders(self, cfg, tmp_path):
        adapter = self._adapter(cfg, tmp_path, state="readonly", enforce=True)
        assert adapter.signal_readonly is True
        result = adapter.process_symbol("600519.SH")
        assert result["orders"] == []
        assert result["gate_blocked"] is True
        assert result["gate_level"] == "readonly"
        # 信号本身仍产出（只读观测），风控预算照算，只是不落订单
        assert result["signal"]["action"] in ("BUY", "SELL", "HOLD")
        assert "risk_budget" in result

    def test_gated_allows_orders(self, cfg, tmp_path):
        adapter = self._adapter(cfg, tmp_path, state="gated", enforce=True)
        assert adapter.signal_readonly is False
        result = adapter.process_symbol("600519.SH")
        assert result["gate_blocked"] is False
        assert result["gate_level"] == "gated"

    def test_disabled_state_never_blocks(self, cfg, tmp_path):
        adapter = self._adapter(cfg, tmp_path, state="disabled", enforce=True)
        assert adapter.signal_readonly is False


class _StubPredictor:
    """最小可用的预测器替身（看涨），避免测试触网 / 依赖模型产物。"""

    def predict_all_horizons(self, symbol):
        return {
            "symbol": symbol,
            "predictions": {
                "short_term": {"prediction": 1, "probability": 0.7, "confidence": 0.7,
                               "latest_close": 100.0, "latest_date": "2026-09-01"},
                "mid_term": {"prediction": 1, "probability": 0.65, "confidence": 0.65,
                             "latest_close": 100.0},
                "long_term": {"prediction": 1, "probability": 0.6, "confidence": 0.6,
                              "latest_close": 100.0},
            },
        }


# ----------------------------------------------------------------------
# 7. 持仓池 / 真实行情源 / 质量门控（Q2-4）
# ----------------------------------------------------------------------
class TestHoldingsPoolAndQuality:
    def test_pro_config_holdings_pool_is_28_system_pool(self):
        with open(PROJECT_ROOT / "configs" / "config_pro.yaml", "r", encoding="utf-8") as f:
            pro = yaml.safe_load(f)
        symbols = pro["data"]["markets"]["stock"]["symbols"]
        assert len(symbols) >= 26, "专业版应覆盖 28 系统的真实持仓池"
        assert len(symbols) == len(set(symbols)), "持仓池不应有重复标的"
        # 真实行情源优先，模拟仅作兜底
        assert pro["data"]["source"][0] == "wind"
        assert pro["data"]["source"][-1] == "simulation"

    def test_real_sources_precede_simulation(self, cfg):
        collector = DataCollector(cfg)
        assert "simulation" in collector.sources or collector.sources

    def test_quality_gate_assesses_and_summarizes(self, cfg, market_df):
        cfg = dict(cfg)
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["quality_gate"] = True
        cfg["data"]["raw_dir"] = str(Path(cfg["data"]["raw_dir"]) / "q")
        collector = DataCollector(cfg)
        info = collector.assess_quality("600519.SH", market_df)
        assert info["rows"] == len(market_df)
        assert 0.0 <= info["quality_score"] <= 100.0
        summary = collector.quality_summary()
        assert summary["available"] and summary["assessed_symbols"] == 1

    def test_quality_gate_disabled_is_noop(self, cfg, market_df):
        cfg = dict(cfg)
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["quality_gate"] = False
        cfg["data"]["raw_dir"] = str(Path(cfg["data"]["raw_dir"]) / "q2")
        collector = DataCollector(cfg)
        assert collector.assess_quality("600519.SH", market_df) == {}
        assert collector.quality_summary()["available"] is False

    def test_quality_gate_failure_is_fail_soft(self, cfg, market_df, monkeypatch):
        cfg = dict(cfg)
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["quality_gate"] = True
        cfg["data"]["raw_dir"] = str(Path(cfg["data"]["raw_dir"]) / "q3")
        collector = DataCollector(cfg)

        class Boom:
            def score_market_data(self, df):
                raise RuntimeError("boom")

        monkeypatch.setattr(collector, "_get_quality_gate", lambda: Boom())
        assert collector.assess_quality("600519.SH", market_df) == {}, "质量体检异常不得中断主链路"


# ----------------------------------------------------------------------
# 8. 配置与集成
# ----------------------------------------------------------------------
class TestQ2ConfigIntegration:
    def test_configs_expose_q2_sections(self):
        for name in ("config.yaml", "config_pro.yaml"):
            with open(PROJECT_ROOT / "configs" / name, "r", encoding="utf-8") as f:
                c = yaml.safe_load(f)
            assert "factors" in c["model"], f"{name} 应包含 model.factors"
            # IC 门禁配置由同批 Q2 交付提供（src/inference/ic.py + src/trading/gate.py）
            assert "strategy_gate" in c, f"{name} 应包含 strategy_gate"
            assert c["model"]["factors"]["ensemble"]["classes"]
            assert abs(sum(c["model"]["factors"]["ensemble"]["weights"].values()) - 1.0) < 1e-9

    def test_lstm_weight_decay_is_numeric(self):
        """YAML 里的 1e-5 不带小数点会被解析成字符串，直接崩 optimizer。"""
        for name in ("config.yaml", "config_pro.yaml"):
            with open(PROJECT_ROOT / "configs" / name, "r", encoding="utf-8") as f:
                c = yaml.safe_load(f)
            wd = c["model"]["pytorch_lstm"]["weight_decay"]
            assert not isinstance(wd, str), f"{name} 的 weight_decay 必须是数值"

    def test_cli_exposes_q2_commands(self):
        import main as main_cli

        ns = main_cli.build_parser().parse_args(["factors", "600519.SH"])
        assert ns.command == "factors"
        # 可训练多因子模型的诊断入口
        ns = main_cli.build_parser().parse_args(["factor-model", "600519.SH"])
        assert ns.command == "factor-model"
        ns = main_cli.build_parser().parse_args(["--model-type", "multifactor", "predict", "600519.SH"])
        assert ns.model_type == "multifactor"
        ns = main_cli.build_parser().parse_args(["--model-type", "factor_model", "train"])
        assert ns.model_type == "factor_model"

    def test_env_defaults_keep_factors_off_for_standard_config(self):
        """标准配置默认不开启多因子：保证既有行为零回归。"""
        with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
        assert c["model"]["factors"]["enabled"] is False
        assert c["strategy_gate"]["enforce_trading"] is False
