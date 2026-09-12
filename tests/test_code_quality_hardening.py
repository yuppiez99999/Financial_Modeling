"""代码质量加固回归（Issue #40：check 系统代码质量 bug 漏洞）。

本文件把本轮**实际复现过**的缺陷固化为断言。每一条都先有复现脚本、
再有修复，避免"改完没有守卫、下次照样退化"。

覆盖 6 类：
  1. ic.py       : 非有限值（inf）静默污染 IC / 未被过滤
  2. signal.py   : probability/confidence 越界放大 score，突破 [-1,1] 契约
  3. notifier.py : 默认关闭 TLS 证书校验（中间人风险）
  4. run_daily   : 阶段完成判定漏 confirmed / 跳过未落定任务
  5. run_daily   : 缺字段或索引越界把整轮推进吞成 runtime_error
  6. orders.py   : 类型注解前置引用（F821）+ 主单价死代码
"""
import importlib.util
import json
import ssl
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "schedule" / "run_daily.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("schedule_run_daily_hardening", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path, plan: dict) -> dict:
    sched = tmp_path / "schedule"
    sched.mkdir(parents=True, exist_ok=True)
    (sched / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    _load_runner().run_day(str(sched / "plan.json"), str(sched / "logs"))
    return json.loads((sched / "plan.json").read_text(encoding="utf-8"))


# 1. IC 非有限值过滤
class TestICRejectsNonFinite:
    def test_isnan_treats_inf_as_unavailable(self):
        """inf 必须与 NaN 同等对待为「不可用值」。

        原实现用 value != value，只能识别 NaN；inf 会被当成"最大的一个
        普通值"进入秩计算，IC 被静默算成一个有限的假值。
        """
        from src.inference.ic import _isnan

        assert _isnan(float("nan")) is True
        assert _isnan(float("inf")) is True, "inf 必须被判为不可用（否则静默污染 IC）"
        assert _isnan(float("-inf")) is True
        assert _isnan(1.0) is False
        assert _isnan("0.5") is False
        assert _isnan(None) is True
        assert _isnan("not-a-number") is True

    def test_inf_samples_excluded_from_evaluation(self):
        """含 inf 的样本不得进入 IC 评估分母（应与 NaN 一样被剔除）。"""
        import numpy as np

        from src.inference.ic import ICCalculator

        rng = np.random.default_rng(0)
        n = 200
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        scores = list(rng.normal(0, 1, n))
        fwd = [float(closes[i + 5] / closes[i] - 1) for i in range(n - 5)] + [None] * 5

        calc = ICCalculator({})
        clean = calc.evaluate("mid_term", 5, scores, fwd, window_size=30)

        dirty = list(scores)
        dirty[10] = float("inf")
        mixed = calc.evaluate("mid_term", 5, dirty, fwd, window_size=30)

        assert mixed.samples == clean.samples - 1, (
            "inf 样本未被剔除：clean=%s mixed=%s" % (clean.samples, mixed.samples)
        )
        assert abs(mixed.ic) <= 1.0


# 2. Signal 概率口径收口
class TestSignalProbabilityContract:
    def test_out_of_range_probability_does_not_break_score_bounds(self):
        """probability 越界不得让 score 冲出 [-1, 1]（文档契约）。"""
        from src.trading.signal import SignalEngine

        engine = SignalEngine({})
        for proba in (2.0, 10.0, 100.0, -5.0, float("inf"), float("nan")):
            sig = engine.build_signal("X", {"predictions": {
                "short_term": {"prediction": 1, "probability": proba, "confidence": 0.5},
            }})
            assert -1.0 <= sig.score <= 1.0, "probability=%s 导致 score=%s 越界" % (proba, sig.score)
            assert 0.0 <= sig.strength <= 1.0
            assert 0.0 <= sig.confidence <= 1.0

    def test_confidence_out_of_range_is_clamped(self):
        from src.trading.signal import SignalEngine

        sig = SignalEngine({}).build_signal("X", {"predictions": {
            "short_term": {"prediction": 1, "probability": 0.9, "confidence": 99.0},
        }})
        assert sig.confidence == pytest.approx(1.0)

    def test_normal_probability_semantics_unchanged(self):
        """收口不得改变正常取值的语义（防止过度夹取）。"""
        from src.trading.signal import SignalEngine

        sig = SignalEngine({}).build_signal("X", {"predictions": {
            "short_term": {"prediction": 1, "probability": 0.75, "confidence": 0.8},
        }})
        # 0.75 -> mag = 0.5；short_term 权重 = 0.30（三周期默认权重归一化后）
        # => score = 0.5 * 0.30 = 0.15，与修复前的语义逐位一致
        assert sig.score == pytest.approx(0.15, abs=1e-9)


# 3. TLS 默认安全
class TestNotifierTLS:
    def test_tls_verification_enabled_by_default(self):
        """默认必须校验证书与主机名（原实现无条件 CERT_NONE）。"""
        from src.notification.notifier import SignalNotifier

        notifier = SignalNotifier({})
        assert notifier._ssl_ctx.verify_mode == ssl.CERT_REQUIRED
        assert notifier._ssl_ctx.check_hostname is True

    def test_tls_can_only_be_disabled_explicitly(self):
        from src.notification.notifier import SignalNotifier

        notifier = SignalNotifier({"notification": {"insecure_skip_tls_verify": True}})
        assert notifier._ssl_ctx.verify_mode == ssl.CERT_NONE
        assert notifier.insecure_skip_tls_verify is True


# 4/5. 排期推进器状态口径
class TestRunnerStatusAccounting:
    def test_confirmed_manual_checkpoint_counts_as_done(self, tmp_path):
        """人工检查点终态是 confirmed（非 completed），必须被认作已落定。

        原实现只认 completed，导致带人工检查点的阶段永远无法收官。
        """
        out = _run(tmp_path, {
            "stages": [
                {"id": "S1", "name": "s1", "status": "in_progress", "tasks": [
                    {"id": "T1.1", "name": "auto", "status": "completed", "auto_run": True},
                    {"id": "T1.2", "name": "人工检查点", "status": "confirmed",
                     "confirmed_by": "u", "confirmed_at": "2026-01-01T00:00:00Z",
                     "decision": "keep"},
                ]},
                {"id": "S2", "name": "s2", "status": "pending",
                 "tasks": [{"id": "T2.1", "name": "auto2", "status": "pending"}]},
            ],
            "current_stage_index": 0, "current_task_index": 1,
        })
        assert out["stages"][0]["status"] == "completed", "confirmed 检查点应让阶段正常收官"
        assert out["current_stage_index"] == 1, "收官后应推进到下一阶段"

    def test_completed_stage_with_unfinished_task_is_not_skipped(self, tmp_path):
        """阶段标 completed 却有未落定任务时必须阻塞，绝不静默跳过。"""
        out = _run(tmp_path, {
            "stages": [
                {"id": "S1", "name": "s1", "status": "completed", "tasks": [
                    {"id": "T1.1", "name": "a", "status": "completed"},
                    {"id": "T1.2", "name": "b", "status": "pending"},
                ]},
                {"id": "S2", "name": "s2", "status": "pending",
                 "tasks": [{"id": "T2.1", "name": "c", "status": "pending"}]},
            ],
            "current_stage_index": 0, "current_task_index": 0,
        })
        assert out["result"] == "blocked_inconsistent_stage"
        assert out["current_stage_index"] == 0, "不得推进到下一阶段"
        assert out["next_run_date"] is None, "异常态不应再排下一次自动推进"
        assert any("未落定" in b for b in out.get("blockers", []))

    def test_missing_task_name_does_not_crash_run(self, tmp_path):
        """任务缺 name 字段时应如实阻塞，而不是被吞成 runtime_error。"""
        out = _run(tmp_path, {
            "stages": [{"id": "S1", "name": "s1", "status": "in_progress",
                        "tasks": [{"id": "T1.1", "status": "pending"}]}],
            "current_stage_index": 0, "current_task_index": 0,
        })
        assert not any("runtime_error" in b for b in out.get("blockers", [])), (
            "缺字段不应产生未捕获异常：%s" % out.get("blockers")
        )

    def test_empty_stage_is_not_treated_as_done(self, tmp_path):
        """空阶段不得被判为完成（否则阶段被真空收官）。"""
        out = _run(tmp_path, {
            "stages": [
                {"id": "S1", "name": "s1", "status": "in_progress", "tasks": []},
                {"id": "S2", "name": "s2", "status": "pending",
                 "tasks": [{"id": "T2.1", "name": "c", "status": "pending"}]},
            ],
            "current_stage_index": 0, "current_task_index": 0,
        })
        assert out["stages"][0]["status"] != "completed"
        assert out["result"] == "blocked_empty_stage"

    def test_task_index_out_of_range_is_reported(self, tmp_path):
        out = _run(tmp_path, {
            "stages": [{"id": "S1", "name": "s1", "status": "in_progress",
                        "tasks": [{"id": "T1.1", "name": "a", "status": "pending"}]}],
            "current_stage_index": 0, "current_task_index": 9,
        })
        assert out["result"] == "blocked_task_index_out_of_range"
        assert not any("runtime_error" in b for b in out.get("blockers", []))


# 6. orders 类型注解与主单价
class TestOrdersModule:
    def test_module_imports_without_risk_import(self):
        """orders.py 的类型注解必须能在不导入 risk 的情况下解析（原为 F821）。"""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c",
             "import src.trading.orders as m; print(m.OrderGenerator)"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, result.stderr

    def test_main_order_price_is_none_not_take_profit(self):
        """主单价恒为 None（执行端决定），不得把止盈价误当主单价。"""
        from src.trading.orders import OrderGenerator
        from src.trading.risk import RiskManager
        from src.trading.signal import SignalEngine

        sig = SignalEngine({}).build_signal("600519.SH", {"predictions": {
            h: {"prediction": 1, "probability": 0.95, "confidence": 0.9}
            for h in ("short_term", "mid_term", "long_term")
        }})
        assert sig.action == "BUY"
        budget = RiskManager({}).budget(sig, price=1700.0)
        orders = OrderGenerator({}).generate(budget)

        assert orders, "应生成主单"
        assert orders[0].price is None, "主单价应为 None"
        assert orders[0].price != budget.take_price, "主单价不得等于止盈价"


# 7. API 不回显内部异常 / CORS 默认非通配
class TestApiErrorSanitization:
    def test_internal_error_hides_exception_text(self):
        from src.api.server import _internal_error

        exc = _internal_error("预测失败", RuntimeError("/etc/secret/path.py 不存在"))
        assert "secret" not in str(exc.detail), "不得把内部异常文本回显给调用方"
        assert exc.status_code == 500

    def test_cors_default_is_not_wildcard(self):
        from src.api.server import DEFAULT_CORS_ORIGINS, _cors_settings

        origins, credentials = _cors_settings()
        assert "*" not in origins, "默认 CORS 不得为通配来源"
        assert origins == DEFAULT_CORS_ORIGINS
        assert credentials is False, "不应开启 credentials"


# 8. main.py 重复定义 / 重复 CLI 分支（静默遮蔽）
class TestNoSilentShadowing:
    def test_main_has_no_duplicate_top_level_definitions(self):
        """main.py 顶层函数不得重名。

        真实缺陷：曾同时存在两个 ``run_overfit_audit``（1919 行与 2475 行），
        后一个静默遮蔽前一个 —— 读代码的人以为在改 A，实际跑的是 B，
        且无任何报错。重复定义属于"静默遮蔽"，必须由静态守卫拦住。
        """
        import ast

        src = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        seen: dict[str, list[int]] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                seen.setdefault(node.name, []).append(node.lineno)
        dups = {k: v for k, v in seen.items() if len(v) > 1}
        assert not dups, f"main.py 存在重复定义（后者会静默遮蔽前者）: {dups}"

    def test_cli_dispatch_has_no_duplicate_command_branch(self):
        """CLI 分派中同一命令不得出现两次（后一个分支恒为死代码）。"""
        src = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        commands = [
            line.split('"')[1]
            for line in src.splitlines()
            if "args.command ==" in line and '"' in line
        ]
        seen: set[str] = set()
        dups = [c for c in commands if c in seen or seen.add(c)]
        assert not dups, f"CLI 分派存在重复命令分支（死代码）: {dups}"
