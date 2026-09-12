# 非有限值贯通检查 · 落地结论（Issue #40）

> 执行方式：延续「代码质量 / bug / 漏洞」检查轮（PR #52）的方法论 ——
> **静态筛查 + 动态复现双轨，先有复现再改代码**，修完把原缺陷回流一遍确认守卫会失败。
> 全局纪律：`strategy_gate` **零改动**、`affects_gate` 恒为 false、不改任何策略判据。

---

## 一、为什么会有这一轮

上一轮修掉了 `ic.py` 里「`_isnan` 只用 `value != value`，**认不出 `inf`**，
导致 inf 被当成"最大的普通值"进入排序、IC 被算成有限的假值」这一类缺陷。

本轮把**同一类缺陷沿下游链路继续查**：信号聚合 → 风控仓位 → 下单。
结论是**这层比上游更危险** —— 上游是"算错一个数"，
下游是**直接决定买卖方向与仓位**。

## 二、确认并修复的 3 组真实缺陷

### P0-1 · `probability = -inf` 会把看涨预测**翻成做空单**

`src/trading/signal.py::_direction_value` 原先：

```python
proba = float(horizon_pred.get("probability", 0.5))
mag = (proba - 0.5) * 2.0          # [-1,1]（注释这么写，但没有任何保证）
return 1.0 * mag if pred == 1 else -1.0 * mag
```

该函数**没有任何概率契约校验**。动态复现（`prediction=1` 即看涨）：

| probability | score | strength | action | 说明 |
|:---:|:---:|:---:|:---:|:---|
| 0.9（正常） | 0.8 | 0.8 | BUY | 正确 |
| 10 | **19.0** | **19.0** | BUY | **越界 19 倍**，污染日报 / 审计 / API |
| +inf | **inf** | **inf** | BUY | 非有限值直通 |
| **-inf** | **-inf** | **inf** | **SELL** | ⚠️ **看涨预测被判成做空** |
| NaN | nan | nan | HOLD | 侥幸落到 HOLD |

`-inf` 那一行是本轮最严重的读数：**一个坏掉的概率悄悄把方向反转了**，
且 `direction_consensus` 输出"看跌" —— 与三个周期各自的
`prediction: 1`（看涨）**完全相反**，事后审计极难看出。

端到端到下单（`RiskManager` + `OrderGenerator`）：

```
-inf → action=SELL pos_pct=0.095 qty=950 stop=103.0 take=94.0
```

即**凭一个坏值开出 950 股空单**，风控还正常给了止损止盈，链路无一处拦截。

**修复口径**：概率是 `[0,1]` 契约。NaN / ±inf / 越界**不是"极端方向"，而是数据已损坏**，
一律 fail-safe 到中性 0.5（方向分 0）+ `WARNING` 留痕。
另对综合得分做一次 `clamp` 到 `[-1,1]` 作契约硬校验（越界即说明上游装配异常，留痕）。

> 顺带修掉一处浮点越界：`probability=1.0` 时原实现算出
> `score = -1.0000000000000002`（恰好戳破 `[-1,1]` 契约下界）。

### P0-2 · `strength = inf` 被 `min()` 夹成"满档仓位"

`src/trading/risk.py::_sizing_fraction` 原先：

```python
f = self.base_position_pct * (0.5 + min(strength, 1.0) * 0.5)
f = f * (0.5 + min(confidence, 1.0) * 0.5)
```

`min(x, 1.0)` **只有上界**：`strength = inf` 被夹到 `1.0`，
等于把"数据坏了"解释成"史上最强信号" → 满档仓位。
`NaN` 则让所有比较为 False，走 `max(0.0, min(f, cap))` 得到不可预期结果。

| strength | 修复前 frac | 修复后 frac |
|:---:|:---:|:---:|
| 0.8（正常） | 0.0855 | 0.0855 |
| **1.0（合法上界）** | 0.095 | 0.095 |
| **inf** | **0.095（= 上界）** | **0.0475（中性，低于正常）** |
| NaN | 0.0（侥幸） | 0.0475 |

**修复后 inf 的仓位低于合法上界 strength 的仓位** —— 这才是正确语义：
坏值不配得到"强信号"的敞口。Kelly 分支的 `proba_win` 同属概率契约，一并收口。

### P1 · `predictor.py` 闭包捕获循环变量（结构性隐患）

`src/inference/predictor.py::load_models` 里两处 `def _instantiate():` 直接引用循环变量
`horizon_days`（ruff `B023` 已报）。当前**恰好**因为
`try_load_timesfm_or_instance` 在循环内立即调用该闭包而没出问题，
但这是**靠调用时序侥幸**。一旦闭包被延迟调用（收集回调、线程池、缓存），
三个周期会统一拿到循环结束后的最后一个值：

```
延迟调用 → short_term_5d = 20 / mid_term_10d = 20 / long_term_20d = 20
```

即 `short_term_5d` 模型**静悄悄按 20 日周期预测**。
**修复**：以默认参数显式按值绑定。修复后：

```
延迟调用 → short_term_5d = 5 / mid_term_10d = 10 / long_term_20d = 20
```

> **如实说明**：这一项在当前仓库中**不可直接触发**（调用路径是立即调用），
> 属"结构性隐患"而非"已发生故障"。修复理由是**把它变成结构性不可能**，
> 而不是依赖调用时序。列为 P1 而非 P0 正因如此。

## 三、双向验证（每条断言都验过失败态）

新增 `tests/test_nonfinite_containment.py`（16 例）。**在原始实现上跑，9 例失败**：

```
FAILED test_score_never_exceeds_unit_interval_on_overflow  AssertionError: prob=10.0 产出越界 score=19.0
FAILED test_inf_does_not_flip_direction                    AssertionError: 非有限概率必须退化为中性方向分
FAILED test_inf_positive_does_not_saturate_to_buy          AssertionError: assert 'BUY' == 'HOLD'
FAILED test_non_numeric_probability_does_not_raise         ValueError: could not convert string to float: 'bad'
FAILED test_boundary_probabilities_are_accepted            AssertionError: assert -1.0 <= -1.0000000000000002
FAILED test_confidence_nonfinite_is_contained              AssertionError: assert False
FAILED test_infinite_strength_is_not_treated_as_max_conviction
FAILED test_deferred_instantiate_keeps_each_horizon        {'short_term_5d': 20} != {'short_term_5d': 5}
```

修复后 **16/16 通过**，全量回归 **1144 passed, 4 skipped**（基线 1128，+16）。

## 四、仍如实说明（本轮未处理）

- **Kelly 分支在当前默认参数下与 `p` 无关**：默认
  `take_profit_pct / stop_loss_pct = 0.06 / 0.03 = 2.0`，
  则任意 `p ≥ 0.5` 都算出 `f ≥ 0.25 ≥ max_kelly` → **恒被截断到 `max_kelly`**。
  这是**既有的设计性质**（非本轮引入），但意味着 `sizing_method: kelly`
  实际并不响应胜率。**属判据语义，未擅自改动**，列此供决定是否立项。
  > 注：`sizing_method` 默认是 `fixed_fractional`，Kelly 当前无测试覆盖。
- `ruff` 剩余存量项（`F401` 未用导入、`DTZ005` 无时区、`I001` 导入排序等）
  与上轮一致，不影响行为，宜独立清理。
- `src/report/daily_report.py` / `src/monitor/health_report.py` 内 `_cell` 闭包
  同样捕获循环变量（`B023`），但**确认在循环内立即调用**，当前无 bug；
  未改（避免噪声改动），如需一并做值绑定可后续处理。

## 五、复现命令

```bash
pytest tests/test_nonfinite_containment.py -q     # 16 passed（原实现 9 failed）
pytest tests -q                                   # 1144 passed, 4 skipped
python main.py signal 600519.SH                   # 链路可用，缺模型如实降级
```

## 六、边界

- `strategy_gate` **零改动**，`affects_gate` 恒 false；
- 不改任何策略判据、不改配置、不动人工检查点状态（仍 `confirmed`）；
- 主线只有一条：**数据坏了要 fail-safe，不得被静默放大、反转或吞掉**。
