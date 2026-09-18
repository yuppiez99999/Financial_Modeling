# 模型退化诊断结论（2026-09-15）

> 诊断对象：全量 26 只 × 3 周期 = **78 条预测全部「看涨」**、概率挤在 50.56%~51.29%、同周期跨标的几乎并列。
> 结论一句话：**不是工程 bug，是模型概率未校准导致的输出常数化**。

---

## 一、结论

| 判定 | 结果 |
|:---|:---|
| 训练/推理特征错位（工程 bug） | **排除**（顺序与集合完全一致，无填 0/截断） |
| 推理期特征取值异常 | **排除**（推理输出 = 模型在测试集上的典型输出） |
| 真实成因 | **模型概率未校准 + 正样本率略偏正 ⇒ 输出被压缩在先验附近（spread≈1pp），固定阈值 0.5 必然全判「看涨」** |
| 方向信号是否有信息 | **无**（恒定看涨） |
| 排序信号是否有信息 | **有微弱价值**（AUC 0.52~0.58） |

---

## 二、排除「特征错位」的证据（实测）

| 检查项 | 结果 |
|:---|:---|
| 推理侧特征数 vs 模型 `n_features_in_` | 107 vs 107（一致，scaler 亦为 107） |
| 训练侧列 vs 推理侧列（顺序敏感） | **SAME_ORDER = True** |
| 训练侧列 vs 推理侧列（集合） | **SAME_SET = True，DIFF_N = 0** |
| 全量日志「特征数」填 0/截断告警 | **0 次**（78 条预测从未触发） |
| `transform` / `get_feature_columns` 是否使用 `horizon_days` | **均未使用**（preprocessor.py L101-132、L135-147）⇒ 5/10/20 日口径一致 |
| 模型 pkl 内容 | 仅 `['model','scaler']`；`feature_name_` = `Column_0..106`（训练喂 ndarray，**无名称级校验**，属隐患但非本次成因） |

> **关于 `data/processed/*.csv`**：该目录（47 个文件，09-10 落盘）**早于**模型（09-13），且含旧口径列（`a_probability`/`completeness`/`future_return_5d`），而当前推理侧含 `factor_*` 多因子特征 ⇒ 二者 `SAME_SET=False`。
> 因此它**不代表当前模型的训练特征**，不能作为判据；本次改用**当前代码复现训练路径**（逐标的 `transform` + `create_target` → `concat` → 取列）与推理侧比对，结论为一致。

---

## 三、量化模型输出（测试集 = 后 30% 时间切分）

| 标的 | 样本数 | AUC | 正样本率 | 概率 std | p5 | **p50** | p95 | min | max |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300308.SZ | 1001 | 0.5845 | 52.25% | 0.00141 | 0.5036 | **0.5056** | 0.5086 | 0.5016 | 0.5115 |
| 601088.SH | 1357 | 0.5244 | 54.02% | 0.00167 | 0.5038 | **0.5056** | 0.5105 | 0.5016 | 0.5124 |

关键读数：

1. **推理输出 = 测试集中位数**：300308.SZ 推理 short_term = 0.5056，测试集 p50 = 0.5056 ⇒ 推理值完全正常，无异常。
2. **概率 spread 极小**：std 仅 0.0014~0.0017，全距约 1pp（0.5016~0.5124）。
3. **跨标的中位数完全相同**（均为 0.5056）⇒ 模型输出近乎常数，与个标的特征关系极弱。
4. **AUC 0.52~0.58**：排序有微弱区分度（优于随机），但概率**绝对值**几乎无信息。

---

## 四、根因链

1. 训练集正样本率 52%~54%（略偏多）⇒ 模型先验略偏正；
2. 概率**未校准**（S19/H4 的 platt/isotonic 校准层默认未启用）⇒ 输出被压缩在先验附近，缺乏 spread；
3. 方向判定用固定阈值 `proba > 0.5` ⇒ 因全部概率落在 0.5016~0.5124（均 >0.5），**必然全部判为「看涨」**。

⇒ 「78 条全看涨」是 **阈值 0.5 + 未校准概率**的必然产物，**不代表模型认为 26 只都上涨**；方向信号本身无信息量。

这与 README 记录的「真实行情 AUC 0.54~0.57、区分度有限、仅作只读观测参考」一致，本诊断用实测量化了这一现象的具体形态。

---

## 五、影响与建议（诚实口径）

- **方向信号（看涨/看跌）：无信息，不可用于交易/选股决策。**
- **排序信号（相对强弱）：有微弱价值**（AUC 0.52~0.58），仅可用于「相对比较」，不可用于绝对方向。
- **改进方向（按性价比）**：
  1. **启用概率校准**（S19/H4 已实现 platt/isotonic，实测 ECE 0.1042→0.0025）—— 校准后概率获得 spread，阈值判定才有意义。该改动属 **T19.4 人工检查点（当前 defer）**，须人工签字后方可进主推理链路。
  2. 改用**截面分位数/排序**产生信号（top-N / bottom-N），绕开绝对阈值 0.5。
  3. 若要进 28 仓二期决策路径，须先按设计文档 §5 过命中率门禁 —— 当前方向信号恒定，无法满足。
- **明确不建议**：为「消除全看涨」而调阈值或关闭对齐 —— 只是掩盖，不产生真实信息。

---

## 六、复现命令

```powershell
$env:PYTHONIOENCODING="utf-8"
cd e:\各种PY程序\16_金融市场预测模型

# 1) 特征对齐比对（训练侧复现 vs 推理侧）
python -c "import sys; sys.path.insert(0,'.'); import pandas as pd, yaml; from src.data.preprocessor import FeatureEngineer; cfg=yaml.safe_load(open('configs/config_pro.yaml',encoding='utf-8')); fe=FeatureEngineer(cfg); parts=[]; [parts.append(fe.create_target(fe.transform(pd.read_csv('data/raw/'+s+'.csv')),5)) for s in ['300308.SZ','601088.SH']]; c=pd.concat(parts, ignore_index=True).sort_values('date').dropna(); tc=fe.get_feature_columns(c,5); dff=fe.transform(pd.read_csv('data/raw/300308.SZ.csv'),5); ic=fe.get_feature_columns(dff,5); print('TRAIN_N',len(tc)); print('INFER_N',len(ic)); print('SAME_ORDER',tc==ic); print('SAME_SET',set(tc)==set(ic))"

# 2) 测试集概率分布与 AUC（以 300308.SZ 为例）
python -c "import sys; sys.path.insert(0,'.'); import pandas as pd, numpy as np, yaml, joblib; from src.data.preprocessor import FeatureEngineer; from sklearn.metrics import roc_auc_score; cfg=yaml.safe_load(open('configs/config_pro.yaml',encoding='utf-8')); fe=FeatureEngineer(cfg); md=joblib.load('models/lightgbm_short_term_5d.pkl'); m=md['model']; sc=md['scaler']; d=pd.read_csv('data/raw/300308.SZ.csv'); f=fe.create_target(fe.transform(d),5).dropna(); cols=fe.get_feature_columns(f,5); te=f.iloc[int(len(f)*0.7):]; y=te['target_5d'].values; p=m.predict_proba(sc.transform(te[cols].values))[:,1]; print('N',len(te),'AUC',round(roc_auc_score(y,p),4),'std',round(float(np.std(p)),5),'p50',round(float(np.percentile(p,50)),4),'min',round(float(p.min()),4),'max',round(float(p.max()),4))"
```

---

## 七、遗留隐患（非本次成因，建议后续处理）

模型 pkl **未持久化 `feature_cols`**，且 `feature_name_` 为 `Column_i` ⇒ 训练/推理之间**无任何名称级校验**；`_align_features`（predictor.py L28-45）**只按数量对齐**（多则截断、少则填 0），数量相同时静默放行。

本次因顺序恰好一致未出问题，但**一旦特征管道变更（列顺序/集合变化）就会静默错位且完全不可见**。

建议（未实施，需另开计划）：训练时把 `feature_cols` 持久化进 pkl；推理时按名称重排，缺失列显式报错或标记 degraded（不再静默填 0/截断）。
