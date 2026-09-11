"""资产类别识别：把标的池拆成「波动结构同质」的分池。

背景（S7 → S9）：
  S7 的 `--stratify` 实测给出关键证据 —— **池化口径被"木桶短板"绑架**：
  把波动结构完全不同的宽基 ETF 与个股混在一条 IC 序列里，short_term 卡在门槛线上。
  S7 的结论写的是「下一步：按标的分层建模 / 按资产类别分别设门禁」，
  但那是**建议**，未落地。S9 就是把这条建议落地。

设计原则：
  - **零网络**：纯代码规则识别，不调任何行情接口、不读股票名称数据库。
    数据源故障时不得影响门禁判定（门禁是准入闸，不能因为分类器挂掉而放行）；
  - **可解释**：命中规则（`rule`）显式返回，人能一眼看出"为什么把它归到 ETF"；
  - **不猜**：无法识别时归入 `unknown` 分池，**不硬塞进 stock** ——
    硬塞等于把未知风险混进已知口径，会让指标失去含义。

A 股代码规则（上交所/深交所公开规则）：
  - `51xxxx` / `56xxxx` / `58xxxx` 上交所基金（ETF / LOF）；
  - `159xxx` 深交所基金（ETF / LOF）；
  - `60xxxx` 上交所主板、`68xxxx` 科创板、`00xxxx` 深主板、
    `30xxxx` 创业板 —— 个股；
  - `11xxxx` / `12xxxx` 可转债（既非个股也非基金，另立分池）。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

# 分池标识（下游门禁 / 报表 / 报告统一用这几个字符串做键）
CLASS_STOCK = "stock"
CLASS_ETF = "etf"
CLASS_FUTURES = "futures"
CLASS_FOREX = "forex"
CLASS_CONVERTIBLE = "convertible"
CLASS_UNKNOWN = "unknown"

# 分池显示名（报表直接读，避免各处硬编码中文）
CLASS_LABELS: Dict[str, str] = {
    CLASS_STOCK: "个股",
    CLASS_ETF: "ETF / 场内基金",
    CLASS_FUTURES: "期货",
    CLASS_FOREX: "外汇",
    CLASS_CONVERTIBLE: "可转债",
    CLASS_UNKNOWN: "未识别",
}

# 分池遍历顺序（报表/报告输出顺序稳定，便于 diff 与快照测试）
CLASS_ORDER = (
    CLASS_STOCK,
    CLASS_ETF,
    CLASS_FUTURES,
    CLASS_FOREX,
    CLASS_CONVERTIBLE,
    CLASS_UNKNOWN,
)

# 交易所后缀：.SH 上交所 / .SZ 深交所 / 其余为期货、外汇等
_CN_EXCHANGES = ("SH", "SZ", "BJ")
_FUTURES_EXCHANGES = ("SHF", "DCE", "CZC", "INE", "CFE", "GFEX")
_FOREX_EXCHANGES = ("FXCM", "FX", "FOREX")

_CODE_RE = re.compile(r"^(?P<code>[0-9A-Za-z]+)(?:\.(?P<suffix>[A-Za-z]+))?$")


def normalize_symbol(symbol: str) -> str:
    """统一大小写与空格（`600519.sh` → `600519.SH`）。"""
    return str(symbol or "").strip().upper()


def classify(symbol: str) -> Dict[str, str]:
    """识别单只标的所属资产类别。

    Returns:
        ``{"symbol": ..., "asset_class": ..., "code": ..., "exchange": ..., "rule": ...}``
        —— `rule` 是命中规则的短说明，供报表解释"为什么这么分"。
    """
    sym = normalize_symbol(symbol)
    m = _CODE_RE.match(sym)
    if not m:
        return _result(sym, CLASS_UNKNOWN, "", "", "无法解析代码格式")
    code = m.group("code") or ""
    suffix = (m.group("suffix") or "").upper()

    if suffix in _FUTURES_EXCHANGES:
        return _result(sym, CLASS_FUTURES, code, suffix, f"交易所后缀 .{suffix}")
    if suffix in _FOREX_EXCHANGES:
        return _result(sym, CLASS_FOREX, code, suffix, f"交易所后缀 .{suffix}")
    if suffix and suffix not in _CN_EXCHANGES:
        return _result(sym, CLASS_UNKNOWN, code, suffix, f"未知交易所后缀 .{suffix}")

    if code.startswith(("51", "56", "58", "159")):
        return _result(sym, CLASS_ETF, code, suffix, "场内基金代码段 51/56/58/159")
    if code.startswith(("60", "68", "00", "30")):
        return _result(sym, CLASS_STOCK, code, suffix, "个股代码段 60/68/00/30")
    if code.startswith(("11", "12")) and len(code) == 6:
        return _result(sym, CLASS_CONVERTIBLE, code, suffix, "可转债代码段 11/12")
    return _result(sym, CLASS_UNKNOWN, code, suffix, "不在已知代码段内")


def _result(symbol: str, asset_class: str, code: str, exchange: str, rule: str) -> Dict[str, str]:
    return {
        "symbol": symbol,
        "asset_class": asset_class,
        "code": code,
        "exchange": exchange,
        "rule": rule,
    }


def classify_symbols(symbols) -> Dict[str, Dict[str, str]]:
    """批量识别，返回 ``{symbol: 识别结果}``（保序）。"""
    return {normalize_symbol(s): classify(s) for s in symbols or []}


def group_symbols(symbols) -> Dict[str, list]:
    """按资产类别分组，返回 ``{asset_class: [symbol, ...]}``。

    分池顺序按 ``CLASS_ORDER``；空分池不出现（避免报表出现一堆空章节）。
    """
    buckets: Dict[str, list] = {}
    for res in classify_symbols(symbols or []).values():
        buckets.setdefault(res["asset_class"], []).append(res["symbol"])
    return {k: buckets[k] for k in CLASS_ORDER if buckets.get(k)}


def label(asset_class: str) -> str:
    """分池显示名（未登记的分池原样返回，不吞掉信息）。"""
    return CLASS_LABELS.get(asset_class, asset_class or CLASS_UNKNOWN)


def describe(symbol: str) -> str:
    """一行人类可读的识别说明（供 CLI / 报表）。"""
    res = classify(symbol)
    return f"{res['symbol']} → {label(res['asset_class'])}（{res['rule']}）"


def summarize_classification(symbols) -> Dict[str, Any]:
    """识别结果汇总：每类标的数 + 明细，供落盘与报表消费。"""
    by_symbol = classify_symbols(symbols or [])
    groups = group_symbols(symbols or [])
    return {
        "total": len(by_symbol),
        "counts": {k: len(v) for k, v in groups.items()},
        "groups": groups,
        "labels": {k: label(k) for k in CLASS_ORDER},
        "unknown": groups.get(CLASS_UNKNOWN, []),
        "items": list(by_symbol.values()),
    }


def to_pool_filter(asset_class: str):
    """返回一个 ``symbol -> bool`` 的分池判定函数（供评估脚本按池过滤）。"""
    target = asset_class or CLASS_UNKNOWN

    def _keep(symbol: str) -> bool:
        return classify(symbol)["asset_class"] == target

    return _keep


def resolve_pool(data: Dict[str, Any], asset_class: str,
                 symbol_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """从行情字典里挑出属于指定分池的标的。

    Args:
        data       : ``{symbol: DataFrame}``；
        asset_class: 目标分池；
        symbol_map : 可选覆盖表 ``{symbol: asset_class}``（配置显式声明优先，
                     便于把识别不了的自定义代码手工归类，仍然零网络）。

    Returns:
        过滤后的 ``{symbol: DataFrame}``（保序；无命中返回空 dict，绝不回退成全池 ——
        「没找到 ETF」和「把所有标的都当 ETF」是两件完全不同的事）。
    """
    override = {normalize_symbol(k): v for k, v in (symbol_map or {}).items()}
    out: Dict[str, Any] = {}
    for symbol, df in (data or {}).items():
        cls = override.get(normalize_symbol(symbol)) or classify(symbol)["asset_class"]
        if cls == asset_class:
            out[symbol] = df
    return out
