"""量化交易适配层：将 TrendCast Pro 预测结果适配为可执行交易信号与订单流。

本子包面向外部量化交易程序（如用户自有的 tradingview 量化程序），提供：
  - 信号聚合（signal）   ：多周期方向预测 → 综合交易动作 BUY / SELL / HOLD
  - 风险管理（risk）     ：仓位计算（固定分数 / Kelly）、止损止盈、风险限额
  - 订单生成（orders）   ：标准化的下单载荷（side / size / price / type）
  - 统一适配（adapter）  ：predict → signal → risk → order 一键编排
  - 快速回测（backtest） ：基于历史信号统计假设成交的胜率 / 盈亏

纯标准库 + numpy / pandas 实现，无额外依赖，可直接被外部程序 import 复用。
"""
