"""qlib 集成层（S13 / G3）：数据转 .bin → 导出横截面因子 → 增量验证。

对外契约（与 qlib 是否安装**解耦**，CI 离线可跑）：
  - dump_bin    : 把本项目 data/raw/*.csv 行情转成 qlib .bin 列存格式；
  - export_alpha158 : 从行情 DataFrame 逐标的计算 Alpha158 风格因子（纯 pandas 实现，
                    与 qlib 官方 Alpha158 表达式**逐因子对齐**，不依赖 qlib 运行时）；
  - export_factors_to_parquet : 因子矩阵落盘 parquet，供 qlib_factor_provider 消费。

边界（README §18A 统一原则）：
  - 所有产出 `affects_gate` 恒为 False，report_only；
  - qlib 未安装 / 数据缺失时 fail-soft，不阻断主链路；
  - 引入的第三方来源已在 docs/THIRD_PARTY.md 登记。
"""
