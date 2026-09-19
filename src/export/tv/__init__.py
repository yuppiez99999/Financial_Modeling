"""TradingView 交付层：把只读决策源契约翻译成 TradingView 可原样读入的资产。

- ``png_writer``  : 零依赖 PNG 编码器（含 tEXt 元数据块）；
- ``signal_card`` : 单张 PNG 信号卡（数值 + 元数据 + 锚点序列）；
- ``pine_json``   : Pine 外挂 JSON 契约层；
- ``handoff``     : 一次生成 png / json / report 的编排入口。
"""
