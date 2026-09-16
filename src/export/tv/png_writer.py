"""零依赖 PNG 编码器（真彩 RGB8 + tEXt 元数据块）。

## 为什么自己写

TradingView 的图片导入是**客户端**行为，平台不做二次渲染 —— 所以交付物必须是
一张**真 PNG**。本项目运行时没有图像库（Pillow 不在 requirements），为了
「图片信号卡」引入 Pillow 只为一个交付物不划算，因此按 PNG 规范直译：

- IHDR：8bit / color type 2（truecolor RGB）/ 无隔行；
- IDAT：逐行 ``filter=0``（None）后 zlib 压缩 —— 纯 numpy 实现，无需外部库；
- tEXt：每个关键字一个块，载荷是 **UTF-8 字节**（TradingView / Chromium /
  ImageMagick / PIL 均按 UTF-8 解 tEXt，中文落盘不乱码，实测见
  ``tests/test_tv_export.py``）。

## 边界

- 只支持 8bit RGB（本模块的用途是信号卡，不需要调色板 / 透明通道）；
- 坐标越界一律**裁剪**而不是抛错：信号卡是报表产物，不能因为多画一个像素
  把整条交付链路打断；
- 单元格内文本放不下时按列宽**切片**，不缩字号、不换行 —— 保证网格不串行。
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

RGB = Tuple[int, int, int]

# ---------------------------------------------------------------- 5x7 点阵字库
# 每个字形 7 行、每行 5 bit（高位在左）。只收录信号卡用得到的字符：
# 数字 / 大写字母 / 少量符号。**未收录的字符渲染为等宽空白**（不猜字形）。
_FONT: Dict[str, Tuple[int, ...]] = {
    "0": (0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110),
    "1": (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "2": (0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111),
    "3": (0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110),
    "4": (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    "5": (0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110),
    "6": (0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110),
    "7": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000),
    "8": (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
    "9": (0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100),
    "A": (0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "B": (0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110),
    "C": (0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110),
    "D": (0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110),
    "E": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111),
    "F": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000),
    "G": (0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111),
    "H": (0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "I": (0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "J": (0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100),
    "K": (0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001),
    "L": (0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111),
    "M": (0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001),
    "N": (0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001),
    "O": (0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "P": (0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000),
    "Q": (0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101),
    "R": (0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001),
    "S": (0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110),
    "T": (0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100),
    "U": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "V": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100),
    "W": (0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001),
    "X": (0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001),
    "Y": (0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100),
    "Z": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111),
    "-": (0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000),
    "+": (0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000),
    ".": (0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100),
    ":": (0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000),
    "/": (0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000),
    "%": (0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011),
    "(": (0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010),
    ")": (0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000),
    "[": (0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110),
    "]": (0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110),
    "<": (0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010),
    ">": (0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000),
    "=": (0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000),
    "|": (0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100),
    "_": (0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111),
    " ": (0, 0, 0, 0, 0, 0, 0),
    "?": (0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100),
}

GLYPH_W = 5
GLYPH_H = 7
CHAR_ADVANCE = 6   # 5 宽 + 1 列间距
LINE_ADVANCE = 9   # 7 高 + 2 行间距


class Canvas:
    """极简 RGB 画布：矩形填充 + 点阵文本 + 折线。"""

    def __init__(self, width: int, height: int, background: RGB = (255, 255, 255)):
        self.width = int(max(1, width))
        self.height = int(max(1, height))
        self.bg = tuple(int(c) for c in background)
        self.rows: List[List[RGB]] = [
            [self.bg] * self.width for _ in range(self.height)
        ]

    # ------------------------------------------------------------ 基础绘制
    def _set(self, x: int, y: int, color: RGB) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.rows[y][x] = (int(color[0]), int(color[1]), int(color[2]))

    def rect(self, x: int, y: int, w: int, h: int, color: RGB,
             fill: bool = True) -> None:
        """填充或描边矩形（越界自动裁剪，不抛错）。"""
        if fill:
            for yy in range(y, y + h):
                for xx in range(x, x + w):
                    self._set(xx, yy, color)
            return
        for xx in range(x, x + w):
            self._set(xx, y, color)
            self._set(xx, y + h - 1, color)
        for yy in range(y, y + h):
            self._set(x, yy, color)
            self._set(x + w - 1, yy, color)

    def hline(self, x: int, y: int, w: int, color: RGB) -> None:
        for xx in range(x, x + w):
            self._set(xx, y, color)

    def polyline(self, points: Sequence[Tuple[int, int]], color: RGB) -> None:
        """折线（Bresenham）——用于锚点走势小图。"""
        pts = list(points)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            x, y = x0, y0
            for _ in range(dx + dy + 2):
                self._set(x, y, color)
                if x == x1 and y == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

    # ------------------------------------------------------------ 文本
    def text(self, x: int, y: int, s: str, color: RGB,
             max_chars: int | None = None) -> int:
        """点阵文本（ASCII 大写化；未收录字符按空格渲染）。

        Returns: 渲染结束后的 x 坐标（供同行续写）。
        """
        text = str(s)
        if max_chars is not None:
            text = text[: int(max_chars)]
        cx = x
        for ch in text:
            glyph = _FONT.get(ch.upper())
            if glyph is not None:
                for ry, bits in enumerate(glyph):
                    for rx in range(GLYPH_W):
                        if bits & (1 << (GLYPH_W - 1 - rx)):
                            self._set(cx + rx, y + ry, color)
            cx += CHAR_ADVANCE
        return cx

    def text_width(self, s: str) -> int:
        return CHAR_ADVANCE * len(str(s)) - 1

    # ------------------------------------------------------------ 导出
    def to_png(self, texts: Dict[str, str] | None = None,
               text_compress: bool = True) -> bytes:
        """编码为 PNG 字节流。

        Args:
            texts: 写入 tEXt 块的键值对（值按 UTF-8 编码）。
            text_compress: tEXt 是否 zlib 压缩（写 ``text_key\\x00\\x01zlib``）。
        """
        raw = bytearray()
        for row in self.rows:
            raw.append(0)                      # filter type: None
            for (r, g, b) in row:
                raw.append(r)
                raw.append(g)
                raw.append(b)
        out = bytearray(b"\x89PNG\r\n\x1a\n")
        out += _chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height,
                                           8, 2, 0, 0, 0))
        out += _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        for key, value in (texts or {}).items():
            k = str(key).encode("ascii", "replace")
            v = str(value).encode("utf-8")
            if text_compress:
                out += _chunk(b"tEXt", k + b"\x00\x01" + zlib.compress(v, 9))
            else:
                out += _chunk(b"tEXt", k + b"\x00" + v)
        out += _chunk(b"IEND", b"")
        return bytes(out)

    def save(self, path: str | Path, texts: Dict[str, str] | None = None,
             text_compress: bool = True) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.to_png(texts, text_compress=text_compress))
        return p


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (struct.pack(">I", len(payload)) + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))


def read_text_chunks(data: bytes) -> Dict[str, str]:
    """解析 PNG 的 tEXt 块（测试与排障用；平台侧也能读到同样的键值）。"""
    out: Dict[str, str] = {}
    pos = 8
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        if kind == b"tEXt":
            key, _, value = payload.partition(b"\x00")
            if value[:1] == b"\x01":
                value = zlib.decompress(value[1:])
            out[key.decode("ascii", "replace")] = value.decode("utf-8", "replace")
        pos += 12 + length
    return out
