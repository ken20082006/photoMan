"""色彩空間轉換。

內部工作空間是**線性光、float32**（見 docs/design.md §6.1b）。理由有兩個：

1. 在 gamma 編碼的空間做混合，邊界會出現暗邊或亮邊——平均值在
   sRGB 空間不等於感知上的平均值。
2. 曝光的數學只有在線性空間才正確。sRGB 值乘以 2 不是「亮一級」。

**模型邊界是唯一離開線性光的地方**（§6.4）。任何其他地方的轉換都是 bug。
"""

from __future__ import annotations

import numpy as np

# sRGB 傳輸函數的分段點。這兩個常數不是約數，改動會令轉換不再可逆。
_SRGB_THRESHOLD = 0.04045
_LINEAR_THRESHOLD = 0.0031308
_SRGB_SLOPE = 12.92
_SRGB_OFFSET = 0.055
_SRGB_GAMMA = 2.4


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    """sRGB [0,1] → 線性光 [0,1]。

    負值以絕對值處理再還原符號。調整層（例如對比、曲線）可以在線性空間
    產生負值，而負數的 2.4 次方在實數域無定義——直接代入會得到 NaN，
    然後 NaN 會沿著 pipeline 擴散到整張圖。
    """
    array = np.asarray(value, dtype=np.float32)
    sign = np.sign(array)
    magnitude = np.abs(array)
    converted = np.where(
        magnitude <= _SRGB_THRESHOLD,
        magnitude / _SRGB_SLOPE,
        ((magnitude + _SRGB_OFFSET) / (1.0 + _SRGB_OFFSET)) ** _SRGB_GAMMA,
    )
    return (sign * converted).astype(np.float32)


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    """線性光 → sRGB，``srgb_to_linear`` 的逆運算。

    同樣以絕對值處理負值，理由與上行相同。
    """
    array = np.asarray(value, dtype=np.float32)
    sign = np.sign(array)
    magnitude = np.abs(array)
    converted = np.where(
        magnitude <= _LINEAR_THRESHOLD,
        magnitude * _SRGB_SLOPE,
        (1.0 + _SRGB_OFFSET) * (magnitude ** (1.0 / _SRGB_GAMMA)) - _SRGB_OFFSET,
    )
    return (sign * converted).astype(np.float32)
