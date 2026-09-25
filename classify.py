from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from config import Config


class Classifier:
    """
    雑草率からレベル判定を行うクラス
    """

    def __init__(
        self,
        level0: float = Config.LEVEL0,
        level1: float = Config.LEVEL1,
        level2: float = Config.LEVEL2
    ):

        self.level0 = level0
        self.level1 = level1
        self.level2 = level2

    # -------------------------------------------------

    @staticmethod
    def weed_ratio(
        rice_pixels: int,
        weed_pixels: int
    ) -> float:
        """
        雑草率[%]を計算（丸めなし）
        """

        total = rice_pixels + weed_pixels

        if total == 0:
            return 0.0

        return weed_pixels / total * 100.0

    # -------------------------------------------------

    @staticmethod
    def round_ratio(ratio: float) -> float:
        """
        出力CSVに書く雑草率。

        仕様書の「小数点以下2桁目を四捨五入」に合わせる。
        Python標準のround()は銀行丸め（0.5を偶数側に丸める）であり
        四捨五入と結果が食い違うことがあるためDecimalで明示的に丸める。
        """

        rounded = Decimal(str(ratio)).quantize(
            Decimal("0.1"),
            rounding=ROUND_HALF_UP
        )

        return float(rounded)

    # -------------------------------------------------

    def level(
        self,
        ratio: float
    ) -> int:
        """
        雑草率から
        0～3を返す
        """

        if ratio < self.level0:
            return 0

        if ratio < self.level1:
            return 1

        if ratio < self.level2:
            return 2

        return 3

    # -------------------------------------------------

    def predict(
        self,
        rice_pixels: int,
        weed_pixels: int
    ):
        """
        判定は丸める前の値で行い、出力用の値のみ丸める。
        丸めてから閾値と比較すると、境界付近で判定が1段ずれうるため。
        """

        ratio = self.weed_ratio(
            rice_pixels,
            weed_pixels
        )

        return {
            "level": self.level(ratio),
            "ratio": self.round_ratio(ratio)
        }