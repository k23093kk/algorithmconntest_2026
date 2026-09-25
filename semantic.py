from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ADE20K（150クラス）のうち、水田そのものではない背景とみなすクラスID。
# 空・遠景の樹木・建物・道路・人物・乗り物など。
# これ以外のクラス（草地・畑・植物・土など）は判定対象として残す。
_BACKGROUND_CLASS_IDS = {
    1,   # building
    2,   # sky
    4,   # tree
    16,  # mountain
    20,  # car
    25,  # house
    26,  # sea
    32,  # fence
    48,  # skyscraper
    60,  # river
    61,  # bridge
    68,  # hill
    69,  # bench
    83,  # truck
    84,  # tower
    87,  # streetlight
    93,  # pole
    102,  # van
    116,  # minibike
    127,  # bicycle
    128,  # lake
}

_INPUT_SIZE = 512

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resource_path(relative: str) -> str:
    """
    PyInstallerでバンドルした際は展開先(_MEIPASS)、通常実行時は
    このファイルと同じディレクトリを基準にリソースを探す。
    """

    base = getattr(sys, "_MEIPASS", None)

    if base is None:
        base = Path(__file__).resolve().parent

    return str(Path(base) / relative)


class SemanticFilter:
    """
    事前学習済みの意味的セグメンテーションモデル(ADE20Kで学習)を用いて、
    画像中の「遠景（空・樹木・建物など）」を判定対象から除外するための
    領域マスクを生成する。

    色（彩度）に基づく判定と異なり、実際にシーンの意味を理解した上で
    背景を判定するため、より正確な除外が期待できる。
    """

    def __init__(self, model_path: str | None = None):

        if model_path is None:
            model_path = _resource_path("models/segformer_ade.onnx")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

    # --------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(
            rgb,
            (_INPUT_SIZE, _INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32) / 255.0

        normalized = (resized - _MEAN) / _STD

        chw = normalized.transpose(2, 0, 1)[None, ...].astype(np.float32)

        return chw

    # --------------------------------------------------

    def foreground_region(self, image: np.ndarray) -> np.ndarray:
        """
        背景（空・遠景の樹木・建物など）を除いた領域を255、
        背景を0としたマスクを返す。
        """

        h, w = image.shape[:2]

        inputs = self._preprocess(image)

        logits = self.session.run(None, {self.input_name: inputs})[0]

        pred_small = logits.argmax(axis=1)[0].astype(np.uint8)

        pred = cv2.resize(
            pred_small,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        background = np.isin(
            pred,
            list(_BACKGROUND_CLASS_IDS)
        )

        result = np.full((h, w), 255, dtype=np.uint8)

        result[background] = 0

        return result
