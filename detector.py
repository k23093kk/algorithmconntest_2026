from __future__ import annotations

import cv2
import numpy as np

from config import Config
from semantic import SemanticFilter


class Detector:
    """
    Segmenterが抽出した植物領域を、遠景を除外したうえで
    彩度により稲と雑草に分離する。
    """

    def __init__(self, semantic_filter: SemanticFilter | None = None):

        self.semantic_filter = semantic_filter or SemanticFilter()

    # ---------------------------------------------------------

    @staticmethod
    def saturation(image: np.ndarray) -> np.ndarray:
        """
        元画像の彩度チャンネルを返す。

        CLAHEをかけた画像ではなく元画像を使う。明度補正は植物領域の
        抽出には有効だが、彩度そのものを歪めるため判定には使わない。
        """

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        return hsv[:, :, 1]

    # ---------------------------------------------------------

    @staticmethod
    def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:

        if min_area <= 0:
            return mask

        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        keep = stats[:, cv2.CC_STAT_AREA] >= min_area

        keep[0] = False

        return np.where(keep[labels], 255, 0).astype(np.uint8)

    # ---------------------------------------------------------

    def split(self,
              image: np.ndarray,
              core_mask: np.ndarray,
              pale_mask: np.ndarray):

        """
        植物領域（core ∪ pale）から遠景を除外し、彩度で稲と雑草に分ける。

        雑草の穂は白っぽく彩度が低いのに対し、稲の葉は彩度が高い。
        植物領域内を彩度閾値で二分し、低彩度側を雑草とする。

        遠景（空・樹木・建物など）の除外には、事前学習済みの意味的
        セグメンテーションモデル（ADE20K）を用いる。
        """

        valid_region = self.semantic_filter.foreground_region(image)

        plant = cv2.bitwise_or(core_mask, pale_mask)

        plant = cv2.bitwise_and(plant, valid_region)

        low_saturation = np.where(
            self.saturation(image) < Config.SATURATION_THRESHOLD,
            255,
            0
        ).astype(np.uint8)

        weed = cv2.bitwise_and(plant, low_saturation)

        weed = self.remove_small(weed, Config.WEED_MIN_COMPONENT)

        rice = cv2.bitwise_and(plant, cv2.bitwise_not(weed))

        return rice, weed

    # ---------------------------------------------------------

    @staticmethod
    def pixel_count(mask):

        return int(cv2.countNonZero(mask))

    # ---------------------------------------------------------

    def statistics(self,
                   rice_mask,
                   weed_mask):

        rice = self.pixel_count(rice_mask)

        weed = self.pixel_count(weed_mask)

        return rice, weed

    # ---------------------------------------------------------

    def create_output(self,
                      rice_mask,
                      weed_mask):

        h, w = rice_mask.shape

        output = np.zeros(
            (h, w, 3),
            dtype=np.uint8
        )

        output[:] = Config.BACKGROUND

        output[rice_mask > 0] = Config.RICE

        output[weed_mask > 0] = Config.WEED

        return output