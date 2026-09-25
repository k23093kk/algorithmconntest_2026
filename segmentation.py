import cv2
import numpy as np

from config import Config


class Segmenter:
    """
    植物領域（稲＋雑草）の抽出

    ・core … ExGとHSVの両方を満たす、確実に緑の植物
    ・pale … coreに囲まれた穴のうち、HSVでは植物と判定されるが
             ExGでは弾かれる色の薄い部分（穂など）

    coreとpaleの和が植物領域である。稲と雑草の区別はここでは行わず、
    Detectorが彩度によって分離する。

    ※ 以前はcore＝稲、pale＝雑草としていたが、ExGの二値化に用いる
      Otsuが「2クラスある」前提で必ず分割するため、画面が植物一色の
      画像では稲自身が明暗で二分されpale（雑草）に流れ込んでいた。
      Otsuはここでは「植物とそれ以外」の分離にのみ用いる。
    """

    def __init__(self):

        self.open_kernel = np.ones(Config.OPEN_KERNEL, np.uint8)
        self.close_kernel = np.ones(Config.CLOSE_KERNEL, np.uint8)

    # --------------------------------------------------

    def clahe(self, img):
        """
        明るさ補正
        """

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=Config.CLAHE_CLIP,
            tileGridSize=Config.CLAHE_GRID
        )

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --------------------------------------------------

    def exg(self, img):
        """
        Excess Green
        """

        img = img.astype(np.float32)

        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        exg = 2 * g - r - b

        exg = cv2.normalize(
            exg,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        return exg.astype(np.uint8)

    # --------------------------------------------------

    def hsv_mask(self, img):

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array(Config.HSV_LOWER)

        upper = np.array(Config.HSV_UPPER)

        return cv2.inRange(hsv, lower, upper)

    # --------------------------------------------------

    def exg_mask(self, img):

        exg = self.exg(img)

        _, mask = cv2.threshold(
            exg,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return mask

    # --------------------------------------------------

    @staticmethod
    def fill_holes(mask):
        """
        植物領域に完全に囲まれた穴を塗りつぶす。

        外周（左上隅）から到達できない0画素（＝穴）を検出し、
        元のマスクに追加する。
        """

        h, w = mask.shape

        flood = mask.copy()

        flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)

        cv2.floodFill(flood, flood_fill_mask, (0, 0), 255)

        holes = cv2.bitwise_not(flood)

        return cv2.bitwise_or(mask, holes)

    # --------------------------------------------------

    def core_mask(self, hsv, exg):
        """
        ExGとHSVの両方を満たす、確実に緑の植物領域。
        """

        mask = cv2.bitwise_and(hsv, exg)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.open_kernel,
            iterations=Config.OPEN_ITER
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.close_kernel,
            iterations=Config.CLOSE_ITER
        )

        return mask

    # --------------------------------------------------

    def pale_mask(self, core, hsv):
        """
        coreに完全に囲まれた穴のうち、HSVでは植物と判定されるが
        ExGでは弾かれる色の薄い部分（雑草の穂などと推定）を抽出する。

        HSVの範囲に限定することで、空や土など無関係な領域が
        誤って取り込まれることを防ぐ。また、1～数画素程度の
        孤立したノイズ片を除去する。
        """

        filled = self.fill_holes(core)

        holes = cv2.bitwise_and(filled, cv2.bitwise_not(core))

        pale = cv2.bitwise_and(holes, hsv)

        return self.remove_noise(pale)

    # --------------------------------------------------

    def remove_noise(
            self,
            mask,
            min_area=Config.MIN_COMPONENT
    ):

        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        # 雑草で分断された画像は連結成分数が非常に多くなりうるため、
        # 成分ごとにPythonループで塗り直すと極端に遅くなる。
        # ラベルIDをキーにした採否テーブルを作り、一括で適用する。
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area

        keep[0] = False

        return np.where(keep[labels], 255, 0).astype(np.uint8)

    # --------------------------------------------------

    def segment(self, img):
        """
        core（確実な緑の植物）とpale（coreに囲まれた色の薄い部分）の
        2つのマスクを生成する。両者の和が植物領域となる。
        """

        img = self.clahe(img)

        hsv = self.hsv_mask(img)

        exg = self.exg_mask(img)

        core = self.core_mask(hsv, exg)

        core = self.remove_noise(core)

        pale = self.pale_mask(core, hsv)

        return core, pale