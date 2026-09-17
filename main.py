import os
from pathlib import Path
import cv2
import numpy as np


def get_unique_filename(base_path: str) -> str:
    """ファイルが既に存在する場合、連番を付けて上書きを防ぐ"""
    path = Path(base_path)
    if not path.exists():
        return str(path)

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    counter = 2
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return str(new_path)
        counter += 1


def segment_field_grabcut(img):
    """画像内の被写体（水田 vs 背景）の色・テクスチャ分布から、

    境界の輪郭に沿って水田領域を抽出する
    """
    h, w = img.shape[:2]

    # GrabCut 用のマスク初期化
    # GC_BGD: 確実な背景, GC_FGD: 確実な前景(水田), GC_PR_BGD: おそらく背景, GC_PR_FGD: おそらく前景
    mask = np.zeros((h, w), dtype=np.uint8)

    # 1. 確実なシード領域を設定（学習用サンプル）
    # 最上部（山・空など）は「確実に背景」
    mask[0 : int(h * 0.12), :] = cv2.GC_BGD

    # 最下部（手前の水田）は「確実に前景」
    mask[int(h * 0.70) : h, :] = cv2.GC_FGD

    # 中間領域（境界線が存在するエリア）は「未知（おそらく前景/背景）」として設定
    mask[int(h * 0.12) : int(h * 0.70), :] = cv2.GC_PR_FGD

    # 2. 右上の通路・人（白っぽい土肌・道）を背景として教える
    # 畦道や通路がある右奥側（上部15%〜35%、幅の右半分）の土色・非緑部分を背景シードに追加
    b, g, r = cv2.split(img.astype(np.float32))
    exg = 2.0 * g - r - b  # 過剰緑指数
    path_candidate = (exg < 5) & (img[:, :, 2] > 80)  # 土・道の露出部分

    # 右奥の非植生エリアを背景としてマーク
    roi_y1, roi_y2 = int(h * 0.15), int(h * 0.35)
    roi_x1 = int(w * 0.5)
    mask[roi_y1:roi_y2, roi_x1:][path_candidate[roi_y1:roi_y2, roi_x1:]] = (
        cv2.GC_BGD
    )

    # 3. GrabCut アルゴリズムの実行
    # 前景・背景それぞれの混合ガウスモデル（GMM）内部バッファ
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 被写体の境界エッジに沿ってセグメンテーション（反復回数: 5回）
    cv2.grabCut(
        img,
        mask,
        None,
        bgd_model,
        fgd_model,
        iterCount=5,
        mode=cv2.GC_INIT_WITH_MASK,
    )

    # 前景（水田）と判定された領域（GC_FGD または GC_PR_FGD）を取り出す
    field_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # 4. 微小な孤立ノイズを整理し、まとまった大きな領域にする
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel)

    # 画面下部に接している最も大きな連結成分のみを残す（奥の孤立した飛び地を除外）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(field_mask)
    if num_labels > 1:
        # 下端中央付近のラベルを採用
        base_label = labels[int(h * 0.95), int(w * 0.5)]
        if base_label == 0:
            # もし0なら最大面積の前景を採用
            base_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        field_mask = np.where(labels == base_label, 255, 0).astype(np.uint8)

    return field_mask


def process_image(input_path="input.jpg", output_base="output.jpg"):
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {input_path}")

    # 1. GrabCut（被写体ベースのグラフカット）による水田領域の抽出
    field_mask = segment_field_grabcut(img)

    # 2. 水田内における白っぽい雑草（出穂部）の抽出
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # 「白〜淡褐色」の穂：水田マスクの範囲内
    is_in_field = field_mask == 255
    low_sat_bright = (sat < 80) & (val > 100)
    pale_brown = (hue < 35) & (sat < 125) & (val > 105)

    weed_mask = is_in_field & (low_sat_bright | pale_brown)
    weed_mask = np.uint8(weed_mask) * 255

    # 穂の微小ノイズ除去
    kernel = np.ones((2, 2), np.uint8)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, kernel)

    # 3. 稲の緑領域
    green_mask = is_in_field & (weed_mask == 0) & (val > 35)

    # 4. 出力画像の構築（背景・山・通路は自動的に黒）
    result = np.zeros_like(img)
    result[green_mask] = [0, 140, 0]  # 緑の稲
    result[weed_mask == 255] = [0, 0, 255]  # 白っぽい雑草（赤色）

    # 5. 連番ファイル名で保存
    save_path = get_unique_filename(output_base)
    cv2.imwrite(save_path, result)
    print(f"保存完了: {save_path}")


if __name__ == "__main__":
    process_image("アルコン正解データ/000/RIMG2746.JPG", "output.jpg")