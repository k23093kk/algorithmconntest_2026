import os
from pathlib import Path
import cv2
import numpy as np


def segment_field_grabcut(img):
    """GrabCutアルゴリズム等を用いて水田領域と背景を分離する関数"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 1. シード領域の設定（上部背景、下部前景）
    mask[0 : int(h * 0.20), :] = cv2.GC_BGD
    mask[int(h * 0.70) : h, :] = cv2.GC_FGD
    mask[int(h * 0.20) : int(h * 0.70), :] = cv2.GC_PR_FGD

    # 2. 右上の通路・人（白っぽい土肌・道）の除外
    b, g, r = cv2.split(img.astype(np.float32))
    exg = 2.0 * g - r - b
    path_candidate = (exg < 5) & (img[:, :, 2] > 80)
    roi_y1, roi_y2 = int(h * 0.15), int(h * 0.35)
    roi_x1 = int(w * 0.5)
    mask[roi_y1:roi_y2, roi_x1:][path_candidate[roi_y1:roi_y2, roi_x1:]] = (
        cv2.GC_BGD
    )

    # 3. GrabCut実行
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        img,
        mask,
        None,
        bgd_model,
        fgd_model,
        iterCount=5,
        mode=cv2.GC_INIT_WITH_MASK,
    )

    field_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # 4. モルフォロジー変換と連結成分によるノイズ除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(field_mask)
    if num_labels > 1:
        base_label = labels[int(h * 0.95), int(w * 0.5)]
        if base_label == 0:
            base_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        field_mask = np.where(labels == base_label, 255, 0).astype(np.uint8)

    return field_mask


def evaluate_weed_ratio(r):
    """雑草比率 r に基づき 4段階のレベル判定を行う

    0: 少ない, 1: 中程度, 2: 多い, 3: 甚大
    """
    if r < 5.0:
        return 0
    elif r < 15.0:
        return 1
    elif r < 30.0:
        return 2
    else:
        return 3


def main():
    input_csv_path = "input.csv"
    output_csv_path = "output.csv"
    # 実際の画像が格納されているルートフォルダ名
    image_root_dir = "アルコン正解データ"

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(
            f"'{input_csv_path}' が見つかりません。カレントフォルダに配置してください。"
        )

    with open(input_csv_path, "r", encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 1行目: 処理すべき画像数 (N)
    total_images = int(lines[0].replace(",", "").split()[0])
    csv_results = []

    # 2行目以降のループ
    for i in range(1, total_images + 1):
        if i >= len(lines):
            break

        # カンマ区切りをパース (width, height, filepath)
        parts = [p.strip() for p in lines[i].split(",")]
        if len(parts) < 3:
            continue
        width = int(parts[0])
        height = int(parts[1])

        # CSVに書かれた相対パス（例: 000/P1060881.JPG）
        raw_path = (
            parts[2]
            .replace("\\", "/")
            .replace("¥", "/")
            .replace("￥", "/")
            .strip()
        )

        # 実際の画像ファイルパス（アルコン正解データ/000/P1060881.JPG など）
        img_path = os.path.join(image_root_dir, raw_path)

        print(f"処理中 ({i}/{total_images}): {img_path}")

        # 画像の読み込み
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 画像を読み込めませんでした -> {img_path}")
            continue

        # サイズ確認・必要に応じたリサイズ
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))

        h, w = img.shape[:2]

        # 1. 水田領域の抽出
        field_mask = segment_field_grabcut(img)

        # 2. 雑草（出穂部・異物）の抽出
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        is_in_field = field_mask == 255
        low_sat_bright = (sat < 80) & (val > 100)
        pale_brown = (hue < 35) & (sat < 125) & (val > 105)

        weed_mask = is_in_field & (low_sat_bright | pale_brown)
        weed_mask = np.uint8(weed_mask) * 255

        # 微小ノイズ除去 & 大きな塊の除外
        kernel = np.ones((2, 2), np.uint8)
        weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, kernel)

        num_w_labels, w_labels, w_stats, _ = cv2.connectedComponentsWithStats(
            weed_mask
        )
        for j in range(1, num_w_labels):
            if w_stats[j, cv2.CC_STAT_AREA] >= 20000:
                weed_mask[w_labels == j] = 0
                field_mask[w_labels == j] = 0

        # 下側の影の補正処理
        lower_limit = int(h * 0.6)
        is_lower = np.zeros((h, w), dtype=bool)
        is_lower[lower_limit:, :] = True
        is_dark_shadow = val < 50

        # 3. 水稲（緑領域）と雑草（赤/白領域）のマスク確定
        weed_indices = (weed_mask == 255) | (
            is_lower & is_dark_shadow & (field_mask == 255)
        )
        rice_indices = (field_mask == 255) & (~weed_indices) & (val > 35)

        # 画素数カウント (p: 水稲, w: 雑草)
        p = int(np.count_nonzero(rice_indices))
        weed_pixel_count = int(np.count_nonzero(weed_indices))

        # 雑草比率 r の計算（小数点以下2桁目を四捨五入して小数点1桁まで算出）
        if (p + weed_pixel_count) > 0:
            r_val = (weed_pixel_count / (p + weed_pixel_count)) * 100.0
        else:
            r_val = 0.0
        r_val = round(r_val, 1)

        # レベル判定 (0, 1, 2, 3)
        level = evaluate_weed_ratio(r_val)

        # 出力画像の構築（仕様通りのカラーコード）
        # その他: (0, 0, 0), 水稲: (0x80, 0x80, 0x80), 雑草: (0xff, 0xff, 0xff)
        out_img = np.zeros((h, w, 3), dtype=np.uint8)
        out_img[rice_indices] = [128, 128, 128]  # 水稲 (B=128, G=128, R=128)
        out_img[weed_indices] = [255, 255, 255]  # 雑草 (B=255, G=255, R=255)

        # 出力ファイル名の生成（拡張子の前に "-output" を追加）
        p_obj = Path(img_path)
        out_file_name = f"{p_obj.stem}-output{p_obj.suffix}"
        out_dir = p_obj.parent

        # 保存先ディレクトリの作成（例: アルコン正解データ/000/ 内に保存）
        os.makedirs(out_dir, exist_ok=True)
        out_img_path = os.path.join(out_dir, out_file_name)

        cv2.imwrite(out_img_path, out_img)

        # 【仕様要件】出力テキスト（output.csv）に書き込む画像ファイル名
        # 仕様書では「入力画像ファイル名の拡張子の前に 'output' を追加したもの」を指定
        # 例: 000/P1060881-output.JPG のように記録する
        csv_img_name = f"{Path(raw_path).parent / out_file_name}"

        # csv行データの追加 (width, height, 出力画像名, 判定, p, w, r)
        csv_results.append(
            f"{width}, {height}, {csv_img_name}, {level}, {p}, {weed_pixel_count}, {r_val}"
        )

    # 4. output.csv の出力（カレントフォルダに出力）
    with open(output_csv_path, "w", encoding="utf-8") as f:
        f.write(f"{len(csv_results)}\n")
        for row in csv_results:
            f.write(row + "\n")

    print(
        f"\nすべての処理が完了しました。'{output_csv_path}' を生成しました。"
    )


if __name__ == "__main__":
    main()