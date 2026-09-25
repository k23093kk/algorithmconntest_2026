import argparse
from pathlib import Path

import cv2

from segmentation import Segmenter
from detector import Detector
from classify import Classifier
from utils import CSVManager, ImageManager, FileManager, RunManager


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--numbered",
        action="store_true",
        help="出力画像とoutput.csvを runs/連番/ に保存し、過去の結果を上書きしない"
             "（指定なしは仕様どおり入力画像と同じフォルダ・カレントのoutput.csv）"
    )

    args = parser.parse_args()

    input_csv = "input.csv"
    output_csv = "output.csv"
    out_dir = None

    if args.numbered:

        out_dir = RunManager.new_run_dir()

        output_csv = str(out_dir / "output.csv")

        print(f"出力先 : {out_dir}")

    # -------------------------------
    # 入力ファイル確認
    # -------------------------------

    FileManager.exists(input_csv)

    image_count, image_list = CSVManager.read_input(input_csv)

    if image_count != len(image_list):

        print(
            f"警告 : input.csvの宣言件数({image_count})と"
            f"実際の行数({len(image_list)})が一致しません。"
        )

    print(f"{image_count}枚の画像を処理します。")

    # -------------------------------
    # クラス生成
    # -------------------------------

    segmenter = Segmenter()

    detector = Detector()

    classifier = Classifier()

    results = []

    # -------------------------------
    # 全画像処理
    # -------------------------------

    for index, info in enumerate(image_list, start=1):

        image_path = Path(info["filename"])

        print(f"[{index}/{image_count}] {image_path}")

        if not image_path.exists():

            print(f"画像が見つかりません : {image_path}")

            continue

        image = cv2.imread(str(image_path))

        if image is None:

            print(f"読み込み失敗 : {image_path}")

            continue

        try:

            # ---------------------------
            # 植物抽出
            # ---------------------------

            core_mask, pale_mask = segmenter.segment(image)

            # ---------------------------
            # 稲・雑草分類
            # ---------------------------

            rice_mask, weed_mask = detector.split(
                image,
                core_mask,
                pale_mask
            )

            rice_pixels, weed_pixels = detector.statistics(
                rice_mask,
                weed_mask
            )

            # ---------------------------
            # 判定
            # ---------------------------

            result = classifier.predict(
                rice_pixels,
                weed_pixels
            )

            # ---------------------------
            # 出力画像
            # ---------------------------

            output_image = detector.create_output(
                rice_mask,
                weed_mask
            )

            output_name = ImageManager.save(
                str(image_path),
                output_image,
                out_dir
            )

        except Exception as e:

            print(f"処理失敗 : {image_path} ({e})")

            continue

        print(
            f"    判定レベル : {result['level']}"
            f"  (雑草率 {result['ratio']}%,"
            f" 稲画素 {rice_pixels}, 雑草画素 {weed_pixels})"
        )

        # ---------------------------
        # CSV保存用
        # ---------------------------

        results.append({

            "width": info["width"],

            "height": info["height"],

            "output_name": output_name,

            "level": result["level"],

            "rice_pixels": rice_pixels,

            "weed_pixels": weed_pixels,

            "ratio": result["ratio"]

        })

    # -------------------------------
    # output.csv保存
    # -------------------------------

    CSVManager.write_output(
        output_csv,
        results
    )

    print()

    print("===================================")
    print("処理完了")
    print(f"処理画像 : {len(results)}")
    print(f"出力CSV : {output_csv}")
    print("===================================")


if __name__ == "__main__":

    main()

    # onnxruntimeのネイティブライブラリが、実行ファイル化した環境の
    # インタプリタ終了処理と競合してクラッシュすることがある
    # （全ての出力は完了した後のクリーンアップ処理でのみ発生）。
    # 全処理完了後は即座にプロセスを終了し、この競合を回避する。
    import os

    os._exit(0)