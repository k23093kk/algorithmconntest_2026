from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict

import cv2


class CSVManager:
    """
    CSVの読み書きを担当
    """

    @staticmethod
    def read_input(csv_path: str | Path):

        csv_path = Path(csv_path)

        images = []

        with csv_path.open(
            mode="r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.reader(f, skipinitialspace=True)

            rows = list(reader)

        image_count = int(rows[0][0].strip())

        for row in rows[1:]:

            if len(row) < 3:
                continue

            width = int(row[0].strip())
            height = int(row[1].strip())

            filename = row[2].strip().replace("\\", "/")

            images.append({

                "width": width,
                "height": height,
                "filename": filename

            })

        return image_count, images

    # -------------------------------------------------------------

    @staticmethod
    def write_output(
        output_csv: str | Path,
        results: List[Dict]
    ):

        output_csv = Path(output_csv)

        with output_csv.open(
            mode="w",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([len(results)])

            for r in results:

                writer.writerow([

                    r["width"],
                    r["height"],
                    r["output_name"],
                    r["level"],
                    r["rice_pixels"],
                    r["weed_pixels"],
                    r["ratio"]

                ])

class ImageManager:

    @staticmethod
    def output_filename(image_path: str, out_dir: str | Path | None = None):

        p = Path(image_path)

        name = p.with_name(
            p.stem + "-output" + p.suffix
        )

        # 通常は入力画像と同じフォルダ（仕様どおり）。
        # out_dir指定時はそのフォルダ配下に同じ階層構造で出力する。
        if out_dir is not None:
            name = Path(out_dir) / name

        return str(name)

    # ----------------------------------------------------------

    @staticmethod
    def save(image_path: str, image, out_dir: str | Path | None = None):

        output_name = ImageManager.output_filename(
            image_path,
            out_dir
        )

        output_path = Path(output_name)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # 出力は稲/雑草/背景の3色のみのはずだが、JPEGはデフォルト品質だと
        # 色の境界がにじみ、微妙にずれた色（例: 128,128,128が127や129に
        # なる等）が混入する。品質を最大にして色のにじみを抑える。
        cv2.imwrite(
            str(output_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 100]
        )

        return output_name
class RunManager:
    """
    実行ごとの連番フォルダ（runs/001, runs/002, ...）を作る。
    再実行しても過去の出力画像・output.csvを上書きしないためのもの。
    """

    @staticmethod
    def new_run_dir(base: str | Path = "runs") -> Path:

        base = Path(base)

        base.mkdir(parents=True, exist_ok=True)

        numbers = [
            int(p.name)
            for p in base.iterdir()
            if p.is_dir() and p.name.isdigit()
        ]

        run_dir = base / f"{max(numbers, default=0) + 1:03d}"

        run_dir.mkdir()

        return run_dir


class FileManager:

    @staticmethod
    def exists(path):

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(
                f"{path} が見つかりません。"
            )