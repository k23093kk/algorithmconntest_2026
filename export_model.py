"""
遠景除外に使う意味的セグメンテーションモデル(ADE20Kで学習済みのSegFormer)を
ONNX形式に変換して models/segformer_ade.onnx を生成するスクリプト。

実行時(main.py)はこのONNXファイルとonnxruntimeのみを使い、
torch/transformersには依存しない
(transformersはPyInstallerでの実行ファイル化と非互換のため)。

実行前に requirements-dev.txt の内容をインストールすること:
    pip install -r requirements-dev.txt
"""

import os

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
OUTPUT_PATH = Path(__file__).resolve().parent / "models" / "segformer_ade.onnx"


def main():

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    model.eval()

    dummy = torch.randn(1, 3, 512, 512)

    torch.onnx.export(
        model,
        dummy,
        str(OUTPUT_PATH),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=14,
        dynamo=False,
    )

    print(f"ONNXモデルを出力しました: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
