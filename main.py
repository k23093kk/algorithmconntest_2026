import os
import time
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# ==========================================
# 1. ここにRoboflowのAPIキーを貼り付けます
# ==========================================
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"

def get_weed_level(client, image_path):
    """APIを呼び出し、面積割合から0〜3のレベルを判定する"""
    try:
        # Roboflowへのリクエスト送信
        result = client.run_workflow(
            workspace_name="e0909mbihara-icloud-com",
            workflow_id="1790164418212",
            images={"image": image_path},
            use_cache=True
        )
        
        # 結果はリストまたは辞書で返るため、構造に合わせて取得
        # ※実際のAPIの返り値を見て、取得キーが違う場合は修正してください
        output = result[0] if isinstance(result, list) else result
        
        # 面積割合の取得
        rice_fraction = output.get("rice_area_fraction", 0.0)
        other_fraction = output.get("other_area_fraction", 0.0)
        
        # 割合の計算 (0除算を防止)
        total = rice_fraction + other_fraction
        if total == 0:
            ratio = 0.0
        else:
            ratio = other_fraction / total  # 水稲に対する「その他(雑草)」の割合

        # 4段階のレベル判定 (※ここの閾値は正解データを見て調整してください)
        if ratio < 0.05:
            return 0 # 少ない
        elif ratio < 0.15:
            return 1 # 中程度
        elif ratio < 0.30:
            return 2 # 多い
        else:
            return 3 # 甚大

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return -1 # エラー時は-1を出力

def main():
    start_time = time.time()
    
    # 2. クライアントの初期化
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    ).configure(InferenceConfiguration(
        api_key_transport="header"
    ))

    # 3. input.csvの読み込み
    csv_path = 'input.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} が見つかりません。")
        return

    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return

    num_images = int(lines[0])

    # 4. 画像ごとの処理ループ
    for i in range(1, num_images + 1):
        if i >= len(lines):
            break
            
        data = lines[i].split(',')
        if len(data) >= 3:
            image_path = data[2].replace('\\', os.sep).replace('/', os.sep)
            
            # APIを呼び出してレベルを取得し、結果を出力
            level = get_weed_level(client, image_path)
            print(f"{image_path},{level}")

    processing_time = time.time() - start_time
    print(f"Total Processing Time: {processing_time:.3f} seconds")

if __name__ == "__main__":
    main()