# app.py

import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import time

from diffusers import StableDiffusionPipeline, ControlNetModel, DDIMScheduler
from insightface.app import FaceAnalysis

print("--- アプリケーションの初期化を開始 ---")

# --- 1. モデルのロード ---
# この部分はFastAPIの時とほぼ同じです。
# アプリ起動時に一度だけ実行されます。
try:
    print("モデルのロードを開始...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

    # 顔分析モデル (InsightFace)
    face_app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    # ControlNetモデル (InstantID)
    controlnet = ControlNetModel.from_pretrained(
        "InstantID/ControlNetModel", 
        torch_dtype=dtype
    )
    
    # ベースとなる画像生成モデル (RealBeautyMix)
    pipe = StableDiffusionPipeline.from_pretrained(
        "SG161222/RealBeautyMix",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        use_safetensors=True,
    ).to(device)

    # IP-Adapter (InstantID)
    pipe.load_ip_adapter("InstantID/ip-adapter", subfolder="models", weight_name="ip-adapter.bin")
    
    print("モデルのロードが完了しました。")
    MODELS_LOADED = True
except Exception as e:
    print(f"モデルのロード中にエラーが発生しました: {e}")
    MODELS_LOADED = False

# --- 2. 画像生成関数 ---
# この関数がUIのボタンクリックとAPIの両方から呼び出されます。
def generate_image(
    face_image, 
    prompt, 
    negative_prompt, 
    guidance_scale, 
    ip_adapter_scale, 
    num_steps,
    progress=gr.Progress(track_ τότε=True)
):
    if not MODELS_LOADED:
        raise gr.Error("モデルがロードされていないため、画像を生成できません。")
    if face_image is None:
        raise gr.Error("顔画像をアップロードしてください。")
    if not prompt:
        raise gr.Error("プロンプトを入力してください。")

    # PIL Imageに変換
    face_image = Image.fromarray(face_image)

    # 顔分析
    face_info = face_app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    if not face_info:
        raise gr.Error("アップロードされた画像から顔を検出できませんでした。")
    
    face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
    face_emb = face_info['embedding']

    # パイプライン設定
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    # 画像生成
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=[face_emb],
        image=face_image,
        controlnet_conditioning_scale=ip_adapter_scale,
        num_inference_steps=int(num_steps),
        guidance_scale=guidance_scale,
    ).images

    return images[0]

# --- 3. GradioのUIとAPIの定義 ---
with gr.Blocks() as demo:
    gr.Markdown("# InstantID 画像生成アプリ")
    
    with gr.Row():
        with gr.Column():
            # 入力コンポーネント
            face_image_input = gr.Image(label="顔写真", type="numpy")
            prompt_input = gr.Textbox(label="プロンプト (例: a photo of a man in a suit)")
            negative_prompt_input = gr.Textbox(label="ネガティブプロンプト", value="(lowres, low quality, worst quality:1.2), ugly")
            
            with gr.Accordion("詳細設定", open=False):
                ip_adapter_scale_slider = gr.Slider(minimum=0, maximum=1.5, step=0.1, value=0.8, label="顔の忠実度 (IP Adapter Scale)")
                guidance_scale_slider = gr.Slider(minimum=1, maximum=10, step=0.5, value=5.0, label="プロンプトへの忠実度 (Guidance Scale)")
                num_steps_slider = gr.Slider(minimum=10, maximum=50, step=1, value=30, label="生成ステップ数 (Steps)")
            
            generate_button = gr.Button("画像を生成", variant="primary")
            
        with gr.Column():
            # 出力コンポーネント
            output_image = gr.Image(label="生成結果")

    # ボタンがクリックされた時の動作を定義
    # ここで api_name を設定するのが最重要ポイント！
    generate_button.click(
        fn=generate_image,
        inputs=[
            face_image_input,
            prompt_input,
            negative_prompt_input,
            guidance_scale_slider,
            ip_adapter_scale_slider,
            num_steps_slider
        ],
        outputs=[output_image],
        api_name="generate"  # APIエンドポイント名を "generate" に設定
    )

# アプリケーションを起動 (キューを有効にして複数リクエストに対応)
demo.queue().launch()

print("--- Gradioアプリの起動準備完了 ---")