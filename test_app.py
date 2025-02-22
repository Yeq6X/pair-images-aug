import gradio as gr
import os
from PIL import Image
import tempfile
from dataset_aug import process_images
import convert_source_to_sketch  # スケッチ変換用のモジュールをインポート
import random

def process_multiple_images(
    source_images,
    target_images,
    output_size,
    num_copies,
    is_flip,
    rotation_range,
    min_scale,
    max_scale,
    source_is_avg_color_fill,
    source_is_edge_mode_fill,
    target_is_avg_color_fill,
    target_is_edge_mode_fill,
    expand_to_long_side
):
    result_source_images = []
    result_target_images = []
    
    # 各画像ペアに対して処理を実行
    for source_path, target_path in zip(source_images, target_images):
        # PILイメージとして読み込み
        source_img = Image.open(source_path.name)
        target_img = Image.open(target_path.name)
        
        # 拡張処理を実行し、PILイメージのリストを取得
        aug_sources, aug_targets = process_images(
            source_img,
            target_img,
            num_copies=num_copies,
            output_size=(output_size, output_size),
            is_flip=is_flip,
            rotation_range=rotation_range,
            min_scale=min_scale,
            max_scale=max_scale,
            source_is_avg_color_fill=source_is_avg_color_fill,
            source_is_edge_mode_fill=source_is_edge_mode_fill,
            target_is_avg_color_fill=target_is_avg_color_fill,
            target_is_edge_mode_fill=target_is_edge_mode_fill,
            expand_to_long_side=expand_to_long_side
        )
        
        # 生成された画像を収集
        result_source_images.extend(aug_sources)
        result_target_images.extend(aug_targets)
    
    return result_source_images, result_target_images

def update_source_preview(source_files):
    preview_images = []
    if source_files:
        for source in source_files:
            preview_images.append(source.name)
    return preview_images

def update_target_preview(target_files):
    preview_images = []
    if target_files:
        for target in target_files:
            preview_images.append(target.name)
    return preview_images

def convert_to_sketch(source_files):
    """sourceをスケッチに変換"""
    converted_images = []
    if source_files:
        # 一時ディレクトリを作成（グローバルに保持）
        temp_dir = tempfile.mkdtemp()
        try:
            for source in source_files:
                # スケッチ変換処理
                image = Image.open(source.name)
                sketch = convert_source_to_sketch.convert_pil_to_sketch(image)
                
                # 一時ファイルとして保存
                temp_path = os.path.join(temp_dir, os.path.basename(source.name))
                sketch.save(temp_path)
                converted_images.append(temp_path)
        except Exception as e:
            print(f"Error during conversion: {e}")
            # エラー時にも一時ディレクトリを削除
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
            return []
    
    return converted_images

# アプリケーション終了時のクリーンアップ処理を修正
def cleanup_temp_files():
    """一時ファイルをクリーンアップ"""
    temp_root = tempfile.gettempdir()
    for item in os.listdir(temp_root):
        if item.startswith('tmp'):
            item_path = os.path.join(temp_root, item)
            try:
                if os.path.isdir(item_path):
                    # ディレクトリ内の画像ファイルをチェック
                    for root, dirs, files in os.walk(item_path):
                        for file in files:
                            if file.endswith(('.jpg', '.png')):
                                file_path = os.path.join(root, file)
                                try:
                                    with Image.open(file_path) as img:
                                        img.verify()  # 画像ファイルの整合性チェック
                                except Exception as e:
                                    print(f"Corrupted image found: {file_path} - {e}")
                    
                    import shutil
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error cleaning up {item_path}: {e}")

def randomize_params():
    """パラメータをランダムに設定"""
    return (
        random.choice([512, 768, 1024, 1536, 2048]),  # output_size
        random.randint(1, 5),                        # num_copies
        random.choice([True, False]),                 # is_flip
        random.randint(0, 180),                      # rotation_range
        round(random.uniform(0.1, 1.0), 1),          # min_scale
        round(random.uniform(1.0, 2.0), 1),          # max_scale
        random.choice([True, False]),                # source_is_avg_color_fill
        random.choice([True, False]),                # source_is_edge_mode_fill
        random.choice([True, False]),                # target_is_avg_color_fill
        random.choice([True, False]),                # target_is_edge_mode_fill
        random.choice([True, False])                 # expand_to_long_side
    )

def reset_params():
    """パラメータを初期設定に戻す"""
    return (
        1024,    # output_size
        1,       # num_copies
        True,    # is_flip
        0,       # rotation_range
        1.0,     # min_scale
        1.0,     # max_scale
        True,    # source_is_avg_color_fill
        False,   # source_is_edge_mode_fill
        False,   # target_is_avg_color_fill
        False,   # target_is_edge_mode_fill
        False    # expand_to_long_side
    )

def test_process_image_pair_with_expand_to_long_side():
    """長辺拡張オプションのテスト"""
    # テスト用の画像を作成（長方形の画像）
    source_image = Image.new('RGB', (800, 400), color='white')
    target_image = Image.new('RGB', (800, 400), color='white')
    
    result_source, result_target = process_image_pair(
        source_image,
        target_image,
        output_size=(512, 512),
        is_flip=False,
        rotation_range=0,
        min_scale=1.0,
        max_scale=1.0,
        source_is_avg_color_fill=True,
        source_is_edge_mode_fill=False,
        target_is_avg_color_fill=True,
        target_is_edge_mode_fill=False,
        expand_to_long_side=True  # 長辺拡張を有効化
    )
    
    # 結果が正方形であることを確認
    assert result_source.size[0] == result_source.size[1]
    assert result_target.size[0] == result_target.size[1]
    
    # 出力サイズが指定通りであることを確認
    assert result_source.size == (512, 512)
    assert result_target.size == (512, 512)

# Gradioインターフェースの作成
with gr.Blocks() as demo:
    gr.Markdown("# データ拡張テスト")
    gr.Markdown("Code : https://github.com/Yeq6X/pair-images-aug")
    
    with gr.Row():
        # 左側のカラム（Source画像とパラメータ）
        with gr.Column():
            with gr.Row():
                # Source画像
                source_files = gr.File(
                    label="Source画像を選択",
                    file_count="multiple",
                    file_types=["image"],
                    height=150
                )
                # Target画像
                target_files = gr.File(
                    label="Target画像を選択",
                    file_count="multiple",
                    file_types=["image"],
                    height=150
                )

            # サンプル画像の追加
            gr.Examples(
                examples=[
                    [["samples/source/sample1.png", "samples/source/sample2.png"],
                     ["samples/target/sample1.png", "samples/target/sample2.png"]],
                ],
                inputs=[source_files, target_files],
                label="サンプル画像セット",
                examples_per_page=5
            )

            source_preview = gr.Gallery(
                label="Source画像プレビュー",
                show_label=True,
                object_fit="contain",
                columns=4,
                height=300,
                preview=True,
            )
            with gr.Row():
                gr.Markdown("### scribble_xdogで変換")
                convert_src_to_tgt_btn = gr.Button("↓", variant="primary", size="sm")
                convert_tgt_to_src_btn = gr.Button("↑", variant="primary", size="sm")
                gr.Markdown("")
            target_preview = gr.Gallery(
                label="Target画像プレビュー",
                show_label=True,
                object_fit="contain",
                columns=4,
                height=300,
                preview=True
            )           
            
        # 右側のカラム（Target画像と出力）
        with gr.Column():
            # パラメータ設定部分
            with gr.Group():
                gr.Markdown("### パラメータ設定")
                
                # パラメータ操作ボタン
                with gr.Row():
                    randomize_btn = gr.Button("🎲 ランダム設定", variant="secondary")
                    reset_btn = gr.Button("↺ 初期設定に戻す", variant="secondary")
                
                with gr.Row():
                    with gr.Column():
                        output_size = gr.Slider(
                            minimum=256,
                            maximum=2048,
                            value=1024,
                            step=256,
                            label="出力画像サイズ"
                        )
                        num_copies = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=1,
                            step=1,
                            label="リピート回数"
                        )
                        is_flip = gr.Checkbox(
                            label="ランダムフリップを適用",
                            value=True
                        )
                        expand_to_long_side = gr.Checkbox(
                            label="長辺に合わせて拡張する",
                            value=False
                        )
                        rotation_range = gr.Slider(
                            minimum=0,
                            maximum=180,
                            value=0,
                            step=1,
                            label="回転角度の範囲"
                        )
                    
                    with gr.Column():
                        min_scale = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=1.0,
                            step=0.1,
                            label="最小スケール"
                        )
                        max_scale = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="最大スケール"
                        )
                        with gr.Row():
                            with gr.Column():
                                source_is_edge_mode_fill = gr.Checkbox(
                                    label="Source: 外周の最頻色で埋める",
                                    value=False
                                )
                                source_is_avg_color_fill = gr.Checkbox(
                                    label="Source: 画像の平均色で埋める",
                                    value=True
                                )
                            with gr.Column():
                                target_is_edge_mode_fill = gr.Checkbox(
                                    label="Target: 外周の最頻色で埋める",
                                    value=False
                                )
                                target_is_avg_color_fill = gr.Checkbox(
                                    label="Target: 画像の平均色で埋める",
                                    value=False
                                )
            
            process_btn = gr.Button("処理開始", variant="primary")

            # 結果表示
            result_source_gallery = gr.Gallery(
                label="生成結果 (Source)",
                show_label=True,
                object_fit="contain",
                columns=4,
                height=250,
                preview=True,
                type="pil"
            )
            result_target_gallery = gr.Gallery(
                label="生成結果 (Target)",
                show_label=True,
                object_fit="contain",
                columns=4,
                height=250,
                preview=True,
                type="pil"
            )
    
    # イベントハンドラ
    source_files.change(
        fn=update_source_preview,
        inputs=[source_files],
        outputs=source_preview
    )
    
    target_files.change(
        fn=update_target_preview,
        inputs=[target_files],
        outputs=target_preview
    )
    
    convert_src_to_tgt_btn.click(
        fn=convert_to_sketch,
        inputs=[source_files],
        outputs=[target_files]
    )
    
    convert_tgt_to_src_btn.click(
        fn=convert_to_sketch,
        inputs=[target_files],
        outputs=[source_files]
    )
    
    param_outputs = [
        output_size,
        num_copies,
        is_flip,
        rotation_range,
        min_scale,
        max_scale,
        source_is_avg_color_fill,
        source_is_edge_mode_fill,
        target_is_avg_color_fill,
        target_is_edge_mode_fill,
        expand_to_long_side
    ]

    randomize_btn.click(
        fn=randomize_params,
        inputs=[],
        outputs=param_outputs
    )

    reset_btn.click(
        fn=reset_params,
        inputs=[],
        outputs=param_outputs
    )
    
    process_btn.click(
        fn=process_multiple_images,
        inputs=[
            source_files,
            target_files,
            output_size,
            num_copies,
            is_flip,
            rotation_range,
            min_scale,
            max_scale,
            source_is_avg_color_fill,
            source_is_edge_mode_fill,
            target_is_avg_color_fill,
            target_is_edge_mode_fill,
            expand_to_long_side
        ],
        outputs=[result_source_gallery, result_target_gallery]
    )

if __name__ == "__main__":
    try:
        demo.launch(
            # server_name="0.0.0.0",
            # server_port=8000,
            debug=True
        )
    finally:
        cleanup_temp_files()  # アプリケーション終了時にクリーンアップ 