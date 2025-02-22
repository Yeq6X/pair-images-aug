"""
python convert_source_to_scribble_xdog,py "input_folder" "output_folder"
画像をスケッチ（線画）に変換する機能を提供します。
"""
import argparse
import os
from lineart_util import scribble_xdog
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

def convert(image_path):
    """
    画像をスケッチに変換します。
    
    Args:
        image_path (str): 入力画像のパス
        
    Returns:
        str: 変換後の画像のパス
    """
    image = Image.open(image_path)
    return convert_pil_to_sketch(image)

def convert_pil_to_sketch(image):
    """
    PIL.Imageをスケッチに変換します。
    
    Args:
        image (PIL.Image): 入力画像
        
    Returns:
        PIL.Image: 変換後の画像
    """
    input_width, input_height = image.size
    image = np.array(image)
    processed_image, _ = scribble_xdog(image, 2048, 16)  # PIL.Image
    processed_image = processed_image.resize((input_width, input_height))
    # make PIL.Image to cv2 and INVERSE
    processed_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
    processed_image = 255 - processed_image
    return Image.fromarray(processed_image)

def process_images(input_folder, output_folder):
    """
    フォルダ内の画像を一括変換します。
    
    Args:
        input_folder (str): 入力フォルダのパス
        output_folder (str): 出力フォルダのパス
    """
    # 入力フォルダ内の全ての画像ファイルを取得
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 出力フォルダを作成（存在しない場合）
    os.makedirs(output_folder, exist_ok=True)

    # 各画像ファイルを処理
    for image_file in tqdm(image_files):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        processed_image = convert_pil_to_sketch(Image.open(input_path))
        processed_image.save(output_path)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='指定したフォルダ内の全ての画像をscribble_xdogで処理し、出力フォルダに保存します。')
    parser.add_argument('input_folder', type=str, help='入力フォルダのパス')
    parser.add_argument('output_folder', type=str, help='出力フォルダのパス')

    args = parser.parse_args()
    process_images(args.input_folder, args.output_folder)
