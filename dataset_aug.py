"""
画像の拡張処理を行うための関数群を提供します。

1. 画像の平均色を計算する関数
2. 画像を指定された角度で回転させ、平均色で余白を埋める関数
3. 回転した画像から最大の長方形を切り出す関数
4. ランダムな正方形を切り出す関数
"""

import os
from PIL import Image, ImageStat, ImageOps
from collections import Counter
import random
import math
from tqdm import tqdm
import argparse

def get_average_color(image):
    """画像の平均色を計算する"""
    stat = ImageStat.Stat(image)
    # 平均色を取得（RGB）
    r, g, b = map(int, stat.mean)
    return (r, g, b)

def get_edge_mode_color(img, edge_width=10):
    """画像の外周の最頻値（mode）を取得する"""
    # 外周の10ピクセル領域を取得
    left = img.crop((0, 0, edge_width, img.height))  # 左端
    right = img.crop((img.width - edge_width, 0, img.width, img.height))  # 右端
    top = img.crop((0, 0, img.width, edge_width))  # 上端
    bottom = img.crop((0, img.height - edge_width, img.width, img.height))  # 下端

    # 各領域のピクセルデータを取得して結合
    colors = list(left.getdata()) + list(right.getdata()) + list(top.getdata()) + list(bottom.getdata())
    # 最頻値（mode）を計算
    mode_color = Counter(colors).most_common(1)[0][0]  # 最も頻繁に出現する色を取得

    return mode_color

def rotate_image(image, angle, fill_color=(255, 255, 255)):
    """画像を指定された角度で回転させ、指定された色で余白を埋める"""
    return image.rotate(angle, expand=True, fillcolor=fill_color)

def crop_square(cropped_rect_image, left, top, crop_size):
    """ランダムな正方形を切り出す"""
    return cropped_rect_image.crop((left, top, left + crop_size, top + crop_size))

def apply_random_flip(image, is_horizontal):
    """画像にランダムなフリップ（水平または垂直）を適用する"""
    if is_horizontal:
        return ImageOps.mirror(image)  # 水平フリップ
    return image

def process_image_pair(
    source_image,
    target_image,
    output_size=(1024, 1024),
    is_flip=False,
    rotation_range=40,
    min_scale=0.6,
    max_scale=1.2,
    source_is_avg_color_fill=False,
    source_is_edge_mode_fill=False,
    target_is_avg_color_fill=True,
    target_is_edge_mode_fill=False,
    expand_to_long_side=False
    ):
    """1組の画像に対して拡張処理を行う"""
    orig_source_width, orig_source_height = source_image.size
    orig_target_width, orig_target_height = target_image.size
    
    # ソース画像の余白の色を決定
    if source_is_edge_mode_fill:
        source_fill_color = get_edge_mode_color(source_image, edge_width=10)
    elif source_is_avg_color_fill:
        source_fill_color = get_average_color(source_image)
    else:
        source_fill_color = (255, 255, 255)
    
    # ターゲット画像の余白の色を決定
    if target_is_edge_mode_fill:
        target_fill_color = get_edge_mode_color(target_image, edge_width=10)
    elif target_is_avg_color_fill:
        target_fill_color = get_average_color(target_image)
    else:
        target_fill_color = (255, 255, 255)

    base_source = source_image
    base_target = target_image

    # 長辺を基準にする場合の処理を追加
    if expand_to_long_side:
        # sourceの長辺を取得して正方形のキャンバスを作成
        source_long_side = max(base_source.width, base_source.height)
        source_canvas = Image.new("RGB", (source_long_side, source_long_side), source_fill_color)
        # 中央に配置
        source_paste_x = (source_long_side - base_source.width) // 2
        source_paste_y = (source_long_side - base_source.height) // 2
        source_canvas.paste(base_source, (source_paste_x, source_paste_y))
        base_source = source_canvas

        # targetも同様に処理
        target_long_side = max(base_target.width, base_target.height)
        target_canvas = Image.new("RGB", (target_long_side, target_long_side), target_fill_color)
        target_paste_x = (target_long_side - base_target.width) // 2
        target_paste_y = (target_long_side - base_target.height) // 2
        target_canvas.paste(base_target, (target_paste_x, target_paste_y))
        base_target = target_canvas

    if rotation_range > 0:
        angle = random.uniform(-rotation_range, rotation_range)
        rotated_source = rotate_image(source_image, angle, source_fill_color)
        rotated_target = rotate_image(target_image, angle, target_fill_color)
        base_source = rotated_source
        base_target = rotated_target

    if is_flip:
        is_horizontal = random.choice([True, False])
        flipped_source = apply_random_flip(base_source, is_horizontal)
        flipped_target = apply_random_flip(base_target, is_horizontal)
        base_source = flipped_source
        base_target = flipped_target

    scale = random.uniform(min_scale, max_scale)
    canvas_scale = 1/scale

    if canvas_scale > 1.0:
        # 新規画像(canvas)を作成し中心に画像を配置
        scaled_source = Image.new("RGB", (int(base_source.width*canvas_scale), int(base_source.height*canvas_scale)), source_fill_color)
        scaled_target = Image.new("RGB", (int(base_target.width*canvas_scale), int(base_target.height*canvas_scale)), target_fill_color)
        scaled_source.paste(base_source, (int((scaled_source.width-base_source.width)/2), int((scaled_source.height-base_source.height)/2)))
        scaled_target.paste(base_target, (int((scaled_target.width-base_target.width)/2), int((scaled_target.height-base_target.height)/2)))
    else:
        scaled_source = base_source
        scaled_target = base_target

    base_source_width, base_source_height = base_source.size
    base_source_max_square_size = min(base_source_height, base_source_width)
    crop_source_size = int(base_source_max_square_size * canvas_scale)

    base_target_width, base_target_height = base_target.size
    base_target_max_square_size = min(base_target_height, base_target_width)
    crop_target_size = int(base_target_max_square_size * canvas_scale)

    scaled_source_width, scaled_source_height = scaled_source.size
    left_source = random.randint(0, scaled_source_width - crop_source_size)
    top_source = random.randint(0, scaled_source_height - crop_source_size)

    # sourceとtargetの位置合わせ. この場合、sourceとtargetのアスペクト比は同じと仮定
    left_target = left_source * orig_target_width // orig_source_width
    top_target = top_source * orig_target_height // orig_source_height

    final_source = crop_square(scaled_source, left_source, top_source, crop_source_size).resize(output_size)
    final_target = crop_square(scaled_target, left_target, top_target, crop_target_size).resize(output_size)

    return final_source, final_target

def process_images(
    source_img,
    target_img,
    num_copies,
    output_size,
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
    aug_sources = []
    aug_targets = []
    
    for i in range(num_copies):
        # 拡張処理を実行
        aug_source, aug_target = process_image_pair(
            source_img,
            target_img,
            output_size,
            is_flip,
            rotation_range,
            min_scale,
            max_scale,
            source_is_avg_color_fill,
            source_is_edge_mode_fill,
            target_is_avg_color_fill,
            target_is_edge_mode_fill,
            expand_to_long_side
        )
        
        aug_sources.append(aug_source)
        aug_targets.append(aug_target)
    
    return aug_sources, aug_targets

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('source_folder', type=str, help='source画像フォルダのパス / Path to source image folder')
    args.add_argument('target_folder', type=str, help='target画像フォルダのパス / Path to target image folder')
    args.add_argument('output_folder', type=str, help='出力先フォルダのパス / Path to output folder')
    args.add_argument('--output_size', '-s', type=int, default=1024, 
                     help='出力画像の一辺のサイズ / Output image size')
    args.add_argument('--num_copies', '-n', type=int, default=1, 
                     help='出力画像の枚数 / Number of augmented copies')
    args.add_argument('--is_flip', '-f', type=bool, default=True, 
                     help='フリップを適用するかどうか / Whether to apply random flip')
    args.add_argument('--rotation_range', '-r', type=int, default=0, 
                     help='回転角度の範囲 / Range of rotation angle')
    args.add_argument('--min_scale', '--ms', type=float, default=1.0, 
                     help='最小の画像サイズ / Minimum scale of the image')
    args.add_argument('--max_scale', '--xs', type=float, default=1.0, 
                     help='最大の画像サイズ / Maximum scale of the image')
    args.add_argument('--source_is_avg_color_fill', '--sa', type=bool, default=True, 
                     help='source画像を平均色で余白を埋めるかどうか / Whether to fill source image padding with average color')
    args.add_argument('--source_is_edge_mode_fill', '--se', type=bool, default=False, 
                     help='source画像を外周の最頻値で余白を埋めるかどうか / Whether to fill source image padding with edge mode color')
    args.add_argument('--target_is_avg_color_fill', '--ta', type=bool, default=False, 
                     help='target画像を平均色で余白を埋めるかどうか / Whether to fill target image padding with average color')
    args.add_argument('--target_is_edge_mode_fill', '--te', type=bool, default=False, 
                     help='target画像を外周の最頻値で余白を埋めるかどうか / Whether to fill target image padding with edge mode color')
    args.add_argument('--expand_to_long_side', '--el', type=bool, default=False,
                     help='長辺まで拡張して正方形にするかどうか / Whether to expand the image to a square using the long side')
    args = args.parse_args()
    
    output_path_source = os.path.join(args.output_folder, 'aug_source')
    output_path_target = os.path.join(args.output_folder, 'aug_target')

    # 画像処理
    for image_name in tqdm(os.listdir(args.source_folder)):
        if image_name.endswith('.jpg'):
            source_path = os.path.join(args.source_folder, image_name)
            target_path = os.path.join(args.target_folder, image_name)
            
            process_images(
                source_path,
                target_path,
                args.num_copies,
                (args.output_size, args.output_size),
                args.is_flip,
                args.rotation_range,
                args.min_scale,
                args.max_scale,
                args.source_is_avg_color_fill,
                args.source_is_edge_mode_fill,
                args.target_is_avg_color_fill,
                args.target_is_edge_mode_fill,
                args.expand_to_long_side
            )
