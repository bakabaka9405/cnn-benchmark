import os
from PIL import Image
import numpy as np


def remove_black_borders(image_path):
	# 打开图片
	img = Image.open(image_path)
	img_array = np.array(img)

	# 获取图片的非黑色部分的坐标
	non_black_pixels = np.argwhere(img_array > 0)  # 假设黑色是值为0的像素

	# 获取裁剪区域的最小矩形框
	top_left = non_black_pixels.min(axis=0)
	bottom_right = non_black_pixels.max(axis=0)

	# 根据最小矩形框裁剪图片
	cropped_img = img.crop((top_left[1], top_left[0], bottom_right[1], bottom_right[0]))

	return cropped_img


src_root = r'C:\Resources\Datasets\SparseReflectance-Maps-for-Fabric-Characterization'
dst_root = r'C:\Resources\Datasets\SparseReflectance-Maps-for-Fabric-Characterization_1'

if not os.path.exists(dst_root):
	os.makedirs(dst_root)

classes = os.listdir(src_root)

for class_name in classes:
	src_class_dir = os.path.join(src_root, class_name)
	dst_class_dir = os.path.join(dst_root, class_name)
	if not os.path.exists(dst_class_dir):
		os.makedirs(dst_class_dir)
	for img_name in os.listdir(src_class_dir):
		src_img_path = os.path.join(src_class_dir, img_name)
		dst_img_path = os.path.join(dst_class_dir, img_name)
		cropped_img = remove_black_borders(src_img_path)
		cropped_img.save(dst_img_path)
		print(f'{src_img_path} -> {dst_img_path}')
