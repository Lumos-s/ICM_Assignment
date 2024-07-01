import cv2
import numpy as np
import os

def gen_trimap(alpha):
    erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,100)]

    max_kernel_size = 100
    fg_mask = (alpha + 1e-5).astype(np.int_).astype(np.uint8)
    bg_mask = (1 - alpha + 1e-5).astype(np.int_).astype(np.uint8)
    fg_mask = cv2.erode(fg_mask, erosion_kernels[np.random.randint(1, max_kernel_size)])
    bg_mask = cv2.erode(bg_mask, erosion_kernels[np.random.randint(1, max_kernel_size)])

    fg_width = np.random.randint(1, 100)
    bg_width = np.random.randint(1, 100)
    fg_mask = (alpha + 1e-5).astype(int).astype(np.uint8)
    bg_mask = (1 - alpha + 1e-5).astype(int).astype(np.uint8)
    fg_mask = cv2.erode(fg_mask, erosion_kernels[fg_width])
    bg_mask = cv2.erode(bg_mask, erosion_kernels[bg_width])

    trimap = np.ones_like(alpha) * 128
    trimap[fg_mask == 1] = 255
    trimap[bg_mask == 1] = 0

    return trimap

# alpha = cv2.imread('alpha.png', cv2.IMREAD_GRAYSCALE) / 255.0
# trimap = gen_trimap(alpha)
# cv2.imwrite('trimap.png', trimap)
# 读取文件夹中的所有图片，生成 trimap
input_dir = 'alpha'
output_dir = 'trimap'
for file in os.listdir(input_dir):
    if file.endswith('.png'):
        alpha = cv2.imread(os.path.join(input_dir, file), cv2.IMREAD_GRAYSCALE) / 255.0
        trimap = gen_trimap(alpha)
        cv2.imwrite(os.path.join(output_dir, file), trimap)