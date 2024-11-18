import openslide
import numpy as np
import cv2
import h5py
from PIL import Image


def create_attention_map(slide, attention_map_save_path):
    # 使用 OpenSlide 打开 WSI 文件
    slide = openslide.OpenSlide(slide)

    # 设置缩放比例
    scale_factor = 0.2  # 缩放到 20%

    # 读取 WSI 图像并缩放
    level = 0  # 使用最高分辨率层级
    wsi_img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    wsi_img = wsi_img.convert("RGB")
    wsi_img = wsi_img.resize((int(wsi_img.width * scale_factor), int(wsi_img.height * scale_factor)), Image.LANCZOS)
    wsi_img = np.array(wsi_img)

    # 转换为 BGR 格式供 OpenCV 使用
    wsi_img_bgr = cv2.cvtColor(wsi_img, cv2.COLOR_RGB2BGR)

    # 获取缩放后的图像尺寸
    scaled_h, scaled_w, _ = wsi_img_bgr.shape

    # 创建空的注意力图
    attention_map = np.zeros((scaled_h, scaled_w), dtype=np.float32)

    # 读取 HDF5 文件数据
    h5_file = '/data/lichangyong/TCGA_FEATURE/BLCA/h5_files/TCGA-2F-A9KO-01A-01-TSA.114373F3-EFE1-4775-A7D9-AB1C858A7BDC.h5'
    with h5py.File(h5_file, 'r') as f:
        # 获取坐标和特征
        coords = np.array(f['coords'])  # 假设 coords 是 [N, 2] 的二维数组
        features = np.array(f['features'])  # 假设 features 是 [N, ...] 的二维数组

        # 计算注意力分数
        patch_attention_scores = np.mean(features, axis=1)

    assert len(coords) == len(features)

    # 填充注意力图
    patch_size = int(256 * scale_factor)
    for score, (x, y) in zip(patch_attention_scores, coords):
        scaled_x, scaled_y = int(x * scale_factor), int(y * scale_factor)
        attention_map[scaled_y:scaled_y + patch_size, scaled_x:scaled_x + patch_size] = np.power(score, 5)

    # 高斯模糊和平滑处理
    attention_map = cv2.GaussianBlur(attention_map, (21, 21), 0)
    attention_map = cv2.normalize(attention_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 转换为彩色热力图
    attention_colormap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)

    # 叠加热力图和 WSI 图像
    alpha = 0.4  # 热力图透明度
    overlay_img = cv2.addWeighted(attention_colormap, alpha, wsi_img_bgr, 1 - alpha, 0)

    # 保存结果
    cv2.imwrite(attention_map_save_path, overlay_img)

    # 关闭 OpenSlide 对象
    slide.close()


if __name__ == '__main__':
    slide_path = '/data/lichangyong/TCGA/BLCA_SVS/TCGA-2F-A9KO-01A-01-TSA.114373F3-EFE1-4775-A7D9-AB1C858A7BDC.svs'
    create_attention_map(slide_path, '/home/lichangyong/code/PTCMA/result/blca_png')

     