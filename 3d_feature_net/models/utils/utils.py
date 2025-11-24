import numpy as np
from scipy.ndimage import label, center_of_mass
from math import exp, sqrt
import cv2
import torch
import os
from PIL import Image, ImageDraw
from scipy import ndimage, misc, signal, spatial, sparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# 定位细节点, 细节点位置图估计
def compute_probability(area, tau, tau_init):
    """
    计算细节点的概率。
    area: 分量的像素面积
    tau: 当前阈值
    tau_init: 初始阈值 0.4
    """
    return sqrt(area) * (exp(tau - tau_init + 0.1) - 1)

def locate_minutiae(location_map, tau_init=0.4, tau_step=0.5, t_area=45):
    """
    批量处理位置图，生成二值化细节点位置图。
    输入:
        location_map: (B, H, W) 的 PyTorch 张量
    返回:
        binary_output: (B, H, W) 的 PyTorch 张量，细节点位置为 1
    """
    device = location_map.device
    batch_size = location_map.shape[0]
    binary_output = torch.zeros_like(location_map)

    for b in range(batch_size):
        # 转换单个样本为 numpy
        np_map = location_map[b].detach().cpu().numpy()
        H, W = np_map.shape
        detected_minutiae = []
        tau = tau_init

        while True:
            # 阈值分割
            binary_np = (np_map >= tau).astype(np.float32)
            labeled, num = label(binary_np)
            split_needed = False

            for i in range(1, num + 1):
                component = (labeled == i)
                area = np.sum(component)
                if area < t_area:
                    continue
                # 检查是否需要分裂
                if np.any(np_map[component] > tau + tau_step):
                    split_needed = True
                # 记录中心点
                cy, cx = center_of_mass(component)
                detected_minutiae.append((int(cx), int(cy)))  # 图像坐标 (x, y)

            if not split_needed:
                break
            tau += tau_step

        # 生成二值图（去重）
        unique_coords = list(set(detected_minutiae))  # 简单去重
        bin_map = np.zeros((H, W), dtype=np.float32)
        for x, y in unique_coords:
            if 0 <= x < W and 0 <= y < H:
                bin_map[y-3:y+3, x-3:x+3] = 1.0  # 注意坐标顺序 (y, x)

        binary_output[b] = torch.from_numpy(bin_map).to(device)

    return binary_output,unique_coords

def generate_minutiae_map(image_shape, minutiae_locations, r1=3, r2=6):
    """
    生成地面真值指纹特征点图（ground truth minutiae map）
    需要以标注的特征点为圆心，设定r1和r2，划分正样本和负样本区域
    参数:
        image_shape (tuple): 图像的尺寸 (height, width)
        minutiae_locations (list): 特征点位置列表，每个元素为 (x, y)
        r1 (int): 正样本区域的半径
        r2 (int): 负样本区域的半径
        
    返回:
        numpy.ndarray: 地面真值特征点图，值为 1（正样本），-1（负样本），0（忽略区域）
    """
    # 初始化标签地图
    height, width = image_shape
    gt_map = np.full((height, width), -1, dtype=np.int32)  # 默认负样本区域

    for tmp in minutiae_locations:
        x=int(tmp[0])
        y=int(tmp[1])
        # 生成正样本区域
        for i in range(-r2, r2 + 1):
            for j in range(-r2, r2 + 1):
                dist = np.sqrt(i**2 + j**2)
                if 0 <= x + i < width and 0 <= y + j < height:
                    if dist <= r1:
                        gt_map[y + j, x + i] = 1  # 正样本区域
                    elif r1 < dist <= r2:
                        gt_map[y + j, x + i] = 0  # 忽略区域
    
    return gt_map


