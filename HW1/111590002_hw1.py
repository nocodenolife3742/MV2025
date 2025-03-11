import os
import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
from scipy.spatial import KDTree


class Question1:

    # color image -> gray image
    @staticmethod
    def grayscaling(img: np.ndarray) -> np.ndarray:
        # 分離RGB通道
        b, g, r = cv2.split(img)
        # 將RGB通道轉換為灰階圖像
        gray_img = 0.3 * r + 0.59 * g + 0.11 * b
        gray_img = np.round(gray_img).astype(np.uint8)
        return gray_img

    # gray image -> binary image
    @staticmethod
    def binarization(img: np.ndarray, threshold: int = 128) -> np.ndarray:
        # 二值化圖像
        binary_img = np.where(img >= threshold, 255, 0).astype(np.uint8)
        return binary_img

    # color image -> index-color image
    @staticmethod
    def index_coloring(img: np.ndarray) -> np.ndarray:
        # 在RGB空間中進行K-means分群，並取得32個顏色
        image_reshaped = img.reshape((-1, 3))  # 壓平圖片的維度
        kmeans = KMeans(n_clusters=32).fit(image_reshaped)  # K-means分群
        centers = kmeans.cluster_centers_  # 取得群集中心
        colors = np.round(centers).astype(np.uint8)  # 四捨五入並轉換為整數
        # 使用KDTree找到每個像素最近的群集中心
        # (不建立新舊顏色對應表，因為作業中圖片不大，直接使用KDTree效率較高)
        # 簡易效率比較：
        # - 使用KDTree: (log 32) * (圖片像素數) = 5 * (圖片像素數) 次運算
        # - 建立新舊顏色對應表: 256^3 * 32 = 536870912 次運算
        tree = KDTree(colors)
        indices = tree.query(image_reshaped)[1]  # 取得最近的群集中心索引
        # 依據索引重建圖像
        index_color_img = colors[indices].reshape(img.shape)
        return index_color_img


class Question2:

    # scaling color image
    @staticmethod
    def scaling_plain(img: np.ndarray, scale: float) -> np.ndarray:
        # 計算新圖像的高度和寬度
        height, width = img.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        # 創建新圖像
        new_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                new_img[i, j] = img[
                    int(i / scale), int(j / scale)
                ]  # 取得原圖像對應座標的像素值
        return new_img

    # scaling color image with interpolation
    @staticmethod
    def scaling_interpolation(img: np.ndarray, scale: float) -> np.ndarray:
        # 計算新圖像的高度和寬度
        height, width = img.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        # 創建新圖像
        new_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        # 對原圖像進行邊緣填充，以避免超出邊界 (圖片的兩個維度各填充一行/列)
        img = np.pad(img, ((0, 1), (0, 1), (0, 0)), mode="edge")
        # 雙線性插值
        for i in range(new_height):
            for j in range(new_width):
                # 計算原圖像座標
                x, y = i / scale, j / scale
                # 計算四個最近的像素座標
                x_min, y_min = math.floor(x), math.floor(y)
                x_max, y_max = x_min + 1, y_min + 1
                # 計算插值權重
                # (由於 min 和 max 差值為 1 => w0 + w1 + w2 + w3 = 1, 因此不需額外除以 1)
                w0 = (x_max - x) * (y_max - y)
                w1 = (x_max - x) * (y - y_min)
                w2 = (x - x_min) * (y_max - y)
                w3 = (x - x_min) * (y - y_min)
                # 計算插值
                new_img[i, j] = (
                    img[x_min, y_min] * w0
                    + img[x_min, y_max] * w1
                    + img[x_max, y_min] * w2
                    + img[x_max, y_max] * w3
                ).astype(np.uint8)
        return new_img


if __name__ == "__main__":
    # 設定 scikit-learn 使用的 CPU 核心數 (避免警告)
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    # 定義路徑
    input_dir = "test_img"
    output_dir = "result_img"
    for i in range(1, 4):
        # 讀取圖像
        img = cv2.imread(os.path.join(input_dir, f"img{i}.jpg"))
        # Question 1
        # 處理圖像
        gray_img = Question1.grayscaling(img)
        binary_img = Question1.binarization(gray_img)
        index_color_img = Question1.index_coloring(img)
        # 儲存結果
        cv2.imwrite(os.path.join(output_dir, f"img{i}_Q1-1.jpg"), gray_img)
        cv2.imwrite(os.path.join(output_dir, f"img{i}_Q1-2.jpg"), binary_img)
        cv2.imwrite(os.path.join(output_dir, f"img{i}_Q1-3.jpg"), index_color_img)
        # Question 2
        # 縮小/放大圖像
        downscaling_img1 = Question2.scaling_plain(img, 0.5)
        upscaling_img1 = Question2.scaling_plain(img, 2)
        downscaling_img2 = Question2.scaling_interpolation(img, 0.5)
        upscaling_img2 = Question2.scaling_interpolation(img, 2)
        # 儲存結果
        cv2.imwrite(os.path.join(output_dir, f"img{i}_Q2-1half.jpg"), downscaling_img1)
        cv2.imwrite(os.path.join(output_dir, f"img{i}_Q2-1double.jpg"), upscaling_img1)
        cv2.imwrite(os.path.join(output_dir, f"img{i}_Q2-2half.jpg"), downscaling_img2)
        cv2.imwrite(os.path.join(output_dir, f"img{i}_Q2-2double.jpg"), upscaling_img2)
        """
        # 檢查縮小的圖像是否相同
        print(np.all(downscaling_img1 == downscaling_img2))
        # 檢查放大的圖像是否相同
        print(np.all(upscaling_img1 == upscaling_img2))
        """
