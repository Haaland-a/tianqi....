
"""
雾天数据增强模块
模拟不同浓度的雾天环境，提升模型在雾天的鲁棒性
"""
import cv2
import numpy as np
import random


class FogAugmentation:
    """
    雾天增强类
    用于在训练时模拟雾天效果
    """

    def __init__(self, fog_intensity_range=(0.3, 0.8)):
        """
        Args:
            fog_intensity_range: 雾浓度范围 (0-1)
        """
        self.fog_intensity_range = fog_intensity_range

    def add_fog(self, image, intensity=None):
        """
        添加雾天效果
        Args:
            image: 输入图像
            intensity: 雾浓度 (0-1)，None 时随机生成
        Returns:
            雾天图像
        """
        if intensity is None:
            intensity = random.uniform(*self.fog_intensity_range)

        # 转换为浮点数
        img_float = image.astype(np.float32) / 255.0

        # 生成大气光（A）
        A = np.ones_like(img_float) * (1 - intensity * 0.3)

        # 生成透射率图（t）
        h, w = img_float.shape[:2]
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)
        depth_map = np.sqrt(X ** 2 + Y ** 2)
        depth_map = depth_map / depth_map.max()

        # 透射率
        t = np.exp(-intensity * depth_map)
        t = np.stack([t] * 3, axis=-1)

        # 雾天模型: I = J * t + A * (1 - t)
        foggy_image = img_float * t + A * (1 - t)

        # 转换为 0-255
        foggy_image = np.clip(foggy_image * 255, 0, 255).astype(np.uint8)

        return foggy_image

    def __call__(self, image):
        return self.add_fog(image)


# 使用示例
def test_fog_augmentation():
    """测试雾天增强效果"""
    # 读取图像
    image = cv2.imread('test.jpg')

    # 创建雾天增强器
    fog_aug = FogAugmentation(fog_intensity_range=(0.2, 0.9))

    # 添加不同浓度的雾
    light_fog = fog_aug.add_fog(image, intensity=0.3)
    medium_fog = fog_aug.add_fog(image, intensity=0.6)
    heavy_fog = fog_aug.add_fog(image, intensity=0.9)

    # 保存结果
    cv2.imwrite('light_fog.jpg', light_fog)
    cv2.imwrite('medium_fog.jpg', medium_fog)
    cv2.imwrite('heavy_fog.jpg', heavy_fog)


if __name__ == '__main__':
    test_fog_augmentation()
