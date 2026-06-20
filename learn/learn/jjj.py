import cv2
import numpy as np
import os
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.ndimage import gaussian_filter


def add_fog(image, fog_intensity=0.5, fog_density=0.5, fog_color=(220, 220, 220),
            depth_effect=True, noise_variance=0.05, color_attenuation=True,
            sun_direction=None, non_uniformity=0.7):
    """
    添加更逼真的雾效到图像

    参数:
        image: 输入图像 (numpy array)
        fog_intensity: 雾强度 (0-1), 值越大雾越浓
        fog_density: 雾密度 (0-1), 控制雾的分布稀疏程度
        fog_color: 雾的颜色 (BGR元组)
        depth_effect: 是否应用深度雾效果
        noise_variance: 雾噪声强度
        color_attenuation: 是否应用色彩衰减
        sun_direction: 太阳方向 (None表示均匀雾)
        non_uniformity: 雾的不均匀性程度 (0-1)
    """
    h, w, c = image.shape
    image = np.float32(image) / 255.0  # 归一化到 [0, 1]

    # 1. 生成基础雾层
    # 转换雾强度为物理上的衰减系数
    beta = 0.1 + fog_intensity * 2.0  # [0.1, 2.1]

    # 2. 创建深度图 (简单模拟，前景物体雾少，背景雾多)
    if depth_effect:
        # 创建从近到远的深度梯度
        depth_map = np.ones((h, w), dtype=np.float32)
        # 添加一些随机物体（模拟前景障碍物）
        for _ in range(5):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            r = np.random.randint(min(h, w) // 10, min(h, w) // 4)
            cv2.circle(depth_map, (x, y), r, 0.3, -1)
        # 高斯模糊以平滑深度过渡
        depth_map = gaussian_filter(depth_map, sigma=min(h, w) // 10)
    else:
        depth_map = np.ones((h, w), dtype=np.float32)

    # 3. 创建雾的不均匀分布
    # 使用多尺度噪声模拟真实雾的不规则性
    noise = np.zeros((h, w), dtype=np.float32)
    for scale in [1, 2, 4, 8]:
        s_h, s_w = h // scale, w // scale
        # 在不同尺度上生成随机噪声
        scale_noise = np.random.normal(0, 1, (s_h, s_w)).astype(np.float32)
        # 上采样到原始尺寸
        scale_noise = cv2.resize(scale_noise, (w, h))
        noise += scale_noise * (1.0 / scale)

    # 归一化噪声到 [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    # 4. 结合深度和噪声创建最终雾分布
    transmission = np.exp(-beta * depth_map * (1 - non_uniformity * noise))

    # 5. 处理太阳方向（如果提供）
    if sun_direction is not None:
        # 创建太阳方向的光照图
        sun_x, sun_y = sun_direction
        sun_map = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                # 计算与太阳方向的角度差
                angle = np.arctan2(y - sun_y, x - sun_x)
                # 太阳附近区域雾更淡
                sun_map[y, x] = 0.8 + 0.2 * np.cos(angle)

        # 合并到透射率图
        transmission = transmission * sun_map

    # 6. 应用色彩衰减（雾会使远处物体颜色变淡）
    if color_attenuation:
        # 计算每个像素的雾浓度
        fog_amount = 1 - transmission
        # 创建色彩衰减系数（红色衰减较少，蓝色衰减较多）
        attenuation = np.array([1.0 - fog_amount * 0.3,  # B
                                1.0 - fog_amount * 0.2,  # G
                                1.0 - fog_amount * 0.1],  # R
                               dtype=np.float32).transpose(1, 2, 0)
        image = image * attenuation

    # 7. 创建雾色（大气光）
    A = np.array(fog_color, dtype=np.float32) / 255.0

    # 8. 应用雾效公式 I = J*t + A*(1-t)
    foggy_image = image * transmission[..., np.newaxis] + A * (1 - transmission[..., np.newaxis])

    # 9. 添加细微噪声，增强真实感
    if noise_variance > 0:
        noise = np.random.normal(0, noise_variance, (h, w, c)).astype(np.float32)
        foggy_image = np.clip(foggy_image + noise, 0, 1)

    # 转换回 uint8
    foggy_image = np.uint8(foggy_image * 255)
    return foggy_image


def process_folder(input_folder, output_folder, fog_intensity_range=(0.3, 0.7),
                   fog_density=0.5, fog_color=(220, 220, 220), depth_effect=True,
                   color_attenuation=True, non_uniformity=0.7, limit=None):
    """
    处理文件夹中的所有图像

    参数:
        input_folder: 输入图像文件夹
        output_folder: 输出图像文件夹
        fog_intensity_range: 雾强度范围 (min, max)
        limit: 处理图像数量限制（用于测试）
    """
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有图像文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]

    if limit:
        image_files = image_files[:limit]

    # 处理每张图像
    for filename in tqdm(image_files, desc="处理图像"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"foggy_{filename}")

        # 读取图像
        image = cv2.imread(input_path)
        if image is None:
            print(f"警告: 无法读取图像 {input_path}")
            continue

        # 随机选择雾强度
        fog_intensity = np.random.uniform(fog_intensity_range[0], fog_intensity_range[1])

        # 添加雾效
        foggy_image = add_fog(
            image,
            fog_intensity=fog_intensity,
            fog_density=fog_density,
            fog_color=fog_color,
            depth_effect=depth_effect,
            color_attenuation=color_attenuation,
            non_uniformity=non_uniformity
        )

        # 保存结果
        cv2.imwrite(output_path, foggy_image)

    print(f"完成! 共处理 {len(image_files)} 张图像")


def main():
    parser = ArgumentParser(description="图像雾效增强工具")
    parser.add_argument("--input", required=True, help="输入图像或文件夹路径")
    parser.add_argument("--output", required=True, help="输出图像或文件夹路径")
    parser.add_argument("--intensity", type=float, default=0.5, help="雾强度 (0-1)")
    parser.add_argument("--intensity-range", type=float, nargs=2, default=(0.3, 0.7),
                        help="雾强度随机范围 (min, max)")
    parser.add_argument("--density", type=float, default=0.5, help="雾密度 (0-1)")
    parser.add_argument("--color", type=int, nargs=3, default=(220, 220, 220),
                        help="雾颜色 (B G R)")
    parser.add_argument("--no-depth", action="store_false", dest="depth_effect",
                        help="不应用深度雾效果")
    parser.add_argument("--no-color-attenuation", action="store_false", dest="color_attenuation",
                        help="不应用色彩衰减")
    parser.add_argument("--non-uniformity", type=float, default=0.7,
                        help="雾的不均匀性程度 (0-1)")
    parser.add_argument("--limit", type=int, default=None, help="处理图像数量限制")

    args = parser.parse_args()

    if os.path.isdir(args.input):
        # 处理文件夹
        process_folder(
            args.input,
            args.output,
            fog_intensity_range=args.intensity_range,
            fog_density=args.density,
            fog_color=tuple(args.color),
            depth_effect=args.depth_effect,
            color_attenuation=args.color_attenuation,
            non_uniformity=args.non_uniformity,
            limit=args.limit
        )
    else:
        # 处理单张图像
        if not os.path.exists(args.input):
            print(f"错误: 文件 {args.input} 不存在")
            return

        image = cv2.imread(args.input)
        if image is None:
            print(f"错误: 无法读取图像 {args.input}")
            return

        foggy_image = add_fog(
            image,
            fog_intensity=args.intensity,
            fog_density=args.density,
            fog_color=tuple(args.color),
            depth_effect=args.depth_effect,
            color_attenuation=args.color_attenuation,
            non_uniformity=args.non_uniformity
        )

        cv2.imwrite(args.output, foggy_image)
        print(f"雾效图像已保存至: {args.output}")


if __name__ == "__main__":
    # 手动设置路径（代替命令行参数）
    input_folder = r"D:\tomcatz\imagesyolov\imagesyolov\val"
    output_folder = r"D:\tomcatz\image2"

    # 调用批量处理函数
    process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        fog_intensity_range=(0.3, 0.7),  # 其他参数按需调整
        limit=None  # 处理全部图像，若测试可设为10
    )