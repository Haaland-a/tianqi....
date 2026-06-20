import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

def min_channel(img):
    """计算图像的最小通道值"""
    return np.min(img, axis=2)

def min_filter(image, r):
    """应用最小值滤波"""
    return cv2.erode(image, np.ones((2 * r + 1, 2 * r + 1)))

def guided_filter(I, p, r, eps):
    """应用引导滤波"""
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b

def select_bright(img_dark, img_origin, w, t0):
    """选择最亮的0.1%像素并估计透射率和大气光"""
    img_origin = img_origin.astype(np.float32) / 255.0

    flat_dark = img_dark.flatten()
    index = int(flat_dark.size * 0.001)
    indices = np.argpartition(-flat_dark, index)[:index]

    rows, cols = img_dark.shape
    candidate_pixels = [img_origin[i // cols, i % cols] for i in indices]

    A = np.max(candidate_pixels, axis=0)

    V = img_dark * w
    t = 1 - V / (np.max(img_dark) + 1e-6)
    t = np.clip(t, t0, 0.9)
    return t, A

def repair(img_norm, t, A):
    """图像修复"""
    t = np.stack([t, t, t], axis=2)
    t = np.maximum(t, 0.08)
    return (img_norm - A) / t + A

def dehaze_image(image_path, r_dark=10, r_guide=20, eps=0.01, w=0.85, t0=0.1):
    """去雾处理主函数"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0

        img_min = min_channel(img_norm)
        img_dark = min_filter(img_min, r_dark)

        img_guided = guided_filter(img_min, img_dark, r_guide, eps)
        t, A = select_bright(img_guided, img_rgb, w, t0)

        dehazed = repair(img_norm, t, A)
        dehazed = np.clip(dehazed, 0, 1)
        dehazed_uint8 = (dehazed * 255).astype(np.uint8)

        dark_channel_mean = np.mean(img_dark)
        if dark_channel_mean < 0.1:
            fog_level_dcp = '轻度雾'
        elif dark_channel_mean < 0.6:
            fog_level_dcp = '中度雾'
        else:
            fog_level_dcp = '重度雾'

        print(f"暗通道先验评估雾浓度等级: {fog_level_dcp}")

        return img_rgb, img_dark, dehazed_uint8, fog_level_dcp

    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None, None, None, None

def main(image_path):
    img_rgb, img_dark, dehazed_uint8, fog_level_dcp = dehaze_image(image_path)
    if img_rgb is not None:
        cv2.imshow('Original', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow('Dark Channel', (img_dark * 255).astype(np.uint8))
        cv2.imshow('Dehazed', cv2.cvtColor(dehazed_uint8, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"C:\Users\Lenovo\Desktop\2.png"
    main(image_path)
