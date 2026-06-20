import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# 初始化YOLO模型
model = YOLO("yolov8n.pt")  # 替换为你的模型路径

# 支持的图像扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def enhance_image(img):
    """图像增强"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def dehaze_image(image, r_dark=10, r_guide=20, eps=0.01, w=0.85, t0=0.1):
    """去雾处理主函数"""
    try:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0

        img_min = min_channel(img_norm)
        img_dark = min_filter(img_min, r_dark)

        img_guided = guided_filter(img_min, img_dark, r_guide, eps)
        t, A = select_bright(img_guided, img_rgb, w, t0)

        dehazed = repair(img_norm, t, A)
        dehazed = np.clip(dehazed, 0, 1)
        dehazed_uint8 = (dehazed * 255).astype(np.uint8)

        dark_channel_mean = np.mean(img_dark)
        fog_level_dcp = get_fog_level(dark_channel_mean)

        # 图像增强
        enhanced_img = enhance_image(dehazed_uint8)

        return img_rgb, img_dark, enhanced_img, fog_level_dcp

    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None, None, None, None

def get_fog_level(dark_channel_mean):
    """根据暗通道均值判断雾气程度"""
    if dark_channel_mean < 0.1:
        return '轻度雾'
    elif dark_channel_mean < 0.6:
        return '中度雾'
    else:
        return '重度雾'

def detect_objects(image):
    """使用YOLO模型进行检测"""
    results = model(image, augment=False)
    object_counts = {cls: 0 for cls in model.names.values()}
    boxes = []

    for result in results:
        boxes_np = result.boxes.cpu().numpy()  # 获取检测框
        cls_ids = result.boxes.cls.cpu().numpy()  # 获取类别ID
        confidences = result.boxes.conf.cpu().numpy()  # 获取置信度

        for box, cls_id, conf in zip(boxes_np, cls_ids, confidences):
            if len(box) >= 4:
                x1, y1, x2, y2 = box[:4]  # 获取边界框坐标
                lbl = model.names.get(int(cls_id), 'unknown')
                if lbl in object_counts:
                    object_counts[lbl] += 1
                boxes.append([x1, y1, x2, y2, float(conf), int(cls_id)])

    detected_objects = [f"{count} {cls}" for cls, count in object_counts.items() if count > 0]
    return object_counts, boxes, detected_objects

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_frame():
    try:
        if 'frame' not in request.files:
            return jsonify(error="未上传视频帧"), 400

        file = request.files['frame']
        if file.filename == '':
            return jsonify(error="未选择文件"), 400

        if file and allowed_file(file.filename):
            # 保存原始文件
            filename = secure_filename(file.filename)
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            # 读取图像
            img = cv2.imread(upload_path)
            if img is None:
                return jsonify(error="无效的图像数据"), 400

            start_time = time.time()
            img_rgb, img_dark, dehazed_img, fog_level_dcp = dehaze_image(img)
            preprocess_time = (time.time() - start_time) * 1000

            if dehazed_img is None:
                return jsonify(error="去雾处理失败"), 500

            start_time = time.time()
            object_counts, boxes, detected_objects = detect_objects(dehazed_img)
            inference_time = (time.time() - start_time) * 1000

            start_time = time.time()
            image_shape = dehazed_img.shape
            postprocess_time = (time.time() - start_time) * 1000

            process_time = preprocess_time + inference_time + postprocess_time

            detected_objects_str = ', '.join(detected_objects) if detected_objects else '无检测到的物体'

            # 保存处理后的图像
            processed_img_path = os.path.join(PROCESSED_FOLDER, f"processed_{filename}")
            cv2.imwrite(processed_img_path, dehazed_img)

            return jsonify({
                'object_counts': object_counts,
                'boxes': boxes,
                'process_time': process_time,
                'preprocess_time': preprocess_time,
                'inference_time': inference_time,
                'postprocess_time': postprocess_time,
                'image_shape': image_shape,
                'detected_objects': detected_objects_str,
                'image_path': f"/{processed_img_path}",
                'model_names': model.names,
                'fog_level': fog_level_dcp
            })
        else:
            return jsonify(error="不支持的文件类型"), 400
    except Exception as e:
        app.logger.error(f"处理失败: {str(e)}")
        return jsonify(error=f"处理失败: {str(e)}"), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)