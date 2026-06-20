import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QUrl, QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
                             QHBoxLayout, QMessageBox, QFileDialog, QProgressBar, QStatusBar,
                             QGridLayout, QTabWidget, QGroupBox, QScrollArea, QComboBox, QTextEdit)
from PyQt5.QtGui import QImage, QPixmap, QPalette, QBrush, QColor, QIcon, QFont
from datetime import datetime

from 源代码文件夹.桌面端 import MainWindow

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


class WorkerThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list, list)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths

    def run(self):
        results = []
        dehazed_images = []
        total = len(self.image_paths)

        for i, image_path in enumerate(self.image_paths):
            try:
                self.progress.emit(int((i + 1) / total * 100), f"正在处理: {os.path.basename(image_path)}")
                img = cv2.imread(image_path)
                _, _, dehazed_img, _ = dehaze_image(img)
                object_counts, boxes, detected_objects = detect_objects(dehazed_img)
                result = {
                    'object_counts': object_counts,
                    'boxes': boxes,
                    'detected_objects': detected_objects
                }
                if result:
                    results.append(result)
                    dehazed_images.append(dehazed_img)
            except Exception as e:
                self.error_occurred.emit(f"处理 {os.path.basename(image_path)} 时出错: {str(e)}")
                continue

        self.finished.emit(results, dehazed_images)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能雾天交通检测系统")
        self.setGeometry(100, 100, 1400, 800)
        self.setWindowIcon(QIcon('traffic_icon.png'))

        self.image_paths = []
        self.results = []
        self.dehazed_images = []
        self.current_image_index = 0
        self.zoom_factor = 1.0

        self.init_ui()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def init_ui(self):
        """初始化用户界面"""
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主布局使用垂直布局
        main_layout = QVBoxLayout(main_widget)

        # 创建选项卡
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 创建检测选项卡
        self.create_detection_tab()

        # 创建统计选项卡
        self.create_statistics_tab()

        # 创建设置选项卡
        self.create_settings_tab()

        # 添加底部按钮布局
        self.create_bottom_buttons(main_layout)

    def create_detection_tab(self):
        """创建检测选项卡"""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # 左侧面板 - 原始图像和去雾图像
        left_panel = QVBoxLayout()

        # 原始图像组
        original_group = QGroupBox("原始图像")
        original_layout = QVBoxLayout()
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setStyleSheet("background-color: black;")
        original_layout.addWidget(self.original_image_label)
        original_group.setLayout(original_layout)
        left_panel.addWidget(original_group)

        # 去雾图像组
        dehazed_group = QGroupBox("去雾后图像")
        dehazed_layout = QVBoxLayout()
        self.dehazed_image_label = QLabel()
        self.dehazed_image_label.setAlignment(Qt.AlignCenter)
        self.dehazed_image_label.setStyleSheet("background-color: black;")
        dehazed_layout.addWidget(self.dehazed_image_label)
        dehazed_group.setLayout(dehazed_layout)
        left_panel.addWidget(dehazed_group)

        layout.addLayout(left_panel, 1)

        # 右侧面板 - 检测结果和操作
        right_panel = QVBoxLayout()

        # 检测结果组
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout()

        # 检测结果图像
        self.detected_image_label = QLabel()
        self.detected_image_label.setAlignment(Qt.AlignCenter)
        self.detected_image_label.setStyleSheet("background-color: black;")
        result_layout.addWidget(self.detected_image_label)

        # 检测信息
        self.result_info = QLabel("等待检测...")
        self.result_info.setAlignment(Qt.AlignCenter)
        self.result_info.setStyleSheet("""
            font-size: 14px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 5px;
        """)
        result_layout.addWidget(self.result_info)

        # 检测详情
        self.result_details = QTextEdit()
        self.result_details.setReadOnly(True)
        self.result_details.setStyleSheet("""
            font-size: 12px;
            background-color: #f8f8f8;
        """)
        result_layout.addWidget(self.result_details)

        result_group.setLayout(result_layout)
        right_panel.addWidget(result_group, 1)

        # 导航按钮
        nav_buttons = QHBoxLayout()

        self.prev_button = QPushButton("上一张")
        self.prev_button.setIcon(QIcon('prev_icon.png'))
        self.prev_button.setEnabled(False)
        self.prev_button.clicked.connect(self.show_previous_image)
        nav_buttons.addWidget(self.prev_button)

        self.next_button = QPushButton("下一张")
        self.next_button.setIcon(QIcon('next_icon.png'))
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.show_next_image)
        nav_buttons.addWidget(self.next_button)

        right_panel.addLayout(nav_buttons)

        layout.addLayout(right_panel, 1)

        self.tab_widget.addTab(tab, "目标检测")

    def create_statistics_tab(self):
        """创建统计选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 统计信息显示
        self.stats_label = QLabel("统计信息将在此显示")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.stats_label)

        # 添加统计图表区域
        self.stats_chart = QLabel()
        self.stats_chart.setAlignment(Qt.AlignCenter)
        self.stats_chart.setStyleSheet("background-color: white; border: 1px solid #ddd;")
        layout.addWidget(self.stats_chart, 1)

        # 更新统计按钮
        update_stats_btn = QPushButton("更新统计")
        update_stats_btn.clicked.connect(self.update_statistics)
        layout.addWidget(update_stats_btn)

        self.tab_widget.addTab(tab, "统计分析")

    def create_settings_tab(self):
        """创建设置选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 模型设置组
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()

        # 模型选择
        model_select_btn = QPushButton("选择模型文件")
        model_select_btn.clicked.connect(self.load_model)
        model_layout.addWidget(model_select_btn)

        # 模型信息
        self.model_info = QLabel("未加载模型")
        self.model_info.setStyleSheet("font-size: 12px; color: #666;")
        model_layout.addWidget(self.model_info)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # 检测设置组
        detect_group = QGroupBox("检测设置")
        detect_layout = QVBoxLayout()

        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_label = QLabel("置信度阈值:")
        self.conf_threshold = QComboBox()
        self.conf_threshold.addItems([f"{i / 10:.1f}" for i in range(1, 10)])
        self.conf_threshold.setCurrentText("0.5")
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_threshold)
        detect_layout.addLayout(conf_layout)

        # 检测速度
        speed_layout = QHBoxLayout()
        speed_label = QLabel("检测速度:")
        self.detect_speed = QComboBox()
        self.detect_speed.addItems(["快速", "平衡", "精确"])
        self.detect_speed.setCurrentText("平衡")
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.detect_speed)
        detect_layout.addLayout(speed_layout)

        detect_group.setLayout(detect_layout)
        layout.addWidget(detect_group)

        # 界面设置组
        ui_group = QGroupBox("界面设置")
        ui_layout = QVBoxLayout()

        # 主题选择
        theme_layout = QHBoxLayout()
        theme_label = QLabel("主题:")
        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["浅色", "深色", "蓝色"])
        self.theme_selector.setCurrentText("浅色")
        self.theme_selector.currentTextChanged.connect(self.change_theme)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_selector)
        ui_layout.addLayout(theme_layout)

        ui_group.setLayout(ui_layout)
        layout.addWidget(ui_group)

        layout.addStretch()

        self.tab_widget.addTab(tab, "系统设置")

    def create_bottom_buttons(self, main_layout):
        """创建底部按钮布局"""
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 操作按钮
        button_layout = QHBoxLayout()

        # 单图检测按钮
        self.detect_btn = QPushButton("单图检测")
        self.detect_btn.setIcon(QIcon('detect_icon.png'))
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self.detect_image)
        button_layout.addWidget(self.detect_btn)

        # 批量检测按钮
        self.batch_btn = QPushButton("批量检测")
        self.batch_btn.setIcon(QIcon('batch_icon.png'))
        self.batch_btn.setEnabled(False)
        self.batch_btn.clicked.connect(self.batch_detect_folder)
        button_layout.addWidget(self.batch_btn)

        # 保存结果按钮
        self.save_btn = QPushButton("保存结果")
        self.save_btn.setIcon(QIcon('save_icon.png'))
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_results)
        button_layout.addWidget(self.save_btn)

        # 退出按钮
        exit_btn = QPushButton("退出系统")
        exit_btn.setIcon(QIcon('exit_icon.png'))
        exit_btn.clicked.connect(self.exit_application)
        button_layout.addWidget(exit_btn)

        main_layout.addLayout(button_layout)

    def load_model(self):
        """加载模型文件"""
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt)"
        )

        if model_path:
            global model
            try:
                model = YOLO(model_path)
                self.detect_btn.setEnabled(True)
                self.batch_btn.setEnabled(True)
                self.model_info.setText(f"已加载模型: {os.path.basename(model_path)}")
                self.status_bar.showMessage("模型加载成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法加载模型: {str(e)}")
                self.status_bar.showMessage("模型加载失败")

    def detect_image(self):
        """单图检测"""
        image_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "", "图片文件 (*.jpg *.jpeg *.png *.webp)"
        )

        if image_path:
            self.process_single_image(image_path)

    def process_single_image(self, image_path):
        """处理单张图片"""
        try:
            # 显示加载状态
            self.status_bar.showMessage(f"正在处理: {os.path.basename(image_path)}...")
            QApplication.processEvents()

            img = cv2.imread(image_path)
            _, _, dehazed_img, fog_level_dcp = dehaze_image(img)
            object_counts, boxes, detected_objects = detect_objects(dehazed_img)

            result = {
                'object_counts': object_counts,
                'boxes': boxes,
                'detected_objects': detected_objects,
                'fog_level': fog_level_dcp
            }

            if result:
                # 显示结果
                self.display_results(image_path, result, dehazed_img)
                self.save_btn.setEnabled(True)
                self.status_bar.showMessage("检测完成")
            else:
                QMessageBox.warning(self, "警告", "未能检测到目标")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理图片时出错: {str(e)}")
            self.status_bar.showMessage("检测失败")

    def batch_detect_folder(self):
        """批量检测文件夹中的图片"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")

        if folder_path:
            # 获取支持的图片文件
            self.image_paths = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
            ]

            if not self.image_paths:
                QMessageBox.warning(self, "警告", "文件夹中没有支持的图片文件")
                return

            # 初始化进度条
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_bar.showMessage("开始批量检测...")

            # 创建工作线程
            self.worker_thread = WorkerThread(self.image_paths)
            self.worker_thread.progress.connect(self.update_progress)
            self.worker_thread.finished.connect(self.on_batch_finished)
            self.worker_thread.error_occurred.connect(self.show_error)
            self.worker_thread.start()

    def update_progress(self, value, message):
        """更新进度条和状态信息"""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(message)

    def on_batch_finished(self, results, dehazed_images):
        """批量检测完成处理"""
        self.results = results
        self.dehazed_images = dehazed_images
        self.current_image_index = 0

        if results:
            self.display_current_image()
            self.prev_button.setEnabled(True)
            self.next_button.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.status_bar.showMessage(f"批量检测完成，共检测到 {len(results)} 张图片")
        else:
            QMessageBox.warning(self, "警告", "没有检测到任何有效结果")
            self.status_bar.showMessage("批量检测完成，但未检测到目标")

        self.progress_bar.setVisible(False)

    def show_error(self, error_msg):
        """显示错误信息"""
        QMessageBox.warning(self, "处理错误", error_msg)

    def show_previous_image(self):
        """显示上一张图片"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()

    def show_next_image(self):
        """显示下一张图片"""
        if self.current_image_index < len(self.results) - 1:
            self.current_image_index += 1
            self.display_current_image()

    def display_current_image(self):
        """显示当前图片"""
        if 0 <= self.current_image_index < len(self.results):
            image_path = self.image_paths[self.current_image_index]
            result = self.results[self.current_image_index]
            dehazed_image = self.dehazed_images[self.current_image_index]
            self.display_results(image_path, result, dehazed_image)

    def display_results(self, image_path, result, dehazed_image):
        """显示检测结果"""
        # 绘制检测框
        img = dehazed_image.copy()
        for box in result['boxes']:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cls_id = int(box[5])
            cls_name = model.names.get(cls_id, 'unknown')
            conf = box[4]
            cv2.putText(img, f"{cls_name}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示原始图像
        self.set_label_image(self.original_image_label, cv2.imread(image_path))

        # 显示去雾后图像
        self.set_label_image(self.dehazed_image_label, dehazed_image)

        # 显示检测结果图像
        self.set_label_image(self.detected_image_label, img)

        # 显示检测信息
        self.show_detection_info(result, image_path)

    def set_label_image(self, label, image):
        """设置标签显示的图像"""
        if image is not None:
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimage)
            label.setPixmap(pixmap.scaled(
                int(width * self.zoom_factor), int(height * self.zoom_factor), Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    def show_detection_info(self, result, image_path):
        """显示检测详细信息"""
        # 基本信息
        info = f"<b>图片:</b> {os.path.basename(image_path)}<br>"
        info += f"<b>雾气程度:</b> {result.get('fog_level', '未知')}<br>"

        # 检测结果统计
        info += "<b>检测结果:</b><br>"
        for cls, count in result['object_counts'].items():
            info += f"  {cls}: {count}个<br>"

        self.result_info.setText(info)

        # 详细检测结果
        details = "<b>详细检测结果:</b>\n"
        for i, box in enumerate(result['boxes'], 1):
            r = box[:4].astype(int)
            cls_id = int(box[5])
            conf = box[4]
            cls_name = model.names.get(cls_id, 'unknown')
            details += f"{i}. {cls_name} (置信度: {conf:.2f}) - 位置: {r}\n"

        self.result_details.setText(details)

    def save_results(self):
        """保存检测结果"""
        if not self.results:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", f"检测结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "文本文件 (*.txt);;所有文件 (*)", options=options
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"雾天交通检测结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                    for i, result in enumerate(self.results):
                        image_path = self.image_paths[i]
                        f.write(f"图片: {os.path.basename(image_path)}\n")
                        f.write(f"路径: {image_path}\n")
                        f.write(f"雾气程度: {result.get('fog_level', '未知')}\n")

                        f.write("检测统计:\n")
                        for cls, count in result['object_counts'].items():
                            f.write(f"  {cls}: {count}个\n")

                        f.write("\n详细检测:\n")
                        for j, box in enumerate(result['boxes'], 1):
                            r = box[:4].astype(int)
                            cls_id = int(box[5])
                            conf = box[4]
                            cls_name = model.names.get(cls_id, 'unknown')
                            f.write(f"{j}. {cls_name} (置信度: {conf:.2f}) - 位置: {r}\n")

                        f.write("\n" + "=" * 50 + "\n\n")

                QMessageBox.information(self, "成功", f"检测结果已保存到:\n{file_path}")
                self.status_bar.showMessage(f"结果已保存: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存结果时出错: {str(e)}")
                self.status_bar.showMessage("保存失败")

    def update_statistics(self):
        """更新统计信息"""
        if not self.results:
            QMessageBox.warning(self, "警告", "没有可统计的结果")
            return

        # 模拟统计信息
        total_images = len(self.results)
        total_objects = sum(sum(result['object_counts'].values()) for result in self.results)

        # 按类别统计
        class_counts = {}
        for result in self.results:
            for cls, count in result['object_counts'].items():
                if cls in class_counts:
                    class_counts[cls] += count
                else:
                    class_counts[cls] = count

        # 生成统计文本
        stats_text = f"<b>统计摘要</b><br>"
        stats_text += f"总图片数: {total_images}<br>"
        stats_text += f"总检测目标数: {total_objects}<br><br>"

        stats_text += "<b>按类别统计:</b><br>"
        for cls, count in class_counts.items():
            stats_text += f"{cls}: {count} ({count / total_objects:.1%})<br>"

        self.stats_label.setText(stats_text)

        # 模拟图表 (实际应用中可以使用matplotlib等库生成真实图表)
        chart_pixmap = QPixmap(600, 400)
        chart_pixmap.fill(Qt.white)
        self.stats_chart.setPixmap(chart_pixmap)
        self.stats_chart.setText("统计图表区域 (实际应用中可显示真实图表)")

        self.status_bar.showMessage("统计信息已更新")

    def change_theme(self, theme):
        """更改界面主题"""
        if theme == "浅色":
            self.set_light_theme()
        elif theme == "深色":
            self.set_dark_theme()
        elif theme == "蓝色":
            self.set_blue_theme()

        self.status_bar.showMessage(f"已切换至{theme}主题")

    def set_light_theme(self):
        """设置浅色主题"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QLabel {
                color: #333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #ddd;
            }
        """)

        def set_dark_theme(self):
            """设置深色主题"""
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #333;
                }
                QGroupBox {
                    border: 1px solid #555;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 15px;
                    background-color: #444;
                    color: white;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 3px;
                    color: white;
                }
                QLabel {
                    color: #eee;
                }
                QPushButton {
                    background-color: #555;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #666;
                }
                QTextEdit {
                    background-color: #444;
                    color: white;
                    border: 1px solid #555;
                }
                QProgressBar {
                    border: 1px solid #555;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    width: 10px;
                }
            """)

        def set_blue_theme(self):
            """设置蓝色主题"""
            self.setStyleSheet("""
                    QMainWindow {
                        background-color: #e6f2ff;
                    }
                    QGroupBox {
                        border: 1px solid #99c2ff;
                        border-radius: 5px;
                        margin-top: 10px;
                        padding-top: 15px;
                        background-color: #cce0ff;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 10px;
                        padding: 0 3px;
                    }
                    QLabel {
                        color: #003366;
                    }
                    QPushButton {
                        background-color: #0066cc;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 4px;
                    }
                    QPushButton:hover {
                        background-color: #0052a3;
                    }
                    QTextEdit {
                        background-color: white;
                        border: 1px solid #99c2ff;
                    }
                    QProgressBar {
                        border: 1px solid #99c2ff;
                        border-radius: 3px;
                        text-align: center;
                    }
                    QProgressBar::chunk {
                        background-color: #4CAF50;
                        width: 10px;
                    }
                """)

        def zoom_in(self):
            """放大图像"""
            self.zoom_factor *= 1.2
            self.display_current_image()

        def zoom_out(self):
            """缩小图像"""
            self.zoom_factor /= 1.2
            self.display_current_image()

        @staticmethod
        def register_protocol(reg=None):
            """静态方法处理协议注册"""
            try:
                if getattr(sys, 'frozen', False):
                    exe_path = sys.executable
                else:
                    exe_path = os.path.abspath(__file__)

                key = reg.CreateKey(reg.HKEY_CLASSES_ROOT, "fogdetector")
                reg.SetValue(key, "", reg.REG_SZ, "URL:FogDetector Protocol")
                reg.SetValueEx(key, "URL Protocol", 0, reg.REG_SZ, "")

                command_key = reg.CreateKey(key, r"shell\open\command")
                reg.SetValue(command_key, "", reg.REG_SZ, f'"{exe_path}" "%1"')
                return True
            except Exception as e:
                print(f"协议注册失败: {e}")
                return False

    def check_protocol(self):
        """独立协议检查函数"""
        try:
            if not MainWindow.register_protocol():
                app = QApplication.instance() or QApplication(sys.argv)
                QMessageBox.warning(None, "警告", "需要管理员权限运行以完成协议注册！")
        except Exception as e:
            print(f"协议检查失败: {e}")

    if __name__ == "__main__":
        # 先执行协议检查
        check_protocol()

        # 处理启动参数
        launch_arg = sys.argv[1] if len(sys.argv) > 1 else None

        # 创建应用实例
        app = QApplication(sys.argv)

        # 设置中文字体
        font = QFont("Microsoft YaHei", 10)
        app.setFont(font)

        # 创建主窗口
        window = MainWindow(launch_args=launch_arg)
        window.show()

        # 启动Flask服务（需要单独线程运行）
        from threading import Thread
        Thread(target=lambda: app.run(debug=False, use_reloader=False)).start()

        sys.exit(app.exec_())