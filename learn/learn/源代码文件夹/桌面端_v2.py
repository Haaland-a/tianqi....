"""
桌面端 v2 - 支持 FCA 模型加载
修复：先设置 ultralytics-main 路径，再导入 YOLO
"""
import sys
import os

# ===== 关键修复：先设置路径，再导入 ultralytics =====
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_BASE_DIR)  # learn/learn/
sys.path.insert(0, os.path.join(_PARENT_DIR, 'ultralytics-main'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import traceback
import logging
from datetime import datetime
from collections import defaultdict, deque
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QObject
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
                             QHBoxLayout, QMessageBox, QFileDialog, QProgressBar, QStatusBar,
                             QGridLayout, QTabWidget, QGroupBox, QScrollArea, QComboBox, QTextEdit,
                             QSpinBox, QStyleFactory, QDoubleSpinBox)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QPalette
from PyQt5.QtMultimedia import QSound

# 路径设置完成后再导入 YOLO
from ultralytics import YOLO

# 日志记录
logging.basicConfig(
    filename='traffic_detection.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# 可用模型列表
AVAILABLE_MODELS = {
    "YOLOv8n (预训练)": os.path.join(_PARENT_DIR, '源代码文件夹', 'yolov8n.pt'),
    "YOLOv8-FCA (训练结果)": os.path.join(_PARENT_DIR, 'runs', 'fca_detect', 'train_fca_cl2', 'weights', 'best.pt'),
}


# === 暗通道先验去雾算法实现 ===
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


def dehaze_image(image_data, r_dark=10, r_guide=20, eps=0.01, w=0.85, t0=0.1):
    """去雾处理主函数"""
    try:
        if image_data is None:
            raise ValueError("输入图像数据为空")

        img_bgr = image_data
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0

        img_min = min_channel(img_norm)
        img_dark = min_filter(img_min, r_dark)

        img_guided = guided_filter(img_min, img_dark, r_guide, eps)
        t, A = select_bright(img_guided, img_rgb, w, t0)

        dehazed = repair(img_norm, t, A)
        dehazed = np.clip(dehazed, 0, 1)
        dehazed_uint8 = (dehazed * 255).astype(np.uint8)

        # 评估雾浓度
        dark_channel_mean = np.mean(img_dark)
        if dark_channel_mean < 0.1:
            fog_level = '轻度雾'
        elif dark_channel_mean < 0.6:
            fog_level = '中度雾'
        else:
            fog_level = '重度雾'

        logging.info(f"去雾成功，雾浓度: {fog_level}")
        return img_rgb, img_dark, dehazed_uint8, fog_level

    except Exception as e:
        logging.error(f"去雾处理错误: {str(e)}")
        return None, None, image_data, "去雾失败"


class WorkerThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list, list, list)
    error_occurred = pyqtSignal(str)

    def __init__(self, worker, image_paths):
        super().__init__()
        self.worker = worker
        self.image_paths = image_paths
        self._is_running = True
        self._mutex = QObject()

    def run(self):
        results = []
        dehazed_images = []
        fog_levels = []
        total = len(self.image_paths)

        try:
            for i, image_path in enumerate(self.image_paths):
                if not self._is_running:
                    break

                try:
                    raw_image = cv2.imread(image_path)
                    if raw_image is None:
                        raise ValueError(f"无法读取图片: {image_path}")

                    _, _, dehazed_uint8, fog_level = dehaze_image(
                        raw_image,
                        self.worker.r_dark.value(),
                        self.worker.r_guide.value(),
                        self.worker.w.value(),
                        self.worker.t0.value()
                    )

                    result = self.worker.detect_image(dehazed_uint8)

                    if result and len(result) > 0:
                        results.append(result[0])
                        dehazed_images.append(dehazed_uint8)
                        fog_levels.append(fog_level)
                    else:
                        results.append(None)
                        dehazed_images.append(dehazed_uint8)
                        fog_levels.append(fog_level)

                    progress = int((i + 1) / total * 100)
                    message = f"处理中: {os.path.basename(image_path)} ({i + 1}/{total})"
                    self.progress.emit(progress, message)

                except Exception as e:
                    error_msg = f"处理 {os.path.basename(image_path)} 时出错: {str(e)}"
                    self.error_occurred.emit(error_msg)
                    continue

            if self._is_running:
                self.finished.emit(results, dehazed_images, fog_levels)

        except Exception as e:
            error_msg = f"批量处理线程崩溃: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)

    def stop(self):
        self._is_running = False
        self.wait()


class Worker:
    def __init__(self):
        self.model = None
        self.class_names = ["自行车", "摩托车", "汽车", "行人"]
        self.alarm_thresholds = {
            "自行车": 3,
            "摩托车": 2,
            "汽车": 10,
            "行人": 5
        }
        self.r_dark = None
        self.r_guide = None
        self.w = None
        self.t0 = None

        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.track_confidence = defaultdict(list)
        self.track_frames = defaultdict(int)
        self.prev_frame_tracks = {}
        self.id_switch_count = 0
        self.total_tracks = 0
        self.false_positives = 0
        self.false_negatives = 0

    def load_model(self, model_path):
        """加载模型 - 支持 FCA 模型"""
        try:
            if not os.path.exists(model_path):
                logging.error(f"模型文件不存在: {model_path}")
                return False
            
            self.model = YOLO(model_path)
            logging.info(f"模型加载成功: {model_path}, 类别: {self.model.names}")
            return True
        except Exception as e:
            logging.error(f"加载模型失败: {str(e)}")
            return False

    def detect_image(self, image):
        try:
            if image is not None:
                results = self.model.predict(
                    source=image,
                    conf=0.25,
                    iou=0.45,
                    verbose=False,
                    classes=None
                )
                return results
            return []
        except Exception as e:
            logging.error(f"检测图片失败: {str(e)}")
            return []

    def track_frame(self, frame, persist=True):
        try:
            if frame is not None and self.model is not None:
                results = self.model.track(
                    source=frame,
                    conf=0.1,
                    iou=0.45,
                    verbose=False,
                    persist=persist,
                    tracker="bytetrack.yaml"
                )
                return results
            return []
        except Exception as e:
            logging.error(f"跟踪帧失败: {str(e)}")
            return []

    def filter_false_detections(self, result):
        if not result or result.boxes is None or len(result.boxes) == 0:
            return result

        valid_indices = []
        for i in range(len(result.boxes)):
            if result.boxes.id is None:
                valid_indices.append(i)
                continue

            track_id = int(result.boxes.id[i].item())
            conf = result.boxes.conf[i].item()

            self.track_confidence[track_id].append(conf)
            self.track_frames[track_id] += 1

            confidence_integral = np.mean(self.track_confidence[track_id]) * min(len(self.track_confidence[track_id]), 10)

            if confidence_integral >= 0.5 or len(self.track_confidence[track_id]) < 3:
                valid_indices.append(i)
            else:
                self.false_positives += 1

        if len(valid_indices) < len(result.boxes):
            result.boxes = result.boxes[valid_indices]

        return result

    def verify_spatiotemporal_consistency(self, result):
        if not result or result.boxes is None or len(result.boxes) == 0:
            return result

        current_tracks = {}
        for i in range(len(result.boxes)):
            if result.boxes.id is not None:
                track_id = int(result.boxes.id[i].item())
                xyxy = result.boxes.xyxy[i].cpu().numpy()
                center = [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]
                current_tracks[track_id] = {
                    'center': center,
                    'bbox': xyxy,
                    'index': i
                }
                self.track_history[track_id].append(center)

        invalid_tracks = []
        for track_id, track_info in current_tracks.items():
            if track_id in self.prev_frame_tracks:
                prev_center = self.prev_frame_tracks[track_id]['center']
                curr_center = track_info['center']

                distance = np.sqrt((curr_center[0] - prev_center[0]) ** 2 +
                                   (curr_center[1] - prev_center[1]) ** 2)

                if distance > 200:
                    invalid_tracks.append(track_info['index'])

        if invalid_tracks:
            valid_indices = [i for i in range(len(result.boxes)) if i not in invalid_tracks]
            result.boxes = result.boxes[valid_indices]

        self.prev_frame_tracks = current_tracks
        return result


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("恶劣天气交通目标检测系统 v2")
        self.setGeometry(100, 100, 1200, 800)
        
        self.worker = Worker()
        self.worker_thread = None
        
        self.init_ui()
        
        # 默认加载第一个可用模型
        self.load_default_model()
    
    def load_default_model(self):
        """加载默认模型"""
        for name, path in AVAILABLE_MODELS.items():
            if os.path.exists(path):
                if self.worker.load_model(path):
                    self.log_message(f"默认模型加载成功: {name}")
                    self.model_combo.setCurrentText(name)
                    return
        self.log_message("警告: 没有找到可用的模型文件")
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 模型选择区域
        model_group = QGroupBox("模型选择")
        model_layout = QHBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(AVAILABLE_MODELS.keys())
        model_layout.addWidget(QLabel("选择模型:"))
        model_layout.addWidget(self.model_combo)
        
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.on_load_model)
        model_layout.addWidget(self.load_model_btn)
        
        self.browse_model_btn = QPushButton("浏览...")
        self.browse_model_btn.clicked.connect(self.on_browse_model)
        model_layout.addWidget(self.browse_model_btn)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # 图像显示区域
        image_layout = QHBoxLayout()
        
        self.original_label = QLabel("原图")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 300)
        self.original_label.setStyleSheet("border: 1px solid gray;")
        
        self.result_label = QLabel("检测结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(400, 300)
        self.result_label.setStyleSheet("border: 1px solid gray;")
        
        image_layout.addWidget(self.original_label)
        image_layout.addWidget(self.result_label)
        main_layout.addLayout(image_layout)
        
        # 去雾参数
        dehaze_group = QGroupBox("去雾参数")
        dehaze_layout = QGridLayout()
        
        dehaze_layout.addWidget(QLabel("暗通道半径:"), 0, 0)
        self.r_dark = QSpinBox()
        self.r_dark.setRange(1, 50)
        self.r_dark.setValue(10)
        dehaze_layout.addWidget(self.r_dark, 0, 1)
        
        dehaze_layout.addWidget(QLabel("引导滤波半径:"), 0, 2)
        self.r_guide = QSpinBox()
        self.r_guide.setRange(1, 100)
        self.r_guide.setValue(20)
        dehaze_layout.addWidget(self.r_guide, 0, 3)
        
        dehaze_layout.addWidget(QLabel("雾浓度权重:"), 1, 0)
        self.w = QDoubleSpinBox()
        self.w.setRange(0.0, 1.0)
        self.w.setValue(0.85)
        self.w.setSingleStep(0.05)
        dehaze_layout.addWidget(self.w, 1, 1)
        
        dehaze_layout.addWidget(QLabel("透射率下限:"), 1, 2)
        self.t0 = QDoubleSpinBox()
        self.t0.setRange(0.01, 0.5)
        self.t0.setValue(0.1)
        self.t0.setSingleStep(0.01)
        dehaze_layout.addWidget(self.t0, 1, 3)
        
        dehaze_group.setLayout(dehaze_layout)
        main_layout.addWidget(dehaze_group)
        
        # 设置 worker 的去雾参数
        self.worker.r_dark = self.r_dark
        self.worker.r_guide = self.r_guide
        self.worker.w = self.w
        self.worker.t0 = self.t0
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        
        self.detect_btn = QPushButton("检测图片")
        self.detect_btn.clicked.connect(self.on_detect_image)
        btn_layout.addWidget(self.detect_btn)
        
        self.batch_btn = QPushButton("批量检测")
        self.batch_btn.clicked.connect(self.on_batch_detect)
        btn_layout.addWidget(self.batch_btn)
        
        main_layout.addLayout(btn_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)
        
        # 日志区域
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
    
    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def on_load_model(self):
        """加载选中的模型"""
        model_name = self.model_combo.currentText()
        model_path = AVAILABLE_MODELS.get(model_name)
        
        if model_path and os.path.exists(model_path):
            if self.worker.load_model(model_path):
                self.log_message(f"模型加载成功: {model_name}")
                self.statusBar().showMessage(f"模型已加载: {model_name}")
            else:
                self.log_message(f"模型加载失败: {model_name}")
                QMessageBox.warning(self, "错误", f"模型加载失败: {model_name}")
        else:
            self.log_message(f"模型文件不存在: {model_path}")
            QMessageBox.warning(self, "错误", f"模型文件不存在: {model_path}")
    
    def on_browse_model(self):
        """浏览并选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt);;所有文件 (*)"
        )
        if file_path:
            if self.worker.load_model(file_path):
                model_name = os.path.basename(file_path)
                # 添加到下拉列表
                if self.model_combo.findText(model_name) == -1:
                    AVAILABLE_MODELS[model_name] = file_path
                    self.model_combo.addItem(model_name)
                self.model_combo.setCurrentText(model_name)
                self.log_message(f"模型加载成功: {model_name}")
            else:
                QMessageBox.warning(self, "错误", "模型加载失败")
    
    def on_detect_image(self):
        """检测单张图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        if file_path:
            self.detect_single_image(file_path)
    
    def detect_single_image(self, image_path):
        """检测单张图片"""
        try:
            raw_image = cv2.imread(image_path)
            if raw_image is None:
                QMessageBox.warning(self, "错误", f"无法读取图片: {image_path}")
                return
            
            # 去雾处理
            _, _, dehazed, fog_level = dehaze_image(
                raw_image,
                self.r_dark.value(),
                self.r_guide.value(),
                self.w.value(),
                self.t0.value()
            )
            
            # 显示原图
            self.display_image(raw_image, self.original_label)
            
            # 检测
            results = self.worker.detect_image(dehazed)
            
            if results and len(results) > 0:
                result = results[0]
                # 绘制检测结果
                annotated = result.plot()
                self.display_image(annotated, self.result_label)
                
                # 统计检测结果
                counts = defaultdict(int)
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = self.worker.class_names[cls_id] if cls_id < len(self.worker.class_names) else f"类别{cls_id}"
                    counts[cls_name] += 1
                
                result_text = ", ".join([f"{k}: {v}" for k, v in counts.items()])
                self.log_message(f"检测完成 ({fog_level}): {result_text}")
                self.statusBar().showMessage(f"检测完成: {result_text}")
            else:
                self.display_image(dehazed, self.result_label)
                self.log_message(f"检测完成 ({fog_level}): 未检测到目标")
                
        except Exception as e:
            self.log_message(f"检测出错: {str(e)}")
            QMessageBox.warning(self, "错误", f"检测出错: {str(e)}")
    
    def on_batch_detect(self):
        """批量检测"""
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            image_files = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            
            if not image_files:
                QMessageBox.warning(self, "警告", "文件夹中没有图片文件")
                return
            
            self.log_message(f"开始批量检测: {len(image_files)} 张图片")
            self.progress_bar.setValue(0)
            
            # 启动工作线程
            self.worker_thread = WorkerThread(self.worker, image_files)
            self.worker_thread.progress.connect(self.on_progress)
            self.worker_thread.finished.connect(self.on_batch_finished)
            self.worker_thread.error_occurred.connect(self.on_error)
            self.worker_thread.start()
    
    def on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)
    
    def on_batch_finished(self, results, dehazed_images, fog_levels):
        self.progress_bar.setValue(100)
        self.log_message(f"批量检测完成: {len(results)} 张图片")
        
        # 显示最后一张结果
        if results and results[-1] is not None:
            annotated = results[-1].plot()
            self.display_image(annotated, self.result_label)
    
    def on_error(self, message):
        self.log_message(f"错误: {message}")
    
    def display_image(self, cv_image, label):
        """在 QLabel 上显示 OpenCV 图像"""
        if cv_image is None:
            return
        
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
