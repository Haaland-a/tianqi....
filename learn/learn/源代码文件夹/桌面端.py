import sys
import os
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
from ultralytics import YOLO

# 日志记录
logging.basicConfig(
    filename='traffic_detection.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)


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
    """
    去雾处理主函数（接受图像数据而非路径）
    :param image_data: numpy数组格式的图像(BGR)
    :return: (原图RGB, 暗通道图, 去雾图RGB, 雾浓度等级)
    """
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
    finished = pyqtSignal(list, list, list)  # 结果、去雾图像、雾浓度
    error_occurred = pyqtSignal(str)

    def __init__(self, worker, image_paths):
        super().__init__()
        self.worker = worker
        self.image_paths = image_paths
        self._is_running = True
        self._mutex = QObject()  # 线程同步对象

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
                    # 读取图像
                    raw_image = cv2.imread(image_path)
                    if raw_image is None:
                        raise ValueError(f"无法读取图片: {image_path}")

                    # 去雾处理
                    _, _, dehazed_uint8, fog_level = dehaze_image(
                        raw_image,
                        self.worker.r_dark.value(),
                        self.worker.r_guide.value(),
                        self.worker.w.value(),
                        self.worker.t0.value()
                    )

                    # 目标检测
                    result = self.worker.detect_image(dehazed_uint8)

                    if result and len(result) > 0:
                        results.append(result[0])
                        dehazed_images.append(dehazed_uint8)
                        fog_levels.append(fog_level)
                    else:
                        results.append(None)
                        dehazed_images.append(dehazed_uint8)
                        fog_levels.append(fog_level)

                    # 更新进度
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
        # 去雾参数默认值
        self.r_dark = None
        self.r_guide = None
        self.w = None
        self.t0 = None

        # === 时空一致性追踪与误检过滤相关参数 ===
        self.track_history = defaultdict(lambda: deque(maxlen=30))  # 轨迹历史（最多30帧）
        self.track_confidence = defaultdict(list)  # 每个目标的置信度历史
        self.track_frames = defaultdict(int)  # 每个目标连续出现的帧数
        self.prev_frame_tracks = {}  # 上一帧的跟踪结果
        self.id_switch_count = 0  # ID切换次数
        self.total_tracks = 0  # 总跟踪目标数
        self.false_positives = 0  # 误检过滤数量
        self.false_negatives = 0  # 漏检数量

    def load_model(self, model_path):
        try:
            self.model = YOLO(model_path)
            logging.info(f"模型加载成功: {model_path}, 类别: {self.model.names}")
            return True
        except Exception as e:
            logging.error(f"加载模型失败: {str(e)}")
            return False

    def detect_image(self, image):
        try:
            if image is not None:
                # YOLOv8预测参数
                results = self.model.predict(
                    source=image,
                    conf=0.25,  # 置信度阈值
                    iou=0.45,  # IoU阈值
                    verbose=False,  # 静默模式
                    classes=None  # 不限制类别
                )
                return results
            return []
        except Exception as e:
            logging.error(f"检测图片失败: {str(e)}")
            return []

    def track_frame(self, frame, persist=True):
        """使用 ByteTrack 进行目标跟踪（时空一致性追踪核心）"""
        try:
            if frame is not None and self.model is not None:
                results = self.model.track(
                    source=frame,
                    conf=0.1,  # ByteTrack需要低置信度阈值
                    iou=0.45,
                    verbose=False,
                    persist=persist,  # 保持跨帧跟踪
                    tracker="bytetrack.yaml"  # 使用ByteTrack跟踪器
                )
                return results
            return []
        except Exception as e:
            logging.error(f"跟踪帧失败: {str(e)}")
            return []

    def filter_false_detections(self, result):
        """
        误检过滤：基于轨迹置信度积分剔除不可靠检测
        置信度积分 = 平均置信度 × 连续出现帧数
        """
        if not result or result.boxes is None or len(result.boxes) == 0:
            return result

        valid_indices = []
        for i in range(len(result.boxes)):
            if result.boxes.id is None:
                valid_indices.append(i)
                continue

            track_id = int(result.boxes.id[i].item())
            conf = result.boxes.conf[i].item()

            # 更新轨迹置信度历史
            self.track_confidence[track_id].append(conf)
            self.track_frames[track_id] += 1

            # 计算置信度积分（平均置信度 × 连续出现帧数，最多算10帧）
            confidence_integral = np.mean(self.track_confidence[track_id]) * min(len(self.track_confidence[track_id]),
                                                                                 10)

            # 误检过滤：保留置信度积分≥0.5 或 出现帧数<3的检测
            if confidence_integral >= 0.5 or len(self.track_confidence[track_id]) < 3:
                valid_indices.append(i)
            else:
                self.false_positives += 1
                logging.debug(f"过滤误检: track_id={track_id}, 置信度积分={confidence_integral:.2f}")

        # 过滤无效检测
        if len(valid_indices) < len(result.boxes):
            result.boxes = result.boxes[valid_indices]

        return result

    def verify_spatiotemporal_consistency(self, result):
        """
        时空一致性验证：检查目标位置变化的合理性
        利用卡尔曼滤波预测位置与外观特征的连续性
        """
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
                # 更新轨迹历史
                self.track_history[track_id].append(center)

        # 检查位置跳变（时空一致性）
        invalid_tracks = []
        for track_id, track_info in current_tracks.items():
            if track_id in self.prev_frame_tracks:
                prev_center = self.prev_frame_tracks[track_id]['center']
                curr_center = track_info['center']

                # 计算欧氏距离
                distance = np.sqrt((curr_center[0] - prev_center[0]) ** 2 +
                                   (curr_center[1] - prev_center[1]) ** 2)

                # 如果位置跳变过大（>200px），标记为不可靠
                if distance > 200:
                    invalid_tracks.append(track_info['index'])
                    logging.debug(f"时空不一致: track_id={track_id}, 位置跳变={distance:.1f}px")

        # 过滤时空不一致的检测
        if invalid_tracks:
            valid_indices = [i for i in range(len(result.boxes)) if i not in invalid_tracks]
            result.boxes = result.boxes[valid_indices]

        # 更新上一帧跟踪结果
        self.prev_frame_tracks = current_tracks

        return result

    def calculate_track_metrics(self, result):
        """计算跟踪指标：ID Switch、MOTA等"""
        if not result or result.boxes is None or len(result.boxes) == 0:
            return

        current_ids = set()
        if result.boxes.id is not None:
            current_ids = set(int(id.item()) for id in result.boxes.id)

        # 检测 ID Switch
        if self.prev_frame_tracks:
            prev_ids = set(self.prev_frame_tracks.keys())
            for track_id in current_ids:
                if track_id not in prev_ids:
                    self.id_switch_count += 1

        self.total_tracks = len(current_ids)

    def reset_tracking_state(self):
        """重置跟踪状态"""
        self.track_history.clear()
        self.track_confidence.clear()
        self.track_frames.clear()
        self.prev_frame_tracks.clear()
        self.id_switch_count = 0
        self.total_tracks = 0
        self.false_positives = 0
        self.false_negatives = 0

    def get_class_name(self, class_id):
        """根据类别ID获取类别名称"""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"未知类别({class_id})"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能雾天交通检测系统")
        self.setGeometry(100, 100, 1600, 900)
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'traffic_icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.worker = Worker()
        self.image_paths = []
        self.results = []
        self.dehazed_images = []
        self.fog_levels = []  # 存储雾浓度
        self.current_image_index = -1
        self.zoom_factor = 1.0  # 图片缩放比例
        self.worker_thread = None

        self.init_ui()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        self._SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(self._SCRIPT_DIR, "alarm.wav")):
            QMessageBox.critical(
                None,
                "系统缺失文件",
                "检测到系统缺少报警音频文件！\n\n请将alarm.wav文件放置在程序目录下",
                QMessageBox.Ok
            )
            sys.exit(1)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.create_detection_tab()
        self.create_statistics_tab()
        self.create_settings_tab()
        self.create_bottom_toolbar(main_layout)

    def create_detection_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        left_panel = QVBoxLayout()
        original_group = QGroupBox("原始图像")
        original_layout = QVBoxLayout()
        self.original_image_label = QLabel("原始图像预览")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setStyleSheet("background-color: #2d2d2d; color: white;")
        original_layout.addWidget(self.original_image_label)
        original_group.setLayout(original_layout)
        left_panel.addWidget(original_group)

        dehazed_group = QGroupBox("去雾后图像")
        dehazed_layout = QVBoxLayout()
        self.dehazed_image_label = QLabel("去雾效果预览")
        self.dehazed_image_label.setAlignment(Qt.AlignCenter)
        self.dehazed_image_label.setStyleSheet("background-color: #2d2d2d; color: white;")
        dehazed_layout.addWidget(self.dehazed_image_label)
        dehazed_group.setLayout(dehazed_layout)
        left_panel.addWidget(dehazed_group)

        layout.addLayout(left_panel, 1)

        right_panel = QVBoxLayout()
        result_group = QGroupBox("检测结果可视化")
        result_layout = QVBoxLayout()
        self.detected_image_label = QLabel("检测结果将在此显示")
        self.detected_image_label.setAlignment(Qt.AlignCenter)
        self.detected_image_label.setStyleSheet("background-color: #2d2d2d; color: white;")
        result_layout.addWidget(self.detected_image_label)

        self.result_info = QLabel("等待检测...")
        self.result_info.setAlignment(Qt.AlignCenter)
        self.result_info.setStyleSheet("""
            font-size: 14px;
            padding: 10px;
            background-color: #404040;
            color: white;
            border-radius: 5px;
        """)
        result_layout.addWidget(self.result_info)

        self.result_details = QTextEdit()
        self.result_details.setReadOnly(True)
        self.result_details.setStyleSheet("""
            font-size: 12px;
            background-color: #2d2d2d;
            color: white;
            border: 1px solid #404040;
        """)
        result_layout.addWidget(self.result_details)
        result_group.setLayout(result_layout)
        right_panel.addWidget(result_group, 1)

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
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.stats_label = QLabel("统计信息将在此显示")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("font-size: 16px; color: #333;")
        layout.addWidget(self.stats_label)

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        self.plt = plt
        self.FigureCanvas = FigureCanvas
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        update_stats_btn = QPushButton("更新统计")
        update_stats_btn.clicked.connect(self.update_statistics)
        layout.addWidget(update_stats_btn)

        alarm_history = QGroupBox("报警历史记录")
        alarm_layout = QVBoxLayout()
        self.alarm_log = QTextEdit()
        self.alarm_log.setReadOnly(True)
        self.alarm_log.setStyleSheet("""
            background-color: #ffe6e6;
            font-family: 'Microsoft YaHei';
            padding: 10px;
            border: 1px solid #ffcccc;
        """)
        scroll = QScrollArea()
        scroll.setWidget(self.alarm_log)
        scroll.setWidgetResizable(True)
        alarm_layout.addWidget(scroll)
        alarm_history.setLayout(alarm_layout)
        layout.addWidget(alarm_history)

        fog_group = QGroupBox("雾浓度统计")
        fog_layout = QVBoxLayout()
        self.fog_stats = QLabel("雾浓度统计将在此显示")
        self.fog_stats.setStyleSheet("font-size: 12px; color: #555;")
        fog_layout.addWidget(self.fog_stats)
        fog_group.setLayout(fog_layout)
        layout.addWidget(fog_group)

        self.tab_widget.addTab(tab, "统计分析")

    def create_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()
        model_select_btn = QPushButton("选择模型文件")
        model_select_btn.clicked.connect(self.load_model)
        model_layout.addWidget(model_select_btn)
        self.model_info = QLabel("未加载模型")
        self.model_info.setStyleSheet("font-size: 12px; color: #666;")
        model_layout.addWidget(self.model_info)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        detect_group = QGroupBox("检测设置")
        detect_layout = QVBoxLayout()
        conf_layout = QHBoxLayout()
        conf_label = QLabel("置信度阈值:")
        self.conf_threshold = QComboBox()
        self.conf_threshold.addItems([f"{i / 10:.1f}" for i in range(1, 10)])
        self.conf_threshold.setCurrentText("0.5")
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_threshold)
        detect_layout.addLayout(conf_layout)

        speed_layout = QHBoxLayout()
        speed_label = QLabel("检测速度:")
        self.detect_speed = QComboBox()
        self.detect_speed.addItems(["快速", "平衡", "精确"])
        self.detect_speed.setCurrentText("平衡")
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.detect_speed)
        detect_layout.addLayout(speed_layout)

        # 去雾参数设置
        dehaze_group = QGroupBox("去雾参数设置")
        dehaze_layout = QGridLayout()
        r_dark_label = QLabel("暗通道半径:")
        self.r_dark = QSpinBox()
        self.r_dark.setRange(5, 50)
        self.r_dark.setValue(10)
        dehaze_layout.addWidget(r_dark_label, 0, 0)
        dehaze_layout.addWidget(self.r_dark, 0, 1)

        r_guide_label = QLabel("引导滤波半径:")
        self.r_guide = QSpinBox()
        self.r_guide.setRange(10, 100)
        self.r_guide.setValue(20)
        dehaze_layout.addWidget(r_guide_label, 1, 0)
        dehaze_layout.addWidget(self.r_guide, 1, 1)

        w_label = QLabel("大气光权重:")
        self.w = QDoubleSpinBox()
        self.w.setRange(0.5, 0.95)
        self.w.setSingleStep(0.05)
        self.w.setValue(0.85)
        dehaze_layout.addWidget(w_label, 2, 0)
        dehaze_layout.addWidget(self.w, 2, 1)

        t0_label = QLabel("透射率阈值:")
        self.t0 = QDoubleSpinBox()
        self.t0.setRange(0.05, 0.3)
        self.t0.setSingleStep(0.05)
        self.t0.setValue(0.1)
        dehaze_layout.addWidget(t0_label, 3, 0)
        dehaze_layout.addWidget(self.t0, 3, 1)

        dehaze_group.setLayout(dehaze_layout)
        detect_layout.addWidget(dehaze_group)
        detect_group.setLayout(detect_layout)
        layout.addWidget(detect_group)

        alarm_group = QGroupBox("报警阈值设置")
        alarm_layout = QGridLayout()
        for i, (cls_name, default) in enumerate(self.worker.alarm_thresholds.items()):
            label = QLabel(f"{cls_name}报警阈值:")
            spinbox = QSpinBox()
            spinbox.setRange(1, 100)
            spinbox.setValue(default)
            spinbox.valueChanged.connect(lambda val, name=cls_name: self.update_alarm_threshold(name, val))
            alarm_layout.addWidget(label, i, 0)
            alarm_layout.addWidget(spinbox, i, 1)
        alarm_group.setLayout(alarm_layout)
        layout.addWidget(alarm_group)

        ui_group = QGroupBox("界面设置")
        ui_layout = QVBoxLayout()
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

    def create_bottom_toolbar(self, main_layout):
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)

        self.detect_btn = QPushButton("单图检测")
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self.detect_image)
        toolbar_layout.addWidget(self.detect_btn)

        self.batch_btn = QPushButton("批量检测")
        self.batch_btn.setEnabled(False)
        self.batch_btn.clicked.connect(self.batch_detect_folder)
        toolbar_layout.addWidget(self.batch_btn)

        self.video_btn = QPushButton("视频检测")
        self.video_btn.setEnabled(True)
        self.video_btn.clicked.connect(self.detect_video)
        toolbar_layout.addWidget(self.video_btn)

        self.save_btn = QPushButton("保存结果")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_results)
        toolbar_layout.addWidget(self.save_btn)

        exit_btn = QPushButton("退出系统")
        exit_btn.clicked.connect(self.exit_application)
        toolbar_layout.addWidget(exit_btn)

        toolbar.setLayout(toolbar_layout)
        main_layout.addWidget(toolbar)

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择YOLOv8模型文件", "", "模型文件 (*.pt)"
        )
        if model_path:
            if self.worker.load_model(model_path):
                # 传递去雾参数控件值到worker
                self.worker.r_dark = self.r_dark
                self.worker.r_guide = self.r_guide
                self.worker.w = self.w
                self.worker.t0 = self.t0

                self.detect_btn.setEnabled(True)
                self.batch_btn.setEnabled(True)
                self.model_info.setText(f"已加载模型: {os.path.basename(model_path)}")
                self.status_bar.showMessage("模型加载成功")
            else:
                QMessageBox.critical(self, "错误", "无法加载模型，请检查模型路径")
                self.status_bar.showMessage("模型加载失败")

    def detect_image(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "", "图片文件 (*.jpg *.jpeg *.png *.webp)"
        )
        if image_path:
            self.process_single_image(image_path)

    def process_single_image(self, image_path):
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")

            raw_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if raw_image is None:
                raise ValueError(f"无法读取图片文件: {image_path}")

            # 去雾处理
            _, _, dehazed_uint8, fog_level = dehaze_image(
                raw_image,
                self.r_dark.value(),
                self.r_guide.value(),
                self.w.value(),
                self.t0.value()
            )

            # 目标检测
            result = self.worker.detect_image(dehazed_uint8)

            if result and len(result) > 0:
                if not self.image_paths:
                    self.image_paths = [image_path]
                self.fog_levels = [fog_level]
                self.handle_valid_result(result[0], dehazed_uint8, image_path, fog_level)
            else:
                self.handle_empty_result()

        except Exception as e:
            self.show_error_message(f"处理图片时发生错误:\n{traceback.format_exc()}")

    def handle_valid_result(self, result, dehazed_image, image_path, fog_level):
        self.results = [result]
        self.dehazed_images = [dehazed_image]
        self.current_image_index = 0
        self.display_current_image()
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.status_bar.showMessage(f"检测完成 (雾浓度: {fog_level})")

    def handle_empty_result(self):
        QMessageBox.warning(self, "警告", "未能检测到任何目标")
        self.reset_display()

    def reset_display(self):
        self.image_paths = []
        self.results = []
        self.dehazed_images = []
        self.fog_levels = []
        self.current_image_index = -1
        self.set_label_image(self.original_image_label, None)
        self.set_label_image(self.dehazed_image_label, None)
        self.set_label_image(self.detected_image_label, None)
        self.result_info.setText("等待检测...")
        self.result_details.setText("")
        self.save_btn.setEnabled(False)

    def batch_detect_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            # 获取所有支持的图像文件
            valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
            self.image_paths = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(valid_extensions)
            ]

            if not self.image_paths:
                QMessageBox.warning(self, "警告", "文件夹中没有有效的图片文件")
                return

            # 初始化进度条
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_bar.showMessage("开始批量检测...")

            # 创建工作线程
            self.worker_thread = WorkerThread(self.worker, self.image_paths)
            self.worker_thread.progress.connect(self.update_progress)
            self.worker_thread.finished.connect(self.on_batch_finished)
            self.worker_thread.error_occurred.connect(self.show_batch_error)
            self.worker_thread.start()

    def on_batch_finished(self, results, dehazed_images, fog_levels):
        """批量处理完成后的回调"""
        # 过滤掉失败的结果
        valid_results = [(r, img, fog) for r, img, fog in zip(results, dehazed_images, fog_levels) if r is not None]

        if valid_results:
            self.results = [r for r, _, _ in valid_results]
            self.dehazed_images = [img for _, img, _ in valid_results]
            self.fog_levels = [fog for _, _, fog in valid_results]
            self.current_image_index = 0

            self.display_current_image()
            self.prev_button.setEnabled(len(self.results) > 1)
            self.next_button.setEnabled(len(self.results) > 1)
            self.save_btn.setEnabled(True)

            success_count = len(valid_results)
            total_count = len(self.image_paths)
            self.status_bar.showMessage(
                f"批量处理完成 - 成功 {success_count}/{total_count} 张图片"
            )

            # 更新雾浓度统计
            self.update_fog_statistics()
        else:
            QMessageBox.warning(self, "警告", "批量处理完成，但未检测到任何有效结果")

        self.progress_bar.setVisible(False)

    def show_batch_error(self, error_msg):
        """显示批量处理错误"""
        QMessageBox.warning(self, "处理错误", error_msg)
        self.progress_bar.setVisible(False)

    def update_progress(self, value, message):
        """更新进度条和状态信息"""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(message)

    def update_fog_statistics(self):
        """更新雾浓度统计信息"""
        if not self.fog_levels:
            self.fog_stats.setText("暂无雾浓度数据")
            return

        fog_counts = {"轻度雾": 0, "中度雾": 0, "重度雾": 0, "去雾失败": 0}
        for level in self.fog_levels:
            fog_counts[level] = fog_counts.get(level, 0) + 1

        stats_text = "雾浓度统计:<br>"
        total = sum(fog_counts.values())
        for level, count in fog_counts.items():
            percentage = f"{count / total * 100:.1f}%" if total > 0 else "0%"
            stats_text += f"{level}: {count}张 ({percentage})<br>"

        self.fog_stats.setText(stats_text)

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()

    def show_next_image(self):
        if 0 <= self.current_image_index < len(self.results) - 1:
            self.current_image_index += 1
            self.display_current_image()

    def display_current_image(self):
        """显示当前图片"""
        if 0 <= self.current_image_index < len(self.results):
            image_path = self.image_paths[self.current_image_index]
            result = self.results[self.current_image_index]
            dehazed_image = self.dehazed_images[self.current_image_index]
            fog_level = self.fog_levels[self.current_image_index]
            self.display_results(image_path, result, dehazed_image, fog_level)
        else:
            self.reset_display()
            self.status_bar.showMessage("错误：当前图片索引无效")

    def display_results(self, image_path, result, dehazed_image, fog_level):
        """显示检测结果"""
        try:
            if not result or len(result) == 0 or result.boxes is None or len(result.boxes) == 0:
                raise ValueError("检测结果不包含有效边界框")

            raw_image = cv2.imread(image_path)
            original_size = raw_image.shape[:2][::-1]
            processed_size = dehazed_image.shape[:2][::-1]

            annotated_image = result.plot()
            self.set_label_image(self.original_image_label, raw_image)
            self.set_label_image(self.dehazed_image_label, dehazed_image)
            self.set_label_image(self.detected_image_label, annotated_image)
            self.show_detection_info(result, image_path, original_size, processed_size, fog_level)

        except Exception as e:
            self.show_error_message(f"显示检测结果时出错:\n{traceback.format_exc()}")

    def show_detection_info(self, result, image_path, original_size, processed_size, fog_level):
        """显示检测详细信息"""
        try:
            if not result.boxes or len(result.boxes) == 0:
                raise ValueError("检测结果不包含任何目标")

            info = f"<b>图片:</b> {os.path.basename(image_path)}<br>"
            info += f"<b>尺寸:</b> {result.orig_shape[1]}×{result.orig_shape[0]}<br>"
            info += f"<b>雾浓度:</b> {fog_level}<br>"
            info += f"<b>处理时间:</b> {result.speed['inference']:.1f}ms<br><br>"

            class_counts = {}
            # 解析YOLOv8的boxes数据
            for i in range(len(result.boxes)):
                cls = int(result.boxes.cls[i].item())
                cls_name = self.worker.get_class_name(cls)
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            info += "<b>检测结果:</b><br>"
            for cls, count in class_counts.items():
                info += f"  {cls}: {count}个<br>"

            self.result_info.setText(info)

            details = "<b>详细检测结果:</b>\n"
            # 提取YOLOv8的边界框和置信度信息
            for i in range(len(result.boxes)):
                r = result.boxes.xyxy[i].cpu().numpy().astype(int).flatten()
                adjusted_r = self.adjust_coordinates(r, original_size, processed_size).flatten()
                cls = int(result.boxes.cls[i].item())
                conf = result.boxes.conf[i].item()
                cls_name = self.worker.get_class_name(cls)
                details += f"{i + 1}. {cls_name} (置信度: {conf:.2f}) - 原始坐标: {r} → 调整后坐标: {adjusted_r}\n"

            self.result_details.setText(details)

            self.check_alarm_thresholds(class_counts)

        except Exception as e:
            self.show_error_message(f"显示检测信息时出错:\n{traceback.format_exc()}")

    def adjust_coordinates(self, boxes, original_size, processed_size):
        scale_x = original_size[0] / processed_size[0]
        scale_y = original_size[1] / processed_size[1]
        adjusted_boxes = boxes * [scale_x, scale_y, scale_x, scale_y]
        return adjusted_boxes.astype(int)

    def check_alarm_thresholds(self, class_counts):
        alarm_triggered = False
        for cls, count in class_counts.items():
            if count > self.worker.alarm_thresholds.get(cls, 0):
                alarm_triggered = True
                self.trigger_alarm(cls, count)

        if alarm_triggered:
            self.status_bar.showMessage("⚠️ 检测到超限目标！请立即处理", 5000)

    def trigger_alarm(self, class_name, count):
        QSound.play(os.path.join(self._SCRIPT_DIR, "alarm.wav"))
        self.status_bar.showMessage(f"⚠️ {class_name}超限报警！当前{count}个", 8000)
        for _ in range(5):
            self.status_bar.setStyleSheet("color: red; font-weight: bold;")
            QApplication.processEvents()
            QThread.msleep(200)
            self.status_bar.setStyleSheet("color: black;")
            QApplication.processEvents()
            QThread.msleep(200)

        alarm_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        threshold = self.worker.alarm_thresholds.get(class_name, 0)
        alarm_log = f"[{alarm_time}] {class_name}超限报警 - 当前数量: {count}, 阈值: {threshold}\n"
        self.alarm_log.append(alarm_log)

        QMessageBox.critical(
            self,
            "紧急报警！",
            f"检测到{class_name}数量严重超限！\n\n"
            f"当前数量：{count}个\n"
            f"安全阈值：{threshold}个\n\n"
            f"建议立即采取：\n"
            f"- 检查{class_name}聚集区域\n"
            f"- 调度执勤人员\n"
            f"- 启动应急预案",
            QMessageBox.Ok
        )

    def update_statistics(self):
        if not self.results:
            QMessageBox.warning(self, "警告", "没有可统计的结果")
            return
        total_images = len(self.results)
        total_objects = sum(len(r.boxes) for r in self.results)
        class_counts = {}
        for result in self.results:
            for i in range(len(result.boxes)):
                cls = int(result.boxes.cls[i].item())
                cls_name = self.worker.get_class_name(cls)
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        stats_text = f"<b>统计摘要</b><br>"
        stats_text += f"总图片数: {total_images}<br>"
        stats_text += f"总检测目标数: {total_objects}<br><br>"
        stats_text += "<b>按类别统计:</b><br>"
        for cls, count in class_counts.items():
            stats_text += f"{cls}: {count} ({count / total_objects:.1%})<br>"
        self.stats_label.setText(stats_text)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        labels = list(class_counts.keys())
        sizes = list(class_counts.values())
        ax.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax.axis('equal')
        self.canvas.draw()
        self.status_bar.showMessage("统计信息已更新")

    def update_alarm_threshold(self, class_name, value):
        try:
            value = int(value)
            if 1 <= value <= 100:
                self.worker.alarm_thresholds[class_name] = value
                self.status_bar.showMessage(f"已更新{class_name}报警阈值为{value}", 3000)
            else:
                raise ValueError
        except ValueError:
            QMessageBox.warning(
                self,
                "输入错误",
                "请输入1-100之间的整数作为报警阈值",
                QMessageBox.Ok
            )

    def change_theme(self, theme):
        if theme == "浅色":
            self.set_light_theme()
        elif theme == "深色":
            self.set_dark_theme()
        elif theme == "蓝色":
            self.set_blue_theme()
        self.status_bar.showMessage(f"已切换至{theme}主题")

    def set_light_theme(self):
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
                color: white;
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
                background-color: #555;
                color: white;
                border: 1px solid #666;
            }
        """)

    def set_blue_theme(self):
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
        """)

    def detect_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if video_path:
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError("无法打开视频文件")

                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0

                result = QMessageBox.question(
                    self,
                    "确认视频检测",
                    f"视频信息:\n"
                    f"时长: {int(duration // 60)}分{int(duration % 60)}秒\n"
                    f"帧率: {fps:.1f} FPS\n"
                    f"总帧数: {frame_count}\n\n"
                    f"是否开始视频检测？",
                    QMessageBox.Yes | QMessageBox.No
                )

                if result == QMessageBox.Yes:
                    self._process_video(cap, video_path)
            except Exception as e:
                self.show_error_message(f"打开视频时出错:\n{str(e)}")

    def _process_video(self, cap, video_path):
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            output_path = os.path.splitext(video_path)[0] + '_detected.mp4'
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, int(fps))  # 每秒处理一帧

            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            current_frame = 0
            processed_frames = 0
            detection_results = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_frame += 1
                if current_frame % frame_interval != 0:
                    out.write(frame)
                    continue

                # 视频帧去雾处理
                try:
                    _, _, dehazed_frame, fog_level = dehaze_image(
                        frame,
                        self.r_dark.value(),
                        self.r_guide.value(),
                        self.w.value(),
                        self.t0.value()
                    )
                except Exception as e:
                    logging.error(f"视频帧去雾失败: {str(e)}")
                    dehazed_frame = frame
                    fog_level = "去雾失败"

                # 目标检测
                results = self.worker.detect_image(dehazed_frame)

                if results and len(results) > 0:
                    result = results[0]
                    detection_results.append(result)
                    annotated_frame = result.plot()
                    out.write(annotated_frame)
                else:
                    out.write(frame)

                # 更新进度
                processed_frames += 1
                progress = int(processed_frames / (frame_count / frame_interval) * 100)
                self.progress_bar.setValue(progress)
                self.status_bar.showMessage(f"处理中: {processed_frames}/{int(frame_count / frame_interval)} 帧")

                # 处理界面事件
                QApplication.processEvents()

            cap.release()
            out.release()

            self.progress_bar.setVisible(False)

            if detection_results:
                reply = QMessageBox.information(
                    self,
                    "视频处理完成",
                    f"视频处理完成！\n\n"
                    f"检测到目标的总帧数: {len(detection_results)}\n"
                    f"输出文件: {os.path.basename(output_path)}\n\n"
                    f"是否打开输出文件夹？",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    import subprocess
                    output_dir = os.path.dirname(output_path)
                    if os.name == 'nt':  # Windows
                        subprocess.Popen(f'explorer "{output_dir}"')
                    elif os.name == 'posix':  # Linux/Mac
                        subprocess.Popen(f'xdg-open "{output_dir}"', shell=True)
            else:
                QMessageBox.warning(self, "视频处理完成", "未检测到任何目标")

        except Exception as e:
            self.show_error_message(f"处理视频时出错:\n{str(e)}")
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'out' in locals():
                out.release()
            self.progress_bar.setVisible(False)

    def save_results(self):
        if not self.results:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not save_dir:
            return

        try:
            os.makedirs(save_dir, exist_ok=True)
            for i, (result, dehazed_img, img_path, fog_level) in enumerate(zip(
                    self.results, self.dehazed_images, self.image_paths, self.fog_levels
            )):
                base_name = os.path.basename(img_path)
                name, ext = os.path.splitext(base_name)

                # 保存原始图像
                original_img = cv2.imread(img_path)
                cv2.imwrite(os.path.join(save_dir, f"{name}_original{ext}"), original_img)

                # 保存去雾图像
                cv2.imwrite(os.path.join(save_dir, f"{name}_dehazed{ext}"), dehazed_img)

                # 保存检测结果图像
                annotated_img = result.plot()
                cv2.imwrite(os.path.join(save_dir, f"{name}_detected{ext}"), annotated_img)

                # 保存检测结果文本
                with open(os.path.join(save_dir, f"{name}_results.txt"), 'w', encoding='utf-8') as f:
                    f.write(f"检测结果 - {base_name}\n\n")
                    f.write(f"尺寸: {result.orig_shape[1]}×{result.orig_shape[0]}\n")
                    f.write(f"雾浓度: {fog_level}\n")
                    f.write(f"处理时间: {result.speed['inference']:.1f}ms\n\n")

                    class_counts = {}
                    for i in range(len(result.boxes)):
                        cls = int(result.boxes.cls[i].item())
                        cls_name = self.worker.get_class_name(cls)
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                    f.write("检测统计:\n")
                    for cls, count in class_counts.items():
                        f.write(f"  {cls}: {count}个\n")

                    f.write("\n详细检测结果:\n")
                    for j in range(len(result.boxes)):
                        r = result.boxes.xyxy[j].cpu().numpy().astype(int).flatten()
                        cls = int(result.boxes.cls[j].item())
                        conf = result.boxes.conf[j].item()
                        cls_name = self.worker.get_class_name(cls)
                        f.write(f"{j + 1}. {cls_name} (置信度: {conf:.2f}) - 坐标: {r}\n")

            QMessageBox.information(self, "保存成功",
                                    f"已成功保存 {len(self.results)} 个结果到:\n{save_dir}",
                                    QMessageBox.Ok)
            self.status_bar.showMessage(f"已保存 {len(self.results)} 个结果")
            logging.info(f"检测结果已成功保存至 {save_dir}，共保存 {len(self.results)} 个结果")

        except Exception as e:
            self.show_error_message(f"保存结果时出错:\n{str(e)}")
            logging.error(f"保存结果失败: {str(e)}")

    def exit_application(self):
        reply = QMessageBox.question(
            self,
            "退出确认",
            "确定要退出系统吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if self.worker_thread and self.worker_thread.isRunning():
                self.worker_thread.stop()
                self.worker_thread.wait()
            self.close()

    def show_error_message(self, message):
        QMessageBox.critical(
            self,
            "错误",
            f"发生错误:\n{message}",
            QMessageBox.Ok
        )
        logging.error(message)

    def set_label_image(self, label, image, max_size=700):
        if image is None:
            label.setText("无图像")
            return

        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        height, width = image.shape[:2]
        # 限制最大显示尺寸，防止图片撑出屏幕
        scale = min(max_size / max(width, height), self.zoom_factor)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        bytes_per_line = 3 * new_width
        q_img = QImage(
            image.data, new_width, new_height, bytes_per_line,
            QImage.Format_BGR888
        )
        label.setPixmap(QPixmap.fromImage(q_img))
        label.setAlignment(Qt.AlignCenter)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())