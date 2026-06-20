import tkinter as tk
from tkinter import filedialog
import cv2
import torch
from ultralytics import YOLO


class FogDetectionGUI:
    def __init__(self, master):
        self.master = master
        master.title("雾天行人车辆检测系统")       #初始化 GUI 窗口，并设置窗口标题为“雾天行人车辆检测系统

        # 创建按钮
        self.load_file_button = tk.Button(master, text="选择文件", command=self.load_file)
        self.load_file_button.pack()

        self.detect_button = tk.Button(master, text="开始检测", command=self.detect_objects, state=tk.DISABLED)
        self.detect_button.pack()

        # 用于显示检测结果的标签
        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.file_path = None
        self.model = YOLO("yolov8n.pt")

    def load_file(self):
        file_path = filedialog.askopenfilename(title="选择图像或视频文件",
                                               filetypes=(("图像文件", "*.jpg;*.png;*.jpeg"),
                                                          ("视频文件", "*.mp4;*.avi")))
        if file_path:
            self.file_path = file_path
            self.detect_button.config(state=tk.NORMAL)
            self.result_label.config(text="已选择文件：{}".format(file_path))
        else:
            self.result_label.config(text="未选择文件")

    def detect_objects(self):
        if self.file_path.endswith(('.jpg', '.png', '.jpeg')):
            # 处理图像
            img = cv2.imread(self.file_path)
            results = self.model(img)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("检测结果", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif self.file_path.endswith(('.mp4', '.avi')):
            # 处理视频
            cap = cv2.VideoCapture(self.file_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.model(frame)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("检测结果", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

#创建 Tkinter 主窗口实例，并启动事件循环，使 GUI 界面保持运行
root = tk.Tk()
my_gui = FogDetectionGUI(root)
root.mainloop()