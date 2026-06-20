import os
import sys

from PyQt5.QtWidgets import QApplication
from flask import Flask, request, jsonify

# 这里可以导入你的 PyQt5 应用代码
# 假设你的 PyQt5 应用在 desktop_app.py 中
from integrated_app import MainWindow

app = Flask(__name__)
app.app = QApplication(sys.argv)
app.main_window = MainWindow()

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # 获取上传的图片
        file = request.files['frame']
        image_path = 'temp_image.jpg'
        file.save(image_path)

        # 调用 PyQt5 应用的单图检测功能
        app.main_window.process_single_image(image_path)

        # 这里可以根据实际情况返回处理结果
        result = {
            'status': 'success',
            'message': '图片处理成功'
        }

        # 删除临时图片
        if os.path.exists(image_path):
            os.remove(image_path)

        return jsonify(result)
    except Exception as e:
        error = {
            'status': 'error',
            'message': str(e)
        }
        return jsonify(error)

if __name__ == '__main__':
    app.run(debug=True)