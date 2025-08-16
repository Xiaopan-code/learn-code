import glob
import os
import shutil
import time
import warnings
import cv2
from PIL import ImageFont
from PyQt5 import QtWidgets,QtCore
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QAbstractItemView, QMainWindow, QTabWidget, QFileDialog, QMessageBox, QTableWidgetItem, \
    QApplication
from FaceRecognition_UI import Ui_MainWindow
from PyQt5 import QtGui
import numpy as np
import csv
import dlib
from PIL import Image, ImageDraw
import datetime
import pandas as pd


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')


class Face_MainWindow(Ui_MainWindow):
    def __init__(self, MainWindow):
        super().__init__()

        # 新增：存储人脸置信度
        self.last_face_confidence = []  # 上一帧人脸识别置信度
        self.current_face_confidence = []  # 当前帧人脸识别置信度

        # 优化参数
        self.reclassify_interval = 30  # 降低重新分类间隔，提高响应速度

        self.ui = Ui_MainWindow()
        self.ui.setupUi(MainWindow)

        self.main_window = MainWindow


        self.path_face_dir = r'FaceRec/face_data'

        self.fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)

        self.timer_camera = QtCore.QTimer()
        self.timer_camera_load = QtCore.QTimer()
        self.timer_video = QtCore.QTimer()

        self.CAM_NUM = 0

        self.cap = cv2.VideoCapture(0)

        self.setupUi(MainWindow)
        self.retranslateUi(MainWindow)

        self.resetUi()
        self.slot_init()

        self.path = os.getcwd()
        self.current_image = None
        self.video_path = ''
        self.flag_timer = 0

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('D:\code\Image-Recognition\data\data_dlib\shape_predictor_68_face_landmarks.dat')
        self.face_reco_model = dlib.face_recognition_model_v1('D:\code\Image-Recognition\data\data_dlib\dlib_face_recognition_resnet_model_v1.dat')

        self.current_face = None

        self.count_face = 0
        self.col_row = []

        self.face_feature_exist = []  # 存储已录入人脸的128维特征向量
        self.face_name_exist = []  # 存储对应人脸名称
        self.last_centroid = []  # 上一帧人脸质心坐标
        self.current_centroid = []  # 当前帧人脸质心坐标
        self.last_face_name = []  # 上一帧识别的人脸名称
        self.current_face_name = []  # 当前帧识别的人脸名称
        self.last_face_cnt = 0  # 上一帧人脸数量
        self.current_face_cnt = 0  # 当前帧人脸数量
        self.current_face_position = []  # 当前帧人脸位置坐标
        self.current_face_feature = []  # 当前帧人脸特征向量

        # 识别控制与优化变量
        self.reclassify_cnt = 0  # 重新分类计数器
        self.reclassify_interval = 60  # 重新分类间隔(帧数)
        self.last_current_distance = 0  # 帧间人脸移动距离
        self.current_face_distance = []  # 当前人脸与库中人脸的距离

        # 统计与状态变量
        self.count = 0  # 识别操作计数
        self.exist_flag = False
        self.main_window.closeEvent = self.close_event_handler

    def close_event_handler(self, event):
        if self.timer_camera and self.timer_camera.isActive():
            self.timer_camera.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.clean_up_table_widgets()
        event.accept()

    def clean_up_table_widgets(self):
        try:
            if self.tableWidget_rec:
                self.tableWidget_rec.disconnect()
            if self.tableWidget_mana:
                self.tableWidget_mana.disconnect()
        except TypeError:
            pass  # 如果已经没有连接，会抛出 TypeError
        self.tableWidget_rec = None
        self.tableWidget_mana = None

    def ini_value(self):
        self.face_feature_exist = []  # 存储已录入人脸的128维特征向量
        self.face_name_exist = []  # 存储对应人脸名称
        self.last_centroid = []  # 上一帧人脸质心坐标
        self.current_centroid = []  # 当前帧人脸质心坐标
        self.last_face_name = []  # 上一帧识别的人脸名称
        self.current_face_name = []  # 当前帧识别的人脸名称
        self.last_face_cnt = 0  # 上一帧人脸数量
        self.current_face_cnt = 0  # 当前帧人脸数量
        self.current_face_position = []  # 当前帧人脸位置坐标
        self.current_face_feature = []  # 当前帧人脸特征向量

        # 识别控制与优化变量
        self.reclassify_cnt = 10  # 重新分类计数器
        self.reclassify_interval = 60  # 重新分类间隔(帧数)
        self.last_current_distance = 0  # 帧间人脸移动距离
        self.current_face_distance = []  # 当前人脸与库中人脸的距离

        # 统计与状态变量
        self.count = 0  # 识别操作计数
        self.exist_flag = 0

    def resetUi(self):
        if self.tableWidget_rec:
            self.tableWidget_rec.horizontalHeader().setVisible(True)
            self.tableWidget_rec.setColumnWidth(0, 80)
            self.tableWidget_rec.setColumnWidth(1, 200)
            self.tableWidget_rec.setColumnWidth(2, 150)
            self.tableWidget_rec.setColumnWidth(3, 200)
            self.tableWidget_rec.setColumnWidth(4, 120)
        if self.tableWidget_mana:
            self.tableWidget_mana.horizontalHeader().setVisible(True)
            self.tableWidget_mana.setColumnWidth(0, 80)
            self.tableWidget_mana.setColumnWidth(1, 350)
            self.tableWidget_mana.setColumnWidth(2, 150)
            self.tableWidget_mana.setColumnWidth(3, 200)
            self.tableWidget_mana.setColumnWidth(4, 120)
            self.tableWidget_mana.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.tabWidget.setCurrentIndex(0)
        self.tabWidget.setTabVisible(1, False)
        self.tabWidget.setTabVisible(2, False)
        self.toolButton_get_pic.setEnabled(False)
        self.toolButton_load_pic.setEnabled(False)
        self.toolButton_file_2.setEnabled(False)
        self.toolButton_camera_load.setEnabled(False)

        self.gif_movie()

    def gif_movie(self):
        movie = QMovie(':/newPrefix/images_test/face_rec.gif')
        self.label_display.setMovie(movie)
        self.label_display.setScaledContents(True)
        movie.start()

    def slot_init(self):
        self.toolButton_run_load.clicked.connect(self.change_size_load)
        self.toolButton_run_rec.clicked.connect(self.change_size_rec)
        self.toolButton_run_manage.clicked.connect(self.change_size_mana)
        self.toolButton_file.clicked.connect(self.choose_rec_img)
        self.toolButton_video.clicked.connect(self.button_open_video_click)
        self.toolButton_camera.clicked.connect(self.button_open_camera_click)
        self.toolButton_camera_load.clicked.connect(self.button_open_camera_load)
        self.toolButton_get_pic.clicked.connect(self.get_img_doing)
        self.toolButton_new_folder.clicked.connect(self.new_face_doing)
        self.toolButton_file_2.clicked.connect(self.choose_file)
        self.toolButton_load_pic.clicked.connect(self.load_img_doing)
        # 新增摄像头/视频定时器信号连接
        self.timer_camera_load.timeout.connect(self.show_camera_load)
        self.timer_camera.timeout.connect(self.show_camera)  # 摄像头定时器超时信号
        self.timer_video.timeout.connect(self.show_video)  # 视频定时器超时信号
        # 新增开始/停止按钮信号连接
        self.toolButton_runing.clicked.connect(self.run_rec)  # 开始运行按钮

        # 1. 连接【更新】按钮点击信号到do_update_face()方法
        self.toolButton_mana_update.clicked.connect(self.do_update_face)

        # 2. 连接表格单元格点击信号到table_review()方法
        self.tableWidget_mana.cellClicked.connect(self.table_review)

        # 3. 连接【删除】按钮点击信号到delete_doing()方法
        self.toolButton_mana_delete.clicked.connect(self.delete_doing)

    def change_size_load(self):
        self.toolButton_run_load.setGeometry(26, 218, 280, 70)
        self.toolButton_run_rec.setGeometry(66, 324, 199, 49)
        self.toolButton_run_manage.setGeometry(66, 414, 199, 49)

        self.tabWidget.setCurrentIndex(1)
        self.tabWidget.setTabVisible(0, False)
        self.tabWidget.setTabVisible(1, True)
        self.tabWidget.setTabVisible(2, False)

        self.flag_timer = 0

        if self.timer_camera and self.timer_camera.isActive():
            self.timer_camera.stop()

        if self.timer_video and self.timer_video.isActive():
            self.timer_video.stop()

        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

        # if hasattr(self, 'cap_video') and self.cap_video:
        #     self.cap_video.release()

        QtWidgets.QApplication.processEvents()

        self.label_display.clear()
        self.gif_movie()
        self.label_pic_newface.clear()
        self.label_pic_org.clear()

    def change_size_rec(self):
        self.toolButton_run_load.setGeometry(26, 218, 280, 70)
        self.toolButton_run_rec.setGeometry(66, 324, 199, 49)
        self.toolButton_run_manage.setGeometry(66, 414, 199, 49)

        self.tabWidget.setCurrentIndex(0)
        self.tabWidget.setTabVisible(0, True)
        self.tabWidget.setTabVisible(1, False)
        self.tabWidget.setTabVisible(2, False)

        if self.timer_camera_load and self.timer_camera_load.isActive():
            self.timer_camera_load.stop()

        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

        QtWidgets.QApplication.processEvents()

        self.label_display.clear()
        self.gif_movie()
        self.label_pic_newface.clear()

        self.lineEdit_face_name.setText("请在此输入人脸名")
        self.label_new_res.setText("等待新建人脸文件夹")
        self.label_loadface.setText("等待点击以录入人脸")

        self.toolButton_get_pic.setEnabled(False)
        self.toolButton_load_pic.setEnabled(False)

    def change_size_mana(self):
        self.toolButton_run_load.setGeometry(26, 218, 280, 70)
        self.toolButton_run_rec.setGeometry(66, 324, 199, 49)
        self.toolButton_run_manage.setGeometry(66, 414, 199, 49)

        self.tabWidget.setCurrentIndex(2)
        self.tabWidget.setTabVisible(0, False)
        self.tabWidget.setTabVisible(1, True)
        self.tabWidget.setTabVisible(2, True)

        self.update_face()

    @staticmethod
    def cv_imread(filePath):
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
        if len(cv_img.shape) > 2:
            if cv_img.shape[2] > 3:
                cv_img = cv_img[:, :, :3]
        return cv_img

    def choose_rec_img(self):
        self.flag_timer = ''

        if hasattr(self, 'timer_camera') and self.timer_camera.isActive():
            self.timer_camera.stop()
        if hasattr(self, 'timer_video') and self.timer_video.isActive():
            self.timer_video.stop()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        # if hasattr(self, 'cap_video') and self.cap_video.isOpened():
        #     self.cap_video.release()

        self.label_plate_result.setText("未知人脸")
        self.label_score_num.setText("0")
        self.label_score_dis.setText("0")

        self.textEdit_camera.setText("实时摄像已关闭")
        self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                           "border-color: rgb(0, 170, 255);\n"  
                                           "color: rgb(0, 170, 255);\n"
                                           "font: regular 12pt \"华为仿宋\";")
        self.textEdit_video.setText("实时视频已关闭")
        self.textEdit_video.setStyleSheet(self.textEdit_camera.styleSheet())

        self.label_display.clear()
        self.label_pic_newface.clear()
        self.gif_movie()

        file_path, _ = QFileDialog.getOpenFileName()

        if file_path:
            self.path = file_path
            self.flag_timer = 'image'
            self.textEdit_file.setText(f"已选中文件:{self.path}")

            image = self.cv_imread(self.path)
            if image is None:
                self.textEdit_file.setText("文件读取失败，请重新选择")
                self.gif_movie()
                return

            image = cv2.resize(image, (500, 500))

            if len(image.shape) != 3 or image.shape[2] != 3:
                self.textEdit_file.setText("请选择彩色图片文件")
                self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")
                self.gif_movie()
                return

            self.current_image = image.copy()
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

            h, w, ch = rgb_image.shape
            bytes_per_line = 3 * w
            q_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            q_pixmap = QtGui.QPixmap.fromImage(q_image)

            self.label_display.setPixmap(q_pixmap)
            self.label_display.setScaledContents(True)
            QtWidgets.QApplication.processEvents()

        else:
            self.path = None
            self.flag_timer = ''
            self.textEdit_file.setText("图片文件未选中")

    def button_open_video_click(self):
        if hasattr(self, 'timer_camera') and self.timer_camera.isActive():
            self.timer_camera.stop()
        if hasattr(self,'cap') and self.cap.isOpened():
            self.cap.release()
        # if hasattr(self,'cap_video') and self.cap_video.isOpened():
        #     self.cap_video.release()

        self.label_display.clear()
        self.gif_movie()
        self.label_pic_newface.clear()
        self.flag_timer = ''
        self.ini_value()

        if not hasattr(self,'timer_video') or not self.timer_video.isActive():
            video_path, _ = QFileDialog.getOpenFileName()

            if video_path:
                self.video_path = video_path
                self.flag_timer = 'video'
                self.textEdit_video.setText(f"已选中文件:{self.video_path}")
                QtWidgets.QApplication.processEvents()

                self.cap = cv2.VideoCapture(self.video_path)

                if not self.cap.isOpened():
                    if isinstance(self.video_path, int):
                        print(f"错误: 无法打开摄像头设备 {self.video_path}")
                        print("请确保摄像头已正确连接并且没有被其他应用程序占用")
                    else:
                        print(f"错误: 无法打开视频文件 {self.video_path}")
                        print("请检查文件路径是否正确以及文件是否已损坏")
                    self.textEdit_video.setText("视频打开失败")
                    return

            else:
                self.video_path = None
                self.textEdit_video.setText("视频文件未选中")
                self.textEdit_camera.setText("实时摄像已关闭")
                self.textEdit_file.setText("图片文件未选中")

                self.label_display.clear()
                self.gif_movie()

                QtWidgets.QApplication.processEvents()

                self.label_plate_result.setText("未知人脸")
                self.label_score_num.setText("0")
                self.label_score_dis.setText("0")

        if hasattr(self,'timer_video') and self.timer_video.isActive():
            self.timer_video.stop()

            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()

            self.textEdit_video.setText("视频文件未选中")
            self.textEdit_camera.setText("实时摄像已关闭")
            self.textEdit_file.setText("图片文件未选中")

            QtWidgets.QApplication.processEvents()

            self.label_plate_result.setText("未知人脸")
            self.label_score_num.setText("0")
            self.label_score_dis.setText("0")

    def button_open_camera_load(self):
        if hasattr(self,'timer_video') and self.timer_video.isActive():
            self.timer_video.stop()
        if hasattr(self,'timer_video') and self.timer_video.isActive():
            self.timer_video.stop()
        if hasattr(self,'cap') and self.cap.isOpened():
            self.cap.release()
        # if hasattr(self,'cap_video') and self.cap_video.isOpened():
        #     self.cap_video.release()

        self.label_display.clear()
        self.gif_movie()
        self.label_pic_newface.clear()
        self.label_pic_org.clear()


        if not hasattr(self,'timer_camera_load') or not self.timer_camera_load.isActive():
            self.cap = cv2.VideoCapture(0)
            camera_opened = self.cap.isOpened()

            if not camera_opened:
                QMessageBox.warning(self.main_window, "摄像头错误", "无法打开摄像头，请检查连接！")
                self.flag_timer = ''
            else:
                self.textEdit_camera.setText("相机准备就绪")
                self.flag_timer = 'camera_load'
                self.textEdit_video.setText("实时视频已关闭")
                self.textEdit_file.setText("图片文件未选中”")

                QtWidgets.QApplication.processEvents()

                self.timer_camera_load.start(50)

        if hasattr(self,'timer_video') and self.timer_video.isActive():
            self.timer_video.stop()
            self.flag_timer = ''

            if hasattr(self,'cap') and self.cap.isActive():
                self.cap.release()

                self.label_display.clear()
                self.label_pic_newface.clear()
                self.label_pic_org.clear()

                self.textEdit_file.setText("文件未选中")
                self.textEdit_camera.setText("实时摄像已关闭")
                self.textEdit_video.setText("实时视频已关闭")

        self.gif_movie()

        self.label_plate_result.setText("未知人脸")
        self.label_score_num.setText("0")
        self.label_score_dis.setText("0")

    def button_open_camera_click(self):
        # 停止视频定时器（如果正在运行）
        if self.timer_video and self.timer_video.isActive():
            self.timer_video.stop()
        # 释放视频捕获对象（如果已经打开）
        if self.cap and self.cap.isOpened():
            self.cap.release()

        # 避免在人脸离开和进入时重新初始化数据
        # 只有在第一次打开摄像头或者摄像头连接有问题重新打开时才初始化
        if not hasattr(self, 'camera_opened') or not self.camera_opened:
            self.ini_value()
            self.camera_opened = True

        # 尝试打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self.main_window, "摄像头错误", "无法打开摄像头，请检查连接！")
            self.flag_timer = ''
            self.camera_opened = False
        else:
            self.textEdit_camera.setText("相机准备就绪")
            self.flag_timer = 'camera'
            self.textEdit_video.setText("实时视频已关闭")
            self.textEdit_file.setText("图片文件未选中")

            QtWidgets.QApplication.processEvents()

            self.label_plate_result.setText("未知人脸")
            self.label_score_num.setText("0")
            self.label_score_dis.setText("0")

            # 启动摄像头定时器
            self.timer_camera.start(30)

    def drawRectBox(self,image,rect,addText):
        left, top, right, bottom = rect
        cv2.rectangle(image,(left,top),(right,bottom),(0,255,0),2)
        text_width = len(addText) * 12  # 估算文字宽度
        cv2.rectangle(image, (left, top - 20), (left + text_width, top), (0, 255, 0), -1)

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype("simhei.ttf", 16)
        except:
            font = ImageFont.load_default()

        draw.text((left + 5, top - 18), addText, font=font, fill=(255, 255, 255))

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def new_face_doing(self):
        self.toolButton_get_pic.setEnabled(False)
        self.toolButton_load_pic.setEnabled(False)

        self.label_display.clear()
        self.label_pic_newface.clear()
        self.current_face = None

        face_name = self.lineEdit_face_name.text().strip()

        if face_name and face_name != self.lineEdit_face_name.placeholderText():
            face_data_root = os.path.join(os.getcwd(), "face_data")
            self.path_face_dir = os.path.join(face_data_root, face_name)

            if not os.path.exists(face_data_root):
                os.makedirs(face_data_root)

            if not os.path.exists(self.path_face_dir):
                os.makedirs(self.path_face_dir)
                self.label_new_res.setText(f"新建人脸文件夹：{face_name}")
                self.label_loadface.setText("请点击【开启摄像设备录入】开始采集人脸")
            else:
                self.label_new_res.setText(f"警告：'{face_name}' 已存在，继续录入将添加照片")
                existing_files = [f for f in os.listdir(self.path_face_dir)
                                  if f.startswith(f"{face_name}_") and f.lower().endswith(('.jpg', '.png'))]

                if existing_files:
                    self.toolButton_load_pic.setEnabled(True)
                    self.label_loadface.setText(f"已存在 {len(existing_files)} 张照片，可继续录入")
                else:
                    self.label_loadface.setText("人脸文件夹已存在，但没有照片，请点击【开启摄像设备录入】开始采集")

            if self.current_face is not None:
                self.toolButton_get_pic.setEnabled(True)

            self.toolButton_file_2.setEnabled(True)
            self.toolButton_camera_load.setEnabled(True)

        else:
            self.label_new_res.setText("请输入有效的人脸名称")
            self.label_loadface.setText("等待点击以录入人脸")
            self.path_face_dir = ''

    def disp_face(self,image):
        self.label_display.clear()

        if image is None or image.size == 0:
            return

        resize_image = cv2.resize(image,(200,200))
        show = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        h, w, ch = show.shape
        bytes_per_line = 3 * w
        showImage = QtGui.QImage(show.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        a = QtGui.QPixmap.fromImage(showImage)

        self.label_pic_newface.setPixmap(a)
        self.label_pic_newface.setScaledContents(True)

        QtWidgets.QApplication.processEvents()

    def disp_image(self, image):
        resize_image = cv2.resize(image, (500, 500))
        show = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        h, w, ch = show.shape
        bytes_per_line = 3 * w
        showImage = QtGui.QImage(show.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        a = QtGui.QPixmap.fromImage(showImage)

        self.label_display.setPixmap(a)
        self.label_display.setScaledContents(True)

        QtWidgets.QApplication.processEvents()

    def show_camera_load(self):
        flag, img_rd = self.cap.read()
        if flag:
            self.current_image = img_rd.copy()
            image = img_rd.copy()

            img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2GRAY)
            faces = self.detector(img_gray)
            self.label_score_num.setText(f"人脸数量: {len(faces)}")
            face_name = self.lineEdit_face_name.text().strip()

            for k, d in enumerate(faces):
                left, top, right, bottom = d.left(), d.top(), d.right(), d.bottom()
                label_text = f"{face_name}_{k + 1}" if face_name else f"人脸_{k + 1}"
                image = self.drawRectBox(image, (left, top, right, bottom), label_text)

                h, w = img_rd.shape[:2]  # 获取图像高度和宽度

                # （1）计算人脸区域坐标并（2）确保在画面范围内
                x1 = max(0, left)  # 左边界不小于0
                y1 = max(0, top)  # 上边界不小于0
                x2 = min(w, right)  # 右边界不超过图像宽度
                y2 = min(h, bottom)  # 下边界不超过图像高度

                # （3）剪切人脸区域并存储
                if x2 > x1 and y2 > y1:  # 确保裁剪区域有效
                    self.current_face = img_rd[y1:y2, x1:x2]

                    # 8. 显示剪切出的人脸
                    self.disp_face(self.current_face)
                else:
                    print(f"无效的人脸区域: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        if self.current_face is not None and face_name:
            self.toolButton_get_pic.setEnabled(True)
        else:
            self.toolButton_get_pic.setEnabled(False)

        self.disp_image(image)

        self.label_plate_result.setText("正在录入")
        self.label_score_dis.setText("None")

    def choose_file(self):
        try:
            fileName_choose, filetype = QFileDialog.getOpenFileName()
            self.path = fileName_choose

            if fileName_choose:
                # 使用自定义方法读取含中文路径的图片
                image_read = self.cv_imread(fileName_choose)
                if image_read is None:
                    QMessageBox.warning(self.main_window, "文件错误", "无法读取所选图片，请选择有效图片文件！")
                    return

                image = image_read.copy()
                face_name = self.lineEdit_face_name.text().strip()

                # **关键修正：将彩色图转为灰度图**
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.detector(img_gray)  # 传入灰度图

                # 清空之前的显示
                self.label_pic_newface.clear()
                self.current_face = None

                for k, d in enumerate(faces):
                    left, top, right, bottom = d.left(), d.top(), d.right(), d.bottom()
                    label_text = f"{face_name}_{k + 1}" if face_name else f"人脸_{k + 1}"
                    image = self.drawRectBox(image, (left, top, right, bottom), label_text)

                    x1 = max(0, left)
                    y1 = max(0, top)
                    x2 = min(image.shape[1], right)  # 使用 image.shape 避免越界
                    y2 = min(image.shape[0], bottom)

                    if x2 > x1 and y2 > y1:
                        self.current_face = image_read[y1:y2, x1:x2]
                        self.disp_face(self.current_face)  # 显示剪切的人脸

                # 显示处理后的图像（循环外统一更新）
                self.disp_image(image)

                # 处理检测结果
                if len(faces) > 0 and face_name:
                    self.toolButton_get_pic.setEnabled(True)
                    self.label_loadface.setText(f"已检测到 {len(faces)} 张人脸，点击【取图】保存")
                elif len(faces) == 0:
                    self.label_loadface.setText("未检测到人脸，请尝试其他图片")
                    self.toolButton_get_pic.setEnabled(False)
                else:
                    self.label_loadface.setText("请输入人脸名称")

            else:
                self.label_loadface.setText("未选择文件")
                self.toolButton_get_pic.setEnabled(False)

        except Exception as e:
            # 捕获所有异常并显示错误信息
            QMessageBox.critical(self.main_window, "程序错误", f"选择图片时发生异常：{str(e)}")
            print(f"异常详情：{e}")

    def extract_features(self,path_img):
        img_rd = self.cv_imread(path_img)
        if img_rd is None:
            print(f"无法读取图片: {path_img}")
            return 0  # 按要求返回 0

        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2GRAY)
        faces = self.detector(img_gray, 0)

        if len(faces) > 0:
            face = faces[0]
            shape = self.predictor(img_rd, face)
            face_descriptor = self.face_reco_model.compute_face_descriptor(img_rd, shape)
            return face_descriptor
        else:
            print(f"在图片 {path_img} 中未检测到人脸")
            return 0

    def disp_load_face(self,image):
        if image is None:
            self.label_pic_org.clear()
            return

        resize_image = cv2.resize(image,(500,500))
        show = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        h, w, ch = show.shape
        bytes_per_line = 3 * w
        showImage = QtGui.QImage(show.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        a = QtGui.QPixmap.fromImage(showImage)

        self.label_pic_org.setPixmap(a)
        self.label_pic_org.setScaledContents(True)

        QtWidgets.QApplication.processEvents()

    def get_img_doing(self):
        try:
            face_name = self.lineEdit_face_name.text().strip()
            if not face_name:
                QMessageBox.warning(self, "警告", "请输入人脸名称")
                return

            if not self.path_face_dir or not os.path.exists(self.path_face_dir):
                face_data_root = os.path.join(os.getcwd(), "face_data")
                self.path_face_dir = os.path.join(face_data_root, face_name)
                os.makedirs(self.path_face_dir, exist_ok=True)

            pattern = os.path.join(self.path_face_dir, f"{face_name}_*.jpg")
            existing_files = glob.glob(pattern)
            img_num = len(existing_files)

            if self.current_face is not None and self.current_face.size > 0:
                img_num += 1
                save_path = os.path.join(self.path_face_dir, f"{face_name}_{img_num}.jpg")
                _, buffer = cv2.imencode('.jpg', self.current_face)
                with open(save_path, 'wb') as f:
                    f.write(buffer.tobytes())

                self.label_loadface.setText(f"已保存人脸图像: {save_path}")
            else:
                QMessageBox.warning(self, "警告", "未检测到有效人脸图像")
                return

            files_path = glob.glob(os.path.join(self.path_face_dir, "*.jpg"))

            self.disp_load_face(None)

            self.current_face = None
            self.toolButton_get_pic.setEnabled(False)

            if len(files_path) > 0:
                self.toolButton_load_pic.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "保存错误", f"保存人脸时发生异常：{str(e)}")
            print(f"异常详情：{e}")

    def load_img_doing(self):
        if hasattr(self, 'timer_camera_load') and self.timer_camera_load.isActive():
            self.timer_camera_load.stop()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

        face_data_root = os.path.join(os.getcwd(), "face_data")
        if not os.path.exists(face_data_root):
            QMessageBox.warning(self, "警告", "人脸数据目录不存在，请先录入人脸")
            self.label_loadface.setText("人脸数据目录不存在")
            return

        person_list = [d for d in os.listdir(face_data_root)
                       if os.path.isdir(os.path.join(face_data_root, d))]

        if not person_list:
            QMessageBox.warning(self, "警告", "未找到人脸文件夹，请先录入人脸")
            self.label_loadface.setText("未找到人脸文件夹")
            return

        csv_path = os.path.join(face_data_root, "features_all.csv")

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                for person in person_list:
                    person_path = os.path.join(face_data_root, person)
                    features_list = []

                    photos_list = [f for f in os.listdir(person_path)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                    if not photos_list:
                        writer.writerow([person] + [0.0] * 128)
                        continue

                    for photo in photos_list:
                        photo_path = os.path.join(person_path, photo)

                        try:
                            features_128D = self.extract_features(photo_path)

                            self.label_loadface.setText(f"正在录入: {person}/{photo}")
                            QtWidgets.QApplication.processEvents()

                            if features_128D != 0:
                                features_list.append(features_128D)
                        except Exception as e:
                            print(f"处理图片 {photo_path} 时出错: {e}")
                            continue

                    if features_list:
                        features_array = np.array([list(feature) for feature in features_list])
                        features_mean = np.mean(features_array, axis=0).tolist()
                    else:
                        features_mean = [0.0] * 128

                    writer.writerow([person] + features_mean)

            self.label_loadface.setText("所有图片的人脸特征已提取并保存完成")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"提取特征时发生异常: {str(e)}")
            print(f"异常详情: {e}")
            self.label_loadface.setText(f"提取失败: {str(e)}")

    def change_table(self, path, res, time_now, distance):

        self.count += 1
        max_rows = 100  # 统一设置最大行数为100

        # 在表格头部插入新行
        self.tableWidget_rec.insertRow(0)

        # 记录序号（显示为1-based）
        Item_num = QTableWidgetItem(str(self.count))
        Item_num.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget_rec.setItem(0, 0, Item_num)

        # 记录人脸路径
        Item_path = QTableWidgetItem(path)
        Item_path.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget_rec.setItem(0, 1, Item_path)

        # 记录识别结果
        Item_res = QTableWidgetItem(res)
        Item_res.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget_rec.setItem(0, 2, Item_res)
        self.tableWidget_rec.setCurrentItem(Item_res)  # 选中当前项

        # 记录识别时间
        Item_time = QTableWidgetItem(time_now)
        Item_time.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget_rec.setItem(0, 3, Item_time)

        # 记录置信度（保留4位小数）
        Item_dis = QTableWidgetItem(f"{distance:.4f}")
        Item_dis.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget_rec.setItem(0, 4, Item_dis)

        # 保持表格最多显示max_rows行
        if self.tableWidget_rec.rowCount() > max_rows:
            self.tableWidget_rec.removeRow(max_rows)

        self.tableWidget_rec.update()
        QApplication.processEvents()

    def get_face_database(self):
        self.face_feature_exist = []
        self.face_name_exist = []
        csv_dir = os.path.join(os.getcwd(),  "face_data")

        # 检查文件夹路径是否存在
        if not os.path.exists(csv_dir):
            print(f"人脸数据文件夹 {csv_dir} 不存在")
            return 0

        path_features_known_csv = os.path.join(csv_dir, "features_all.csv")

        if os.path.exists(path_features_known_csv):
            try:
                print(f"开始加载人脸特征文件: {path_features_known_csv}")

                # 读取CSV文件
                csv_rd = pd.read_csv(path_features_known_csv, header=None)
                print(f"成功读取CSV文件，共 {csv_rd.shape[0]} 行数据")

                # 遍历每一行数据
                for i in range(csv_rd.shape[0]):
                    name = csv_rd.iloc[i][0]
                    self.face_name_exist.append(name)
                    features = csv_rd.iloc[i][1:].tolist()
                    self.face_feature_exist.append(features)
                    print(f"已加载第 {i + 1} 行: 姓名={name}, 特征长度={len(features)}")

                print(f"成功加载 {len(self.face_name_exist)} 条人脸特征数据")
                return 1

            except pd.errors.ParserError as e:
                print(f"CSV解析错误: {str(e)}")
                print(f"错误位置: {path_features_known_csv}")
                return 0
            except FileNotFoundError as e:
                print(f"文件未找到错误: {str(e)}")
                print(f"预期路径: {path_features_known_csv}")
                return 0
            except PermissionError as e:
                print(f"权限错误: {str(e)}")
                print(f"无法访问: {path_features_known_csv}")
                return 0
            except Exception as e:
                print(f"加载人脸数据库出错: {str(e)}")
                print(f"错误类型: {type(e).__name__}")
                print(f"错误详情: {e}")
                return 0
        else:
            print(f"人脸特征文件不存在: {path_features_known_csv}")
            print(f"文件路径详情: 完整路径={os.path.abspath(path_features_known_csv)}")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"目录内容检查: {os.listdir(csv_dir) if os.path.exists(csv_dir) else '目录不存在'}")
            return 0

    def euclidean_distance(self,feature1,feature2):
        vec1 = np.array(feature1, dtype=np.float64)
        vec2 = np.array(feature2, dtype=np.float64)

        # 2. 计算欧式距离（向量差的L2范数）
        dist = np.linalg.norm(vec1 - vec2)

        # 3. 返回距离值
        return dist

    def do_choose_file(self):
        try:
            # 验证图片路径
            if not os.path.exists(self.path):
                self.textEdit_file.setText(f"错误：文件不存在 - {self.path}")
                print(f"文件不存在: {self.path}")
                return

            # 清空显示区域
            self.label_display.clear()
            self.label_pic_newface.clear()
            QApplication.processEvents()  # 处理界面更新

            # 加载人脸特征数据库
            exist_flag = self.get_face_database()

            # 读取图片
            img_rd = self.cv_imread(self.path)
            if img_rd is None:
                self.textEdit_file.setText(f"错误：图片读取失败 - {self.path}")
                print(f"图片读取失败: {self.path}")
                return

            # 复制图片用于处理
            image = img_rd.copy()

            # 人脸检测
            detector = dlib.get_frontal_face_detector()
            faces = detector(image, 1)
            self.current_face_cnt = len(faces)

            # 处理检测到的人脸
            if self.current_face_cnt > 0:
                self.label_score_num.setText(f"检测到 {self.current_face_cnt} 张人脸")
                face_feature_list = []
                face_name_list = ["未知人脸"] * self.current_face_cnt
                face_position_list = []
                face_distance_list = []

                # 提取所有人脸的特征
                for i, face in enumerate(faces):
                    # 计算人脸坐标并确保在图像范围内
                    x1 = max(0, face.left())
                    y1 = max(0, face.top())
                    x2 = min(image.shape[1], face.right())
                    y2 = min(image.shape[0], face.bottom())

                    # 保存人脸位置
                    face_position_list.append((x1, y1, x2, y2))

                    # 提取人脸区域和特征
                    crop_face = image[y1:y2, x1:x2]
                    shape = self.predictor(image, face)
                    face_descriptor = self.face_reco_model.compute_face_descriptor(image, shape)
                    face_feature_list.append(np.array(face_descriptor))

                    # 显示当前人脸
                    if i == 0:  # 只显示第一张人脸
                        self.current_face = crop_face
                        self.disp_face(self.current_face)  # 修改这里，移除多余的参数

                if exist_flag:
                    for i, feat in enumerate(face_feature_list):
                        current_distances = []

                        # 计算与所有库中特征的距离
                        for db_feat in self.face_feature_exist:
                            if len(db_feat) == 0:
                                current_distances.append(999999999)
                            else:
                                dist = self.euclidean_distance(feat, db_feat)
                                current_distances.append(dist)

                        min_dis = min(current_distances)
                        similar_idx = np.argmin(current_distances)
                        face_distance_list.append(min_dis)

                        # 根据阈值判断是否匹配
                        if min_dis < 0.3:  # 识别阈值
                            face_name_list[i] = self.face_name_exist[similar_idx]
                        else:
                            face_name_list[i] = "未知人脸"

                    # 更新表格记录
                    if face_distance_list:
                        min_idx = np.argmin(face_distance_list)
                        best_name = face_name_list[min_idx]  # 获取最佳匹配的名字
                        self.label_score_dis.setText(f"最佳相似度：{face_distance_list[min_idx]:.2f}")

                        # 记录识别结果
                        date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.change_table(
                            path=self.path,
                            res=best_name,  # 传递最佳匹配的名字
                            time_now=date_now,
                            distance=face_distance_list[min_idx]
                        )
                # 绘制识别结果
                for i, (x1, y1, x2, y2) in enumerate(face_position_list):
                    name = face_name_list[i]
                    image = self.drawRectBox(image, (x1, y1, x2, y2), name)  # 修正这里的参数格式

                    # 只显示第一张人脸的结果到主标签
                    if i == 0:
                        self.label_plate_result.setText(name)

                # 显示处理后的图像
                self.disp_image(image)  # 修改这里，移除多余的参数
            else:
                # 未检测到人脸
                self.label_display.setText("提示：未在图片中检测到人脸，请重新选择图片")
                self.label_plate_result.setText("未检测到人脸")

            # 恢复初始动画
            self.gif_movie()

        except Exception as e:
            self.textEdit_file.setText(f"错误：人脸识别失败 - {str(e)}")
            print(f"人脸识别错误: {str(e)}")

    def centroid_tracker(self):
        self.last_face_feature = self.current_face_feature.copy()
        if not self.last_centroid:
            self.current_face_name = ["未知"] * len(self.current_centroid)
            return

        # 1. 遍历当前帧中的所有人脸质心坐标
        for i, current_centroid in enumerate(self.current_centroid):
            # (1) 初始化距离列表
            distance_current_person = []

            # (2) 遍历上一帧的所有质心坐标
            for j, last_centroid in enumerate(self.last_centroid):
                # (2.1) 计算欧式距离并保存
                self.last_current_distance = self.euclidean_distance(
                    current_centroid,
                    last_centroid
                )

                distance_current_person.append(self.last_current_distance)

            if distance_current_person:
                min_distance = min(distance_current_person)
                last_frame_num = distance_current_person.index(min_distance)

                self.current_face_name[i] = self.last_face_name[last_frame_num]

    def show_video(self):
        self.last_face_feature = self.current_face_feature.copy()  # 保存上一帧特征
        self.last_face_name = self.current_face_name.copy()

        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 加载人脸特征数据库（确保每帧都有最新的特征库）
                if not self.exist_flag:
                    self.exist_flag = self.get_face_database()

                # 转换为灰度图像，提高人脸检测性能
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 人脸检测
                faces = self.detector(gray)
                self.current_face_cnt = len(faces)
                self.current_face_position = []
                self.current_face_feature = []
                self.current_face_name = []
                self.current_centroid = []
                self.current_face_distance = []
                self.current_face_confidence = []  # 新增：存储每个人脸的置信度

                # 处理检测到的人脸
                for face in faces:
                    # 计算人脸坐标并确保在图像范围内
                    x1 = max(0, face.left())
                    y1 = max(0, face.top())
                    x2 = min(frame.shape[1], face.right())
                    y2 = min(frame.shape[0], face.bottom())

                    # 保存人脸位置
                    self.current_face_position.append((x1, y1, x2, y2))

                    # 计算人脸质心
                    centroid_x = (x1 + x2) // 2
                    centroid_y = (y1 + y2) // 2
                    self.current_centroid.append((centroid_x, centroid_y))

                    # 提取人脸特征
                    shape = self.predictor(frame, face)
                    face_descriptor = self.face_reco_model.compute_face_descriptor(frame, shape)
                    self.current_face_feature.append(face_descriptor)

                    # 人脸识别（如果有特征库）
                    matched_name = "Unknown"
                    min_distance = float('inf')
                    best_match_index = -1

                    if self.exist_flag and self.face_feature_exist:
                        # 计算与所有库中特征的距离
                        distances = []
                        for i, db_feat in enumerate(self.face_feature_exist):
                            if len(db_feat) == 0:
                                distances.append(float('inf'))
                                continue
                            try:
                                # 计算欧氏距离
                                dist = self.euclidean_distance(face_descriptor, db_feat)
                                distances.append(dist)
                                if dist < min_distance:
                                    min_distance = dist
                                    best_match_index = i
                            except Exception as e:
                                print(f"计算距离时出错: {e}")
                                distances.append(float('inf'))
                                continue

                        # 自适应阈值调整
                        if distances:
                            # 计算平均距离和标准差
                            valid_distances = [d for d in distances if d != float('inf')]
                            if valid_distances:
                                mean_dist = np.mean(valid_distances)
                                std_dist = np.std(valid_distances)

                                # 根据数据分布动态调整阈值
                                # 基础阈值为0.5，根据标准差进行微调
                                threshold = 0.5 + 0.1 * (std_dist / 0.2)
                                threshold = min(0.7, max(0.3, threshold))  # 限制在合理范围内

                                # 计算置信度 (0-1之间，值越高越确定)
                                if min_distance < threshold:
                                    # 距离越小，置信度越高；距离接近阈值，置信度降低
                                    confidence = 1.0 - (min_distance / threshold)
                                    confidence = max(0.0, confidence)  # 确保不为负

                                    matched_name = self.face_name_exist[best_match_index]
                                    self.current_face_name.append(matched_name)
                                    self.current_face_confidence.append(confidence)
                                else:
                                    self.current_face_name.append("Unknown")
                                    self.current_face_confidence.append(0.0)
                            else:
                                self.current_face_name.append("Unknown")
                                self.current_face_confidence.append(0.0)
                        else:
                            self.current_face_name.append("Unknown")
                            self.current_face_confidence.append(0.0)
                    else:
                        self.current_face_name.append("Unknown")
                        self.current_face_confidence.append(0.0)

                    # 记录当前人脸与库中人脸的最小距离
                    self.current_face_distance.append(min_distance)

                    # 在画面上显示人脸框和置信度信息
                    confidence = self.current_face_confidence[-1] if self.current_face_confidence else 0.0
                    addText = f"{matched_name}: {confidence:.2f}"
                    frame = self.drawRectBox(frame, (x1, y1, x2, y2), addText)

                    # 输出识别结果和置信度
                    print(f"Matched: {matched_name}, Confidence: {confidence:.2f}, Distance: {min_distance:.4f}")

                # 人脸跟踪优化：如果上一帧有人脸，尝试关联当前帧
                if self.last_face_cnt > 0 and self.current_face_cnt > 0:
                    self.centroid_tracker()

                    # 改进的身份关联逻辑
                    for i in range(self.current_face_cnt):
                        # 如果当前识别为Unknown且上一帧有结果，尝试通过特征匹配
                        if self.current_face_name[i] == "Unknown" and self.last_face_name:
                            best_dist = float('inf')
                            best_name = "Unknown"
                            best_index = -1

                            for j in range(self.last_face_cnt):
                                # 计算当前特征与上一帧特征的距离
                                if len(self.current_face_feature[i]) > 0 and len(self.last_face_feature[j]) > 0:
                                    dist = self.euclidean_distance(self.current_face_feature[i],
                                                                   self.last_face_feature[j])
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_name = self.last_face_name[j]
                                        best_index = j

                            # 放宽阈值并增加时间平滑
                            if best_dist < 0.6:  # 比正常识别放宽阈值
                                # 检查上一帧的识别置信度
                                if best_index >= 0 and best_index < len(self.last_face_confidence):
                                    last_confidence = self.last_face_confidence[best_index]
                                    # 只有上一帧置信度高的情况下才继承身份
                                    if last_confidence > 0.6:
                                        self.current_face_name[i] = best_name
                                        # 降低当前帧的置信度，因为是通过跟踪继承的
                                        adjusted_confidence = last_confidence * (1 - best_dist / 0.6)
                                        self.current_face_confidence[i] = adjusted_confidence

                # 更新表格记录（仅当检测到人脸时）
                if self.current_face_cnt > 0 and self.current_face_distance:
                    # 找到最佳匹配（最小距离）
                    min_idx = np.argmin(self.current_face_distance)
                    best_name = self.current_face_name[min_idx]
                    best_distance = self.current_face_distance[min_idx]
                    best_confidence = self.current_face_confidence[min_idx] if min_idx < len(
                        self.current_face_confidence) else 0.0

                    # 更新界面标签
                    self.label_plate_result.setText(best_name)
                    self.label_score_num.setText(f"检测到 {self.current_face_cnt} 张人脸")
                    self.label_score_dis.setText(f"最佳相似度：{best_confidence:.2f}")

                    # 记录识别结果到表格
                    date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.change_table(
                        path=self.video_path,
                        res=best_name,
                        time_now=date_now,
                        distance=best_confidence  # 使用置信度而非距离
                    )

                # 更新重新分类计数器
                self.reclassify_cnt += 1
                if self.reclassify_cnt >= self.reclassify_interval:
                    self.reclassify_cnt = 0
                    # 定期重新加载数据库，确保数据更新
                    self.exist_flag = self.get_face_database()

                # 更新上一帧的信息
                self.last_face_cnt = self.current_face_cnt
                self.last_face_name = self.current_face_name.copy()
                self.last_centroid = self.current_centroid.copy()
                self.last_face_feature = self.current_face_feature.copy()
                self.last_face_confidence = self.current_face_confidence.copy()  # 新增：保存上一帧置信度

                # 将处理后的图像显示到界面
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = 3 * w
                q_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                q_pixmap = QtGui.QPixmap.fromImage(q_image)
                self.label_display.setPixmap(q_pixmap)
                self.label_display.setScaledContents(True)
            else:
                # 视频播放结束，释放资源
                self.cap.release()
                self.timer_video.stop()
                self.textEdit_video.setText("视频播放结束")
                self.gif_movie()
        else:
            self.timer_video.stop()
            self.textEdit_video.setText("视频无法打开")
            self.gif_movie()

    def show_camera(self):
        self.last_face_feature = self.current_face_feature.copy()  # 保存上一帧特征
        self.last_face_name = self.current_face_name.copy()
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 加载人脸特征数据库（确保每帧都有最新的特征库）
                if not self.exist_flag:
                    self.exist_flag = self.get_face_database()

                # 转换为灰度图像，提高人脸检测性能
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 人脸检测
                faces = self.detector(gray)
                self.current_face_cnt = len(faces)
                self.current_face_position = []
                self.current_face_feature = []
                self.current_face_name = []
                self.current_centroid = []
                self.current_face_distance = []  # 初始化距离列表

                # 处理检测到的人脸
                for face in faces:
                    # 计算人脸坐标并确保在图像范围内
                    x1 = max(0, face.left())
                    y1 = max(0, face.top())
                    x2 = min(frame.shape[1], face.right())
                    y2 = min(frame.shape[0], face.bottom())

                    # 保存人脸位置
                    self.current_face_position.append((x1, y1, x2, y2))

                    # 计算人脸质心
                    centroid_x = (x1 + x2) // 2
                    centroid_y = (y1 + y2) // 2
                    self.current_centroid.append((centroid_x, centroid_y))

                    # 提取人脸特征
                    shape = self.predictor(frame, face)
                    face_descriptor = self.face_reco_model.compute_face_descriptor(frame, shape)
                    self.current_face_feature.append(face_descriptor)

                    # 人脸识别
                    matched_name = "Unknown"
                    min_distance = float('inf')

                    if self.exist_flag and self.face_feature_exist:
                        # 计算与所有库中特征的距离
                        for i, db_feat in enumerate(self.face_feature_exist):
                            if len(db_feat) == 0:
                                continue
                            try:
                                # 计算欧氏距离
                                dist = self.euclidean_distance(face_descriptor, db_feat)
                                if dist < min_distance:
                                    min_distance = dist
                                    matched_name = self.face_name_exist[i]
                            except Exception as e:
                                print(f"计算距离时出错: {e}")
                                continue

                        # 根据阈值判断是否匹配
                        if min_distance < 0.3:
                            self.current_face_name.append(matched_name)
                        else:
                            self.current_face_name.append("Unknown")
                    else:
                        self.current_face_name.append("Unknown")

                    # 记录当前人脸与库中人脸的最小距离
                    self.current_face_distance.append(min_distance)

                    # 计算置信度（距离越小，置信度越高）
                    confidence = 1 / (1 + min_distance)

                    # 在画面上显示人脸框和置信度信息
                    addText = f"{matched_name}: {confidence:.2f}"
                    frame = self.drawRectBox(frame, (x1, y1, x2, y2), addText)

                    # 输出识别结果和置信度
                    print(f"Matched: {matched_name}, Confidence: {confidence:.2f}")

                # 人脸跟踪优化：如果上一帧有人脸，尝试关联当前帧
                if self.last_face_cnt > 0 and self.current_face_cnt > 0:
                    self.centroid_tracker()
                    for i in range(self.current_face_cnt):
                        if self.current_face_name[i] == "Unknown" and self.last_face_name:
                            best_dist = float('inf')
                            best_name = "Unknown"
                            for j in range(self.last_face_cnt):
                                # 计算当前特征与上一帧特征的距离
                                if len(self.current_face_feature[i]) > 0 and len(self.last_face_feature[j]) > 0:
                                    dist = self.euclidean_distance(self.current_face_feature[i],self.last_face_feature[j])
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_name = self.last_face_name[j]
                            if best_dist < 0.5:  # 放宽阈值
                                self.current_face_name[i] = best_name
                # 更新表格记录（仅当检测到人脸时）
                if self.current_face_cnt > 0 and self.current_face_distance:
                    # 找到最佳匹配（最小距离）
                    min_idx = np.argmin(self.current_face_distance)
                    best_name = self.current_face_name[min_idx]
                    best_distance = self.current_face_distance[min_idx]

                    # 更新界面标签
                    self.label_plate_result.setText(best_name)
                    self.label_score_num.setText(f"检测到 {self.current_face_cnt} 张人脸")
                    self.label_score_dis.setText(f"最佳相似度：{best_distance:.2f}")

                    # 记录识别结果到表格
                    date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.change_table(
                        path="摄像头实时识别",
                        res=best_name,
                        time_now=date_now,
                        distance=best_distance
                    )

                # 更新重新分类计数器
                self.reclassify_cnt += 1
                if self.reclassify_cnt >= self.reclassify_interval:
                    self.reclassify_cnt = 0

                # 更新上一帧的信息
                self.last_face_cnt = self.current_face_cnt
                self.last_face_name = self.current_face_name
                self.last_centroid = self.current_centroid

                # 将处理后的图像显示到界面
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = 3 * w
                q_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                q_pixmap = QtGui.QPixmap.fromImage(q_image)
                self.label_display.setPixmap(q_pixmap)
                self.label_display.setScaledContents(True)
            else:
                print("Failed to read frame from camera.")
        else:
            print("Camera is not opened.")

    def run_rec(self):
        if self.flag_timer == "image":
            # 图片模式处理
            self.do_choose_file()

        elif self.flag_timer == "video":
            # 视频模式处理
            if not self.timer_video.isActive():
                # 启动视频识别
                self.exist_flag = self.get_face_database()
                if self.exist_flag:
                    self.timer_video.start(30)
            else:
                # 停止视频识别
                self.timer_video.stop()

        elif self.flag_timer == "camera":
            # 摄像头模式处理
            if not self.timer_camera.isActive():
                # 启动摄像头识别
                QtWidgets.QApplication.processEvents()
                self.exist_flag = self.get_face_database()
                if self.exist_flag:
                    self.timer_camera.start(30)
            else:
                # 停止摄像头识别
                self.timer_camera.stop()
                self.flag_timer = ""
                if hasattr(self, 'cap') and self.cap.isOpened():
                    self.cap.release()
                self.textEdit_camera.setStyleSheet("border: 1px solid #ccc;")
                QtWidgets.QApplication.processEvents()

        else:
            # 未选择有效数据源处理
            self.textEdit_file.setText("图片文件未选中")
            self.textEdit_camera.setText("实时摄像已关闭")
            self.textEdit_video.setText("实时视频已关闭")
            self.label_display.clear()
            self.gif_movie()
            self.label_pic_newface.clear()
            self.label_plate_result.setText("未知人脸")
            self.label_score_fps.setText("0")
            self.label_score_num.setText("0")
            self.label_score_dis.setText("0")

    def table_review(self, row, col):
        """记录用户点击的表格位置"""
        self.col_row = [row, col]
        print(f"选中表格第 {row + 1} 行，第 {col + 1} 列")

    def delete_doing(self):
        # 检查是否选择了表格中的行
        if self.col_row:
            # 弹出确认对话框
            msg = QtWidgets.QMessageBox.question(self.main_window, "Warning", "确定删除该人脸数据吗?",QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,QtWidgets.QMessageBox.No)
            # 如果用户点击"是"按钮
            if msg == QtWidgets.QMessageBox.Yes:
                # 获取用户点击的表格的行索引和列索引
                row, col = self.col_row
                # 获取人脸数据文件夹的路径
                r_path = self.tableWidget_mana.item(row, 1).text()
                # 检查文件夹路径是否存在
                if os.path.exists(r_path):
                    # 删除文件夹及其所有内容
                    shutil.rmtree(r_path)
                    # 更新界面提示信息
                    self.label_mana_info.setText("已删除该人脸")
                    # 更新人脸数据
                    self.do_update_face()
                    # 更新界面提示信息
                    self.label_mana_info.setText("开始重新录入")
                    # 获取人脸数据库路径下的所有文件夹名称
                    person_list = [f for f in os.listdir(self.path_face_dir)
                                   if os.path.isdir(os.path.join(self.path_face_dir, f))]
                    # 打开CSV文件准备写入更新后的人脸特征数据
                    with open('features_all.csv', 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)

                        # 遍历所有人脸数据文件夹
                        for person in person_list:
                            # 初始化特征列表
                            features_list = []

                            # 获取当前人脸文件夹内的所有图片文件
                            person_path = os.path.join(self.path_face_dir, person)
                            photos_list = [f for f in os.listdir(person_path)
                                           if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                            # 如果文件夹内有图片文件
                            if photos_list:
                                for photo in photos_list:
                                    # 提取图片中的人脸特征向量
                                    img_path = os.path.join(person_path, photo)
                                    features_128D = self.extract_features(img_path)
                                    # 更新界面提示信息
                                    self.label_mana_info.setText(f"正在录入: {photo}")
                                    QtWidgets.QApplication.processEvents()  # 处理界面事件
                                    # 如果未检测到人脸，跳过当前图片
                                    if len(features_128D) == 0:
                                        continue
                                    # 将有效特征向量添加到列表中
                                    features_list.append(features_128D)

                            # 计算特征向量的平均值
                            if features_list:
                                features_mean = np.mean(features_list, axis=0)
                            else:
                                # 创建默认的全零特征向量
                                features_mean = np.zeros(128)

                            # 准备写入CSV的数据
                            str_face = [person]
                            str_face.extend(features_mean)

                            # 写入CSV文件
                            writer.writerow(str_face)

                    # 更新界面提示信息
                    self.label_mana_info.setText("已重新录入人脸！")
        else:
            # 如果没有选择行，显示提示
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择要删除的人脸数据")

    def change_table_mana(self, path, face_name, time_now):

        # 获取当前表格的行数，这将是新行要插入的位置
        row_position = self.tableWidget_mana.rowCount()
        # 在表格末尾插入新行
        self.tableWidget_mana.insertRow(row_position)

        # 3. 添加序号列（居中对齐）
        Item_num = QTableWidgetItem(str(row_position + 1))
        Item_num.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget_mana.setItem(row_position, 0, Item_num)

        # 4. 添加路径列（垂直居中对齐）
        Item_way = QTableWidgetItem(path)
        Item_way.setTextAlignment(QtCore.Qt.AlignVCenter)
        self.tableWidget_mana.setItem(row_position, 1, Item_way)

        # 5. 添加名称列（居中对齐并选中）
        Item_name = QTableWidgetItem(face_name)
        Item_name.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget_mana.setItem(row_position, 2, Item_name)
        self.tableWidget_mana.setCurrentItem(Item_name)  # 选中名称项

        # 6. 添加时间列（居中对齐并选中）
        Item_time = QTableWidgetItem(time_now)
        Item_time.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget_mana.setItem(row_position, 3, Item_time)
        self.tableWidget_mana.setCurrentItem(Item_time)  # 选中时间项

    def update_face(self):
        # 1. 将 self.count_face 重置为 0，用于重新计数表格中的人脸数据行数
        self.count_face = 0

        # 2. 调用 os.listdir()函数，获取人脸数据库文件夹（self.path_face_dir）下的所有子文件夹名称
        person_list = []
        if os.path.exists(self.path_face_dir) and os.path.isdir(self.path_face_dir):
            person_list = [d for d in os.listdir(self.path_face_dir) if
                           os.path.isdir(os.path.join(self.path_face_dir, d))]

        # 3. 计算 person_list 的长度，即人脸数据的数量，并将结果赋值给变量 num_faces
        num_faces = len(person_list)

        # 4. 将人脸数量 num_faces 转换为字符串，设置到管理界面中的 label_mana_face_num 标签上
        if hasattr(self, 'label_mana_face_num'):
            self.label_mana_face_num.setText(str(num_faces))

        # 5. 获取人脸数据库文件夹 self.path_face_dir 的最后修改时间的时间戳，存储在变量 timestamp 中
        try:
            timestamp = os.path.getmtime(self.path_face_dir)
        except FileNotFoundError:
            timestamp = None

        if timestamp is not None:
            # 6. 将时间戳 timestamp 转换为本地时间结构体，存储在变量 timeStruct 中
            timeStruct = time.localtime(timestamp)

            # 7. 将本地时间结构体 timeStruct 格式化为字符串，格式为“月-日 时:分:秒”，结果覆盖存储在 timeStruct 变量中
            timeStruct = time.strftime("%m-%d %H:%M:%S", timeStruct)

            # 8. 将格式化后的时间字符串 timeStruct 设置到管理界面中的 label_mana_time 标签上，显示人脸数据库的最后修改时间
            if hasattr(self, 'label_mana_time'):
                self.label_mana_time.setText(timeStruct)

        # 9. 遍历 person_list 中的每个人脸数据文件夹名称
        for face_name in person_list:

            dir_path = os.path.join(self.path_face_dir, face_name)

            try:
                timestamp = os.path.getmtime(dir_path)
            except FileNotFoundError:
                timestamp = None

            if timestamp is not None:

                timeStruct = time.localtime(timestamp)

                timeStruct = time.strftime("%m-%d %H:%M:%S", timeStruct)
            else:
                timeStruct = "未知时间"

            if hasattr(self, 'change_table_mana') and callable(self.change_table_mana):
                self.change_table_mana(dir_path, face_name, timeStruct)

    def do_update_face(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.path_face_dir = os.path.join(current_dir, 'face_data')
        # 1. 重置计数
        self.count_face = 0

        # 调试：打印当前路径
        print(f"当前人脸数据库路径: {self.path_face_dir}")
        print(f"当前工作目录: {os.getcwd()}")  # 检查相对路径基准

        # 2. 获取子文件夹（严格校验路径）
        person_list = []
        if not os.path.exists(self.path_face_dir):
            print(f"错误：路径 {self.path_face_dir} 不存在！")
            self.label_mana_face_num.setText("当前人脸数量: 0")
            self.label_mana_time.setText("最后修改时间: 未知")
            self.label_mana_info.setText("人脸数据库路径不存在")
            self.tableWidget_mana.setRowCount(0)
            return

        if not os.path.isdir(self.path_face_dir):
            print(f"错误：{self.path_face_dir} 不是文件夹！")
            self.label_mana_face_num.setText("当前人脸数量: 0")
            self.label_mana_time.setText("最后修改时间: 未知")
            self.label_mana_info.setText("路径不是文件夹")
            self.tableWidget_mana.setRowCount(0)
            return

        # 仅保留子文件夹
        person_list = [
            d for d in os.listdir(self.path_face_dir)
            if os.path.isdir(os.path.join(self.path_face_dir, d))
        ]
        print(f"找到子文件夹: {person_list}")  # 调试：打印实际获取的子文件夹

        # 3. 更新人脸数量
        num_faces = len(person_list)
        self.label_mana_face_num.setText(f"当前人脸数量: {num_faces}")

        # 4. 更新数据库最后修改时间
        try:
            timestamp = os.path.getmtime(self.path_face_dir)
            time_str = time.strftime("%m-%d %H:%M:%S", time.localtime(timestamp))
            self.label_mana_time.setText(f"最后修改时间: {time_str}")
        except Exception as e:
            self.label_mana_time.setText("最后修改时间: 未知")
            print(f"获取文件夹时间失败: {e}")

        # 5. 清空表格
        self.tableWidget_mana.setRowCount(0)

        # 6. 遍历子文件夹更新表格
        for face_name in person_list:
            dir_path = os.path.join(self.path_face_dir, face_name)
            print(f"处理子文件夹: {dir_path}")  # 调试：打印当前处理的路径

            # 校验子文件夹路径（防御性编程）
            if not os.path.exists(dir_path):
                print(f"警告：子文件夹 {dir_path} 不存在，跳过")
                continue

            # 获取修改时间
            try:
                timestamp = os.path.getmtime(dir_path)
                mod_time = time.strftime("%m-%d %H:%M:%S", time.localtime(timestamp))
            except Exception as e:
                mod_time = "未知时间"
                print(f"子文件夹 {face_name} 获取时间失败: {e}")

            # 更新表格
            self.change_table_mana(
                path=dir_path,
                face_name=face_name,
                time_now=mod_time
            )

        # 7. 更新提示
        self.label_mana_info.setText("人脸已更新")



