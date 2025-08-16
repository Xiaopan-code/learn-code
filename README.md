# 人脸识别系统

## 项目简介
本项目是一个基于Python和PyQt5的人脸识别系统，支持人脸录入、实时识别和数据管理功能。通过简洁的图形界面，用户可以轻松完成人脸数据采集与识别任务。

### 核心功能
- 从摄像头或本地图片录入人脸数据
- 实时摄像头人脸识别
- 支持图片和视频文件的离线识别
- 人脸数据管理（查看、删除记录）

## 环境要求
- Python 3.x
- 依赖库：PyQt5、dlib、opencv-python、numpy、pandas、Pillow

安装依赖：pip install PyQt5 dlib opencv-python numpy pandas Pillow
需要额外下载dlib模型文件并放置在`data/data_dlib/`目录：
- shape_predictor_68_face_landmarks.dat
- dlib_face_recognition_resnet_model_v1.dat

## 使用方法
1. 克隆仓库并进入项目目录
2. 安装所需依赖库
3. 配置dlib模型文件路径
4. 运行主程序：
   ```bash
   python runMain.py
   ```

### 人脸录入
1. 在"录入"标签页输入姓名
2. 选择"开启摄像设备录入"或"选择人脸照片文件"
3. 检测到人脸后点击"取图"保存

### 人脸识别
1. 在"识别"标签页选择识别方式：
   - 摄像头实时识别
   - 本地图片识别
   - 本地视频识别
2. 系统将自动显示识别结果及置信度

### 数据管理
1. 在"管理"标签页查看所有已录入人脸
2. 可删除不需要的人脸数据
3. 点击"更新"刷新数据列表

## 项目结构├── FaceRecognition_UI.py       # 界面代码
├── FaceRqecognition.py        # 核心识别逻辑
├── features_all.csv           # 人脸特征数据
├── runMain.py                 # 程序入口
└── FaceRec/
    └── face_data/             # 人脸图片存储目录
---

# Face Recognition System

## Project Introduction
This is a face recognition system based on Python and PyQt5, supporting face registration, real-time recognition and data management. With a user-friendly GUI, users can easily perform face data collection and recognition tasks.

### Core Features
- Register face data from camera or local images
- Real-time face recognition via camera
- Offline recognition for image and video files
- Face data management (view, delete records)

## Environment Requirements
- Python 3.x
- Dependencies: PyQt5, dlib, opencv-python, numpy, pandas, Pillow

Install dependencies:pip install PyQt5 dlib opencv-python numpy pandas Pillow
Additional dlib model files required in `data/data_dlib/` directory:
- shape_predictor_68_face_landmarks.dat
- dlib_face_recognition_resnet_model_v1.dat

## Usage
1. Clone the repository and enter project directory
2. Install required dependencies
3. Configure dlib model file paths
4. Run the main program:
   ```bash
   python runMain.py
   ```

### Face Registration
1. Enter name in the "Registration" tab
2. Choose "Enable camera for registration" or "Select face photo file"
3. Click "Capture" to save when face is detected

### Face Recognition
1. Select recognition method in the "Recognition" tab:
   - Real-time camera recognition
   - Local image recognition
   - Local video recognition
2. The system will automatically display recognition results and confidence

### Data Management
1. View all registered faces in the "Management" tab
2. Delete unwanted face data
3. Click "Update" to refresh data list

## Project Structure├── FaceRecognition_UI.py       # Interface code
├── FaceRqecognition.py        # Core recognition logic
├── features_all.csv           # Face feature data
├── runMain.py                 # Program entry
└── FaceRec/
    └── face_data/             # Face image storage directory
