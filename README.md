# 仓库分支说明  

| 分支名称              | 核心功能与说明                                                                                     |  
|-----------------------|--------------------------------------------------------------------------------------------------|  
| bird-recognition      | 鸟类识别：基于 Python 3.12 识别 325 种鸟类，数据集推荐百度 AI Studio 地址，需独立虚拟环境配置                |  
| face-recognition      | 人脸识别系统：PyQt5 界面，支持摄像头/本地图片录入、实时识别、数据管理，依赖 dlib 模型文件（需手动放置到指定目录）       |  
| Intelligent-language-processing      | 智能家居语音控制：语音指令控制设备（灯、风扇等），支持位置指定（如“客厅灯”），集成百度语音识别 API，需配置密钥          |  
| tianchi-coupon-pred   | 天池优惠券预测：参与天池新人实战赛，用随机森林模型预测优惠券使用，提供 `demo.py`（适配旧 Pandas）和 `revise.py`（适配新版） |  


# Repository Branch Overview  

| Branch Name           | Core Features & Notes                                                                 |  
|-----------------------|---------------------------------------------------------------------------------------|  
| bird-recognition      | Bird Recognition: Identifies 325 bird species with Python 3.12. Dataset via Baidu AI Studio (recommended). Use a dedicated virtual environment. |  
| face-recognition      | Face Recognition System: PyQt5 GUI for face enrollment (camera/local images), real-time recognition, and data management. Depends on dlib model files (manually place in specified paths). |  
| Intelligent-language-processing      | Smart Home Voice Control: Controls devices (lights, fans) via voice commands (supports location specs like “living room light”). Integrates Baidu Speech API (API key/secret required). |  
| tianchi-coupon-pred   | Tianchi Coupon Usage Prediction: Competes in Alibaba Tianchi’s O2O coupon prediction challenge. Uses Random Forest. Includes `demo.py` (for old Pandas) and `revise.py` (for latest Pandas). |  
