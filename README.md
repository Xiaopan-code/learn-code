# Smart Home Voice Control System  
# 智能家居语音控制系统  


## Project Overview  
## 项目概述  
A simple voice-controlled system for managing home devices (lights, fans, air conditioners, etc.), supporting location-specific commands (e.g., "Turn on the living room light").  
一个简单的语音控制系统，用于管理家居设备（灯、风扇、空调等），支持指定位置的精准控制（例如：“打开客厅的灯”）。  


## Core Features  
## 核心功能  
- Control home devices via voice commands  
  通过语音指令控制家居设备  
- Support location-specific device control (e.g., bedroom, living room)  
  支持指定位置的设备控制（如卧室、客厅）  
- Interactive graphical user interface (GUI)  
  交互式图形用户界面（GUI）  
- Integrates speech recognition and command parsing  
  集成语音识别与指令解析功能  


## Project Structure  
## 项目结构  

| File Name               | Description (English)                                  | 描述（中文）                                      |  
|-------------------------|--------------------------------------------------------|---------------------------------------------------|  
| `Gui.py`                | Main program with graphical user interface             | 主程序，提供图形用户界面                          |  
| `model_binary.py`       | Trains binary classification model (for on/off commands) | 训练二分类模型（用于识别开关指令）                |  
| `model_multiclass.py`   | Trains multi-class model (for device type/location)    | 训练多分类模型（用于识别设备类型和位置）          |  
| `VoiceRecognitionCore.py` | Integrates Baidu Speech Recognition API for voice input | 集成百度语音识别API，处理语音输入                 |  
| `Translate.py`          | Generates and processes training datasets              | 生成和处理训练数据集                              |  
| `data_processor.py`     | Defines Chinese word segmenter for command parsing     | 定义中文分词器，用于指令解析                      |  


## Usage Guide  
## 使用指南  

1. Install required dependencies (see `requirements.txt` if provided)  
   安装必要的依赖库（如有`requirements.txt`请参考）  
2. Configure Baidu Speech Recognition API credentials (API key and secret key)  
   配置百度语音识别API密钥（API Key和Secret Key）  
3. Run `Gui.py` to start the system  
   运行`Gui.py`启动系统  
4. Follow on-screen prompts for voice input or manual operation  
   按照界面提示进行语音输入或手动操作  


## Example Commands  
## 指令示例  

| English Example               | Chinese Example             |  
|-------------------------------|-----------------------------|  
| "Turn on the living room light" | "打开客厅的灯"             |  
| "Turn off the bedroom fan"      | "关闭卧室的风扇"           |  
| "Start the study room air conditioner" | "启动书房的空调"       |  
| "Turn off all devices"         | "关闭所有设备"             |  


## Notes  
## 注意事项  
- Requires network connection for Baidu Speech Recognition service  
  使用百度语音识别服务需保持网络连接  
- Model files need to be pre-trained via `model_binary.py` and `model_multiclass.py`  
  模型文件需通过`model_binary.py`和`model_multiclass.py`提前训练生成  
- API credentials must be correctly configured in `VoiceRecognitionCore.py`  
  需在`VoiceRecognitionCore.py`中正确配置API密钥  


Feel free to submit issues or pull requests for improvements!  
欢迎提交问题反馈或代码改进建议！
