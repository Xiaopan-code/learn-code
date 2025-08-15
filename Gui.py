import tkinter as tk
from tkinter import ttk, messagebox
import threading
import joblib
import jieba
import os
from VoiceRecognitionCore import VoiceRecognitionCore  # 导入重构后的语音识别核心类


# 必须与模型训练时相同的预处理类
class DataProcessor:
    _jieba_user_dict_loaded = False
    _user_dict_path = None  # 用于记录加载的词典路径

    @staticmethod
    def chinese_tokenizer(text):
        """中文分词器，必须与模型训练时保持一致"""
        # 注意：cut_all=False (精确模式), HMM=True (使用HMM模型识别新词)
        return jieba.lcut(text, cut_all=False, HMM=True)

    @staticmethod
    def load_jieba_user_dict(dict_path):
        if not DataProcessor._jieba_user_dict_loaded or DataProcessor._user_dict_path != dict_path:
            if os.path.exists(dict_path):
                jieba.load_userdict(dict_path)
                DataProcessor._jieba_user_dict_loaded = True
                DataProcessor._user_dict_path = dict_path
                print(f"Jieba自定义词典已从 '{dict_path}' 加载。")
            else:
                print(f"警告：Jieba自定义词典文件 '{dict_path}' 未找到，将使用默认词典。")
                DataProcessor._jieba_user_dict_loaded = False


class SmartHomeVoiceControl:
    def __init__(self, root):
        self.root = root
        self.root.title("智能家居语音控制系统 (单次执行模式)")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.confirm_exit)

        # 系统状态
        self.analyzing = False
        self.device_states = {}

        # 模型组件
        self.binary_vectorizer = None
        self.binary_model = None
        self.multi_vectorizer = None
        self.multi_model = None
        self.label_encoders = None


        current_dir = os.path.dirname(os.path.abspath(__file__))
        user_dict_path = os.path.join(current_dir, 'datasets', 'user_dict.txt')
        DataProcessor.load_jieba_user_dict(user_dict_path)

        if not self.initialize_models():
            self.root.after(100, self.root.destroy)
            return

        self.create_widgets()

        self.voice_recognizer = VoiceRecognitionCore(
            sample_rate=16000,
            energy_threshold=2000,
            silence_timeout=2.5,
            denoise_level=0.8,
            recognition_callback=self.on_recognition_result  # 只保留这个回调
        )
        self.update_status("系统就绪，请点击“开始聆听”", "green")  # 初始状态更新
        self.update_result_display("--", "--", "--")  # 初始化结果显示

    def initialize_models(self):
        try:
            # 确保模型文件路径正确
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            binary_model_path = os.path.join(models_dir, 'binary_classifier.pkl')
            multi_model_path = os.path.join(models_dir, 'multiclass_classifier.pkl')

            # 检查必要文件是否存在
            required_files = [binary_model_path, multi_model_path]
            for f_path in required_files:
                if not os.path.exists(f_path):
                    messagebox.showerror("文件缺失", f"必要的模型文件未找到: {f_path}\n请先训练模型。")
                    return False

            binary_state = joblib.load(binary_model_path)
            self.binary_vectorizer = binary_state['vectorizer']
            self.binary_model = binary_state['model']
            print(f"已加载二分类模型: {binary_model_path}")

            multi_state = joblib.load(multi_model_path)
            self.multi_vectorizer = multi_state['vectorizer']
            self.multi_model = multi_state['model']
            self.label_encoders = multi_state['label_encoders']

            locations = self.label_encoders["location"].classes_
            devices = self.label_encoders["device"].classes_
            for loc in locations:
                for dev in devices:
                    self.device_states[f"{loc}_{dev}"] = "off"

            print(f"已加载多分类模型: {multi_model_path}")
            return True
        except Exception as e:
            messagebox.showerror("初始化错误", f"模型加载失败: {str(e)}\n请确保模型文件存在且依赖库版本一致。")
            return False

    def create_widgets(self):
        """创建GUI界面"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        self.status_label = ttk.Label(top_frame, text="准备就绪", font=('Helvetica', 12, 'bold'), foreground="green")
        self.status_label.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.record_button = ttk.Button(top_frame, text="开始聆听", command=self.start_or_force_analyze)
        self.record_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(top_frame, text="退出系统", command=self.confirm_exit).pack(side=tk.LEFT, padx=5)

        log_frame = ttk.LabelFrame(main_frame, text="执行日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, font=('Helvetica', 11), state=tk.DISABLED)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text['yscrollcommand'] = log_scroll.set
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=5)

        input_frame = ttk.LabelFrame(bottom_frame, text="指令输入/识别结果", padding=10)
        input_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.input_entry = ttk.Entry(input_frame, font=('Helvetica', 12))
        self.input_entry.pack(fill=tk.X, expand=True, ipady=4)
        self.input_entry.bind("<Return>", lambda e: self.start_analysis_from_input())

        result_frame = ttk.LabelFrame(bottom_frame, text="分析结果", padding=10)
        result_frame.pack(side=tk.LEFT, padx=10)
        self.result_labels = {}
        for i, header in enumerate(["地点", "设备", "状态"]):
            ttk.Label(result_frame, text=header, font=('Helvetica', 10, 'bold')).grid(row=0, column=i, padx=8)
            lbl = ttk.Label(result_frame, text="--", font=('Helvetica', 11), width=8, anchor='center')
            lbl.grid(row=1, column=i, padx=8)
            self.result_labels[header] = lbl

        self.update_log("系统初始化完成。")

    def start_or_force_analyze(self):
        engine_is_off = self.voice_recognizer.exit_flag or \
                        not getattr(self.voice_recognizer, '_audio_thread', None) or \
                        not self.voice_recognizer._audio_thread.is_alive()

        if engine_is_off:
            self.voice_recognizer.start()
            self.update_status("正在聆听...", "blue")
            self.record_button.config(text="停止并分析")
            self.update_log("语音引擎已启动，请说话...")
        else:
            self.record_button.config(state=tk.DISABLED)

            if getattr(self.voice_recognizer, '_is_recording', False):  # 假设_is_recording存在
                self.update_log("手动停止录音，开始识别...")
                self.update_status("正在识别...", "orange")
                self.voice_recognizer.stop_recording_manually()
            else:
                self.update_log("当前未在录音，无内容可立即分析。")
                self.finish_analysis()

    def on_recognition_result(self, text, error=None):
        """语音识别结果的回调函数。"""
        if self.root.winfo_exists():
            if error:
                self.update_log(f"语音识别错误: {error}", "red")
                self.update_status("识别出错", "red")
                self.finish_analysis()
                return

            if text:
                self.update_log(f"识别到指令: '{text}'")
                self.input_entry.delete(0, tk.END)  # 先清空，再插入识别结果
                self.input_entry.insert(0, text)
                self.start_analysis_from_input()
            else:
                self.update_log("未识别到有效指令。", "orange")
                self.finish_analysis()


    def start_analysis_from_input(self):
        """从输入框获取文本并开始分析"""
        if self.analyzing:
            return
        input_text = self.input_entry.get().strip()
        if not input_text:
            messagebox.showwarning("输入为空", "请输入或说出指令文本")
            self.finish_analysis()  # 如果输入为空，也结束流程并重置状态
            return

        self.analyzing = True
        self.update_status("正在分析...", "orange")
        self.update_log(f"开始分析指令: {input_text}")

        analysis_thread = threading.Thread(
            target=self.analyze_text_task,
            args=(input_text,),
            daemon=True
        )
        analysis_thread.start()

    def analyze_text_task(self, text):
        """在子线程中执行模型分析"""
        try:
            text_vec_bin = self.binary_vectorizer.transform([text])
            is_control = bool(self.binary_model.predict(text_vec_bin)[0])
            proba = self.binary_model.predict_proba(text_vec_bin)[0][1]

            if not is_control:
                result = {"is_control": False, "confidence": float(proba), "message": "非智能家居控制指令"}
            else:
                text_vec_multi = self.multi_vectorizer.transform([text])
                preds = self.multi_model.predict(text_vec_multi)[0]
                location = self.label_encoders["location"].inverse_transform([preds[0]])[0]
                device = self.label_encoders["device"].inverse_transform([preds[1]])[0]
                command = "on" if preds[2] == 1 else "off"

                device_key = f"{location}_{device}"
                self.device_states[device_key] = command

                result = {"is_control": True, "confidence": float(proba), "location": location, "device": device,
                          "command": command}

            self.root.after(0, self.handle_analysis_result, result)
        except Exception as e:
            self.root.after(0, self.handle_analysis_error, str(e))

    def handle_analysis_result(self, result):
        """在主线程处理分析结果并更新GUI"""
        if result.get("is_control"):
            loc, dev, cmd = result["location"], result["device"], result["command"]
            state_text = '打开' if cmd == 'on' else '关闭'
            log_msg = f"分析结果 -> 位置: {loc}, 设备: {dev}, 指令: {state_text} (置信度: {result['confidence']:.2%})"
            self.update_log(log_msg, "green")
            self.update_result_display(loc, dev, state_text)
            self.execute_control(loc, dev, cmd)
        else:
            log_msg = f"分析结果 -> {result.get('message', '非控制指令')} (置信度: {1 - result['confidence']:.2%})"
            self.update_log(log_msg, "orange")
            self.update_result_display("--", "--", "--")

        self.finish_analysis()  # 分析完成后调用，将清除输入框

    def handle_analysis_error(self, error_msg):
        """处理分析中的错误"""
        self.update_log(f"分析出错: {error_msg}", "red")
        self.finish_analysis()

    def execute_control(self, location, device, command):
        """模拟执行控制指令"""
        state_text = '打开' if command == 'on' else '关闭'
        self.update_log(f"执行成功: {location}的{device}已{state_text}。")

    def finish_analysis(self):
        """
        完成分析过程，停止语音引擎，并将UI完全重置为初始状态。
        """
        self.analyzing = False
        engine_is_running = getattr(self.voice_recognizer, '_audio_thread', None) is not None and \
                            self.voice_recognizer._audio_thread.is_alive() and \
                            not getattr(self.voice_recognizer, 'exit_flag', True)  # 默认为True表示已退出，防止误操作

        if engine_is_running:
            self.voice_recognizer.stop_engine()  # 假设这个方法会停止所有内部线程
            self.update_log("语音引擎已在此次分析后自动停止。", "gray")

        # 恢复UI到初始的“准备就绪”状态
        if self.root.winfo_exists():
            self.record_button.config(state=tk.NORMAL)
            self.update_status("准备就绪", "green")
            self.record_button.config(text="开始聆听")
            # --- 新增行：清除输入框中的文本 ---
            self.input_entry.delete(0, tk.END)
            # -------------------------------

    def update_status(self, text, color="black"):
        """更新顶部状态栏"""
        self.status_label.config(text=text, foreground=color)

    def update_log(self, text, color_tag=None):
        """向日志区域追加文本"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{text}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_result_display(self, location, device, state):
        """更新右下角的结果显示"""
        self.result_labels["地点"].config(text=location)
        self.result_labels["设备"].config(text=device)
        self.result_labels["状态"].config(text=state,
                                          foreground="green" if state == "打开" else "red" if state == "关闭" else "black")

    def confirm_exit(self):
        """确认退出并清理资源"""
        if messagebox.askyesno("退出系统", "确定要退出智能家居语音控制系统吗？"):
            self.update_log("正在关闭系统...")
            self.voice_recognizer.cleanup()
            self.root.destroy()


if __name__ == '__main__':
    # 确保'models'和'datasets'文件夹存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'models')
    datasets_dir = os.path.join(current_dir, 'datasets')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)

    required_model_files = ['binary_classifier.pkl', 'multiclass_classifier.pkl']
    missing_models = [f for f in required_model_files if not os.path.exists(os.path.join(models_dir, f))]

    user_dict_file = os.path.join(datasets_dir, 'user_dict.txt')
    if not os.path.exists(user_dict_file):
        missing_models.append('user_dict.txt (在datasets文件夹中)')

    if missing_models:
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        messagebox.showerror("文件缺失",
                             f"缺少以下必要的模型或数据文件:\n{', '.join(missing_models)}\n请确保它们存在于正确的路径下。")
        root.destroy()  # 关闭隐藏的窗口
    else:
        root = tk.Tk()
        app = SmartHomeVoiceControl(root)
        root.mainloop()
