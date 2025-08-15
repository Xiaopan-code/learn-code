import threading
import time
import numpy as np
import pyaudio
import speech_recognition as sr
import noisereduce as nr
import collections
from aip import AipSpeech
import os

# 请替换为你自己的百度语音识别 API Key、Secret Key 和 App ID
APP_ID = '119211234'
API_KEY = 'lHzADm8qt0FqB6wyMyo82HMt'
SECRET_KEY = 'BaDF0CPRuuq4AOpK7ljQniiqUwa816FX'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


class VoiceRecognitionCore:
    """
    语音识别核心引擎（重构版）
    - 使用回调函数将识别结果实时传递给调用者。
    - 在一次识别后自动重新进入聆听状态，而不是退出。
    """

    def __init__(self, sample_rate=16000, energy_threshold=2000,
                 silence_timeout=2.5, denoise_level=0.8, recognition_callback=None):
        # --- 音频参数配置 ---
        self.sample_rate = sample_rate
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1

        # --- 语音检测参数 ---
        self.energy_threshold = energy_threshold
        self.silence_timeout = silence_timeout
        self.denoise_level = denoise_level
        self.exit_flag = False

        # --- 回调函数 ---
        self.recognition_callback = recognition_callback

        # --- 状态与控制事件 ---
        self._frames = []
        self._is_recording = False
        self._manual_stop_recording_event = threading.Event()

        # --- 线程对象 ---
        self._audio_thread = None

        # --- 初始化核心组件 ---
        self._audio = pyaudio.PyAudio()
        self._recognizer = sr.Recognizer()
        self._energy_history = collections.deque(maxlen=20)
        self.last_recognized_text = ""

    def _is_speech(self, audio_chunk):
        energy = np.abs(np.frombuffer(audio_chunk, dtype=np.int16)).mean()
        is_speech_by_energy = energy > self.energy_threshold

        if len(self._energy_history) > 0:
            min_energy = max(min(self._energy_history), 1e-5)
            energy_ratio = energy / min_energy
            is_speech_by_ratio = energy_ratio > 2.0
        else:
            is_speech_by_ratio = False

        self._energy_history.append(energy)
        return is_speech_by_energy or is_speech_by_ratio

    def _recognize_and_process(self):
        try:
            stream = self._audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
        except IOError:
            print("错误：无法打开麦克风设备。程序退出。")
            if self.recognition_callback:
                self.recognition_callback(None, "错误：无法打开麦克风设备。")
            self.exit_flag = True
            return

        print("语音识别系统启动，持续聆听中...")

        while not self.exit_flag:
            self._frames = []
            self._manual_stop_recording_event.clear()
            self._is_recording = False
            last_sound_time = time.time()

            # 1. 等待语音开始
            while not self.exit_flag:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    if self._is_speech(data):
                        print("检测到语音，开始录音...(超时或手动停止)")
                        self._is_recording = True
                        self._frames = [data]
                        last_sound_time = time.time()
                        break
                except Exception:
                    time.sleep(0.1)

            if self.exit_flag: break

            # 2. 录音阶段
            while self._is_recording and not self.exit_flag:
                if self._manual_stop_recording_event.is_set():
                    print("用户手动停止录音。")
                    break
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self._frames.append(data)
                    if self._is_speech(data):
                        last_sound_time = time.time()

                    if time.time() - last_sound_time > self.silence_timeout:
                        print(f"静音超时({self.silence_timeout}秒)，停止录音。")
                        break
                except Exception as e:
                    print(f"读取音频错误: {e}")
                    self._is_recording = False
                    break

            self._is_recording = False

            # 3. 处理已录制的音频
            if self._frames and not self.exit_flag:
                print("录音结束，正在处理和识别...")
                try:
                    raw_data = b''.join(self._frames)
                    audio_np = np.frombuffer(raw_data, dtype=np.int16)
                    reduced_noise = nr.reduce_noise(y=audio_np, sr=self.sample_rate, prop_decrease=self.denoise_level)

                    result = client.asr(reduced_noise.tobytes(), 'pcm', self.sample_rate, {'dev_pid': 1537})

                    if result and result['err_no'] == 0:
                        text = result['result'][0]
                        print(f"识别结果: {text}")
                        self.last_recognized_text = text
                        self.save_to_file(text)  # 保留文件保存功能
                        if self.recognition_callback:
                            self.recognition_callback(text)  # <<< 核心改动：调用回调
                    else:
                        error_msg = result.get('err_msg', '未知识别错误')
                        print(f"百度语音识别错误: {error_msg}")
                        if self.recognition_callback:
                            self.recognition_callback(None, error_msg)
                except Exception as e:
                    print(f"处理音频时出错: {e}")
                    if self.recognition_callback:
                        self.recognition_callback(None, str(e))

            print("\n系统重置，返回聆听状态...")

        # 关闭音频流
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()

    def save_to_file(self, text):
        try:
            with open('voice_command.txt', 'w', encoding='utf-8') as file:
                file.write(text)
            print(f"已将识别结果保存到 voice_command.txt")
        except Exception as e:
            print(f"保存文件时出错: {e}")

    def start(self):
        if self._audio_thread and self._audio_thread.is_alive():
            print("系统已在运行中。")
            return

        self.exit_flag = False
        self._audio_thread = threading.Thread(target=self._recognize_and_process, daemon=True)
        self._audio_thread.start()
        print("语音识别核心线程已启动。")

    def stop_recording_manually(self):
        """手动停止单次录音，进入识别流程"""
        if self._is_recording:
            self._manual_stop_recording_event.set()

    def stop_engine(self):
        """完全停止语音识别引擎"""
        print("正在停止语音识别引擎...")
        self.exit_flag = True
        if self._is_recording:
            self._manual_stop_recording_event.set()  # 确保录音循环退出
        if self._audio_thread:
            self._audio_thread.join(timeout=2.0)
        print("语音识别引擎已停止。")

    def cleanup(self):
        self.stop_engine()
        if self._audio:
            self._audio.terminate()
        print("系统资源已释放。")