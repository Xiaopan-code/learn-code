import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random  # 引入random模块


# ----------------------
# 1. 数据预处理模块（添加标签编码和Jieba词典加载）
# ----------------------
class DataProcessor:
    # 静态变量，用于存储jieba是否已加载自定义词典的状态
    _jieba_user_dict_loaded = False
    _user_dict_path = None  # 用于记录加载的词典路径

    @staticmethod
    def chinese_tokenizer(text):
        """中文分词器，必须与模型训练时保持一致"""
        # 注意：cut_all=False (精确模式), HMM=True (使用HMM模型识别新词)
        return jieba.lcut(text, cut_all=False, HMM=True)

    @staticmethod
    def load_jieba_user_dict(dict_path):
        """
        加载Jieba自定义词典。
        建议将智能家居相关的设备名、位置名等添加到词典中，确保它们被正确分词。
        """
        if not DataProcessor._jieba_user_dict_loaded or DataProcessor._user_dict_path != dict_path:
            if os.path.exists(dict_path):
                jieba.load_userdict(dict_path)
                DataProcessor._jieba_user_dict_loaded = True
                DataProcessor._user_dict_path = dict_path
                print(f"Jieba自定义词典已从 '{dict_path}' 加载。")
            else:
                print(f"警告：Jieba自定义词典文件 '{dict_path}' 未找到，将使用默认词典。")
                DataProcessor._jieba_user_dict_loaded = False

    @staticmethod
    def load_multiclass_dataset(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件未找到：{file_path}")
        df = pd.read_csv(file_path, encoding='utf-8')

        required_columns = {'input_text', 'location', 'device', 'command'}
        if not set(df.columns).issuperset(required_columns):
            raise ValueError("数据集必须包含input_text、location、device、command列")

        # --------------------------
        # 新增：处理input_text缺失值
        # --------------------------
        # 检测input_text列的缺失值
        if df['input_text'].isna().any():
            missing_count = df['input_text'].isna().sum()
            print(f"警告：input_text列存在 {missing_count} 个缺失值，将自动删除这些行。")
            # 删除input_text为空的行
            df = df.dropna(subset=['input_text']).reset_index(drop=True)
            # 如果删除后数据为空，抛出错误
            if len(df) == 0:
                raise ValueError("删除缺失值后数据集为空，请检查原始数据。")

        # 类型校验：确保command为整数
        df["command"] = df["command"].astype(int)
        if not df["command"].isin([0, 1]).all():
            raise ValueError("command列必须为0或1的整数")

        # 标签编码（字符串→整数）
        le_location = LabelEncoder()  # 设备位置编码器
        le_device = LabelEncoder()  # 设备名称编码器
        df["location_encoded"] = le_location.fit_transform(df["location"])
        df["device_encoded"] = le_device.fit_transform(df["device"])

        # 保存编码器，用于预测时反向转换
        label_encoders = {
            "location": le_location,
            "device": le_device
        }

        # 返回编码后的标签（location和device为整数，command保持整数）
        return (
            df["input_text"].tolist(),
            df[["location_encoded", "device_encoded", "command"]].values,
            label_encoders,  # 返回编码器
            df  # 返回原始DataFrame，以便后续扩展
        )

    @staticmethod
    def generate_diverse_commands(location, device, command_text):
        """
        生成多样化的“开”或“关”指令。

        Args:
            location: 设备位置 (例如 "卧室", "客厅").
            device: 设备名称 (例如 "灯", "空调").
            command_text: "开" 或 "关".

        Returns:
            包含多样化指令的列表。
        """
        commands = []
        if command_text == "开":
            open_phrases = ["打开", "开启", "启动", "开一下", "把...打开", "让...亮起来", "开", "请打开", "劳驾打开",
                            "请把", "帮我打开"]
            commands.append(f"{random.choice(open_phrases)}{location}的{device}")
            commands.append(f"把{location}的{device}{random.choice(open_phrases[0:4])}")
            commands.append(f"{location}{device}{random.choice(open_phrases[0:3])}")
            commands.append(f"请问可以{random.choice(open_phrases)}{location}的{device}吗")
            commands.append(f"请{random.choice(open_phrases)}{location}的{device}")
            commands.append(f"{random.choice(open_phrases[0:3])}{location}的{device}一下")
            commands.append(f"{location}的{device}{random.choice(open_phrases[0:3])}一下")
        elif command_text == "关":
            close_phrases = ["关闭", "关掉", "关", "停用", "把...关了", "请关闭", "劳驾关闭", "请把", "帮我关闭"]
            commands.append(f"{random.choice(close_phrases)}{location}的{device}")
            commands.append(f"把{location}的{device}{random.choice(close_phrases[0:3])}")
            commands.append(f"{location}{device}{random.choice(close_phrases[0:3])}")
            commands.append(f"请问可以{random.choice(close_phrases)}{location}的{device}吗")
            commands.append(f"请{random.choice(close_phrases)}{location}的{device}")
            commands.append(f"{random.choice(close_phrases[0:3])}{location}的{device}一下")
            commands.append(f"{location}的{device}{random.choice(close_phrases[0:3])}一下")
        return commands

    @staticmethod
    def expand_and_save_dataset(df_original, file_path):
        """
        扩展数据集，添加多样化的指令，并保存覆盖原文件。

        Args:
            df_original: Pandas DataFrame，包含原始数据集。
            file_path: 数据集文件的路径。
        """
        expanded_data = []
        # 使用set来避免在本次扩展中产生完全重复的指令
        existing_inputs = set(df_original["input_text"].tolist())

        # 添加原始指令 (确保所有原始指令都包含在内)
        for index, row in df_original.iterrows():
            expanded_data.append(row.to_dict())

        for index, row in df_original.iterrows():  # 再次遍历原始数据以进行扩展
            location = row["location"]
            device = row["device"]
            command_code = row["command"]
            command_text = "开" if command_code == 1 else "关"

            # 生成多样化的指令并添加
            diverse_commands = DataProcessor.generate_diverse_commands(location, device, command_text)
            for diverse_command in diverse_commands:
                if diverse_command not in existing_inputs:  # 避免添加本次循环中生成但已经存在的重复指令
                    expanded_data.append({
                        "input_text": diverse_command,
                        "location": location,
                        "device": device,
                        "command": command_code,
                    })
                    existing_inputs.add(diverse_command)  # 将新添加的指令也加入到已存在的集合中

        df_expanded = pd.DataFrame(expanded_data)
        # 将扩展后的数据集保存覆盖原文件
        df_expanded.to_csv(file_path, index=False, encoding='utf-8')
        print(f"数据集已扩展并保存至：{file_path} (新行数: {len(df_expanded)})")
        return df_expanded


# ----------------------
# 2. 迭代式多分类模型（支持标签编码）
# ----------------------
class IterativeMulticlassClassifier:
    def __init__(self, label_encoders):
        self.label_encoders = label_encoders  # 保存编码器
        self.vectorizer = TfidfVectorizer(
            tokenizer=DataProcessor.chinese_tokenizer,  # 使用静态分词器
            max_features=1000,  # 适当增加特征数量，100可能太少，建议通过交叉验证调整
            ngram_range=(1, 2),  # 考虑单字和双字词，有助于捕获短语。可尝试(1,3)
            min_df=2,  # 忽略在少于2个文档中出现的词
            max_df=0.8,  # 忽略在超过80%文档中出现的词
            # stop_words=your_chinese_stop_words_list # 可选：加入中文停用词列表以提高特征质量
        )
        self.model = MultiOutputClassifier(
            estimator=SGDClassifier(
                loss='log_loss',  # 使用对数损失，支持概率预测
                penalty='l2',  # L2正则化，防止过拟合
                alpha=0.001,  # 正则化强度，建议通过交叉验证调整
                max_iter=1500,  # 最大迭代次数，建议根据收敛情况和数据集大小调整
                random_state=42,
                tol=1e-3,  # 容忍度，损失变化小于此值则停止，建议通过交叉验证调整
            ),
            n_jobs=-1  # 使用所有可用CPU核心进行并行处理
        )
        self.is_trained = False
        # self._jieba_user_dict_path_at_save = None # 可选：记录模型保存时使用的jieba词典路径

    def initialize_training(self, X_train, y_train):
        print("正在进行特征向量化 (TfidfVectorizer.fit_transform)...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        # 确保y_train为二维数组（Scikit-learn要求）
        print("正在训练多分类模型 (MultiOutputClassifier.fit)...")
        self.model.fit(X_train_vec, np.array(y_train))
        self.is_trained = True
        print("模型训练完成。")
        # 打印学习到的词汇表数量
        print(f"Vectorizer学习到的特征数量: {len(self.vectorizer.vocabulary_)}")

    def incremental_training(self, X_new, y_new):
        if not self.is_trained:
            raise RuntimeError("请先完成首次全量训练")

        print("正在进行增量特征向量化 (TfidfVectorizer.transform)...")
        X_new_vec = self.vectorizer.transform(X_new)
        print("正在进行模型增量训练 (MultiOutputClassifier.partial_fit)...")
        self.model.partial_fit(X_new_vec, np.array(y_new))  # 增量训练
        print("模型增量训练完成。")

    def predict_intent(self, text):
        if not self.is_trained:
            raise RuntimeError("模型未训练，无法预测")

        text_vec = self.vectorizer.transform([text])
        # 预测结果是编码后的整数
        predictions_encoded = self.model.predict(text_vec)[0]
        loc_encoded, dev_encoded, cmd = predictions_encoded[0], predictions_encoded[1], predictions_encoded[2]

        # 反向转换编码后的预测结果
        location = self.label_encoders["location"].inverse_transform([loc_encoded])[0]
        device = self.label_encoders["device"].inverse_transform([dev_encoded])[0]

        # 直接映射command为具体操作指令
        command_mapping = {
            1: "打开",
            0: "关闭",
        }
        command_text = command_mapping.get(cmd, "未知操作")  # 默认显示"未知操作"

        return {
            "设备位置": location,
            "设备名称": device,
            "控制指令（打开/关闭）": command_text
        }

    def save_model(self, model_path="multiclass_classifier.pkl"):
        joblib.dump({
            "vectorizer": self.vectorizer,
            "model": self.model,
            "is_trained": self.is_trained,
            "label_encoders": self.label_encoders,  # 保存编码器
        }, model_path)
        print(f"模型已保存至：{model_path}")

    def load_model(self, model_path="multiclass_classifier.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到：{model_path}")

        state = joblib.load(model_path)
        self.vectorizer = state["vectorizer"]
        self.model = state["model"]
        self.is_trained = state["is_trained"]
        self.label_encoders = state["label_encoders"]  # 加载编码器
        print(f"已从{model_path}加载模型")


# ----------------------
# 3. 业务逻辑接口（整合编码器）
# ----------------------
class HomeControlMulticlassClassifier:
    def __init__(self, train_dataset_path, user_dict_path=None):
        self.train_dataset_path = train_dataset_path

        if user_dict_path:
            DataProcessor.load_jieba_user_dict(user_dict_path)
        else:
            print("未指定Jieba自定义词典路径，将使用Jieba默认词典。")

        # 加载数据集，并获取原始DataFrame以便扩展
        self.X_train, self.y_train, self.label_encoders, self.df_original = DataProcessor.load_multiclass_dataset(
            self.train_dataset_path
        )
        self.classifier = IterativeMulticlassClassifier(label_encoders=self.label_encoders)

    def train_model(self):
        print("\n=== 开始多分类模型首次训练 ===")

        # --- 数据集扩展和原地保存逻辑 ---
        print("正在扩展数据集并覆盖原始文件...")
        self.df_original = DataProcessor.expand_and_save_dataset(self.df_original, self.train_dataset_path)

        self.X_train, self.y_train, _, _ = DataProcessor.load_multiclass_dataset(self.train_dataset_path)
        # ------------------------------------

        self.classifier.initialize_training(self.X_train, self.y_train)
        self.classifier.save_model()
        print("训练完成，模型已保存。")

    def update_model(self, new_dataset_path):
        print(f"\n=== 加载增量数据集：{new_dataset_path} ===")
        X_new, y_new, _, _ = DataProcessor.load_multiclass_dataset(new_dataset_path)

        self.classifier.load_model()
        self.classifier.incremental_training(X_new, y_new)
        self.classifier.save_model()
        print("模型已更新。")

    def predict(self, input_text):
        if not self.classifier.is_trained:
            try:
                self.classifier.load_model()
            except FileNotFoundError:
                print("模型未训练且模型文件不存在，请先训练模型。")
                return {"设备位置": "N/A", "设备名称": "N/A", "控制指令（打开/关闭）": "N/A"}
            except Exception as e:
                print(f"加载模型时发生错误: {e}")
                return {"设备位置": "N/A", "设备名称": "N/A", "控制指令（打开/关闭）": "N/A"}

        return self.classifier.predict_intent(input_text)

    def user_input_interactive(self):
        print("\n=== 智能家居指令分析 ===")
        print("输入文本（如'打开卧室的空调'），输入'exit'退出")
        while True:
            user_text = input("请输入指令：")
            if user_text.lower() == 'exit':
                break

            if not DataProcessor._jieba_user_dict_loaded and DataProcessor._user_dict_path:
                DataProcessor.load_jieba_user_dict(DataProcessor._user_dict_path)

            result = self.predict(user_text)

            print(f"设备位置：{result['设备位置']}")
            print(f"设备名称：{result['设备名称']}")
            print(f"控制指令（打开/关闭）：{result['控制指令（打开/关闭）']}\n")


# ----------------------
# 4. 主程序入口
# ----------------------
if __name__ == "__main__":
    DATASET_DIR = "datasets"
    os.makedirs(DATASET_DIR, exist_ok=True)
    TRAIN_DATASET = os.path.join(DATASET_DIR, "multiclass_dataset.csv")
    USER_DICT_PATH = os.path.join(DATASET_DIR, "user_dict.txt")  # 自定义词典路径


    if not os.path.exists(TRAIN_DATASET):
        print(f"错误：训练数据集文件 '{TRAIN_DATASET}' 未找到。请先手动创建此文件并包含初始数据。")
        exit()
    if not os.path.exists(USER_DICT_PATH):
        print(f"错误：Jieba自定义词典文件 '{USER_DICT_PATH}' 未找到。请先手动创建此文件。")
        exit()

    classifier_app = HomeControlMulticlassClassifier(TRAIN_DATASET, USER_DICT_PATH)

    # 首次训练模型（将包含数据集扩展逻辑）
    classifier_app.train_model()

    # 交互式测试
    classifier_app.user_input_interactive()