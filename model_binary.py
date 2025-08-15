import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import joblib
import os


# ----------------------
# 1. 数据预处理模块
# ----------------------
class DataProcessor:
    @staticmethod
    def chinese_tokenizer(text):
        return jieba.lcut(text, cut_all=False, HMM=True)

    @staticmethod
    def load_dataset(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件未找到：{file_path}")
        df = pd.read_csv(file_path, encoding='utf-8')

        required_columns = {'input_text', 'label'}
        if not set(df.columns).issuperset(required_columns):
            raise ValueError("数据集必须包含input_text和label列")

        df["label"] = df["label"].astype(int)
        if not df["label"].isin([0, 1]).all():
            raise ValueError("label列必须为0或1的整数")

        return df["input_text"].tolist(), df["label"].values


# ----------------------
# 2. 控制类型分类器
# ----------------------
class ControlTypeClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=DataProcessor.chinese_tokenizer,
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.model = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.001,
            max_iter=1500,
            random_state=42,
            tol=1e-3
        )
        self.is_trained = False

    def train(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True

    def predict(self, text):
        if not self.is_trained:
            raise RuntimeError("模型未训练，无法预测")

        text_vec = self.vectorizer.transform([text])
        cmd = self.model.predict(text_vec)[0]

        # 只返回是否为智能家居控制类
        return {
            "is_smart_home_control": bool(cmd),
            "control_type": "智能家居控制类" if cmd == 1 else "非智能家居控制类"
        }

    def save_model(self, model_path="binary_classifier.pkl"):
        joblib.dump({
            "vectorizer": self.vectorizer,
            "model": self.model,
            "is_trained": self.is_trained
        }, model_path)
        print(f"模型已保存至：{model_path}")

    def load_model(self, model_path="binary_classifier.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到：{model_path}")

        state = joblib.load(model_path)
        self.vectorizer = state["vectorizer"]
        self.model = state["model"]
        self.is_trained = state["is_trained"]
        print(f"已从{model_path}加载模型")


# ----------------------
# 3. 业务逻辑接口
# ----------------------
class HomeControlClassifier:
    def __init__(self, train_dataset_path):
        self.train_dataset_path = train_dataset_path
        self.X_train, self.y_train = DataProcessor.load_dataset(self.train_dataset_path)
        self.classifier = ControlTypeClassifier()

    def train_model(self):
        print("开始控制类型分类模型训练...")
        self.classifier.train(self.X_train, self.y_train)
        self.classifier.save_model()
        print("训练完成，模型已保存")

    def predict(self, input_text):
        return self.classifier.predict(input_text)

    def user_input_interactive(self):
        print("\n=== 智能家居指令分析 ===")
        print("输入文本（如'打开卧室的空调'），输入'exit'退出")
        while True:
            user_text = input("请输入指令：")
            if user_text.lower() == 'exit':
                break
            result = self.predict(user_text)

            # 输出格式
            print(f"预测结果：{result['control_type']}")


# ----------------------
# 4. 主程序入口
# ----------------------
if __name__ == "__main__":
    DATASET_DIR = "datasets"
    os.makedirs(DATASET_DIR, exist_ok=True)
    TRAIN_DATASET = os.path.join(DATASET_DIR, "binary_dataset.csv")

    classifier = HomeControlClassifier(TRAIN_DATASET)
    classifier.train_model()
    classifier.user_input_interactive()