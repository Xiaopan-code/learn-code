import jieba

class DataProcessor:
    @staticmethod
    def chinese_tokenizer(text):
        """中文分词器"""
        return list(jieba.cut(text))