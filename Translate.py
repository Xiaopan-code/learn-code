import pandas as pd
import random
import os
import jieba  # Ensure jieba is imported for tokenization in augmentation

# Disable Jieba logging to avoid console clutter
import logging

jieba.setLogLevel(logging.ERROR)


class DatasetGenerator:
    def __init__(self, output_dir="datasets", seed=42):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        random.seed(seed)  # 设置随机种子，保证可复现性

        # 步骤1: 明确数据需求与类别定义
        # 智能家居控制的地点、设备和指令定义
        self.locations = ["厨房", "卧室", "客厅", "卫生间", "书房", "阳台"]
        self.devices = ["灯", "空调", "电视", "窗帘", "风扇", "音响", "插座"]
        self.commands_on = ["打开", "开启", "启动", "开"]
        self.commands_off = ["关闭", "关掉", "关", "停用"]

        # 同义词和句式变换模板
        self.on_synonyms = {
            "打开": ["开启", "启动", "开", "接通"],
            "开": ["打开", "开启", "启动", "接通"],
        }
        self.off_synonyms = {
            "关闭": ["关掉", "关", "切断", "停用"],
            "关": ["关闭", "关掉", "切断", "停用"],
        }
        self.request_phrases = ["请", "麻烦", "帮我", "能否", "可以"]
        self.ending_phrases = ["一下", "吗", "好吗", "行吗"]
        # 扩展噪声词和通用词，用于负例增强
        self.noise_words = ["那个", "嗯", "啊", "就是说", "然后", "接着", "对了", "哦", "啊呀", "好的", "嗯嗯",
                            "真的吗", "其实", "总之"]
        self.generic_verbs = ["查看", "询问", "了解", "知道", "告诉", "查找", "显示", "听", "看看", "说说"]
        self.generic_nouns = ["信息", "数据", "状态", "情况", "消息", "内容", "什么", "事情"]
        self.punctuation_marks = ["，", "。", "！", "？", "、"]

        # 负例生成模板（多样化，不含控制指令的词）
        self.negative_templates = {
            "weather": [
                "今天会下雨吗？", "明天天气怎么样？", "外面冷不冷？", "有太阳吗？",
                "今天需要带伞吗？", "明天的气温是多少？", "未来一周天气预报。",
            ],
            "chat": [
                "你最喜欢的电影是什么？", "有什么好玩的事情吗？", "讲个笑话。", "你叫什么名字？",
                "我有点无聊。", "我们聊聊天吧。", "你觉得人生有什么意义？", "你好", "谢谢", "再见", "很高兴认识你",
                "你来自哪里？"
            ],
            "shopping": [
                "帮我订一杯咖啡。", "哪里有卖这个的？", "我想买一件新衣服。", "这个多少钱？",
                "网上有优惠吗？", "推荐个好用的产品。", "我需要买些食物。", "附近有超市吗？"
            ],
            "general_noise": [  # 包含一些可能被误判的词，但整体语境非控制指令
                "把这个盒子打开。", "请你把门关上。", "我的心扉已经打开。", "这个文件你关了吗？",
                "关上电脑。", "关灯睡觉吧。", "把音乐关小点。", "打开我的微信。",
                "查看一下电量", "显示一下日期", "帮我看看这个", "了解一下情况", "告知我一声",
                "这个故事很精彩。", "今天的工作完成了。", "帮我处理一下邮件。", "我想学习编程。",
                "这本书很好看。", "请问现在几点了？", "查询一下最新新闻。"
            ]
        }

        print("数据集生成器初始化完成。")
        print(f"定义地点: {self.locations}")
        print(f"定义设备: {self.devices}")

    # 步骤2: 数据生成
    def _generate_positive_example(self, location, device, command_text, command_label):
        """生成一个家居控制语句的正例"""
        template_phrases = {
            "打开": [f"{location}的{device}{command_text}", f"{command_text}{location}的{device}"],
            "关闭": [f"{location}的{device}{command_text}", f"{command_text}{location}的{device}"],
        }
        # 基础组合
        sentences = []
        for phrase in template_phrases.get(command_text, []):
            sentences.append(phrase)

        # 示例：打开厨房的灯 -> location=厨房, device=灯, command=1
        return [{"text": s, "location": location, "device": device, "command": command_label} for s in sentences]

    # 步骤3: 数据增强
    def _augment_sentence(self, text, location=None, device=None, command_text=None, augmentation_type="positive"):
        """对句子进行多种增强操作"""
        augmented_texts = [text]  # 包含原始文本

        # 1. 同义词替换和句式变换 (主要针对正例)
        if augmentation_type == "positive":
            # 引入请求词
            for req in self.request_phrases:
                augmented_texts.append(f"{req}{text}")
                augmented_texts.append(f"{req}，{text}")

            # 引入结尾词
            for end in self.ending_phrases:
                augmented_texts.append(f"{text}{end}")
                augmented_texts.append(f"{text}，{end}")

            # 混合请求词和结尾词
            for req in self.request_phrases:
                for end in self.ending_phrases:
                    augmented_texts.append(f"{req}{text}{end}")

            # 替换动词同义词
            # 对于 '打开'
            if command_text == "开" and "打开" in text:
                for syn in self.on_synonyms["打开"]:
                    augmented_texts.append(text.replace("打开", syn))
            # 对于 '关闭'
            elif command_text == "关" and "关闭" in text:
                for syn in self.off_synonyms["关闭"]:
                    augmented_texts.append(text.replace("关闭", syn))
            # 对于 '开' (作为动词)
            elif command_text == "开" and "开" in text and not "打开" in text:
                for syn in self.on_synonyms["开"]:
                    # 避免替换掉“开启”中的“开”
                    if "开启" in text and syn == "开启":
                        pass  # 跳过
                    else:
                        augmented_texts.append(text.replace("开", syn, 1))  # 只替换第一个“开”
            # 对于 '关' (作为动词)
            elif command_text == "关" and "关" in text and not "关闭" in text:
                for syn in self.off_synonyms["关"]:
                    # 避免替换掉“关闭”中的“关”
                    if "关闭" in text and syn == "关闭":
                        pass  # 跳过
                    else:
                        augmented_texts.append(text.replace("关", syn, 1))  # 只替换第一个“关”
        elif augmentation_type == "negative":  # 对负例进行更复杂的增强以增加多样性
            words = list(jieba.cut(text))  # 使用jieba分词

            # 1. 随机插入噪声词 (提高插入次数和概率)
            if random.random() < 0.8 and self.noise_words:
                num_insertions = random.randint(1, 4)  # 插入1到4个噪声词
                temp_words = list(words)  # 复制一份，避免在循环中修改原列表
                for _ in range(num_insertions):
                    noise_word = random.choice(self.noise_words)
                    pos = random.randint(0, len(temp_words))
                    temp_words.insert(pos, noise_word)
                augmented_texts.append("".join(temp_words))

            # 2. 随机替换通用词
            if random.random() < 0.6:
                temp_words = list(words)
                found_and_replaced = False
                for i, word in enumerate(temp_words):
                    if word in ["看", "查", "问", "说", "想", "做"] and self.generic_verbs:  # 扩展通用动词
                        temp_words[i] = random.choice(self.generic_verbs)
                        augmented_texts.append("".join(temp_words))
                        found_and_replaced = True
                        break  # 只替换一个
                if not found_and_replaced and self.generic_nouns:  # 尝试替换名词
                    for i, word in enumerate(temp_words):
                        if word in ["事情", "东西", "情况"] and self.generic_nouns:  # 扩展通用名词
                            temp_words[i] = random.choice(self.generic_nouns)
                            augmented_texts.append("".join(temp_words))
                            break

            # 3. 随机删除非关键词或标点
            if random.random() < 0.5:
                temp_words = list(words)
                removable_elements = [w for w in temp_words if
                                      w in ["吗", "呢", "啊", "了", "吧", "的", "，", "。", "！", "？"]]
                if removable_elements:
                    element_to_remove = random.choice(removable_elements)
                    temp_words.remove(element_to_remove)
                    augmented_texts.append("".join(temp_words))

            # 4. 随机重排相邻词 (小范围)
            if random.random() < 0.3 and len(words) > 2:
                temp_words = list(words)
                idx1 = random.randint(0, len(temp_words) - 2)
                idx2 = idx1 + 1
                temp_words[idx1], temp_words[idx2] = temp_words[idx2], temp_words[idx1]
                augmented_texts.append("".join(temp_words))

        # 过滤掉重复或空的句子
        return list(set([t.strip() for t in augmented_texts if t.strip()]))

    # 步骤4: 数据平衡处理
    def _balance_data(self, df_binary, df_multi):
        print("\n开始数据平衡处理...")

        # **二分类模型平衡: 正负样本1:1**
        df_pos = df_binary[df_binary['label'] == 1].copy()  # 使用.copy()避免SettingWithCopyWarning
        df_neg = df_binary[df_binary['label'] == 0].copy()

        pos_count = df_pos.shape[0]
        neg_count = df_neg.shape[0]
        print(f"二分类原始正例: {pos_count}, 负例: {neg_count}")

        # 找到两个类别中数量较多的那个作为目标数量
        target_count = max(pos_count, neg_count)
        # 增加一个缓冲，确保平衡后即使有去重也能达到目标
        target_count = int(target_count * 1.1)  # 增加10%的缓冲

        # 定义一个函数，用于对少数类进行过采样，并生成新的独特增强样本
        def oversample_and_augment(df_minority, current_count, target_count, augmentation_type):
            if current_count >= target_count:
                return df_minority  # 已经足够，无需过采样

            needed_samples = target_count - current_count
            augmented_samples = []

            # 使用一个集合来跟踪已经添加的文本，确保唯一性
            existing_texts = set(df_minority['text'].tolist())

            # 策略：循环从现有样本中抽取，并对它们进行多次增强，直到达到目标数量
            # 引入一个最大尝试次数，避免无限循环
            max_attempts_per_sample = 10  # 对每个样本尝试增强更多次

            temp_df_minority = df_minority.copy()  # 复制一份用于迭代，避免修改原始df_minority

            # 优先从现有少数类样本中随机选择并增强
            while needed_samples > 0 and not temp_df_minority.empty:
                sample_row = temp_df_minority.sample(1, random_state=random.randint(0, 10000)).iloc[0]
                text_to_augment = sample_row['text']

                num_generated_from_this_sample = 0
                for _ in range(max_attempts_per_sample):  # 对同一个原始样本尝试多次增强
                    new_augmented_texts = self._augment_sentence(text_to_augment, augmentation_type=augmentation_type)

                    # 尝试将每个新生成的文本添加到结果中，确保唯一性
                    for new_text in new_augmented_texts:
                        if new_text not in existing_texts:
                            augmented_samples.append({'text': new_text, 'label': sample_row['label']})
                            existing_texts.add(new_text)
                            needed_samples -= 1
                            num_generated_from_this_sample += 1
                            if needed_samples <= 0:
                                break
                    if needed_samples <= 0:
                        break

            # 如果增强仍然不足，则进行简单复制 (确保最终数量达标)
            if needed_samples > 0:
                print(
                    f"警告: 增强不足以达到目标数量 ({target_count})，正在进行简单复制以补充 {augmentation_type} 样本 ({needed_samples} 份)。")
                # 需要确保replace=True，才能进行复制
                if current_count > 0:  # 只有当有原始样本时才能复制
                    # 复制时也进行去重，避免复制出已经存在的
                    additional_copied_samples = df_minority.sample(needed_samples, replace=True,
                                                                   random_state=random.randint(0, 10000))
                    # 过滤掉已经存在的文本
                    additional_copied_samples = additional_copied_samples[
                        ~additional_copied_samples['text'].isin(existing_texts)]
                    augmented_samples.extend(additional_copied_samples.to_dict('records'))
                else:
                    print(f"警告: 无法对 {augmentation_type} 样本进行简单复制，因为原始数量为0。")

            df_augmented = pd.DataFrame(augmented_samples)
            # 将原始df和新生成的独特增强样本合并
            return pd.concat([df_minority, df_augmented], ignore_index=True).drop_duplicates(
                subset=['text']).reset_index(drop=True)

        # 对正例进行过采样和增强
        df_pos_balanced = oversample_and_augment(df_pos, pos_count, target_count, augmentation_type="positive")

        # 对负例进行过采样和增强
        df_neg_balanced = oversample_and_augment(df_neg, neg_count, target_count, augmentation_type="negative")

        # 最终合并
        df_binary_balanced = pd.concat([df_pos_balanced, df_neg_balanced], ignore_index=True)
        df_binary_balanced = df_binary_balanced.drop_duplicates(subset=['text']).reset_index(drop=True)  # 最终去重

        print(
            f"二分类平衡后正例: {df_binary_balanced[df_binary_balanced['label'] == 1].shape[0]}, 负例: {df_binary_balanced[df_binary_balanced['label'] == 0].shape[0]}")

        # 多分类模型平衡 (保持原逻辑，因为这部分主要是针对location/device/command内部的平衡)
        print("\n多分类模型类别分布检查与平衡:")
        # 控制指令平衡 (0/1)
        cmd_counts = df_multi['command'].value_counts()
        print(f"指令类别原始分布:\n{cmd_counts}")
        if len(cmd_counts) == 2:
            max_cmd_count = cmd_counts.max()
            min_cmd_label = cmd_counts.idxmin()
            min_cmd_count = cmd_counts.min()

            if min_cmd_count < max_cmd_count:
                df_minority_cmd = df_multi[df_multi['command'] == min_cmd_label].copy()
                # 对少数类进行过采样和增强
                needed_aug_times = (max_cmd_count // min_cmd_count) - 1
                if needed_aug_times > 0:
                    for _ in range(needed_aug_times):
                        # 对每次抽样的样本都进行增强
                        for _, row in df_minority_cmd.sample(frac=0.5, random_state=random.randint(0, 10000),
                                                             replace=True).iterrows():
                            text = row['input_text']  # 统一使用 input_text
                            location = row['location']
                            device = row['device']
                            command = row['command']  # 0 or 1
                            command_text = "开" if command == 1 else "关"

                            augmented_texts = self._augment_sentence(text, location, device, command_text,
                                                                     augmentation_type="positive")
                            for aug_text in augmented_texts:
                                df_multi = pd.concat([df_multi, pd.DataFrame([{
                                    "input_text": aug_text,  # 统一使用 input_text 作为列名
                                    "location": location,
                                    "device": device,
                                    "command": command
                                }])], ignore_index=True)
                # 随机抽取以达到精确平衡 (如果增强仍然不足)
                remaining = max_cmd_count - df_multi[df_multi['command'] == min_cmd_label].shape[0]
                if remaining > 0:
                    df_multi = pd.concat([df_multi, df_minority_cmd.sample(remaining, replace=True,
                                                                           random_state=random.randint(0, 10000))],
                                         ignore_index=True)
            print(f"指令类别平衡后分布:\n{df_multi['command'].value_counts()}")

        # 设备位置和设备名称平衡
        for col in ['location', 'device']:
            counts = df_multi[col].value_counts()
            print(f"\n{col} 类别原始分布:\n{counts}")
            max_count = counts.max()
            for label, count in counts.items():
                if count < max_count:
                    df_minority = df_multi[df_multi[col] == label].copy()
                    needed_aug_times = (max_count // count) - 1
                    if needed_aug_times > 0:
                        for _ in range(needed_aug_times):
                            for _, row in df_minority.sample(frac=0.5,
                                                             random_state=random.randint(0, 10000),
                                                             replace=True).iterrows():
                                text = row['input_text']  # 统一使用 input_text
                                location = row['location']
                                device = row['device']
                                command = row['command']
                                command_text = "开" if command == 1 else "关"

                                augmented_texts = self._augment_sentence(text, location, device, command_text,
                                                                         augmentation_type="positive")
                                for aug_text in augmented_texts:
                                    df_multi = pd.concat([df_multi, pd.DataFrame([{
                                        "input_text": aug_text,  # 统一使用 input_text 作为列名
                                        "location": location,
                                        "device": device,
                                        "command": command
                                    }])], ignore_index=True)
                    remaining = max_count - df_multi[df_multi[col] == label].shape[0]
                    if remaining > 0:
                        df_multi = pd.concat([df_multi, df_minority.sample(remaining, replace=True,
                                                                           random_state=random.randint(0, 10000))],
                                             ignore_index=True)
            print(f"{col} 类别平衡后分布:\n{df_multi[col].value_counts()}")

        # 确保平衡后没有完全重复的行
        df_multi = df_multi.drop_duplicates(subset=['input_text', 'location', 'device', 'command']).reset_index(
            drop=True)

        print("\n数据平衡处理完成。")
        return df_binary_balanced, df_multi

    def generate_datasets(self, num_positive_templates=1, num_negative_samples_per_template=200,  # 再次大幅度增加每个负例模板的生成数量
                          num_augmentation_per_positive=5):

        all_multi_data = []
        all_binary_positive_data = []

        # 步骤2: 正例生成 (家居控制语句)
        print("开始生成正例...")
        for loc in self.locations:
            for dev in self.devices:
                for cmd_text, cmd_label in zip(["打开", "关闭"], [1, 0]):
                    base_positive_samples = self._generate_positive_example(loc, dev, cmd_text, cmd_label)
                    for sample_dict in base_positive_samples:
                        text = sample_dict["text"]
                        all_multi_data.append(sample_dict)
                        all_binary_positive_data.append({"text": text, "label": 1})

                        # 步骤3: 数据增强 (针对正例)
                        # 确保 positive augmentation 每次都产生多个变体
                        augmented_texts = self._augment_sentence(text, loc, dev, cmd_text, augmentation_type="positive")
                        for aug_text in augmented_texts:
                            all_multi_data.append({
                                "input_text": aug_text,  # 统一列名
                                "location": loc,
                                "device": dev,
                                "command": cmd_label
                            })
                            all_binary_positive_data.append({"text": aug_text, "label": 1})

        # 将多分类数据的 'text' 列名统一为 'input_text'
        df_multi_raw = pd.DataFrame(all_multi_data)
        if 'text' in df_multi_raw.columns and 'input_text' not in df_multi_raw.columns:
            df_multi_raw.rename(columns={'text': 'input_text'}, inplace=True)
        df_multi_raw = df_multi_raw.drop_duplicates(subset=['input_text', 'location', 'device', 'command'])

        df_binary_positive = pd.DataFrame(all_binary_positive_data).drop_duplicates(subset=['text'])
        print(f"初步生成正例总数 (二分类): {len(df_binary_positive)}")
        print(f"初步生成多分类正例总数: {len(df_multi_raw)}")

        # 步骤2: 负例生成 (非家居控制语句)
        print("\n开始生成负例...")
        all_binary_negative_data = []
        for category, templates in self.negative_templates.items():
            for template in templates:
                # 对负例也进行增强，并循环收集所有增强后的文本
                # 这里确保每次增强都能产生多个独特变体
                for _ in range(num_negative_samples_per_template):  # 每个模板生成多份，再对每份进行增强
                    augmented_negative_texts = self._augment_sentence(template, augmentation_type="negative")
                    for aug_text in augmented_negative_texts:
                        all_binary_negative_data.append({"text": aug_text, "label": 0})

        df_binary_negative = pd.DataFrame(all_binary_negative_data).drop_duplicates(subset=['text'])
        print(f"初步生成负例总数 (二分类): {len(df_binary_negative)}")

        # 合并二分类数据
        df_binary = pd.concat([df_binary_positive, df_binary_negative], ignore_index=True)
        # 再次去重，因为增强可能产生重复
        df_binary = df_binary.drop_duplicates(subset=['text']).reset_index(drop=True)
        print(f"合并后二分类数据集总数 (去重): {len(df_binary)}")

        # 步骤4: 数据平衡处理
        df_binary_balanced, df_multi_balanced = self._balance_data(df_binary, df_multi_raw)

        # 步骤5: (人工审核 - 代码不实现，但需在实际流程中执行)
        print("\n数据生成完成。")
        print(f"最终二分类数据集大小: {len(df_binary_balanced)}")
        print(f"最终多分类数据集大小: {len(df_multi_balanced)}")

        # 保存数据集
        df_binary_balanced.to_csv(os.path.join(self.output_dir, "binary_dataset.csv"), index=False, encoding="utf-8")
        df_multi_balanced.to_csv(os.path.join(self.output_dir, "multiclass_dataset.csv"), index=False, encoding="utf-8")
        print(f"数据集已保存到 '{self.output_dir}' 目录。")

        return df_binary_balanced, df_multi_balanced


# 主程序入口
if __name__ == "__main__":
    generator = DatasetGenerator(output_dir="datasets")

    # 生成数据集
    binary_df, multi_df = generator.generate_datasets(
        num_positive_templates=1,
        num_negative_samples_per_template=200  # 再次大幅度增加每个负例模板的生成数量
    )

    print("\n--- 二分类数据集示例 (前5行) ---")
    print(binary_df.head())
    print("\n--- 多分类数据集示例 (前5行) ---")
    print(multi_df.head())

    print("\n--- 二分类数据集标签分布 ---")
    print(binary_df['label'].value_counts())
    print("\n--- 多分类数据集指令分布 ---")
    print(multi_df['command'].value_counts())
    print("\n--- 多分类数据集地点分布 ---")
    print(multi_df['location'].value_counts())
    print("\n--- 多分类数据集设备分布 ---")
    print(multi_df['device'].value_counts())