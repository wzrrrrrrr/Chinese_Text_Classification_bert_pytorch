import pandas as pd
import time

from transformers import MarianMTModel, MarianTokenizer



class TextDataAugmentation:
    def __init__(self, data_path, augment_ratio=0.2,
                 target_labels={0: '科技', 1: '娱乐', 2: '时事', 3: '社会讨论/话题', 4: '时政', 5: '科普', 6: '经济',
                                7: '体育'}):
        self.data_path = data_path
        self.augment_ratio = augment_ratio
        self.target_labels = target_labels

        # 读取数据
        self.df = pd.read_csv(data_path)

        # 初始化 Marian 翻译模型
        self.zh_en_model_name = 'Helsinki-NLP/opus-mt-zh-en'
        self.en_zh_model_name = 'Helsinki-NLP/opus-mt-en-zh'

        self.zh_en_tokenizer = MarianTokenizer.from_pretrained(self.zh_en_model_name)
        self.en_zh_tokenizer = MarianTokenizer.from_pretrained(self.en_zh_model_name)

        self.zh_en_model = MarianMTModel.from_pretrained(self.zh_en_model_name)
        self.en_zh_model = MarianMTModel.from_pretrained(self.en_zh_model_name)

    def back_translation(self, text, src_lang='zh', mid_lang='en'):
        """
        使用 Marian 模型进行回译增强

        :param text: 输入文本
        :param src_lang: 源语言
        :param mid_lang: 中间语言
        :return: 增强后文本
        """
        try:
            # 第一次翻译：中文 -> 英文
            zh_inputs = self.zh_en_tokenizer([text], return_tensors="pt", padding=True)
            zh_to_en_translated = self.zh_en_model.generate(**zh_inputs)
            en_text = self.zh_en_tokenizer.batch_decode(zh_to_en_translated, skip_special_tokens=True)[0]

            # 回译：英文 -> 中文
            en_inputs = self.en_zh_tokenizer([en_text], return_tensors="pt", padding=True)
            en_to_zh_translated = self.en_zh_model.generate(**en_inputs)
            back_translated = self.en_zh_tokenizer.batch_decode(en_to_zh_translated, skip_special_tokens=True)[0]

            return back_translated
        except Exception as e:
            print(f"翻译错误: {e}")
            return text

    # ... 其他方法保持不变 ...

    def augment_data(self):
        """
        数据增强主方法
        """
        # 筛选需要增强的数据
        target_data = self.df[
            self.df['标签（时政、科技、科普、娱乐、体育、社会讨论/话题、时事、经济）'].isin(self.target_labels.values())]

        # 按比例抽样
        sample_size = int(len(target_data) * self.augment_ratio)
        sample_data = target_data.sample(n=sample_size)

        # 创建增强数据集
        augmented_data = []

        # # 同义词替换增强
        # print("开始同义词替换增强...")
        # start_time = time.time()
        # synonym_augmented = sample_data.copy()
        # synonym_augmented['热搜词条'] = synonym_augmented['热搜词条'].apply(self.synonym_replacement)
        # synonym_augmented['augment_type'] = 'synonym'
        # augmented_data.append(synonym_augmented)
        # synonym_time = time.time() - start_time

        # 回译增强
        print("开始回译增强...")
        start_time = time.time()
        backtrans_augmented = sample_data.copy()
        backtrans_augmented['热搜词条'] = backtrans_augmented['热搜词条'].apply(self.back_translation)
        backtrans_augmented['augment_type'] = 'back_translation'
        augmented_data.append(backtrans_augmented)
        backtrans_time = time.time() - start_time

        # 保存
        augmented_df = pd.concat(augmented_data, ignore_index=True)
        augmented_df.to_csv(f"{self.data_path.replace('.csv', '_augmented.csv')}", index=False)

        return {
            # 'synonym': {
            #     'time': synonym_time,
            #     'data_size': len(synonym_augmented)
            # },
            'back_translation': {
                'time': backtrans_time,
                'data_size': len(backtrans_augmented)
            }
        }

def main():
    # 数据增强
    augmenter = TextDataAugmentation(
        data_path='../data/train_weibo-hot-search-labeled.csv',
        augment_ratio=1,
        target_labels={0: '科技', 1: '娱乐', 2: '时事', 3: '社会讨论/话题', 4: '时政', 5: '科普', 6: '经济', 7: '体育'}
    )

    # 执行增强
    augment_stats = augmenter.augment_data()

    # 打印结果
    print("数据增强统计：", augment_stats)

if __name__ == "__main__":
    main()
