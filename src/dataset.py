from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_and_process_data(data_path, tokenizer, label_column, text_column,
                          label_map=None, sample_size=None, test_size=0.2, selection_method="random"):
    """
    读取数据文件并处理标签和文本数据，将其分为训练集和测试集。

    Args:
        data_path (str): 数据文件的路径。
        tokenizer (BertTokenizer): 用于文本编码的分词器。
        label_column (str): 标签所在的列名。
        text_column (str): 文本所在的列名。
        label_map (dict, optional): 自定义标签映射，将标签名称映射到整数值。
        sample_size (int, optional): 每个标签类保留的样本数量。
        test_size (float): 测试集所占的比例，默认为0.2。
        selection_method (str): 选择数据的方式，可选值为 "random"、"top"、"bottom"。

    Returns:
        train_dataset (TextDataset): 处理后的训练集。
        test_dataset (TextDataset): 处理后的测试集。
        num_labels (int): 标签类别总数。
    """
    # 读取数据文件
    df = pd.read_csv(data_path)

    # 如果提供了 label_map，则对标签进行筛选和编码
    if label_map:
        # 只保留标签在 label_map 中的行
        df = df[df[label_column].isin(label_map.values())]

        # 将标签映射为指定的整数
        df[label_column] = df[label_column].map({v: k for k, v in label_map.items()})
        labels = df[label_column].astype(int).tolist()

        # 打印标签映射关系
        print("标签与文本标签的自定义映射关系：")
        for idx, label in label_map.items():
            print(f"{idx} -> {label}")
    else:
        # 如果未提供 label_map，则自动生成整数标签
        labels = df[label_column].astype(str).tolist()
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        # 打印自动生成的标签映射关系
        print("标签与文本标签的自动映射关系：")
        for idx, label in enumerate(label_encoder.classes_):
            print(f"{idx} -> {label}")

        # 更新 label_map 为自动生成的映射字典
        label_map = {idx: label for idx, label in enumerate(label_encoder.classes_)}

    # 只保留每类前 sample_size 条数据，根据选择方式选择数据
    if sample_size:
        df['Encoded_Labels'] = labels  # 创建中间列以便分组

        if selection_method == "random":
            df = df.groupby('Encoded_Labels').apply(lambda x: x.sample(n=sample_size, random_state=42)).reset_index(drop=True)
        elif selection_method == "top":
            df = df.groupby('Encoded_Labels').head(sample_size).reset_index(drop=True)
        elif selection_method == "bottom":
            df = df.groupby('Encoded_Labels').tail(sample_size).reset_index(drop=True)
        else:
            raise ValueError("selection_method 参数必须为 'random'、'top' 或 'bottom'")

        labels = df['Encoded_Labels'].tolist()  # 更新 labels 列

    # 提取文本列
    texts = df[text_column].astype(str).tolist()

    # 将数据集划分为训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )

    # 创建训练集和测试集对象
    train_dataset = TextDataset(train_texts, train_labels, tokenizer=tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer=tokenizer)

    # 返回训练集、测试集以及标签类别总数
    return train_dataset, test_dataset, len(label_map)