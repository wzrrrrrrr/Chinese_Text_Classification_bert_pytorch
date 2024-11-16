import pandas as pd
from sklearn.model_selection import train_test_split


def generate_label_map(input_path, label_column):
    # 读取CSV文件
    df = pd.read_csv(input_path)

    # 获取唯一标签并创建映射
    unique_labels = df[label_column].unique()
    label_map = {index: label for index, label in enumerate(unique_labels)}

    return label_map


def split_data_by_specified_labels(input_path, output_train_path, output_test_path, label_column,
                                   specified_labels=None):
    # 读取CSV文件
    df = pd.read_csv(input_path)

    # 如果没有指定标签，则自动生成
    if specified_labels is None:
        label_map = generate_label_map(input_path, label_column)
    else:
        label_map = specified_labels

    # 存储最终的训练集和测试集
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    # 对每个指定的标签进行分层抽样
    for label_name in label_map.values():
        # 筛选特定标签的数据
        label_data = df[df[label_column] == label_name]

        # 检查该标签的数据是否为空
        if len(label_data) == 0:
            print(f"警告：标签 '{label_name}' 没有对应的数据，跳过该标签")
            continue

        # 如果数据量太少，调整抽样策略
        if len(label_data) < 10:  # 根据实际情况调整阈值
            print(f"警告：标签 '{label_name}' 数据量过少 ({len(label_data)} 条)")
            train_subset = label_data
            test_subset = pd.DataFrame()
        else:
            # 分层抽样
            train_subset, test_subset = train_test_split(
                label_data,
                test_size=0.2,
                random_state=42  # 设置随机种子以保证可重复性
            )

        # 合并数据
        train_data = pd.concat([train_data, train_subset])
        test_data = pd.concat([test_data, test_subset])

    # 检查最终数据集是否为空
    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError("无法创建训练集或测试集。请检查数据和标签设置。")

    # 保存训练集和测试集
    train_data.to_csv(output_train_path, index=False)
    test_data.to_csv(output_test_path, index=False)

    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    print(f"训练集标签分布:\n{train_data[label_column].value_counts()}")
    print(f"测试集标签分布:\n{test_data[label_column].value_counts()}")

    return train_data, test_data

if __name__ == "__main__":
    input_path = 'data/weibo-hot-search-labeled.csv'
    train_output_path = 'data/train_weibo-hot-search-labeled.csv'
    test_output_path = 'data/test_weibo-hot-search-labeled.csv'
    label_column = '标签（时政、科技、科普、娱乐、体育、社会讨论/话题、时事、经济）'

    # 方法1：指定特定标签
    # specified_labels = {0: '科技', 1: '娱乐', 2: '时事'}  # 这里的键值可以是任意的，重要的是值要匹配实际标签
    specified_labels = None

    if specified_labels:
        split_data_by_specified_labels(
            input_path,
            train_output_path,
            test_output_path,
            label_column,
            specified_labels
        )
    else:
        label_map = generate_label_map(input_path, label_column)
        split_data_by_specified_labels(
            input_path,
            train_output_path,
            test_output_path,
            label_column
        )
