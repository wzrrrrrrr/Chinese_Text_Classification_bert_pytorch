import pandas as pd


def generate_label_map(data_path, cls):
    # 读取 CSV 文件
    df = pd.read_csv(data_path)

    # 获取 'Movie_Name_CN' 列的唯一值并创建字典
    unique_labels = df[cls].unique()
    return {idx: label for idx, label in enumerate(unique_labels)}



if __name__ == "__main__":
    data_path = "data/weibo-hot-search-labeled.csv"  # 指定你的 CSV 文件路径
    label_map = generate_label_map(data_path, '标签（时政、科技、科普、娱乐、体育、社会讨论/话题、时事、经济）')
    # 打印生成的标签字典
    print("生成的标签字典：")
    print(label_map)