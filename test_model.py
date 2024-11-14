import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader

from src.dataset import load_and_process_data
from tqdm import tqdm  # 导入 tqdm

def test_model_accuracy():
    # 设置六个参数，这些参数可以根据不同的需求进行修改
    MODEL_PATH = 'bert_text_classification_final.pth'  # 最佳模型的路径，保存了训练过程中表现最好的模型
    BERT_PATH = 'model/bert-base-chinese'  # BERT 基础模型的路径，用于加载预训练的BERT模型和分词器
    data_path = 'data/test_data.csv'  # 测试数据集的路径，需要是 CSV 格式
    csv_map_str = "{0: '科技', 1: '娱乐', 2: '时事'}"  # 标签与类别之间的映射，格式是字典的字符串表示
    label_column = '标签（时政、科技、科普、娱乐、体育、社会讨论/话题、时事、经济）'  # 标签列名，指示文本的类别
    text_column = '热搜词条'  # 文本列名，包含我们要进行分类的文本内容
    selection_method = "bottom"  # selection_method (str): 选择数据的方式，可选值为 "random"、"top"、"bottom"。
    sample_size = 10

    # 将 csv_map_str 字符串解析为字典
    label_map = eval(csv_map_str)  # 将标签字符串转换为字典

    # 配置设备：判断是否有 GPU，如果有 GPU 使用 GPU，否则使用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载预训练的 BERT 分词器
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    # 加载和处理测试数据集
    test_dataset, _, _ = load_and_process_data(
        data_path=data_path,            # 数据路径
        tokenizer=tokenizer,            # 分词器
        label_column=label_column,      # 标签列
        text_column=text_column,        # 文本列
        label_map=label_map,            # 标签映射
        sample_size=sample_size,               # 不进行采样，使用全部数据
        test_size=None,                    # 将所有数据作为测试集
        selection_method = selection_method
    )

    # 创建 DataLoader，用于批量加载数据集
    test_loader = DataLoader(test_dataset, batch_size=8)

    # 加载 BERT 模型
    model = BertForSequenceClassification.from_pretrained(BERT_PATH, num_labels=len(label_map), use_safetensors=False)  # 加载BERT模型
    model.load_state_dict(torch.load(MODEL_PATH))  # 加载训练好的模型参数
    model.to(device)  # 将模型移动到合适的设备（GPU/CPU）

    # 开始测试模型性能
    model.eval()  # 设置模型为评估模式

    # 使用 tqdm 包装 DataLoader 显示进度条
    test_accuracy = 0.0
    j = 0
    with torch.no_grad():  # 不需要计算梯度
        for batch in tqdm(test_loader, desc="Testing Model"):  # 使用 tqdm 显示进度条
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 获取对应文本
            texts = [test_dataset.texts[j+i] for i in range(batch['input_ids'].size(0))]  # 从 Dataset 中获取文本
            j = j+ batch['input_ids'].size(0)
            # 模型预测
            outputs = model(input_ids=inputs, attention_mask=attention_mask)  # 仅传递 input_ids 和 attention_mask

            logits = outputs.logits  # 获取预测的logits

            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            correct_preds = (preds == labels).sum().item()
            test_accuracy += correct_preds

            # 打印每个文本及其预测结果
            for i in range(len(texts)):
                predicted_label = label_map[preds[i].item()]  # 将预测结果映射到标签
                true_label = label_map[labels[i].item()]  # 将真实标签映射到标签
                print(f"Text: {texts[i]}")
                print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
                print("-" * 50)

    # 计算准确率
    test_accuracy /= len(test_dataset)

    # 输出评估结果
    print(f"Test Accuracy: {test_accuracy:.4f}")  # 打印测试准确率