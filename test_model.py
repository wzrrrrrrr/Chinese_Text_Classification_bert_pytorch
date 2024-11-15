import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from src import config
from src.dataset import load_and_process_data
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model_path, num_labels):
    """
    加载预训练的BERT模型和分词器。

    Parameters:
    - model_path (str): 模型文件的路径
    - num_labels (int): 标签数量

    Returns:
    - model: 加载的BERT模型
    - tokenizer: 加载的BERT分词器
    """
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(config.MODEL_PATH, num_labels=num_labels)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, tokenizer


def evaluate_model(model, test_loader, device, label_map):
    """
    详细评估模型在测试集上的表现。

    Parameters:
    - model: 待评估的模型
    - test_loader (DataLoader): 测试数据集的 DataLoader
    - device: 设备 (CPU or GPU)
    - label_map (dict): 标签映射

    Returns:
    - 评估指标字典
    """
    model.to(device)
    all_preds = []
    all_labels = []
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", ncols=100):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    avg_loss = total_loss / total_samples
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    # 生成分类报告
    report = classification_report(
        all_labels,
        all_preds,
        labels=list(label_map.keys()),  # 指定标签
        target_names=list(label_map.values()),
        digits=4
    )

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_labels
    }


def plot_confusion_matrix(cm, label_names, save_path):
    """
    绘制并保存混淆矩阵。

    Parameters:
    - cm (numpy.ndarray): 混淆矩阵
    - label_names (list): 标签名称列表
    - save_path (str): 保存路径
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def test_model_performance(model_path=None):
    """
    测试模型性能的主函数。

    Parameters:
    - model_path (str, optional): 模型路径，默认为 None
    """
    # 如果没有提供模型路径，使用配置中的默认路径
    if model_path is None:
        model_path = os.path.join(config.ARTIFACTS_DIR, '20241115_214629', 'models', 'bert_epoch13_val_loss0.1352.pth')

    # 解析标签映射
    label_map = eval(config.CSV_MAP_STR)

    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型和分词器
    model, tokenizer = load_model(model_path, num_labels=len(label_map))

    # 加载和处理测试数据集
    test_dataset, _, _ = load_and_process_data(
        data_path=config.TEST_DATA_PATH,  # 使用测试数据集路径
        tokenizer=tokenizer,
        label_column=config.LABEL_COLUMN,
        text_column=config.TEXT_COLUMN,
        csv_map_str=config.CSV_MAP_STR,
        sample_size=None,  # 测试集通常使用全部数据
        test_size=None
    )

    # 创建DataLoader
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    # 评估模型
    results = evaluate_model(model, test_loader, device, label_map)

    # 打印基本指标
    print(f"Average Loss: {results['avg_loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:\n", results['classification_report'])

    # 根据配置决定是否绘制混淆矩阵
    if config.EVALUATION_CONFIG['draw_confusion_matrix']:
        plot_confusion_matrix(
            results['confusion_matrix'],
            list(label_map.values()),
            os.path.join(config.ARTIFACTS_DIR, 'confusion_matrix.png')
        )


if __name__ == '__main__':
    test_model_performance()
