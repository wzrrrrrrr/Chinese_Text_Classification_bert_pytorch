import os

from tqdm import tqdm
import torch
from src.visualization_utils import  plot_classification_report_epoch, plot_confusion_matrix_epoch
from src.config import EVALUATION_CONFIG  # 导入配置


def evaluate(model, test_loader, device, label_map, epoch=None, visualizations_dir=None):
    """
    评估模型在测试集上的表现，返回平均损失和准确率，并根据需要绘制混淆矩阵和分类报告。

    Args:
        model (torch.nn.Module): 待评估的模型。
        test_loader (DataLoader): 测试数据集的 DataLoader。
        device (torch.device): 模型和数据的运行设备（CPU 或 GPU）。
        label_map (dict): 标签映射的字典，例如 {0: '科技', 1: '娱乐', 2: '时事'}。
        epoch (int, optional): 当前的训练 epoch。
        visualizations_dir (str, optional): 存储图表的目录路径。

    Returns:
        tuple: (平均损失 (float), 准确率 (float))
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

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

            _, preds = torch.max(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    # 创建文件夹存储混淆矩阵和分类报告图表
    if visualizations_dir:
        confusion_matrix_dir = os.path.join(visualizations_dir, 'confusion_matrix')
        classification_report_dir = os.path.join(visualizations_dir, 'classification_report')

        os.makedirs(confusion_matrix_dir, exist_ok=True)
        os.makedirs(classification_report_dir, exist_ok=True)

        # 绘制混淆矩阵并保存
        if EVALUATION_CONFIG.get('draw_confusion_matrix'):
            class_names = [label_map[i] for i in range(len(label_map))]
            plot_confusion_matrix_epoch(all_labels, all_preds, class_names, epoch=epoch,
                                        visualizations_dir=confusion_matrix_dir)

        # 绘制分类报告并保存
        if EVALUATION_CONFIG.get('draw_classification_report'):
            class_names = [label_map[i] for i in range(len(label_map))]
            plot_classification_report_epoch(all_labels, all_preds, class_names, epoch=epoch,
                                             visualizations_dir=classification_report_dir)

    return avg_loss, accuracy
