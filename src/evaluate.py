from tqdm import tqdm
import torch
from src.visualization_utils import  plot_classification_report_epoch, plot_confusion_matrix_epoch


def evaluate(model, test_loader, device, label_map, epoch=None, draw_confusion_matrix=False, draw_classification_report=False):
    """
    评估模型在测试集上的表现，返回平均损失和准确率，并根据需要绘制混淆矩阵和分类报告。

    Args:
        model (torch.nn.Module): 待评估的模型。
        test_loader (DataLoader): 测试数据集的 DataLoader。
        device (torch.device): 模型和数据的运行设备（CPU 或 GPU）。
        label_map (dict): 标签映射的字典，例如 {0: '科技', 1: '娱乐', 2: '时事'}。
        epoch (int, optional): 当前的训练 epoch。
        draw_confusion_matrix (bool): 是否绘制混淆矩阵，默认为 False。
        draw_classification_report (bool): 是否绘制分类报告，默认为 False。

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

    # 绘制混淆矩阵
    if draw_confusion_matrix:
        class_names = [label_map[i] for i in range(len(label_map))]
        plot_confusion_matrix_epoch(all_labels, all_preds, class_names, epoch=epoch)

    # 绘制分类报告并显示当前 epoch
    if draw_classification_report:
        class_names = [label_map[i] for i in range(len(label_map))]
        plot_classification_report_epoch(all_labels, all_preds, class_names, epoch=epoch)

    return avg_loss, accuracy