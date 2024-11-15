import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as fm
import torch

# 加载支持中文的字体，例如SimHei
font_path = '/System/Library/Fonts/STHeiti Light.ttc'  # 替换为你的字体路径
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = font_prop.get_name()  # 使用该字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
"""
在基于BERT和PyTorch的文本分类任务中，通常有以下几种可视化操作对理解和优化模型至关重要：
	1.	训练和验证损失的变化曲线：显示训练过程中的损失随epoch的变化，有助于观察模型是否收敛、是否存在过拟合。
	2.	训练和验证准确率曲线：帮助判断模型的准确性，以及是否在验证集上表现良好。
	3.	混淆矩阵（Confusion Matrix）：展示分类结果的具体分布情况，尤其适合多分类任务，能直观地显示哪些类别易被混淆。
	4.	学习率调度曲线：如果使用了学习率调度器，跟踪学习率的变化趋势可以帮助理解模型在不同阶段的训练效果。
	5.	分类报告（Classification Report）：以表格形式展示各类别的精确率、召回率和F1分数，细化分类性能分析。
	6.	训练时间记录：可以跟踪每个epoch的训练时间，帮助了解模型的效率。
"""


def plot_loss_curve(train_losses, val_losses):
    """
    绘制训练和验证损失曲线。

    参数：
    - train_losses (list): 训练集每轮的损失值
    - val_losses (list): 验证集每轮的损失值
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy_curve(train_accuracies, val_accuracies):
    """
    绘制训练和验证准确率曲线。

    参数：
    - train_accuracies (list of float): 每个 epoch 的训练准确率
    - val_accuracies (list of float): 每个 epoch 的验证准确率
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Training Accuracy", color='blue')
    plt.plot(val_accuracies, label="Validation Accuracy", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy Curve")
    plt.legend()
    plt.grid()
    plt.show()


def plot_confusion_matrix(true_labels, pred_labels, class_names):
    """
    绘制混淆矩阵图，用于显示模型的分类性能。

    参数：
    - true_labels (list or array): 验证集的真实标签
    - pred_labels (list or array): 模型的预测标签
    - class_names (list of str): 类别名称列表，用于标签矩阵

    """
    # 生成混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)

    # 使用 Seaborn 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


def plot_confusion_matrix(model, data_loader, class_names_map, device='cpu', epoch=None, model_name="Model"):
    """
    绘制模型在验证集上的混淆矩阵，并在标题中显示模型名称和当前 epoch。

    Args:
        model (torch.nn.Module): 要评估的模型。
        data_loader (DataLoader): 验证数据集的 DataLoader。
        class_names_map (dict): 类别索引到类别名称的映射字典。
        device (torch.device): 模型和数据的运行设备（CPU 或 GPU）。
        epoch (int, optional): 当前的 epoch 号。默认为 None。
        model_name (str, optional): 模型的名称。默认为 "Model"。
    """
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []

    # 禁用梯度计算
    with torch.no_grad():
        for batch in data_loader:
            # 获取输入和标签
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 模型预测
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            # 收集预测和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names_map.values(),
                yticklabels=class_names_map.values())
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # 设置标题，包含模型名称和 epoch 信息
    title = f"Confusion Matrix for {model_name}"
    if epoch is not None:
        title += f" - Epoch {epoch}"
    plt.title(title)

    # 显示混淆矩阵
    plt.show()

def plot_lr_curve(lr_list):
    """
    绘制学习率曲线。

    Args:
        lr_list (list): 每个 epoch 结束时的学习率列表。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(lr_list) + 1), lr_list, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Scheduler Curve')
    plt.legend()
    plt.grid(True)
    plt.show()