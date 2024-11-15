import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.font_manager as fm
import torch
import os


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



def plot_loss_curve(train_losses, val_losses, save_path=None):
    """
    绘制训练和验证损失曲线，并可选择保存为图片。

    参数：
    - train_losses (list): 训练集每轮的损失值。
    - val_losses (list): 验证集每轮的损失值。
    - save_path (str, optional): 保存图片的路径。如果为 None，则不保存图片。
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)

    # 如果提供了保存路径，则保存图片
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"损失曲线已保存至 {save_path}")

    # 显示图形
    plt.show()



def plot_accuracy_curve(train_accuracies, val_accuracies, save_path=None):
    """
    绘制训练和验证准确率曲线，并可选择保存为图片。

    参数：
    - train_accuracies (list of float): 每个 epoch 的训练准确率。
    - val_accuracies (list of float): 每个 epoch 的验证准确率。
    - save_path (str, optional): 保存图片的路径。如果为 None，则不保存图片。
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Training Accuracy", color='blue', marker='o')
    plt.plot(val_accuracies, label="Validation Accuracy", color='orange', marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy Curve")
    plt.legend()
    plt.grid(True)

    # 如果提供了保存路径，则保存图片
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"准确率曲线已保存至 {save_path}")

    # 显示图形
    plt.show()


def plot_confusion_matrix_epoch(true_labels, pred_labels, class_names, epoch=None, visualizations_dir=None):
    """
    绘制混淆矩阵图，并保存到指定目录。

    参数：
    - true_labels (list or array): 验证集的真实标签
    - pred_labels (list or array): 模型的预测标签
    - class_names (list of str): 类别名称列表，用于标签矩阵
    - epoch (int, optional): 当前的训练 epoch，用于保存文件名。
    - visualizations_dir (str, optional): 存储图表的目录路径
    """
    # 生成混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)

    # 使用 Seaborn 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix (Epoch {epoch})" if epoch is not None else "Confusion Matrix")

    # 保存图表
    if visualizations_dir:
        filename = f"confusion_matrix_epoch_{epoch}.png" if epoch is not None else "confusion_matrix.png"
        plt.savefig(os.path.join(visualizations_dir, filename))
    plt.close()

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

def plot_lr_curve(lr_list, save_path=None):
    """
    绘制学习率曲线，并可选择保存为图片。

    参数：
    - lr_list (list): 每个 epoch 结束时的学习率列表。
    - save_path (str, optional): 保存图片的路径。如果为 None，则不保存图片。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(lr_list) + 1), lr_list, label='Learning Rate', color='green', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Scheduler Curve')
    plt.legend()
    plt.grid(True)

    # 如果提供了保存路径，则保存图片
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"学习率曲线已保存至 {save_path}")

    # 显示图形
    plt.show()

def plot_classification_report_epoch(y_true, y_pred, class_names, epoch=None, figsize=(10, 6), visualizations_dir=None):
    """
    绘制分类报告的热力图，并保存到指定目录。

    Args:
        y_true (list or array): 真实标签。
        y_pred (list or array): 预测标签。
        class_names (list): 类别名称列表。
        epoch (int, optional): 当前的训练 epoch，用于显示在图表标题中。
        figsize (tuple): 图表尺寸。
        visualizations_dir (str, optional): 存储图表的目录路径
    """
    # 生成分类报告并转换为 DataFrame
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).iloc[:-1, :].T

    # 动态生成标题
    title = f"Classification Report (Epoch {epoch})" if epoch is not None else "Classification Report"

    # 绘制热力图
    plt.figure(figsize=figsize)
    sns.heatmap(report_df, annot=True, cmap="Blues", fmt=".2f", cbar=False)
    plt.title(title)
    plt.xlabel("Metrics")
    plt.ylabel("Classes")
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图表
    if visualizations_dir:
        filename = f"classification_report_epoch_{epoch}.png" if epoch is not None else "classification_report.png"
        plt.savefig(os.path.join(visualizations_dir, filename))
    plt.close()


def plot_classification_report(model, data_loader, class_names_map, device, epoch=None, model_name=None):
    """
    绘制分类报告的可视化图，显示每个类的精确度、召回率和 F1 分数，并在标题中包含 epoch 和模型名称。

    Args:
        model (torch.nn.Module): 待评估的模型。
        data_loader (DataLoader): 测试数据集的 DataLoader。
        class_names_map (dict): 标签映射字典，例如 {0: '科技', 1: '娱乐', 2: '时事'}。
        device (torch.device): 模型和数据的运行设备（CPU 或 GPU）。
        epoch (int, optional): 当前的 epoch，用于图像标题显示。
        model_name (str, optional): 模型名称，用于图像标题显示。
    """
    # 设置模型为评估模式
    model.eval()

    all_preds = []
    all_labels = []

    # 禁用梯度计算
    with torch.no_grad():
        for batch in data_loader:
            # 将数据移至指定设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 获取模型预测
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            # 收集所有预测和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 生成分类报告
    report = classification_report(all_labels, all_preds, target_names=list(class_names_map.values()), output_dict=True)

    # 将报告转换为 DataFrame 格式
    report_df = pd.DataFrame(report).transpose()

    # 设置图像大小
    plt.figure(figsize=(10, 6))

    # 绘制热力图，显示各项指标
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f", cbar=False)

    # 设置标题，包含 epoch 和模型名称
    title = f"Classification Report (Epoch {epoch}) - Model: {model_name}" if epoch is not None and model_name else "Classification Report"
    plt.title(title)
    plt.xlabel("Metrics")
    plt.ylabel("Classes")

    # 显示图像
    plt.tight_layout()
    plt.show()