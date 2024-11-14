from tqdm import tqdm
import torch

def evaluate(model, test_loader, device):
    """
    评估模型在测试集上的表现，返回平均损失和准确率。

    Args:
        model (torch.nn.Module): 待评估的模型。
        test_loader (DataLoader): 测试数据集的 DataLoader。
        device (torch.device): 模型和数据的运行设备（CPU 或 GPU）。

    Returns:
        tuple: (平均损失 (float), 准确率 (float))
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0  # 累积损失
    correct_predictions = 0  # 预测正确的样本数
    total_samples = 0  # 总样本数

    # 创建测试集的进度条
    pbar = tqdm(test_loader, desc="Evaluating", ncols=100)

    # 禁用梯度计算，加快推理速度并降低内存使用
    with torch.no_grad():
        for batch in pbar:
            # 将批次数据转移到指定设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播，计算模型输出和损失
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # 获取损失值
            logits = outputs.logits  # 获取模型的预测结果

            # 累加当前批次的损失
            total_loss += loss.item() * labels.size(0)  # 乘上批次样本数，累积总损失
            total_samples += labels.size(0)

            # 获取预测结果，计算准确率
            _, preds = torch.max(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()

            # 更新进度条
            pbar.set_postfix(loss=loss.item(), accuracy=correct_predictions / total_samples)

    # 计算平均损失和准确率
    avg_loss = total_loss / total_samples  # 计算总样本的平均损失
    accuracy = correct_predictions / total_samples  # 计算总样本的准确率

    return avg_loss, accuracy