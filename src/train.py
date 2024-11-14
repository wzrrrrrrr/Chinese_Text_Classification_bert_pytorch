from tqdm import tqdm
import torch
from torch.amp import autocast, GradScaler  # 自动混合精度所需

def train(model, train_loader, optimizer, device, scheduler=None, accumulation_steps=1, use_amp=False):
    """
    训练模型一个 epoch，并返回平均损失和准确率。

    Args:
        model (torch.nn.Module): 待训练的模型。
        train_loader (DataLoader): 训练数据集的 DataLoader。
        optimizer (torch.optim.Optimizer): 优化器，用于更新模型参数。
        device (torch.device): 模型和数据的运行设备（CPU 或 GPU）。
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器，默认无。
        accumulation_steps (int): 梯度累积的步数，默认1表示每个batch都更新梯度。
        use_amp (bool): 是否使用自动混合精度（AMP），默认不使用。

    Returns:
        tuple: (平均损失 (float), 准确率 (float))
    """
    model.train()  # 启用训练模式
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    optimizer.zero_grad()  # 初始化梯度
    scaler = GradScaler() if use_amp else None  # 自动混合精度缩放器

    # 使用 tqdm 包装训练数据加载器，显示进度条信息
    pbar = tqdm(train_loader, desc="Training", leave=True, ncols=100)

    for batch_idx, batch in enumerate(pbar):
        # 将数据加载到设备上
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with autocast("cuda", enabled=use_amp):  # 在 AMP 模式下前向传播
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps  # 平均到每个累积步的损失
            logits = outputs.logits

        # 反向传播
        if use_amp:
            scaler.scale(loss).backward()  # 使用缩放后的损失进行反向传播
        else:
            loss.backward()

        # 梯度累积：每 accumulation_steps 执行一次参数更新
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            if use_amp:
                scaler.step(optimizer)  # 使用缩放的梯度更新
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # 累加当前步损失
        total_loss += loss.item() * accumulation_steps

        # 计算准确率
        _, predicted = torch.max(logits, dim=1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # 更新进度条显示当前损失、准确率和学习率
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(loss=loss.item() * accumulation_steps, accuracy=correct_predictions / total_samples, lr=current_lr)

    # 计算平均损失和准确率
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    # 更新学习率
    if scheduler:
        scheduler.step()

    return avg_loss, accuracy