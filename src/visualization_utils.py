import matplotlib.pyplot as plt

def plot_loss_curve(train_losses, val_losses):
    """
    绘制训练和验证损失曲线，帮助观察模型的收敛情况。

    参数：
    - train_losses (list of float): 每个epoch的训练损失值
    - val_losses (list of float): 每个epoch的验证损失值
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.show()
