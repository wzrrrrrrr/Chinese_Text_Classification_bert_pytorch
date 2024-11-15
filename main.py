import yaml
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from src import config
from src.dataset import load_and_process_data
from src.train import train
from src.evaluate import evaluate
from src.visualization_utils import plot_loss_curve, plot_accuracy_curve, plot_lr_curve
import os

# 创建训练目录
def create_training_directories():
    """
    创建存储训练过程中数据的目录结构。

    Parameters:
    - config.ARTIFACTS_DIR: 存放训练过程生成文件的根目录
    """
    # 创建主训练目录
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_training_dir = os.path.join(config.ARTIFACTS_DIR, timestamp)
    os.makedirs(current_training_dir, exist_ok=True)

    # 创建子目录：模型和可视化图表
    models_dir = os.path.join(current_training_dir, 'models')
    visualizations_dir = os.path.join(current_training_dir, 'visualizations')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    return models_dir, visualizations_dir, current_training_dir


# 加载数据集和模型
def load_data_and_model():
    """
    加载并处理数据集，初始化模型和分词器。

    返回:
    - train_dataset: 训练数据集
    - test_dataset: 测试数据集
    - model: 初始化的BERT模型
    - label_map: 标签映射
    """
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_PATH)

    # 加载和处理数据集
    train_dataset, test_dataset, label_map = load_and_process_data(
        data_path=config.DATA_PATH,
        tokenizer=tokenizer,
        label_column=config.LABEL_COLUMN,
        text_column=config.TEXT_COLUMN,
        csv_map_str=config.CSV_MAP_STR,
        sample_size=config.SAMPLE_SIZE,
        test_size=config.TEST_SIZE,
        selection_method=config.SELECTION_METHOD
    )

    num_labels = len(label_map)
    model = BertForSequenceClassification.from_pretrained(config.MODEL_PATH, num_labels=num_labels)
    return train_dataset, test_dataset, model, label_map


# 训练过程管理
def train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler, device, models_dir, visualizations_dir, label_map):
    """
    执行模型的训练和评估，保存最佳模型，并生成可视化图表。

    Parameters:
    - model: 初始化的BERT模型
    - train_loader: 训练数据加载器
    - test_loader: 测试数据加载器
    - optimizer: 优化器
    - scheduler: 学习率调度器
    - device: 训练设备（CPU/GPU）
    - models_dir: 存储模型的目录
    - visualizations_dir: 存储可视化图表的目录
    - label_map: 标签映射

    返回:
    - 训练和验证过程中的损失、准确率和学习率数据
    """
    best_val_loss = float('inf')  # 记录最佳验证损失
    epochs_without_improvement = 0  # 记录验证损失未改进的epoch数
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    learning_rates = []

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()

        # 训练阶段
        time.sleep(1)
        train_loss, train_accuracy = train(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        # 验证阶段
        time.sleep(1)
        val_loss, val_accuracy = evaluate(model, test_loader, device, label_map, epoch=epoch + 1, visualizations_dir = visualizations_dir)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # 记录学习率
        learning_rates.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_path = os.path.join(models_dir, f"bert_epoch{epoch + 1}_val_loss{val_loss:.4f}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"验证损失有改进，最佳模型已保存至 {best_model_path}！")
        else:
            epochs_without_improvement += 1
            print(f"验证损失无改进，已连续 {epochs_without_improvement} 个 epoch 无改善")

        if epochs_without_improvement >= config.PATIENCE:
            print("早停触发，训练提前结束！")
            break

        end_time = time.time()
        print(f"Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds.")
        print("=" * 90)

    return train_losses, val_losses, train_accuracies, val_accuracies, learning_rates


# 保存模型和可视化图表
def save_model_and_visualizations(model, models_dir, visualizations_dir, train_losses, val_losses, train_accuracies, val_accuracies, learning_rates):
    """
    保存最终模型和可视化图表。

    Parameters:
    - model: 训练后的模型
    - models_dir: 模型保存目录
    - visualizations_dir: 可视化图表保存目录
    - train_losses: 训练损失列表
    - val_losses: 验证损失列表
    - train_accuracies: 训练准确率列表
    - val_accuracies: 验证准确率列表
    - learning_rates: 学习率列表
    """
    final_model_path = os.path.join(models_dir, "bert_text_classification_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存至 {final_model_path}！")

    # 绘制并保存可视化图表
    plot_loss_curve(train_losses, val_losses, save_path=os.path.join(visualizations_dir, "loss_curve.png"))
    plot_accuracy_curve(train_accuracies, val_accuracies, save_path=os.path.join(visualizations_dir, "accuracy_curve.png"))
    plot_lr_curve(learning_rates, save_path=os.path.join(visualizations_dir, "lr_curve.png"))


# 保存训练参数的函数
def save_training_params_yaml(config_module, current_training_dir):
    """
    提取并保存训练参数为 YAML 格式。

    Parameters:
    - config_module: 包含配置的模块对象
    - current_training_dir: 当前训练目录，用于保存训练参数
    """
    # 提取配置模块中的可序列化部分
    config_dict = {
        key: value
        for key, value in vars(config_module).items()
        if not key.startswith("__") and not callable(value)
    }

    # 保存为 YAML 文件
    params_file = os.path.join(current_training_dir, "training_params.yaml")
    with open(params_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, allow_unicode=True, sort_keys=False)

    print(f"训练参数已保存至 {params_file}！")

def load_config_from_yaml(config_path):
    """
    从 YAML 文件加载配置，并返回配置对象。

    Parameters:
    - config_path (str): 配置文件路径

    Returns:
    - config (object): 包含配置内容的对象
    """
    # 读取 YAML 文件内容
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # 创建动态配置对象
    class Config:
        pass

    config = Config()
    for key, value in config_dict.items():
        setattr(config, key, value)  # 动态添加配置项
    return config


def initialize_training_directories(base_dir="./artifacts"):
    """
    初始化训练目录，用于保存模型、可视化图表等。

    Parameters:
    - base_dir (str): 基础目录路径

    Returns:
    - models_dir (str): 模型保存目录
    - visualizations_dir (str): 可视化图表保存目录
    - current_training_dir (str): 当前训练会话目录
    """
    from datetime import datetime

    # 生成当前时间戳的训练会话目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_training_dir = os.path.join(base_dir, timestamp)

    # 定义子目录路径
    models_dir = os.path.join(current_training_dir, "models")
    visualizations_dir = os.path.join(current_training_dir, "visualizations")

    # 创建目录
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    return models_dir, visualizations_dir, current_training_dir


def main(config):
    """
    主函数，包含模型训练和评估的完整流程。

    Parameters:
    - config (object): 配置对象
    """
    # Step 1: 初始化训练目录
    models_dir, visualizations_dir, current_training_dir = initialize_training_directories()

    # Step 2: 加载数据集和模型（函数需自行实现）
    train_dataset, test_dataset, model, label_map = load_data_and_model()

    # Step 3: 保存训练参数到 YAML 文件
    save_training_params_yaml(config, current_training_dir)

    # Step 4: 配置设备（CPU 或 GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Step 5: 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    # Step 6: 定义优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA)

    # Step 7: 训练和评估模型（需自行实现训练逻辑）
    train_losses, val_losses, train_accuracies, val_accuracies, learning_rates = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        models_dir=models_dir,
        visualizations_dir=visualizations_dir,
        label_map=label_map
    )

    # Step 8: 保存最终模型和可视化图表
    save_model_and_visualizations(
        model=model,
        models_dir=models_dir,
        visualizations_dir=visualizations_dir,
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        learning_rates=learning_rates
    )


if __name__ == '__main__':
    # 加载配置
    # config_path = "./artifacts/20241115_212647/training_params.yaml"
    # config = load_config_from_yaml(config_path)

    # 启动主函数
    main(config)
