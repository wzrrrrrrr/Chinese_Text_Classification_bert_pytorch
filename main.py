import sys
import yaml
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from src.dataset import load_and_process_data
from src.train import train
from src.evaluate import evaluate
from src.visualization_utils import plot_loss_curve, plot_accuracy_curve, plot_lr_curve
import os


# 创建训练目录
def create_training_directories(base_dir):
    """
    创建存储训练过程中数据的目录结构。

    Parameters:
    - base_dir: 存放训练过程生成文件的根目录
    """
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_training_dir = os.path.join(base_dir, timestamp)
    os.makedirs(current_training_dir, exist_ok=True)

    models_dir = os.path.join(current_training_dir, 'models')
    visualizations_dir = os.path.join(current_training_dir, 'visualizations')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    return models_dir, visualizations_dir, current_training_dir


# 加载和处理数据集
def load_datasets(tokenizer):
    """
    加载并处理数据集。

    Parameters:
    - tokenizer: BERT分词器

    Returns:
    - train_dataset: 训练数据集
    - test_dataset: 测试数据集
    - label_map: 标签映射
    """
    return load_and_process_data(
        data_path=config.DATA_PATH,
        tokenizer=tokenizer,
        label_column=config.LABEL_COLUMN,
        text_column=config.TEXT_COLUMN,
        csv_map_str=config.CSV_MAP_STR,
        sample_size=config.SAMPLE_SIZE,
        test_size=config.TEST_SIZE,
        selection_method=config.SELECTION_METHOD
    )

def train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler, device, models_dir, visualizations_dir,
                       label_map, config):
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

    Returns:
    - 训练和验证过程中的损失、准确率和学习率数据
    :param config:
    """
    best_val_loss = float('inf')
    epochs_without_improvement = 0
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
        print(
            f"Epoch {epoch + 1}/{config.NUM_EPOCHS}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        # 验证阶段
        time.sleep(1)
        val_loss, val_accuracy = evaluate(model, test_loader, device, label_map, config, epoch=epoch + 1,
                    visualizations_dir=visualizations_dir)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(
            f"Epoch {epoch + 1}/{config.NUM_EPOCHS}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

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
def save_model_and_visualizations(model, models_dir, visualizations_dir, train_losses, val_losses, train_accuracies,
                                  val_accuracies, learning_rates):
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
    plot_accuracy_curve(train_accuracies, val_accuracies,
                        save_path=os.path.join(visualizations_dir, "accuracy_curve.png"))
    plot_lr_curve(learning_rates, save_path=os.path.join(visualizations_dir, "lr_curve.png"))


# 保存训练参数的函数
def save_training_params_yaml(config_module, current_training_dir):
    """
    提取并保存训练参数为 YAML 格式。

    Parameters:
    - config_module: 包含配置的模块对象
    - current_training_dir: 当前训练目录，用于保存训练参数
    """
    config_dict = {key: value for key, value in vars(config_module).items() if
                   not key.startswith("__") and not callable(value)}

    params_file = os.path.join(current_training_dir, "training_params.yaml")
    with open(params_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, allow_unicode=True, sort_keys=False)

    print(f"训练参数已保存至 {params_file}！")


def load_config_from_yaml(config_path):

    class DotDict(dict):
        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    # 加载 YAML 配置
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = DotDict(config_dict)

    # 现在可以使用 config.MODEL_PATH
    return config


def main(config):
    """
    主函数，包含模型训练和评估的完整流程。

    Parameters:
    - config (object): 配置对象
    """
    # 打印配置信息，便于调试
    print("训练配置:")
    print(config)

    try:
        # Step 1: 初始化训练目录
        models_dir, visualizations_dir, current_training_dir = create_training_directories(
            base_dir=config.ARTIFACTS_DIR
        )

        # Step 2: 加载分词器和数据集
        tokenizer = BertTokenizer.from_pretrained(config.MODEL_PATH)
        train_dataset, test_dataset, label_map = load_datasets(tokenizer)

        # Step 3: 初始化模型
        num_labels = len(label_map)
        model = BertForSequenceClassification.from_pretrained(config.MODEL_PATH, num_labels=num_labels)

        # Step 4: 保存训练参数到 YAML 文件
        save_training_params_yaml(config, current_training_dir)

        # Step 5: 配置设备（CPU 或 GPU）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        model.to(device)

        # Step 6: 定义数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE
        )

        # Step 7: 定义优化器和学习率调度器
        optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE
        )
        scheduler = StepLR(
            optimizer,
            step_size=config.SCHEDULER_STEP_SIZE,
            gamma=config.SCHEDULER_GAMMA
        )

        # Step 8: 训练和评估模型
        train_losses, val_losses, train_accuracies, val_accuracies, learning_rates = train_and_evaluate(model=model,
                                                                                                        train_loader=train_loader,
                                                                                                        test_loader=test_loader,
                                                                                                        optimizer=optimizer,
                                                                                                        scheduler=scheduler,
                                                                                                        device=device,
                                                                                                        models_dir=models_dir,
                                                                                                        visualizations_dir=visualizations_dir,
                                                                                                        label_map=label_map,
                                                                                                        config=config)

        # Step 9: 保存最终模型和可视化图表
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

        print("训练完成!")

    except Exception as e:
        print(f"训练过程出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 支持多种配置文件路径
    config_paths = [
        "./src/training_params.yaml",
        "./artifacts/20241115_212647/training_params.yaml"
    ]

    config_path = config_paths[0]


    if config_path:
        print(f"使用配置文件: {config_path}")
        config = load_config_from_yaml(config_path)
        main(config)
    else:
        print("未找到配置文件，请检查配置文件路径")
        sys.exit(1)
