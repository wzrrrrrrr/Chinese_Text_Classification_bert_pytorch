from src.dataset import load_and_process_data
from src.train import train
from src.evaluate import evaluate
from src.visualization_utils import plot_loss_curve, plot_accuracy_curve, \
    plot_lr_curve  # 导入可视化函数
import os
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 初始化时间戳和训练目录
artifacts_dir = 'artifacts'
os.makedirs(artifacts_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
current_training_dir = os.path.join(artifacts_dir, timestamp)
os.makedirs(current_training_dir, exist_ok=True)

# 创建子目录
models_dir = os.path.join(current_training_dir, 'models')
visualizations_dir = os.path.join(current_training_dir, 'visualizations')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

# 记录总的开始时间
total_start_time = time.time()

# 配置文件路径和映射字典
data_path = 'data/weibo-hot-search-labeled.csv'
csv_map_str = "{0: '科技', 1: '娱乐', 2: '时事'}"
label_column = '标签（时政、科技、科普、娱乐、体育、社会讨论/话题、时事、经济）'
text_column = '热搜词条'
sample_size = 30
model_path = 'bert-base-chinese'
selection_method = "random"

# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained(model_path)

# 加载并处理数据集
train_dataset, test_dataset, label_map = load_and_process_data(
    data_path=data_path,
    tokenizer=tokenizer,
    label_column=label_column,
    text_column=text_column,
    csv_map_str=csv_map_str,
    sample_size=sample_size,
    test_size=0.2,
    selection_method=selection_method
)

# 模型和数据集超参数
num_labels = len(label_map)
batch_size = 8
num_epochs = 50
learning_rate = 2e-5
patience = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 BERT 模型并调整分类层
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, use_safetensors=False)
model.to(device)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 定义优化器与学习率调度器
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

# 初始化早停参数和全局变量
best_val_loss = float('inf')
epochs_without_improvement = 0
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
learning_rates = []
draw_confusion_matrix = False
draw_classification_report = False

# 开始训练和验证循环
for epoch in range(num_epochs):
    start_time = time.time()
    time.sleep(1)

    # 训练阶段
    train_loss, train_accuracy = train(model, train_loader, optimizer, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

    # 验证阶段
    test_loss, test_accuracy = evaluate(
        model, test_loader, device, label_map, epoch=epoch + 1,
        draw_confusion_matrix=draw_confusion_matrix,
        draw_classification_report=draw_classification_report
    )
    val_losses.append(test_loss)
    val_accuracies.append(test_accuracy)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.4f}")

    # 记录学习率
    learning_rates.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

    # 保存最佳模型
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        epochs_without_improvement = 0
        best_model_path = os.path.join(models_dir, f"bert_epoch{epoch + 1}_val_loss{test_loss:.4f}.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"验证损失有改进，最佳模型已保存至 {best_model_path}！")
    else:
        epochs_without_improvement += 1
        print(f"验证损失无改进，已连续 {epochs_without_improvement} 个 epoch 无改善")

    if epochs_without_improvement >= patience:
        print("早停触发，训练提前结束！")
        break

    end_time = time.time()
    print(f"Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds.")
    print("=" * 90)

# 总时间记录
total_end_time = time.time()
total_time = total_end_time - total_start_time
print(f"Total training time: {total_time:.2f} seconds.")

# 保存最终模型
final_model_path = os.path.join(models_dir, "bert_text_classification_final.pth")
torch.save(model.state_dict(), final_model_path)
print(f"最终模型已保存至 {final_model_path}！")

# 绘制并保存可视化图表
plot_loss_curve(train_losses, val_losses, save_path=os.path.join(visualizations_dir, "loss_curve.png"))
plot_accuracy_curve(train_accuracies, val_accuracies, save_path=os.path.join(visualizations_dir, "accuracy_curve.png"))
plot_lr_curve(learning_rates, save_path=os.path.join(visualizations_dir, "lr_curve.png"))