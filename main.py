import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from src.dataset import load_and_process_data
from src.train import train
from src.evaluate import evaluate

# 配置文件路径和映射字典
data_path = 'data/weibo-hot-search-labeled.csv'  # 数据集路径
csv_map_str = "{0: '科技', 1: '娱乐', 2: '时事'}"  # 标签映射，字符串格式
label_column = '标签（时政、科技、科普、娱乐、体育、社会讨论/话题、时事、经济）'  # 标签列
text_column = '热搜词条'  # 文本内容列
sample_size = 10  # 每个类别样本数量限制
model_path = 'model/bert-base-chinese'  # BERT 模型路径
selection_method = "random"  # 数据选择方式，"random"、"top" 或 "bottom"
label_map = eval(csv_map_str)  # 将标签映射字符串解析为字典

# 模型和数据集超参数
batch_size = 8  # 每批数据大小
num_epochs = 50  # 最大训练轮数
learning_rate = 2e-5  # 学习率
patience = 3  # 早停的耐心参数（若损失无改进的最大轮数）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU 或 CPU 训练

# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained(model_path)

# 加载并处理数据集
train_dataset, test_dataset, num_labels = load_and_process_data(
    data_path=data_path,
    tokenizer=tokenizer,
    label_column=label_column,
    text_column=text_column,
    label_map=None,
    sample_size=None,
    test_size=0.2,
    selection_method=selection_method
)

# 加载 BERT 模型并调整分类层
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, use_safetensors=False)
model.to(device)  # 模型移动到计算设备

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 定义优化器与学习率调度器
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)  # 每个 epoch 后将学习率乘以 0.9

# 初始化早停参数
best_val_loss = float('inf')  # 设置验证损失的初始最优值
epochs_without_improvement = 0  # 连续无改进 epoch 计数器

# 开始训练和验证循环
for epoch in range(num_epochs):
    # 训练阶段
    start_time = time.time()
    time.sleep(1)
    train_loss, train_accuracy = train(model, train_loader, optimizer, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

    # 验证阶段
    time.sleep(1)
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.4f}")

    # 学习率调整
    scheduler.step()

    # 检查验证损失是否有改进以实现早停机制
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'bert_text_classification_best.pth')
        print("验证损失有改进，已保存当前最佳模型！")
    else:
        epochs_without_improvement += 1
        print(f"验证损失无改进，已连续 {epochs_without_improvement} 个 epoch 无改善")

    # 若验证损失无改进超过 patience 阈值，停止训练
    if epochs_without_improvement >= patience:
        print("早停触发，训练提前结束！")
        break

    # 输出当前轮次运行时间
    end_time = time.time()
    print(f"Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds.")
    print("=" * 90)

# 保存最终模型
torch.save(model.state_dict(), 'bert_text_classification_final.pth')
print("最终模型已保存！")