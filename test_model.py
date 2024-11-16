import os
import sys
from datetime import datetime

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from main import load_config_from_yaml
from src.dataset import load_and_process_data


def evaluate_model(model, test_loader, device, label_map):
    model.to(device)
    all_preds = []
    all_labels = []
    total_loss = 0
    total_samples = 0

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

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(label_map.keys()),
        target_names=list(label_map.values()),
        digits=4
    )

    cm = confusion_matrix(all_labels, all_preds)

    # Add accuracy to the report
    report += f"\nAccuracy: {accuracy:.4f}"

    return {
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_labels
    }


def plot_confusion_matrix(cm, label_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_model_performance():
    config_paths = [
        "./src/training_params.yaml",
        "./artifacts/20241116_204746/training_params.yaml"
    ]
    best_model_path = "./artifacts/20241116_204746/models/bert_epoch2_val_loss0.4677.pth"

    config_path = config_paths[1]
    print(f"使用配置文件: {config_path}")
    config = load_config_from_yaml(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(config.MODEL_PATH)

    test_dataset, _, label_map = load_and_process_data(
        data_path=config.TEST_DATA_PATH,
        tokenizer=tokenizer,
        label_column=config.LABEL_COLUMN,
        text_column=config.TEXT_COLUMN,
        csv_map_str=config.CSV_MAP_STR,
        sample_size=None,
        test_size=None
    )

    # Load the model with the specified path
    model = BertForSequenceClassification.from_pretrained(config.MODEL_PATH, num_labels=len(label_map))


    model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()  # Set the model to evaluation mode

    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    results = evaluate_model(model, test_loader, device, label_map)

    assert results['avg_loss'] is not None
    assert results['accuracy'] >= 0  # 确保准确率为非负值
    print(f"Average Loss: {results['avg_loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:\n", results['classification_report'])

    if config.EVALUATION_CONFIG['draw_confusion_matrix']:
        # 获取当前时间并格式化为字符串
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建带有时间戳的文件名
        confusion_matrix_filename = f'test_confusion_matrix_{timestamp}.png'

        plot_confusion_matrix(
            results['confusion_matrix'],
            list(label_map.values()),
            os.path.join(config.ARTIFACTS_DIR, confusion_matrix_filename)
        )
