# 数据集配置
DATA_PATH = 'data/weibo-hot-search-labeled.csv'
LABEL_COLUMN = '标签（时政、科技、科普、娱乐、体育、社会讨论/话题、时事、经济）'
TEXT_COLUMN = '热搜词条'
SAMPLE_SIZE = 30
TEST_SIZE = 0.2
SELECTION_METHOD = "random"
CSV_MAP_STR = "{0: '科技', 1: '娱乐', 2: '时事'}"

# 模型配置
MODEL_PATH = 'bert-base-chinese'

# 训练配置
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 2e-5
PATIENCE = 3

# 学习率调度器参数
SCHEDULER_STEP_SIZE = 1
SCHEDULER_GAMMA = 0.9
# 目录配置
ARTIFACTS_DIR = 'artifacts'  # 放在项目根目录的文件夹 artifacts

# 是否每轮绘制混淆矩阵和分类报告的配置
EVALUATION_CONFIG = {
    'draw_confusion_matrix': True,  # 是否绘制混淆矩阵
    'draw_classification_report': True  # 是否绘制分类报告
}
