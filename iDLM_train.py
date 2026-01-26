# iDLM（Individual Deep Learning Model）

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from utils import data_loader, train_APIN, Prediction

Path('output/rDLM').mkdir(parents=True, exist_ok=True)
Path('logs').mkdir(parents=True, exist_ok=True)
Path('weight/iDLM_weights').mkdir(parents=True, exist_ok=True)
Path('weight/repair').mkdir(parents=True, exist_ok=True)
Path('weight/rDLM_weights').mkdir(parents=True, exist_ok=True)

# Hyper-parameters
num_epochs = 30
batch_size = 128
train_true_file = 'data/DAMP/AMP.tr.fa' 
train_false_file = 'data/DAMP/Non-AMP.tr.fa'
valid_true_file = 'data/DAMP/AMP.eval.fa'
valid_false_file = 'data/DAMP/Non-AMP.eval.fa'
test_true_file = 'data/DAMP/AMP.te.fa'
test_false_file = 'data/DAMP/Non-AMP.te.fa'
attention_num=64
attention_range=14
embed_length=128
max_length=200

classifier = 'APIN'
assign = 'iDLM'

# log recording
log_file = Path(f'logs/{classifier}_iDLM_train.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 加载训练集
output = data_loader(
    # 先加载Preprocess，再加载DataLoader
    max_length, 
    train_batch_size=batch_size, 
    train_true_file=train_true_file, 
    train_false_file=train_false_file,
    valid_true_file=valid_true_file, 
    valid_false_file=valid_false_file,
    true_test_file=test_true_file, 
    false_test_file=test_false_file
)
train_loader = output.get('train_loader')
val_loader = output.get('val_loader')
test_loader = output.get('test_loader')

# Draw_distribution(X_train, Y_train, X_val, Y_val, X_test, Y_test)
logger.info('---Training APIN iDLM---')
model_pth = train_APIN(
    num_epochs,
    classifier,
    assign,
    train_loader,
    val_loader,
    attention_num=attention_num,
    attention_range=attention_range,
    embed_length=embed_length,
    max_length=max_length
)

logger.info('---Testing APIN iDLM---')
Prediction(model_pth, 
           test_loader, 
           classifier,
           assign,
           attention_num=attention_num, 
           attention_range=attention_range, 
           embed_length=embed_length, 
           max_length=max_length)
