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
num_epochs = 5
clip_proportion = 0.5 # 使用部分数据进行训练
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

##########################################################################
classifier = 'APIN'
assign = 'rDLM'
rDLM_num = 20
model_pth_list = []

# log recording
log_file = Path(f'logs/{classifier}_rDLM_train.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

output = data_loader(
    # 先加载Preprocess，再加载DataLoader
    max_length, 
    train_batch_size=batch_size, # train 可以控制 valid 和 test batch size
    # train_true_file=train_true_file,  # 如果没有单独的验证集的话, 需要train dataset来分割验证集
    # train_false_file=train_false_file,
    valid_true_file=valid_true_file, 
    valid_false_file=valid_false_file,
    true_test_file=test_true_file, 
    false_test_file=test_false_file
)
test_loader = output.get('test_loader')
val_loader = output.get('val_loader')

logger.info('---Training APIN rDLM---')
for rDLM in range(rDLM_num):
    # 加载训练集
    output = data_loader(
        max_length, 
        clip_proportion=clip_proportion,
        train_batch_size=batch_size, 
        train_true_file=train_true_file, 
        train_false_file=train_false_file
    )
    train_loader = output.get('train_loader')

    logger.info('---Training rDLM #%d---', rDLM)
    model_pth = train_APIN(
        num_epochs,
        classifier,
        assign,
        train_loader,
        val_loader,
        rDLM=rDLM,
        attention_num=attention_num,
        attention_range=attention_range,
        embed_length=embed_length,
        max_length=max_length
    )
    model_pth_list.append(model_pth)


logger.info('---Testing APIN rDLM---')
for rDLM in range(rDLM_num):
    logger.info('---Testing APIN rDLM #%d---', rDLM)
    model_pth = model_pth_list[rDLM]
    Prediction(model_pth, 
            test_loader, 
            classifier,
            assign,
            rDLM=rDLM,
            attention_num=attention_num, 
            attention_range=attention_range, 
            embed_length=embed_length, 
            max_length=max_length)