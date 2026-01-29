import os
import torch
import torch.nn as nn
import time
from pathlib import Path
import logging
from utils import data_loader, APIN, Prediction


# 对所有权重进行调整, 重新赋值
def setWeights(model, weight_list): # 把一组外部给定的权重列表逐个写回模型对应的参数位置
    for weight, (name, v) in zip(weight_list, model.named_parameters()): #导出带名称的参数迭代器
        attrs = name.split('.') # name = "conv1.weight" 可能有bias, 所有weigh需要保存
        obj = model # 这里内存占用一块, 不能新建一个obj, 不然不会修改到model本体
        for attr in attrs:
            obj = getattr(obj, attr) # getattr(对象, 属性名), 访问, 相当于 obj.attr
            # 逐层定位到具体的参数对象
                # getattr(model, "conv1") → 得到 conv1 层
                # getattr(conv1, "weight") → 得到 weight 参数
        obj.data = weight

def AdjustWeights(baseWeights, corrDiff, incorrDiff, a, b, strategy='both-org', lr=1e-3):
    #  baseWeights 主模型iDLM的权重
    # 主模型与"预测正确的子模型"平均权重的差值
    # 主模型与"预测错误的子模型"平均权重的差值
    # a 预测正确的子模型数量
    # b 预测错误的子模型数量
    # strategy 调整策略：both / both-org（两者都用）、corr / corr-org（只用正确）、incorr / incorr-org（只用错误）
    
    if 'org' in strategy:
        sign = 1 # 沿着正确子模型方向微调
    else:
        sign = -1 # 逆向
    
    p_corr, p_incorr = a/(a+b), b/(a+b) # 正确/错误占比
    
    if 'both' in strategy:
        return [b_w + sign*lr*(p_corr*cD - p_incorr*iD) 
                for b_w, cD, iD in zip(baseWeights, corrDiff, incorrDiff)]
    elif 'corr' in strategy:
        return [b_w + sign*lr*p_corr*cD 
                for b_w, cD in zip(baseWeights, corrDiff)]
    elif 'incorr' in strategy:
        return [b_w - sign*lr*p_incorr*iD 
                for b_w, iD in zip(baseWeights, incorrDiff)]
    else:
        raise ValueError(f'Unrecognized strategy {strategy}')

def EvaluateModel(model, loader):   # 比较acc
    with torch.no_grad():
        total = 0.
        correct = 0.
        model.eval()
        for tx, tx_class in loader:
            tx,tx_class = tx.to(device), tx_class.to(device)
            tclass_logits = model(tx).squeeze()
            result = (torch.sigmoid(tclass_logits) > 0.5)
            total += tx.size(0)
            correct += torch.sum(result == tx_class) # 两个 Tensor 必须在同一个设备上才能比较。
    return 100*float(correct)/total

def TrainingModel(model, epoch_num): # 短期微调训练
    model.train()
    for i in range(1,epoch_num+1):
        logger.info('Process... Retraining model %.2f%%', i/epoch_num*100)
        for i, (x, x_class) in enumerate(train_loader):
            # Forward pass
            x,x_class = x.to(device), x_class.to(device)
            class_logits = model(x).squeeze()

            # Backprop and optimize
            class_loss = loss_fn(class_logits, x_class)
            loss = class_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()


Path('output_repair/rDLM').mkdir(parents=True, exist_ok=True)
Path('logs').mkdir(parents=True, exist_ok=True)
Path('weight/iDLM_weights').mkdir(parents=True, exist_ok=True)
Path('weight/repair').mkdir(parents=True, exist_ok=True)
Path('weight/rDLM_weights').mkdir(parents=True, exist_ok=True)


retrain_epoch_num = 2
assign='repaired'
classifier = 'APIN'
idlm_path = f'weight/iDLM_weights/{classifier}_classifier_iDLM.pth'
rdlm_dir = 'weight/rDLM_weights/'
save_model_pth = f'weight/repair/{classifier}_fixed_patch.pth'
device = 'cuda'

# log recording
log_file = Path(f'logs/{classifier}_weight_adjust.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

max_length=200
train_batch_size = 20 
vali_batch_size = 256
test_batch_size = 256 
train_true_file = 'data/DAMP/AMP.tr.fa' 
train_false_file = 'data/DAMP/Non-AMP.tr.fa'
valid_true_file = 'data/DAMP/AMP.eval.fa'
valid_false_file = 'data/DAMP/Non-AMP.eval.fa'
test_true_file = 'data/DAMP/AMP.te.fa'
test_false_file = 'data/DAMP/Non-AMP.te.fa'

# 加载训练集
output = data_loader(
    # 先加载Preprocess，再加载DataLoader
    max_length, 
    train_batch_size=train_batch_size,
    vali_batch_size=vali_batch_size,
    test_batch_size=test_batch_size,
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

attention_num = 64
attention_range = 14
embed_length = 128

# neural net initialization
iDLM = APIN(attention_num, attention_range, embed_length, max_length)
iDLM.load_state_dict(torch.load(idlm_path))
iDLM.to(device)
iDLM.eval()
optimizer = torch.optim.RMSprop(iDLM.parameters(), lr=0.0002)
loss_fn = nn.BCEWithLogitsLoss()

rDLMs = []
for filename in os.listdir(rdlm_dir):
    rDLM = APIN(attention_num, attention_range, embed_length, max_length)
    rDLM.load_state_dict(torch.load(rdlm_dir + filename))
    rDLM.to(device)
    rDLM.eval()
    rDLMs.append(rDLM)
###############################

# --- actual process --- #
with torch.no_grad(): # 保存初始权重, 用于回溯
    best_weights = list(map(lambda x: x.data.clone(), iDLM.parameters())) # 不可以用.data, 是浅拷贝

tr_acc = EvaluateModel(iDLM, train_loader)
val_acc = EvaluateModel(iDLM, val_loader)
t_acc = EvaluateModel(iDLM, test_loader)
logger.info('org best acc (train): %.2f%% | (val): %.2f%% | (test): %.2f%%\n', tr_acc, val_acc, t_acc)
last_improvement = 0

s = time.time()

all_batches = len(train_loader)
for b_idx, (x, x_class) in enumerate(train_loader): # 一个batch
    # Forward pass
    with torch.no_grad():
        x, x_class = x.to(device), x_class.to(device)
        yOrigin = (torch.sigmoid(iDLM(x)) > 0.5).float()  # 主模型对当前batch的预测结果
        ySubList = [(torch.sigmoid(rdlm(x)) > 0.5).float() for rdlm in rDLMs] # 所有子模型对当前batch的预测结果 
        for i_idx in range(x.size(0)): # (用train_batch_size最后一个batch不够可能会越界) 遍历一个batch中的每个样本 
            single_class = x_class[i_idx] # 一个样本的真实标签
            if yOrigin[i_idx].item() == single_class.item():
                continue
            else: # rdlm 是模型
                correctSubModels = [rdlm for r_idx, rdlm in enumerate(rDLMs) if ySubList[r_idx][i_idx].item() == single_class.item()]
                incorrectSubModels = [rdlm for r_idx, rdlm in enumerate(rDLMs) if ySubList[r_idx][i_idx].item() != single_class.item()]
                if len(correctSubModels) == 0 or len(incorrectSubModels) == 0: # ???
                    continue # slightly different from paper
                
                # 原始：按模型组织
                # 模型1: [w0, w1, w2, w3]
                # 模型2: [w0, w1, w2, w3]
                # 模型3: [w0, w1, w2, w3]

                # ziw(*...) 后：按参数位置组织
                # 参数0: (模型1的w0, 模型2的w0, 模型3的w0) → sum → 参数0的总和
                # 参数1: (模型1的w1, 模型2的w1, 模型3的w1) → sum → 参数1的总和
                # 统计数字
                ## 正确模型的平均权重
                correctWeightSum = [sum(t) for t in zip(*[m.parameters() for m in correctSubModels])]
                correctWeights = [e/len(correctSubModels) for e in correctWeightSum]
                # 错误模型的平均权重
                incorrWeightSum = [sum(t) for t in zip(*[m.parameters() for m in incorrectSubModels])]
                incorrWeights = [e/len(incorrectSubModels) for e in incorrWeightSum]

                baseWeights = list(map(lambda x: x.data, iDLM.parameters()))
                # 每个权重位置的调整量
                corrDiff = [b_w - c_w for b_w, c_w in zip(baseWeights, correctWeights)] 
                incorrDiff = [b_w - i_w for b_w, i_w in zip(baseWeights, incorrWeights)] 
                baseWeights = AdjustWeights(
                    baseWeights, corrDiff, incorrDiff,
                    len(correctSubModels), len(incorrectSubModels),
                    strategy='both-org', lr=1e-3
                )
                setWeights(iDLM, baseWeights) # 以一个样本为单位调整权重
    
    #一个batch评估一次, 验证集
    currAcc = EvaluateModel(iDLM, val_loader)
    currTrain = EvaluateModel(iDLM, train_loader)
    logger.info('batch %d/%d done, last improvement %d batches ago', b_idx, all_batches, b_idx-last_improvement)
    logger.info('New accuracy prior training in (train): %.2f%% | (val) %.2f%%', currTrain, currAcc)

    if val_acc < currAcc:
        with torch.no_grad():
            best_weights = list(map(lambda x: x.data.clone(), iDLM.parameters()))
        
        logger.info('Found new best weights!')
        t_acc = EvaluateModel(iDLM, test_loader) # 用测试集评估一次
        logger.info('new best acc (val): %.2f%% | (test): %.2f%%', currAcc, t_acc)
        val_acc = currAcc
        last_improvement = b_idx
    else:
        if val_acc > currAcc:
            with torch.no_grad():
                setWeights(iDLM, best_weights) # 还原到最佳权重, ==就没必要换了, 贪心
        if last_improvement + 100 < b_idx:
            logger.info('No improvement for too long, terminating')
            break

    TrainingModel(iDLM,retrain_epoch_num) # 每batch之后, 每次adjustment之后, 进行短期微调
    currAcc_after_train = EvaluateModel(iDLM, val_loader)
    currTrain_after_train = EvaluateModel(iDLM, train_loader)
    logger.info('New accuracy (train) post training: %.2f%% | (val) %.2f%%\n', currTrain_after_train, currAcc_after_train)

e = time.time()
logger.info('Total execution time: %s', e-s)
    
setWeights(iDLM, best_weights)
val_acc = EvaluateModel(iDLM, val_loader)
logger.info('new best acc (val): %.2f%%\n', val_acc)
logger.info('test accuracy as follows:')
torch.save(iDLM.state_dict(), save_model_pth)

Prediction(save_model_pth, 
            test_loader, 
            classifier,
            assign,
            attention_num=64, 
            attention_range=14, 
            embed_length=128, 
            max_length=200)
