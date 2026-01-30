import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)


# 序列编码函数
def seq_to_num(line,max_length): # max_length: 长度, line: 单条序列（蛋白质字符串）
    protein_dict = { 
                'Z':0, 
                'A':1, 
                'C':2, 
                'D':3, 
                'E':4, 
                'F':5, 
                'G':6, 
                'H':7, 
                'I':8,
                'K':9, 
                'L':10, 
                'M':11, 
                'N':12, 
                'P':13, 
                'Q':14, 
                'R':15, 
                'S':16, 
                'T':17, 
                'V':18, 
                'W':19, 
                'Y':20}
    seq = np.zeros(max_length) # 创建长度 200 的全零数组
    for j in range(len(line)): # 倒序填充
        seq[max_length - 1 - j] = protein_dict[line[len(line) - 1 - j]]
    return seq


def Preprocess(max_length, true_test_file=None, false_test_file=None, train_true_file=None, train_false_file=None, valid_true_file=None, valid_false_file=None): # 预处理函数
    result = {}

    #################################
    # X_test 中加载测试序列 来自 true_test_file 和 false_test_file
    if true_test_file is not None and false_test_file is not None:
        X_test = [] # 所有测试序列编码
        Y_test = [] # 测试标签
        text = []  # temp存储

        file =open(true_test_file,'r')
        read_text = file.readlines()
        file.close()
        text.extend(read_text) # 将另一个可迭代对象中的所有元素添加到列表末尾
        Y_test.extend((np.zeros(len(read_text)//2) + 1).tolist()) # true_test_file(正样本)部分为1

        file =open(false_test_file,'r')
        read_text = file.readlines()
        file.close()
        text.extend(read_text)
        Y_test.extend(np.zeros(len(read_text)//2).tolist()) # false_test_file(负样本)部分为0

        for i in range(len(text)//2):
            line = text[i*2+1] # 单条序列（蛋白质字符串）
            line = line[0:len(line)-1] # 去掉换行符
            seq = seq_to_num(line,max_length) # seq_to_num 调用 编码后的序列
            X_test.append(seq)

        result['X_test'] = np.array(X_test)
        result['Y_test'] = np.array(Y_test).astype(int)

    #################################
    # X_train 中加载训练序列 来自 train_true_file 和 train_false_file
    if train_true_file is not None and train_false_file is not None:
        X_train = [] # 所有训练序列编码
        Y_train = [] # 训练标签
        text = []
        
        file =open(train_true_file,'r')
        read_text = file.readlines()
        file.close()
        text.extend(read_text)
        Y_train.extend((np.zeros(len(read_text)//2) + 1).tolist()) # train_true_file(正样本)部分为1
        
        file =open(train_false_file,'r')
        read_text = file.readlines()
        file.close()
        text.extend(read_text)
        Y_train.extend(np.zeros(len(read_text)//2).tolist()) # train_false_file(负样本)部分为0

        for i in range(len(text)//2):
            line = text[i*2+1]
            line = line[0:len(line)-1]
            seq = seq_to_num(line,max_length)
            X_train.append(seq)

        result['X_train'] = np.array(X_train)
        result['Y_train'] = np.array(Y_train).astype(int)
    #################################

    # X_val 中加载验证序列 来自 valid_true_file 和 valid_false_file
    if valid_true_file is not None and valid_false_file is not None:
        X_val = [] # 所有验证序列编码
        Y_val = [] # 验证标签
        text = []
        
        file =open(valid_true_file,'r')
        read_text = file.readlines()
        file.close()
        text.extend(read_text)
        Y_val.extend((np.zeros(len(read_text)//2) + 1).tolist()) # valid_true_file(正样本)部分为1
        
        file =open(valid_false_file,'r')
        read_text = file.readlines()
        file.close()
        text.extend(read_text)
        Y_val.extend(np.zeros(len(read_text)//2).tolist()) # valid_false_file(负样本)部分为0

        for i in range(len(text)//2):
            line = text[i*2+1]
            line = line[0:len(line)-1]
            seq = seq_to_num(line,max_length)
            X_val.append(seq)

        result['X_val'] = np.array(X_val)
        result['Y_val'] = np.array(Y_val).astype(int)
        #################################
    return result
 

##############################################
## torch style 
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

class APIN(nn.Module):
    def __init__(self, attention_num, attention_range, embed_length, max_length):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=embed_length, padding_idx=0)
         # transpose(1, 2) forward 时使用
        self.attention_layers = nn.ModuleList()
        for i in range(attention_range):
            attention_layer = nn.Sequential(
                nn.Conv1d(in_channels=embed_length, out_channels=attention_num, kernel_size=i*2+1, padding='same',stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=max_length),

            )
            self.attention_layers.append(attention_layer)
        # concatenate
        self.maxpool = nn.MaxPool1d(kernel_size=attention_range)
        self.reshape = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(attention_num, 1)
        self.sigmoid = nn.Sigmoid()        

    def forward(self, x):
        out = self.embedding(x)
        out_trans= out.transpose(1, 2)  # 转换为 (batch_size, embed_length, max_length)
        attention_outputs = []
        for attention_layer in self.attention_layers:
            attention_output = attention_layer(out_trans)  # (batch_size, attention_num, 1)
            attention_outputs.append(attention_output)
        # concatenate
        out = torch.cat(attention_outputs, dim=2)  # (batch_size, attention_num, attention_range)
        out = self.maxpool(out)  # (batch_size, attention_num, 1)
        out = self.reshape(out)  # (batch_size, attention_num)
        out = self.dropout(out)
        out = self.dense(out) # 不添加sigmoid
        return out

def train_APIN(epochs,
               classifier,
               assign,
               train_data_loader,
               validation_data_loader, # 默认test_data_loader,
               rDLM=0,
               attention_num=64,
               attention_range=14,
               embed_length=128,
               max_length=200,
               device='cuda'):
    
    if assign == 'iDLM':
        model_pth = f'weight/{assign}_weights/{classifier}_classifier_{assign}.pth'
    else:
        model_pth = f'weight/{assign}_weights/{classifier}{rDLM}_classifier_{assign}.pth'

    max_acc = 0
    max_epoch = 0
    model = APIN(attention_num, attention_range, embed_length, max_length)
    model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0002)
    loss_fn = nn.BCEWithLogitsLoss() # 单输出, 二分类, 
        
    for epoch in range(epochs):
        train_corr = 0   # 累计正确预测数
        train_total = 0   # 累计样本总数
        valid_corr = 0
        valid_total = 0
        for i, (x, Y) in enumerate(train_data_loader):
            # Forward pass
            x,Y = x.to(device), Y.to(device)
            
            train_logits = model(x).squeeze()
            loss = loss_fn(train_logits, Y)

            pred_class = (torch.sigmoid(train_logits) > 0.5)
            train_corr += torch.sum(pred_class == Y)  # 累计正确数
            train_total += Y.size(0)  # size(0) batch_size > 累计总样本数，这里 +128

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info("Epoch[{}/{}], CE Loss: {:.4f}".format(epoch+1, epochs, loss.item()))
        
        # 在epoch中的评估, 不是模型完全训练结束后, 而在rDLM中是在结束后评估
        ## 两个脚本目标不同, iDLM是要找到最佳模型, rDLM只是训练多个模型, 不管好坏, 用作后续参考
        with torch.no_grad():

            model.eval() # 设置为评估模式, 里面有dropout和batchnorm等
            for valid_x, valid_Y in validation_data_loader:
                valid_x, valid_Y = valid_x.to(device), valid_Y.to(device)

                valid_logits = model(valid_x).squeeze()
                valid_result = (torch.sigmoid(valid_logits) > 0.5)
                valid_corr += torch.sum(valid_result == valid_Y)
                valid_total += valid_x.size(0)
            model.train() # 切换回训练模式

            train_acc = 100*float(train_corr)/train_total
            validation_acc = 100*float(valid_corr)/valid_total 
            logger.info('train: %d/%d correct (%.2f%%)' 
                    % (train_corr, train_total, train_acc))
            logger.info("validation: %d/%d correct (%.2f%%) | prev max %.2f%% @ %d epoch\n" 
                    % (valid_corr, valid_total, validation_acc, max_acc, max_epoch))

        if validation_acc > max_acc:
            max_acc = validation_acc
            max_epoch = epoch
            torch.save(model.state_dict(), model_pth)
        if epoch-max_epoch > 100: # 因为为validation不提升超过100个epoch就停止训练
            break
    return model_pth

def Prediction(model_pth, 
               test_data_loader, 
               classifier,
               assign,
               rDLM=0,
               attention_num=64, 
               attention_range=14, 
               embed_length=128, 
               max_length=200, 
               device='cuda'): # 预测函数

    if assign == 'iDLM':
        save_pth = f'output_repair/prediction_output_{classifier}_{assign}'
    elif assign == 'rDLM':
        save_pth = f'output_repair/rDLM/prediction_output_{classifier}_{assign}_{rDLM}'
    else:
        save_pth = f'output_repair/repaired_prediction_output_{classifier}_iDLM'
    model = APIN(attention_num, attention_range, embed_length, max_length)
    model.load_state_dict(torch.load(model_pth))
    model.to(device)
    model.eval()

    Y_pred = []
    Y_true = []
    Y_prob = []
    for x,Y in test_data_loader:
        x, Y = x.to(device), Y.to(device).int()
        result = model(x).squeeze()
        prob = torch.sigmoid(result)
        pred = (prob > 0.5).int()
        Y_pred.extend(pred.cpu().tolist())
        Y_true.extend(Y.cpu().tolist())
        Y_prob.extend(prob.cpu().tolist())
    # 保存预测结果
    df = pd.DataFrame({
        'id': range(len(Y_pred)),
        'Pred': Y_pred,
        'Theory': Y_true,
        'Prob': Y_prob
    })
    df.to_csv(f'{save_pth}.csv', index=False)

    accuracy = np.mean(np.array(Y_pred) == np.array(Y_true)) if Y_pred else 0.0
    logger.info('Prediction accuracy: %.2f%%'% (accuracy * 100))


def data_loader(max_length, 
                clip_proportion=1, 
                
                train_batch_size=None,
                vali_batch_size=None,
                test_batch_size=None,

                train_true_file=None, 
                train_false_file=None, 

                valid_true_file=None, 
                valid_false_file=None, 
                
                true_test_file=None, 
                false_test_file=None):
    output = {}
    data = Preprocess(max_length, true_test_file, false_test_file, train_true_file, train_false_file, valid_true_file, valid_false_file)  # 变成list
    
    # 默认 batch size 一致
    if vali_batch_size is None:
        vali_batch_size = train_batch_size
    if test_batch_size is None:
        test_batch_size = train_batch_size

    # 加载到data loader中
    try:
        X_train = data.get('X_train')
        Y_train = data.get('Y_train')

        # 切割数据集 rDLM
        if clip_proportion < 1:
            X_train, _, Y_train, _= train_test_split(
                X_train, Y_train, 
                train_size=clip_proportion,
                stratify=Y_train,
            )
        X_train_tensor = torch.tensor(X_train, dtype=torch.long)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

        data_train = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
        
        train_loader = torch.utils.data.DataLoader(
            dataset=data_train,
            batch_size=train_batch_size,  
            shuffle=True,     
        )
        output['train_loader'] = train_loader

        logger.info("训练数据集大小: %s", len(Y_train))
    except:
        logger.info("No training data loaded.")

    try:
        X_val = data.get('X_val')
        Y_val = data.get('Y_val')

        X_val_tensor = torch.tensor(X_val, dtype=torch.long)
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
        data_val = torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor)
        
        val_loader = torch.utils.data.DataLoader(
            dataset=data_val,
            batch_size=vali_batch_size,
            shuffle=False,     
        )
        output['val_loader'] = val_loader
        
        logger.info("验证数据集大小: %s", len(Y_val))
    except:
        logger.info("No validation data loaded.")

    try:
        X_test = data.get('X_test')
        Y_test = data.get('Y_test')

        X_test_tensor = torch.tensor(X_test, dtype=torch.long)
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
        data_test = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)
    
        test_loader = torch.utils.data.DataLoader(
            dataset=data_test,
            batch_size=test_batch_size,
            shuffle=False, 
        )
        output['test_loader'] = test_loader

        logger.info("测试数据集大小: %s", len(Y_test))
    except:
        logger.info("No test data loaded.")
    
    return output

#################################

import numpy as np
import math
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_classification_metrics(y_true, y_pred, y_prob=None):
    """
    :param y_true: 真实标签 (0/1)
    :param y_pred: 预测标签 (0/1)
    :param y_prob: 预测概率
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob) if y_prob is not None else None
    
    # 1. 计算混淆矩阵基础项 (TP, FP, FN, TN)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    ## Sn / Recall 真阳性率
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    ## 假阴性率 1-Recall
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    ## Sp / Specificity 真阴性率
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    ## 假阳性率 1-Specificity
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0

    ## Precision / Pr
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    # Balanced Accuracy (Ba)
    ba = (tpr + tnr) / 2

    # F1-Score
    f1 = 2 * (ppv * tpr) / (ppv + tpr) if (ppv + tpr) > 0 else 0
    
    # MCC
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
    
    # Kappa (卡帕系数)
    p0 = acc
    pe_num = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn))
    pe_den = (tp + tn + fp + fn) ** 2
    pe = pe_num / pe_den if pe_den > 0 else 0
    kappa = (p0 - pe) / (1 - pe) if (1 - pe) > 0 else 0

    # 面积指标 (AUC-ROC, AUC-PR)
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)

    # 5. 结果封装
    return {
        "Conf_Matrix": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
        "Sensitivity (Sn|Recall)": round(tpr, 4), 
        "Specificity (Sp|Spec)": round(tnr, 4),
        "Precision (Pr)": round(ppv, 4),
        "Accuracy (Acc)": round(acc, 4),
        "Balanced Accuracy (Ba)": round(ba, 4),
        "F1-Score": round(f1, 4),
        "Matthews Correlation Coefficient (MCC)": round(mcc, 4),
        "Kappa": round(kappa, 4),
        "AUC-ROC": round(auc_roc, 4) if y_prob is not None else "N/A",
        "AUC-PR": round(auc_pr, 4) if y_prob is not None else "N/A",
    }