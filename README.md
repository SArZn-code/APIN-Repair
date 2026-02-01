Neural Network Repair of Convolutional Neural Networks (CNN) for Antimicribial Peptide Prediction Model
# Neural network
Derived from APIN (MultipleScale-CNN)
reference: 10.1186/s12859-019-3327-y
# Repair technique
Derived from Apricot (heavy repair method)
reference: 10.1109/ASE.2019.00043
# Pepline
1. original model train (if have not trained .pth)
`iDLM.py`
2. repair prepared
`rDLM.py`
3. repair
`weight_adjust.py`
4. repair evaluation
`repair_evaluation.py`
5. virtual screening using genetic algorithm 
`GA.py`
6. GA evaluation
`GA_evaluation.py`
# data collection

# Repair performance metric
Conf_Matrix
Sensitivity (Sn|Recall)
Specificity (Sp|Spec)
Precision (Pr)
Accuracy (Acc)
Balanced Accuracy (Ba)
F1-Score
Matthews Correlation Coefficient (MCC)
Kappa
AUC-ROC
AUC-PR

# Genetic algorithm performance metric
## GA 性能指标评估
### 收敛性（Convergence）
Best Fitness, Average Fitness 曲线
### 种群多样性（Diversity）
Diff (Best Fitness - Average Fitness) 如果接近0, 算法已经失去了探索新空间的能力
### 成功率（Success Rate）
最终选出的候选者占总评估序列的比例。
### 群体适应度方差 (Fitness Variance)
### 变异系数 (Coefficient of Variation, CV)
如果 CV低，且最高分已经满足要求（如 > 0.9）  赶紧停，去拿结果。
如果 CV 低，但最高分很差（如 0.4） 这不是早停的时机，而是失败的信号。说明算法陷入了局部最优（死胡同）。这时不该停止，而应该大幅调高变异率或者引入新鲜血液（重新随机生成一部分个体）。
### 遗传距离 (Genetic Distance / Hamming Distance)
### 新颖性（Novelty / Levenshtein Distance）
计算生成的序列与初始种群中已知 AMP 的平均编辑距离[Levenshtein Distance更严格比起Hamming Distance]。如果距离太近，说明算法只是在“复读”已知答案。

## Solution (Sequences) quality
### 全局序列比对
CD-HIT
`cd /mnt/e/Main/Dissertation/data_process/Apricot/APIN/`
`cd-hit-2d -i ./data/DAMP/AMP.te.fasta -i2 ./output_GA/top_sequences.fasta -o ./output_GA/result -c 0.9 -n 5`
BLAST
### 理化性质
1. 等电点 (pI) 与 净电荷 (Net Charge)
肽链与细菌膜的静电吸引力会过弱，导致抗菌活性显著下降; 电荷数过高, 破坏人类正常的红细胞，产生毒性
> reference: charge[2-7]; pI[10-12]
> 10.1016/j.biotechadv.2025.108570
> 10.1136/egastro-2025-100253 
2. 分子量 (Molecular Weight, MW)
太小可能杀伤力不够，太大则合成成本高且难以穿透复杂的生物膜。
> 20-50
3. 氨基酸组成
### 核心理化机能
1. 平均疏水性 (Hydrophobicity, H) 
决定了肽进入脂质双分子层的深度。疏水性过低无法穿孔，过高则会像“毒药”一样不分敌我地溶解红细胞
> reference: 0.4-0.6 
> 10.1016/j.jare.2024.02.016
2. 疏水力矩 (Hydrophobic Moment, μH) 
这是衡量两亲性的核心指标。它计算的是疏水侧链在结构分布上的对称性。越高，意味着肽折叠后“一面疏水、一面亲水”的特征越明显，这种结构最容易在细菌膜上打洞。
> reference: 0.4-0.6
> 10.1016/j.jare.2024.02.016
3. Boman 指数 (Potential for Protein Interaction)
如果说 疏水力矩 关注的是“肽与膜”的作用，Boman 指数则关注**“肽与蛋白”**的作用。数值大于 2.48 代表该肽可能具有多功能性（如信号传导、调节免疫），而不仅仅是物理杀菌。
> reference: < 2.5
> 10.1016/j.biotechadv.2025.108570
4. 不稳定性指数 (Instability Index)
预测该肽在细胞内环境是否容易被降解。数值< 40 通常被认为是稳定的。
### 构象与结构预测
1. 二级结构预测 (Secondary Structure)
alpha螺旋轮图
2. 3D 结构  
AlphaFold/RoseTTAFold, PEP-Fold(适合短肽), MD, docking
3. 二级结构组成（百分比）
实验: 圆二色谱 (CD Spectroscopy)
预测: DSSP 算法（针对 3D 结构）/ PSIPRED（针对序列）
### 生物活性预测与评价
1. MIC浓度
2. 选择性指数 (Selectivity Index / Therapeutic Index, TI)
一个好的抗菌肽必须是“只杀细菌，不伤人体”。TI 值越高，代表安全性越高。
