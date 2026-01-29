import pygad
import numpy as np
import torch
from utils import APIN,Preprocess
import logging
from pathlib import Path

# decode中的Z, initial_population, num_parents_mating, num_genes, gene_space, sol_per_pop


def encode_to_indices(seq_input,max_length): # 单序列编码
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
            'Y':20
            }
    seq = np.zeros(max_length, dtype=int)
    for j in range(min(len(seq_input), max_length)):
        seq[max_length - 1 - j] = protein_dict[seq_input[len(seq_input) - 1 - j]]
    return seq

def decode_seq(solution): # 单个序列解码, 用于GA只能一次传一个序列
    reverse_protein_dict = {
            0: 'Z',
            1: 'A',
            2: 'C',
            3: 'D',
            4: 'E',
            5: 'F',
            6: 'G',
            7: 'H',
            8: 'I',
            9: 'K',
            10: 'L',
            11: 'M',
            12: 'N',
            13: 'P',
            14: 'Q',
            15: 'R',
            16: 'S',
            17: 'T',
            18: 'V',
            19: 'W',
            20: 'Y'
            }
    seq = [reverse_protein_dict[num] for num in solution]
    return "".join(seq)


def fitness_func(ga_instance, solution, solution_idx):
    
    temp_decode = decode_seq(solution)
    solution = encode_to_indices(temp_decode, max_length)

    current_seq_tensor = torch.tensor(solution, dtype=torch.long).unsqueeze(0)
    current_seq_tensor = current_seq_tensor.to(device)
    
    with torch.no_grad():
        prediction = amp_model(current_seq_tensor)
        # fitness = torch.sigmoid(prediction).item()
        fitness = prediction.item() # 直接用原始输出作为适应度分数
    
    return fitness


def get_average_hamming_distance(ga_instance, sample_size=100):
    import random
    population = ga_instance.population
    pop_size = population.shape[0]
    
    # 如果种群太小，直接全测；如果很大，随机抽样计算
    actual_sample_size = min(sample_size, pop_size)
    sample_indices = random.sample(range(pop_size), actual_sample_size)
    sampled_pop = population[sample_indices]
    
    distances = []
    # 计算采样对之间的汉明距离
    for i in range(actual_sample_size):
        for j in range(i + 1, actual_sample_size):
            dist = np.sum(sampled_pop[i] != sampled_pop[j])
            distances.append(dist)
    
    return np.mean(distances) if distances else 0

def get_average_novelty(ga_instance, initial_peptides, top_n=10):
    import Levenshtein
    population = ga_instance.population
    current_fitness = ga_instance.last_generation_fitness
    
    # 2. 找到表现最好的 top_n 个个体的索引
    # argsort 是升序排列，取最后 top_n 个
    top_indices = np.argsort(current_fitness)[-top_n:]
    
    novels = []
    
    for idx in top_indices:
        # 解码当前序列 (确保你定义了 decode_seq)
        current_seq = decode_seq(population[idx])
        
        # 计算该序列与所有初始序列的编辑距离
        distances = [Levenshtein.distance(current_seq, start_seq) for start_seq in initial_peptides]
        # min(distances) 代表它与最像它的那个“祖先”的距离
        novels.append(min(distances))
    
    return np.mean(novels) if novels else 0.0


def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    current_fitness = ga_instance.last_generation_fitness
    population = ga_instance.population

    # 1. 计算适应度指标
    best_fit = np.max(current_fitness)
    avg_fit = np.mean(current_fitness)
    fit_var = np.var(current_fitness)
    # 计算变异系数
    fit_cv = np.std(current_fitness) / avg_fit 
    # 2. 计算遗传多样性（每代抽样 50 对进行估算） 汉明距离
    avg_dist = get_average_hamming_distance(ga_instance, sample_size=50)

    # 3. 计算成功率
    success_threshold = 0.9
    success_count = np.sum(current_fitness >= success_threshold)
    success_rate = (success_count / len(current_fitness)) * 100

    # 4. 计算新颖性 (Novelty)
    avg_novelty = get_average_novelty(ga_instance, initial_peptides, top_n=10)

    # 4. 打印详细评估日志
    logger.info(f"Generation {gen} | Best Fitness: {best_fit:.4f} | Average Fitness = {avg_fit:.4f} | Variance: {fit_var:.6f} | Coefficient of Variation: {fit_cv:.6f} | Average Hamming Distance: {avg_dist:.2f} | Average Novelty: {avg_novelty:.2f} | Success Rate: {success_rate:.2f}% ")
    
    # 存入历史记录
    best_outputs.append(best_fit)
    avg_outputs.append(avg_fit)
    fit_var_outputs.append(fit_var)
    avg_dist_outputs.append(avg_dist)
    success_rate_outputs.append(success_rate)
    avg_novelty_outputs.append(avg_novelty)
    fit_cv_outputs.append(fit_cv)

    # 4. 逻辑判断：如果多样性太低，可以手动干预
    # if fit_var < 1e-6:
    #     print("警告：种群已失去多样性，可能陷入局部最优。")


def plot_analysis():
    import matplotlib.pyplot as plt
    
    best_outputs_np = np.array(best_outputs)
    avg_outputs_np = np.array(avg_outputs)
    fit_var_outputs_np = np.array(fit_var_outputs)
    avg_dist_outputs_np = np.array(avg_dist_outputs)
    success_rate_outputs_np = np.array(success_rate_outputs)
    avg_novelty_outputs_np = np.array(avg_novelty_outputs)
    fit_cv_outputs_np = np.array(fit_cv_outputs)

    # 创建 2x2 的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Genetic Algorithm Convergence Analysis", fontsize=16)

    # --- 子图 1: Fitness Trends ---
    axs[0, 0].plot(best_outputs_np, label="Best Fitness")
    axs[0, 0].plot(avg_outputs_np, label="Average Fitness", linestyle='--')
    axs[0, 0].plot(best_outputs_np - avg_outputs_np, label="Diff (Best-Avg)", linestyle='--')
    axs[0, 0].set_title("Fitness Convergence")
    axs[0, 0].set_ylabel("Fitness Score")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # --- 子图 2: Fitness Variation ---
    axs[0, 1].plot(fit_var_outputs_np, label="Fitness Variance", color='orange', linestyle='-.')
    axs[0, 1].plot(fit_cv_outputs_np, label="Fitness CV", color='red', linestyle='dashdot')
    axs[0, 1].set_title("Fitness Variance & CV")
    axs[0, 1].set_ylabel("Variation Metric")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # --- 子图 3: Diversity Metrics ---
    axs[1, 0].plot(avg_novelty_outputs_np, label="Avg Novelty (Levenshtein Distance)", color='green', linestyle='dotted')
    axs[1, 0].plot(avg_dist_outputs_np, label="Avg Hamming Distance", color='purple', linestyle=':')
    axs[1, 0].set_title("Population Diversity")
    axs[1, 0].set_xlabel("Generation")
    axs[1, 0].set_ylabel("Distance/Novelty")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # --- 子图 4: Success Rate ---
    axs[1, 1].plot(success_rate_outputs_np, label="Success Rate (%)", color='brown', linestyle='solid')
    axs[1, 1].set_title("Target Success Rate")
    axs[1, 1].set_xlabel("Generation")
    axs[1, 1].set_ylabel("Percentage (%)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 调整布局防止标签重叠
    plt.savefig('./output_GA/GA_convergence_subplots.png', dpi=150)
    plt.close()


# --- 准备工作 ---
best_outputs = []
avg_outputs = []
fit_var_outputs = []
avg_dist_outputs = []
success_rate_outputs = []
avg_novelty_outputs = []
fit_cv_outputs = []
classifier = 'APIN'
repaired_pth = f'weight/repair/{classifier}_fixed_patch.pth'
attention_num = 64
attention_range = 14
embed_length = 128
max_length = 200 # 模型构建的最大序列长度
device = 'cuda'
amp_model = APIN(attention_num, attention_range, embed_length, max_length)
amp_model.load_state_dict(torch.load(repaired_pth))
amp_model.to(device)
amp_model.eval()

# log recording
log_file = Path(f'logs/{classifier}_GA.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# true_test_file = 'data/DAMP/AMP.te.fa'
# false_test_file = 'data/DAMP/Non-AMP.te.fa'
# data = Preprocess(max_length, true_test_file=true_test_file, false_test_file=false_test_file)
# my_initial_pop = data.get('X_test')

initial_peptides = ["SLWLDKRDTNT", "FIGAVAGLLSKIF", "DISLPILVVQHMPAGFTKAFATR"]
my_initial_pop = [encode_to_indices(seq, max_length) for seq in initial_peptides]

num_generations=100    # 进化代数
num_parents_mating=5   # 每次选 多少 个优秀个体做父母
sol_per_pop=100        # 种群大小（每代序列数量） initial_population传入不足, 会自动生成
ga_length = 20 # GA 搜索的序列长度

# --- 配置 PyGAD ---
ga_instance = pygad.GA(
    # initial_population=my_initial_pop,
    num_generations=num_generations,          
    num_parents_mating=num_parents_mating,  
    sol_per_pop=sol_per_pop,    

    num_genes=ga_length,                  # 序列长度（假设寻找长度 ga_length 肽）
    gene_space=list(range(1, 21)),    # 基因取值范围：1-20 (对应 20 种氨基酸)
    
    fitness_func=fitness_func,     # 刚才定义的适应度函数  
    gene_type=int,                 # 基因必须是整数
    
    parent_selection_type="sss",   # 稳态选择
    crossover_type="single_point", # 单点交叉
    mutation_type="random",        # 随机变异
    mutation_probability=0.5,       # 变异概率
    on_generation=on_generation,
)

# --- 运行进化 ---
ga_instance.run()

# --- 结果展示 ---
solution, solution_fitness, solution_idx = ga_instance.best_solution()
best_seq = decode_seq(solution)
logger.info(f"Best Score: {solution_fitness}")
logger.info(f"Best AMPs Sequence: {best_seq}")

# 保存最后一轮前top_n的序列到fasta
# top_n存全部
top_n = sol_per_pop 
top_n = 10
current_fitness = ga_instance.last_generation_fitness
top_indices = np.argsort(current_fitness)[-top_n:][::-1]  # 从高到低
population = ga_instance.population

with open('./output_GA/top_sequences.fasta', 'w') as f:
    for rank, idx in enumerate(top_indices):
        seq = decode_seq(population[idx])
        score = current_fitness[idx]
        f.write(f'>rank{rank}_score{score:.4f}\n{seq}\n')

# 绘图
plot_analysis()

