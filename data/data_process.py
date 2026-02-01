from Bio import SeqIO
import matplotlib.pyplot as plt

def process_fasta(input_path, output_path, min_len=10, max_len=200):
    stats = {"total": 0, "kept": 0, "len_rem": 0, "aa_rem": 0}
    # 需去除的特殊氨基酸
    FORBIDDEN_AA = set("BJOUXZ")
    
    kept_records = []
    
    for record in SeqIO.parse(input_path, "fasta"):
        stats["total"] += 1
        
        # 1. 统一处理序列
        seq_str = str(record.seq).upper()
        seq_len = len(seq_str)
        
        # 2. 长度过滤
        if not (min_len <= seq_len <= max_len):
            stats["len_rem"] += 1
            continue
            
        # 3. 特殊氨基酸过滤
        if any(aa in FORBIDDEN_AA for aa in seq_str):
            stats["aa_rem"] += 1
            continue
        
        # 更新 record 的序列为大写（保证输出一致性）
        record.seq = record.seq.upper()
        kept_records.append(record)
        stats["kept"] += 1

    # 4.  fasta-2line 一次性写入
    SeqIO.write(kept_records, output_path, "fasta")
            
    return stats

def plot_length_distribution(amp_fasta, non_amp_fasta,path):
    # 提取长度数据
    amp_lengths = [len(record.seq) for record in SeqIO.parse(amp_fasta, "fasta")]
    non_amp_lengths = [len(record.seq) for record in SeqIO.parse(non_amp_fasta, "fasta")]

    # 设置画布
    plt.figure(figsize=(10, 6))

    # 绘制直方图
    # alpha=0.5 使颜色透明, 方便观察重叠部分
    # bins 可以根据你的长度范围调整, 这里设为 30
    plt.hist(amp_lengths, bins=80, alpha=0.6, label='AMP (Positive)', color='royalblue', edgecolor='black')
    plt.hist(non_amp_lengths, bins=80, alpha=0.5, label='Non-AMP (Negative)', color='orange', edgecolor='black')

    # 添加轴标签和标题
    plt.xlabel('Sequence Length (aa)', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    plt.title('Sequence Length Distribution Comparison', fontsize=14)
    
    # 显示图例
    plt.legend(loc='upper right')
    
    # 添加网格线以增强可读性
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 保存图片
    plt.savefig(path, dpi=300)
    plt.savefig(path)
    logger.info(f"统计图已保存至: {path}")
    
    plt.show()


def data_split(train_path, valid=1, test=1, valid_clip_proportion=0.2, test_clip_proportion=0.2, prefix_save=None):
    # 分割fasta文件为train/valid/test三份, 并保存为fasta文件
    # train: 输入fasta文件路径
    # valid, test: 输出fasta文件路径（不含扩展名, 自动加后缀）
    # valid_clip_proportion, test_clip_proportion: 比例（如0.2）
    from Bio import SeqIO
    import random

    records = list(SeqIO.parse(train_path, "fasta"))
    total = len(records)
    idx = list(range(total))
    random.shuffle(idx)

    valid_n = 0
    test_n = 0
    
    if valid:
        valid_n = int(total * valid_clip_proportion)
        valid_idx = idx[:valid_n]
        valid_records = [records[i] for i in valid_idx]
        valid_out = prefix_save + '.valid.fasta'
        
        SeqIO.write(valid_records, valid_out, "fasta")
        logger.info(f"Valid Clipped: {len(valid_records)}")
    if test:
        test_n = int(total * test_clip_proportion)
        test_idx = idx[valid_n:valid_n+test_n]
        test_records = [records[i] for i in test_idx]
        test_out = prefix_save + '.te.fasta'
        SeqIO.write(test_records, test_out, "fasta")
        logger.info(f"Test Clipped: {len(test_records)}")

    # 输出文件名
    train_idx = idx[valid_n+test_n:]
    train_records = [records[i] for i in train_idx]
    train_out = prefix_save + '.tr.fasta'
    SeqIO.write(train_records, train_out, "fasta")
    logger.info(f"Train: {len(train_records)}")


def run_cdhit_wsl(in_file,out_file,threshold,task='cd-hit'):
    import subprocess
    import os
    # 路径转换逻辑
    def to_wsl(win_path):
        p = os.path.abspath(win_path).replace('\\', '/')
        return f"/mnt/{p[0].lower()}{p[2:]}"
    # 补充工作目录
    cwd = os.getcwd()

    if task == 'cd-hit':
        in_file = os.path.join(cwd,in_file)
        out_file = os.path.join(cwd, out_file)
        cmd = [
            "wsl", "/home/sarzn/cdhit-4.8.1/cd-hit",
            "-i", to_wsl(in_file),
            "-o", to_wsl(out_file),
            "-c", str(threshold), "-n", "5", "-T", "8"
        ]
        logger.info(f"正在 WSL 中启动 CD-HIT 去重...")
        subprocess.run(cmd, check=True)
    else:
        i = os.path.join(cwd,in_file[0])
        i2 = os.path.join(cwd,in_file[1])
        out_file = os.path.join(cwd, out_file)
        cmd = [
        "wsl", "/home/sarzn/cdhit-4.8.1/cd-hit-2d",
        "-i", to_wsl(i),
        "-i2", to_wsl(i2),
        "-o", to_wsl(out_file),
        "-c", str(threshold), "-n", "5", "-T", "8"
        ]
        logger.info(f"正在 WSL 中启动 CD-HIT-2D 去重...")
        subprocess.run(cmd, check=True)
    return 


if __name__ == "__main__":
    # log recording
    import logging
    from pathlib import Path
    classifier = 'APIN'
    dataset = "dbAMP"
    log_file = Path(f'logs/{classifier}_filter_data.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # 过滤数据集
    tasks = [
        (f"data/{dataset}/AMP.fasta", f"data/{dataset}/AMP_clean.fasta"),
        (f"data/{dataset}/uniprot.fasta", f"data/{dataset}/Non-AMP_clean.fasta")
    ]
    # CD-HIT 相似度阈值
    threshold = 0.7
    for i, (INPUT_FASTA, OUTPUT_FASTA) in enumerate(tasks):
        if i == 0:
            logger.info("AMP processing...")
        else:
            logger.info("Non-AMP processing...")

        res = process_fasta(INPUT_FASTA, OUTPUT_FASTA)
        
        logger.info("-" * 30)
        logger.info(f"处理完成！报告如下：")
        logger.info(f"总计输入: {res['total']} 条") 
        logger.info(f"符合要求: {res['kept']} 条 (已存至单行格式)")
        logger.info(f"长度不符: {res['len_rem']} 条") 
        logger.info(f"含非法AA: {res['aa_rem']} 条") 
        logger.info("-" * 30)

        # CD-HIT 去冗余
        INPUT_FASTA = OUTPUT_FASTA
        OUTPUT_FASTA = OUTPUT_FASTA.replace('.fasta', f'_{int(threshold*100)}.fasta')
        
        run_cdhit_wsl(
            INPUT_FASTA, 
            OUTPUT_FASTA, 
            threshold
        )
        with open(OUTPUT_FASTA, 'r') as f:
            target_count = sum(1 for line in f if line.startswith('>'))
        logger.info(f"CD-HIT 去冗余完成, 剩余数量: {target_count}, 输出文件: {OUTPUT_FASTA}")
        # CD-HIT-2D 去冗余
        if i == 1:
            INPUT_FASTA = [f'{tasks[0][1].replace(".fasta", f"_{int(threshold*100)}.fasta")}', OUTPUT_FASTA]
            OUTPUT_FASTA = OUTPUT_FASTA.replace('.fasta', f'_2D.fasta')
            run_cdhit_wsl(
                INPUT_FASTA, # i & i2
                OUTPUT_FASTA, # out_file
                threshold,
                task='cd-hit-2d'
            )
            with open(OUTPUT_FASTA, 'r') as f:
                target_count = sum(1 for line in f if line.startswith('>'))
            logger.info(f"CD-HIT-2D 去冗余完成, 剩余数量: {target_count}, 输出文件: {OUTPUT_FASTA}")

    # 绘制分布图 
    amp_file = f"{tasks[0][1].replace('.fasta', f'_{int(threshold*100)}.fasta')}"
    non_amp_file = f"{tasks[1][1].replace('.fasta', f'_{int(threshold*100)}_2D.fasta')}"
    plot_length_distribution(amp_file, non_amp_file,path=f"data/{dataset}/length_dist.png")

    # 下采样 non-AMP
    import random
    with open(amp_file, 'r') as f:
        target_count = sum(1 for line in f if line.startswith('>'))
    non_amp_records = list(SeqIO.parse(non_amp_file, "fasta"))
    sampled_records = random.sample(non_amp_records, target_count)

    non_amp_file = f"{tasks[1][1].replace('.fasta', f'_{int(threshold*100)}_2D_downsampled.fasta')}"

    SeqIO.write(sampled_records, non_amp_file, "fasta")
    logger.info(f"Non-AMP 下采样完成, 保留数量: {target_count}, 输出文件: {non_amp_file}")
    plot_length_distribution(amp_file, non_amp_file,path=f"data/{dataset}/length_dist_downsampled.png")

    # 数据集划分
    train_path = amp_file
    data_split(train_path=train_path, prefix_save=f"data/{dataset}/AMP")
    train_path = non_amp_file
    data_split(train_path=train_path, prefix_save=f"data/{dataset}/Non-AMP")


''' 
AMP
------------------------------
处理完成！报告如下：
总计输入: 35599 条
符合要求: 27780 条 (已存至单行格式)
长度不符: 5755 条
含非法AA: 2064 条
Non-AMP
------------------------------
处理完成！报告如下：
总计输入: 566105 条
符合要求: 174030 条 (已存至单行格式)
长度不符: 390972 条
含非法AA: 1103 条
------------------------------
CD-HIT
# AMP 剩余 7537 (27780-7537=20243 去冗余)
# Non_AMP 剩余 60622 (174030-60622=113408 去冗余)
cd /mnt/e/Main/Dissertation/data_process/Apricot/APIN/data/dbAMP
cd-hit -i AMP_clean.fasta -o AMP_clean_70.fasta -c 0.7 -n 5 -T 8 
cd-hit -i Non-AMP_clean.fasta -o Non-AMP_clean_70.fasta -c 0.7 -n 5 -T 8 

CD-HIT-2D
# Non-AMP 剩余 60286 (60622-55317=5305 去冗余)
cd-hit-2d -i AMP_clean_70.fasta -i2 Non-AMP_clean_70.fasta -o Non-AMP_clean_70_2D.fasta -c 0.7 -n 5 -T 8
'''