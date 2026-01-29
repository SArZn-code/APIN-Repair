'''
等电点 (pI)	
净电荷 (Net Charge)
氨基酸组成分析
分子量 (MW)	
平均疏水性 (H)
疏水力矩 (μH)
Boman 指数
螺旋轮图 (Helical Wheel)
不稳定性指数 (II)
----------------
二级结构百分比
'''
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
from modlamp.plot import helical_wheel
from collections import Counter
import logging
from pathlib import Path

class AMPAnalyzer:
    def __init__(self, seq):
        self.seq = seq
        # 初始化 modlamp 的全局描述符
        self.global_descriptor = GlobalDescriptor([self.seq])
        self.peptide_descriptor = PeptideDescriptor([self.seq], 'eisenberg')
    def calculate_all_metrics(self):
        """计算所有要求的理化指标"""
        results = {}
        # --- 第一部分：全局描述符 (基于公式) ---
        # GlobalDescriptor

        # 1. 净电荷 (Net Charge)
        self.global_descriptor.calculate_charge()
        results['Net Charge'] = self.global_descriptor.descriptor[0][0]
        
        # 2. 等电点 (pI)
        self.global_descriptor.isoelectric_point()
        results['pI'] = self.global_descriptor.descriptor[0][0]
        
        # 3. 分子量 (MW)
        self.global_descriptor.calculate_MW()
        results['MW'] = self.global_descriptor.descriptor[0][0]
        
        # 4. 不稳定性指数 (Instability Index)
        self.global_descriptor.instability_index()
        results['Instability Index'] = self.global_descriptor.descriptor[0][0]
        
        # 5. Boman 指数 - 使用 peptides 库计算
        self.global_descriptor.boman_index()
        results['Boman Index'] = self.global_descriptor.descriptor[0][0]
        
        # --- 第二部分：肽段描述符 (基于氨基酸量表) ---
        # 6. 平均疏水性 (H) - 使用 Eisenberg 刻度
        self.peptide_descriptor.calculate_global()
        results['Hydrophobicity'] = self.peptide_descriptor.descriptor[0][0]

        # 7. 疏水力矩 (μH)
        self.peptide_descriptor.calculate_moment()
        results['Hydrophobic Moment (μH)'] = self.peptide_descriptor.descriptor[0][0]
        
        # --- 第三部分：氨基酸组成 (原生 Python 最简法) ---
        # 8. 氨基酸组成分析 (AA Composition)
        length = len(self.seq)
        counts = Counter(self.seq)
        # 生成类似 {'A': 0.1, 'C': 0.05...} 的字典
        results['Composition'] = {aa: count / length for aa, count in counts.items()}
        
        return results

    def plot_wheel(self, filename=None):
        """生成并显示/保存螺旋轮图"""
        logger.info(f"Alpha helical wheel, save to {filename}")
        helical_wheel(self.seq, filename=filename, colorcoding='charge')


if __name__ == "__main__":
    classifier = 'APIN'
    log_file = Path(f'logs/{classifier}_GA_evaluation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # 遍历 top_sequences.fasta，评估每个序列
    with open('./output_GA/top_sequences.fasta', 'r') as f:
        lines = f.readlines()
    for idx in range(0, len(lines)//2):
        name = lines[idx*2].strip()[1:]
        seq = lines[idx*2+1].strip()
        analyzer = AMPAnalyzer(seq)
        metrics = analyzer.calculate_all_metrics()
        logger.info(f"--- AMPs evaluation {name} ---")
        for key, value in metrics.items():
            if key != 'Composition':

                logger.info(f"{key}: {value:.3f}")
            else:
                logger.info(f"{key}: {value}")

        wheel_path = f"./output_GA/helical_wheel_{idx}.png"
        analyzer.plot_wheel(filename=wheel_path)
