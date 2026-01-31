# 过滤 dbAMP3.fasta 中序列长度小于100的序列
input_fasta = "data/StarPep/dbAMP3.fasta"
output_fasta = "data/StarPep/dbAMP3_length100.fasta"

def filter_fasta_by_length(input_path, output_path, max_length=100):
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        seq_id = None
        seq_lines = []
        for line in infile:
            line = line.rstrip()
            if line.startswith(">"):
                # 处理前一个序列
                if seq_id is not None:
                    seq = "".join(seq_lines)
                    if len(seq) < max_length:
                        outfile.write(seq_id + "\n")
                        outfile.write(seq + "\n")
                seq_id = line
                seq_lines = []
            else:
                seq_lines.append(line)
        # 处理最后一个序列
        if seq_id is not None:
            seq = "".join(seq_lines)
            if len(seq) < max_length:
                outfile.write(seq_id + "\n")
                outfile.write(seq + "\n")

if __name__ == "__main__":
    filter_fasta_by_length(input_fasta, output_fasta)
    print(f"已保存长度小于100的序列到 {output_fasta}")
