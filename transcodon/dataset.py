"""Common class for sequence datasets."""

from typing import Tuple
import csv,os
import sys
import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .alphabet import ORGANISM2ID
from typing import Dict
from .sequence import (
    Sequence,
    CodonSequence,
    WithUtr_CodonSequence,
    RNA2DSequence,
    AminoAcidSequence
)



def get_sequence_length(fasta_file_path):
    """
    获取 FASTA 文件中序列的长度
    """
    with open(fasta_file_path, "r") as f:
        lines = f.readlines()
        # 跳过第一行（header），计算序列长度
        #sequence = "".join(line.strip() for line in lines[1])
        return len(lines[1].strip())
def get_rna_2d(fasta_file_path):
    """
    获取 FASTA 文件中序列的长度
    """
    with open(fasta_file_path, "r") as f:
        lines = f.readlines()
        # 跳过第一行（header），计算序列长度
        sequence = "".join(line.strip() for line in lines[2:])
        return sequence
class SequenceDataset(torch.utils.data.Dataset):
    """Common class for sequence datasets."""

    def __init__(self, fasta_file: str, codon_sequence: bool=True):
        self.fasta_file = fasta_file
        self.codon_sequence = codon_sequence
        self._sequences, self._titles, self.organism ,self.struct_label= [], [], [],[]

        # for record in SeqIO.parse(fasta_file, 'fasta'):
        #     self._titles.append(record.id)
        #     if self.codon_sequence:
        #         self._sequences.append(CodonSequence(record.seq))
        #     else:
        #         self._sequences.append(AminoAcidSequence(record.seq))
        csv.field_size_limit(sys.maxsize)
        #with open() as struct_label_file:
        # self.struct_label.append(row['struct_label'])
        count=0
        with open(fasta_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                count+=1
                # 提取 dna 列作为序列
                sep_len=0
                if False:
                  utr_sequence = row['5utr_sequence']
                utr_sequence=''
                if utr_sequence!='':
                    sep_len=1
                cds_sequence =row['cds_sequence']
                # print("5utr_sequence",row['5utr_sequence'])
                # print("cds_sequence",row['cds_sequence'])
                len_all=len(utr_sequence)+len(cds_sequence)+sep_len

                # print("len 5utr_sequence",len(row['5utr_sequence']))
                # print("len cds_sequence",len(row['cds_sequence']))
             
                struct_label='*'*len_all
                # print(len(struct_label))
                if False:
                  struct_label_2d_file=row['rna_2d'].split('|')[0]
                else:
                    struct_label_2d_file='*'
                #print("struct_label_2d_file",struct_label_2d_file)

                struct_label_list = list(struct_label)  

                # 计算插入位置
                start_index = len_all - len(struct_label_2d_file)
                #print("start_index",start_index)

                # 替换指定位置的内容
                struct_label_list[start_index:] = list(struct_label_2d_file)  

                # 重新转换回字符串
                struct_label = ''.join(struct_label_list)
                #print("final struct_label",struct_label)

                # if os.path.exists(struct_label_matrix_file) and os.stat(struct_label_matrix_file).st_size > 0:
                if False:
                    if os.path.exists(struct_label_2d_file):
                        #continue
                        len_label=len(dna_sequence)

                        with open(struct_label_2d_file, 'r', encoding='utf-8') as file:
                            lines = file.readlines()  # 读取所有行
                            if len(lines) > 1:  # 确保文件至少有两行
                                temp = lines[1].strip()  # 读取第二行并去除换行符
                                #print("第二行内容:", temp)
                            else:
                                print("文件行数不足，无法读取第二行。")
                        if len(temp)==len_label:
                            struct_label=temp
                            #print("count",count,": ",struct_label)
                    # 创建一个新的 SeqRecord 对象，假设我们将 'name' 作为标题
                    # 这里可以根据你的需求选择合适的列作为标题
                    if False:
                        description=row['description']
                        description_id="".join(c if c.isalnum() or c in ('_', '-') else '_' for c in description)
                        record = SeqRecord(Seq(dna_sequence), id=row['GeneID'], description=row['description'])

                        # 将序列标题添加到 _titles
                        self._titles.append(record.id)
                if self.codon_sequence:
                    # 假设 CodonSequence 是一个类，用来处理序列（比如分割为密码子）
                    # print("record.seq",record.seq)
                    # print("struct_label",struct_label)
                    # self._sequences.append(CodonSequence(record.seq))
                
                    # print("dna_sequence",dna_sequence)
                    # print("row['species_name']",row['species_name'])
                    # print("ORGANISM2ID[row['species_name']]",ORGANISM2ID[row['species_name']])
                    if utr_sequence!='':
                        seq=WithUtr_CodonSequence(utr_sequence,cds_sequence)
                    else:
                        seq=CodonSequence(cds_sequence)
                        if count<10:
                            print("test seq:",seq._seq)

                    # print("seq:",seq._seq)
                    # print("len seq:",len(seq._seq))
                    self._sequences.append(seq)
                    #species_name="Escherichia coli str. K-12 substr. MG1655"
                    if True:
                      species_name=row['species_name']
                    
                    self.organism.append(ORGANISM2ID[species_name])
                    self.struct_label.append(RNA2DSequence(struct_label))
                    #print("RNA2DSequence(struct_label):",RNA2DSequence(struct_label)._seq)
                    
                else:
                    self._sequences.append(AminoAcidSequence(record.seq))
                    self.organism.append(ORGANISM2ID[row['species_name']])
                    self.struct_label.append(RNA2DSequence(struct_label))
       

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx) -> dict:
        return {
             "sequence":self._sequences[idx],
             "organism": self.organism[idx],
             "struct_label":self.struct_label[idx]
            }
        #return self._sequences[idx],self.organism[idx]