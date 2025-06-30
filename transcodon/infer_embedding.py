"""Module to use transcodon as a pretrained model."""

import os
import pickle
import requests
from typing import Optional, Union, List
import numpy as np
import torch
from .alphabet import Alphabet,ORGANISM2ID,AMINO_ACID_TO_INDEX
from .sequence import CodonSequence,CodonSequence_infer
from .model import ProteinBertModel
import numpy as np
import random



def embed_sequence(model,sequence: Union[str, CodonSequence_infer], species_type_ids: int = 0,AA_seq:str="",average: bool = True) -> torch.Tensor:
        """Embeds an individual sequence using transcodon. If the ``average''
        flag is True, then the representation is averaged over all
        possible odons, providing a vector representation of the
        sequence."""
        #print(sequence.type)
        if isinstance(sequence, str):
            seq = CodonSequence_infer(sequence)
        elif isinstance(sequence, CodonSequence_infer):
            seq = sequence
        else:
            raise ValueError('Input sequence must be string or CodonSequence.')
        
        if isinstance(species_type_ids,str):
            species_type_ids = torch.tensor([ORGANISM2ID[species_type_ids]])
        elif isinstance(species_type_ids,int):
            species_type_ids=torch.tensor([species_type_ids])
        else:
            raise ValueError('species_type_ids must be str or int')
    
        #print("seq",seq,seq.seq)
        tokens = tokenize(seq)
        #print("tokens",tokens)
        #print("species_type_ids",species_type_ids)  
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokens =tokens.to(device)
        species_type_ids=species_type_ids.to(device)
        # tokens.to(device)
        # species_type_ids.to(device)
        # model.to(device)
        # print("device",device)
        output = model(tokens,species_type_ids)
        #output.to("cpu")
        logits = output['logits']
        #print("logits",logits)
        repr_ =output['representations'][12]
        logits=logits[:,1:-1,:]
        amino_acid_to_codon = {
            'F': ['UUU', 'UUC'],
            'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'], #1
            'I': ['AUU', 'AUC', 'AUA'],
            'M': ['AUG'],  # 起始密码子
            'V': ['GUU', 'GUC', 'GUA', 'GUG'],
            'P': ['CCU', 'CCC', 'CCA', 'CCG'],
            'T': ['ACU', 'ACC', 'ACA', 'ACG'],
            'A': ['GCU', 'GCC', 'GCA', 'GCG'],
            'Y': ['UAU', 'UAC'],
            '_': ['UAA', 'UAG', 'UGA'],  # 终止密码子            #2
            'H': ['CAU', 'CAC'],
            'Q': ['CAA', 'CAG'],
            'N': ['AAU', 'AAC'],
            'K': ['AAA', 'AAG'],
            'D': ['GAU', 'GAC'],
            'E': ['GAA', 'GAG'],
            'C': ['UGU', 'UGC'],
            'W': ['UGG'],
            'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],        #3
            'S': ['AGU', 'AGC', 'UCU', 'UCC', 'UCA', 'UCG'],           #4
            'G': ['GGU', 'GGC', 'GGA', 'GGG']
        }
        all_codons = [codon for sublist in amino_acid_to_codon.values() for codon in sublist]

        # 创建 codon_to_index 映射
        codon_to_index = {codon: idx for idx, codon in enumerate(all_codons)}
        #codon_to_index = {codon: idx for idx, codon in enumerate(amino_acid_to_codon.values())}
        #print("codon_to_index",codon_to_index)
        selected_codons = ""
        #All_toks_list = ["<cls>", "<pad>", "<eos>", "<unk>",'A','U','C','G',"<mask>"]
        #base_to_index = {'A': 4, 'U': 5, 'C': 6, 'G': 7}
        base_to_index = {'A': 5, 'U': 6, 'C': 7, 'G': 8}
        if True:
            #**************************************************************
            # logits为密码子长度，3m
            # AA_seq长度为m
            #AA_seq+='#'
            #print("AA_seq",AA_seq)
            #exit()
            for i in range(len(AA_seq)):
                 AA=AA_seq[i]
                 possible_codons = amino_acid_to_codon[AA] 
                 possible_indices = [codon_to_index[codon] for codon in possible_codons]
                 #print("possible_indices",possible_indices)
                 start = i * 3  # 当前氨基酸对应的密码子起始位置
                 end = start + 3  # 当前氨基酸对应的密码子结束位置
                 # 获取当前位置的 logits
                 #print("logits",logits)
                 current_logits = logits[:, start:end, :] 
                 #print("current_logits",current_logits) 
                 max_prob = float("-inf")
                 best_codon = None
                 for codon in possible_codons:
                    # 计算当前密码子的总概率
                    prob = 1.0
                    for j, base in enumerate(codon):
                        base_idx = base_to_index[base]  # 获取碱基的索引
                        # print("base_idx",base_idx)
                        # print("current_logits[:, j, :]",current_logits[:, j, :])
                        # print("torch.softmax(current_logits[:, j, :], dim=-1)",torch.softmax(current_logits[:, j, :], dim=-1))
                        prob *= torch.softmax(current_logits[:, j, :], dim=-1)[0, base_idx].item()  # 计算碱基概率
                        # print("prob",prob)
                    if prob > max_prob:
                        max_prob = prob
                        best_codon = codon
                        # print("max_prob",max_prob)
                 selected_codons+=best_codon
                #  print("selected_codons",selected_codons)
                #  exit()
        # exit()
            

        # predicted_indices = logits.argmax(dim=-1).squeeze().tolist()
        # #print("predicted_indices",predicted_indices)

        # All_toks_list = np.array(All_toks_list)

        # # 使用 predicted_indices 作为索引获取对应的预测结果
        # predicted_dna = [All_toks_list[idx] for idx in predicted_indices]
        

        # #predicted_dna = list(All_toks_list[predicted_indices])
        # predicted_dna = (
        #     "".join([token[-3:] for token in predicted_dna]).strip().upper()
        # )
        predicted_dna=selected_codons


        if average:
            #return repr_.mean(axis=1),logits,predicted_dna
            return repr_,logits,predicted_dna
        else:
            return repr_,logits,predicted_dna

def embed_sequences( sequences: List[Union[str, CodonSequence_infer]]) -> torch.Tensor:
        """Embeds a set of sequences using transcodon."""
        return torch.cat([embed_sequence(seq, average=True) for seq in sequences], dim=0)

def tokenize( seq: CodonSequence_infer) -> torch.Tensor:
        assert isinstance(seq, CodonSequence_infer), 'seq must be CodonSequence'
        alphabet = Alphabet.from_architecture('CodonModel')
        bc = alphabet.get_batch_converter()
        _, _, tokens = bc([('', seq.seq)])
        return tokens

