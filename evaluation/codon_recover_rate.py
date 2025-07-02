"""PyTorch Lightning module for standard training."""

import math
import argparse
import os
import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import time


import pandas as pd
from collections import defaultdict
from dtw import accelerated_dtw

codon_to_amino_acid = {
    'UUU': 'F', 'UUC': 'F',
    'UUA': 'L', 'UUG': 'L',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I',
    'AUG': 'M',  # start condon
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'UAU': 'Y', 'UAC': 'Y',
    'UAA': '#', 'UAG': '#', 'UGA': '#',
    'CAU': 'H', 'CAC': 'H',
    'CAA': 'Q', 'CAG': 'Q',
    'AAU': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K',
    'GAU': 'D', 'GAC': 'D',
    'GAA': 'E', 'GAG': 'E',
    'UGU': 'C', 'UGC': 'C',
    'UGG': 'W',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGU': 'S', 'AGC': 'S','UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'AGA': 'R', 'AGG': 'R',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

# 氨基酸到密码子的映射
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
    '#': ['UAA', 'UAG', 'UGA'],  # 终止密码子            #2
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

def _split_into_codons(seq: str,AA_seq:str):
    """Yield successive 3-letter chunks of a string/sequence."""
    res=''
    for i in range(0, len(seq), 3):
            #if codon_to_amino_acid[seq[i:i + 3]] == 'Stop':
            #     #continue
            #     yield codon_to_amino_acid[seq[i:i + 3]]+seq[i:i + 3]
            # else:
            res+=codon_to_amino_acid[seq[i:i + 3]]
    AA_seq=AA_seq[:len(res)]
    print("generate AA seq",res)
    print("original AA seq",AA_seq)
    return res[:-1]==AA_seq[:-1]

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
            'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],           #3
            'S': ['AGU', 'AGC', 'UCU', 'UCC', 'UCA', 'UCG'],           #4
            'G': ['GGU', 'GGC', 'GGA', 'GGG']
        }
def AA_tokenize(seq: str):
    token=''
    for i in range(len(seq)-1):
        if seq[i]=='L':
           token+=("*U*")
        elif seq[i]=='R':
           token+=("*G*")
        elif seq[i]=='S':
           token+=("***")
        else:
            token+=amino_acid_to_codon[seq[i]][0][:2]
            token+='*'
    token+="U**"
    print("seq",seq)
    print("token",token)
    return token

def calculate_similarity(dna1, dna2):
            """
            Calculate similarity score between two DNA sequences.
            For every three codons (triplets), if they are the same, similarity score increases by 1.

            Args:
                dna1 (str): First DNA sequence.
                dna2 (str): Second DNA sequence.

            Returns:
                int: Similarity score.
            """
            # Ensure both DNA sequences are of the same length
            min_length = min(len(dna1), len(dna2))
            dna1 = dna1[:min_length]
            dna2 = dna2[:min_length]

            # Calculate similarity
            similarity_score = 0
            for i in range(0, min_length, 3):  # Iterate in steps of 3 (codon)
                if dna1[i:i+3] == dna2[i:i+3]:
                    similarity_score += 1

            return similarity_score/(min_length/3)  # Divide by number of codons

