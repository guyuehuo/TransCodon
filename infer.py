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

from data_module import CodonDataModule
from checkpointing import PeriodicCheckpoint
from transcodon.sequence import CodonSequence
from transcodon.alphabet import Alphabet,ORGANISM2ID
from transcodon.model import ProteinBertModel
from transcodon.infer import embed_sequence, embed_sequences, tokenize
import time
import pandas as pd
from collections import defaultdict
from dtw import accelerated_dtw
import safetensors.torch

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
    #print("seq",seq)
    #print("token",token)
    return token

    
class CodonModel(pl.LightningModule):
    """PyTorch Lightning module for standard training."""
    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet = alphabet
        self.model = ProteinBertModel(args, alphabet)

        def init_weights(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.normal_(module.weight, std=.02)

            if isinstance(module, (torch.nn.Linear)):
                module.bias.data.fill_(0)
        self.model.apply(init_weights)

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
        self.loss_fn_2d = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
        #self.existence_loss =nn.functional.binary_cross_entropy_with_logits
        #self.struct_loss_fn = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def forward(self, x, y):
        return self.model(x,y)
    
    #*********************************transcodon setting*********************************
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay)

        if self.args.lr_scheduler == 'none':
            return optimizer
        elif self.args.lr_scheduler == 'warmup_sqrt':
            def schedule(global_step):
                if global_step < self.args.warmup_steps:
                    return (global_step+1) / self.args.warmup_steps
                else:
                    return np.sqrt(self.args.warmup_steps / global_step)
        elif self.args.lr_scheduler == 'warmup_cosine':
            def schedule(global_step):
                if global_step < self.args.warmup_steps:
                    return (global_step+1) / self.args.warmup_steps
                else:
                    progress = (global_step - self.args.warmup_steps) / self.args.num_steps
                    return max(0., .5 * (1. + math.cos(math.pi * progress)))
        else:
            raise ValueError('Unrecognised learning rate scheduler')

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        data,organism, labels  = \
            train_batch['input'].to(), \
            train_batch['organism'].to(dtype=torch.int64),\
            train_batch['labels'].to(dtype=torch.int64)
           # train_batch['struct_label'].to(dtype=torch.int64)
        

        output = self.model(data,organism)
        likelihoods = output['logits']
        #struct_label_output=output['struct_label_output']

        #print("organism size",organism.size())

        # 计算序列重建损失（例如交叉熵损失）
        loss = self.loss_fn(
            likelihoods.view(-1, len(self.alphabet.all_toks)),
            labels.view(-1)
        )
        self.log('train_mask_loss', loss)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        data,organism, labels = \
            val_batch['input'].to(), \
            val_batch['organism'].to(dtype=torch.int64),\
            val_batch['labels'].to(dtype=torch.int64)
            #val_batch['struct_label'].to(dtype=torch.int64)
            
        
        output = self.model(data,organism)
        likelihoods = output['logits']
        #struct_label_output=output['struct_label_output']
        loss = self.loss_fn(
            likelihoods.view(-1, len(self.alphabet.all_toks)),
            labels.view(-1)
        )
        self.log('val_mask_loss', loss)
        return loss
    

if __name__ == '__main__':
    # Parsing command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str,default="./Codon-AA-run/latest-120000.ckpt" , help='Path to the trained model checkpoint')
    parser.add_argument('--input_data', type=str,default="input.csv",help='Path to the input data for inference')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for inference')
    parser.add_argument('--output_file', type=str, default='predictions.csv', help='File to save the predictions')

    ProteinBertModel.add_args(parser)

    args = parser.parse_args()

    # Load the alphabet and data module
    alphabet = Alphabet.from_architecture('CodonModel')
    # args.model_checkpoint="/sugon_store/huqiuyue/calm/new_alphabet/2d_mask_predict-5M/CaLM-run28_V100/epoch=2-step=225258.ckpt"
    # model = CodonModel.load_from_checkpoint(args.model_checkpoint, alphabet=alphabet)

    model = CodonModel(args, alphabet)
    state_dict = safetensors.torch.load_file(args.model_checkpoint)
    model.load_state_dict(state_dict)

    #model.to(device)
    model.eval()  # Set the model to evaluation mode
    model.freeze()  # Ensure no gradients are calculated
    data=pd.read_csv(args.input_data)
    prediction_dna_list = []

    # Calculate similarity scores for each pair of sequences
    is_valid = True

    for _, row in data.iterrows():
        AA_seq = row["protein_seq"]
        input=AA_tokenize(AA_seq)
        # if not use_repo:
        if len(input)>2048*3:
                #count+=1
                AA_seq=AA_seq[:2048]
                input=input[:2048*3]
        organism = ORGANISM2ID[row["organism"]]
       

        embedding,logits,prediction_dna = embed_sequence(model,input,organism,AA_seq)

        # Append results to lists
        prediction_dna_list.append(prediction_dna.replace('U','T'))

        is_valid = _split_into_codons(prediction_dna,AA_seq)

        if not is_valid:
            print(f"prediction_dna:{prediction_dna} is not valid")
            exit()
    

    # Create a new DataFrame with similarity scores
    predict_data = pd.DataFrame({
        "organism": data["organism"],
        "protein_seq": data["protein_seq"],
        "prediction_dna": prediction_dna_list,
    })

    # Save to a new CSV file

    file_path = args.output_file
    # 确保目录存在
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    predict_data.to_csv(file_path, index=False)
