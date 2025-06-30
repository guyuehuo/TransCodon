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


class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 可传入类别权重（与CrossEntropy的weight类似）

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', ignore_index=-100
        )
        pt = torch.exp(-ce_loss)                     # 计算预测概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # Focal Loss核心公式
        # if self.alpha is not None:
        #     alpha_weights = self.alpha[targets]       # 根据目标标签选择权重
        #     focal_loss = alpha_weights * focal_loss
        
        return focal_loss.mean()
    
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
        
        # if (struct_label != -100).any(): 
        #     loss_2d = self.loss_fn_2d(
        #         struct_label_output.view(-1,struct_label_output.size(2)),
        #         struct_label.view(-1)
        #     )
        #     self.log('train_2d_mask_loss', loss_2d)

        #     loss += 0.2*loss_2d
       
        #     self.log('train_total_loss', loss)

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
        #print("struct_label_output.size(2)",struct_label_output.size(2))
        # if (struct_label != -100).any(): 
        #     loss_2d = self.loss_fn_2d(
        #         struct_label_output.view(-1,struct_label_output.size(2)),
        #         struct_label.view(-1)
        #     )
        #     self.log('val_2d_mask_loss', loss_2d)
        #self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    

if __name__ == '__main__':
    # Parsing command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str,default="./Codon-AA-run/latest-120000.ckpt" , help='Path to the trained model checkpoint')
    parser.add_argument('--input_data', type=str,default="AAAAAAAA",help='Path to the input data for inference')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for inference')
    parser.add_argument('--output_file', type=str, default='predictions.txt', help='File to save the predictions')

    args = parser.parse_args()

    # Load the alphabet and data module
    alphabet = Alphabet.from_architecture('CodonModel')
    # print(alphabet.all_toks,len(alphabet.all_toks))
 

    weight_name="transcodon-pretrain/latest-330000"
    name_list=[10,50,100,125,42]
    name_list=[42]
    new_data=False
    for i in name_list:
        use_repo=True
        weight_name="transcodon-run28_fintune-epoch15-top10/epoch=14-step=7626"
        args.model_checkpoint=weight_name+".ckpt"
        print(weight_name)

        #device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the trained model
        # model = CodonModel.load_from_checkpoint(args.model_checkpoint, alphabet=alphabet,strict=False)
        model = CodonModel.load_from_checkpoint(args.model_checkpoint, alphabet=alphabet)
        #model.to(device)
        model.eval()  # Set the model to evaluation mode
        model.freeze()  # Ensure no gradients are calculated



        name=['A','E','H','M','S','D','E1','E2']

        dataA = pd.read_csv("test_test_data/test_Arabidopsis_thaliana.csv")
        dataE = pd.read_csv("test_test_data/test_Escherichia_coli_general.csv")
        dataH = pd.read_csv("est_test_data/test_Homo_sapiens.csv")
        dataM = pd.read_csv("test_test_data/test_Mus_musculus.csv")
        dataS = pd.read_csv("test_test_data/test_Saccharomyces_cerevisiae.csv")
        dataD = pd.read_csv("test_test_data/test_Drosophila_melanogaster.csv")
        dataE1 = pd.read_csv("test_test_data/test_Escherichia_coli_1.csv")
        dataE2 = pd.read_csv("test_test_data/test_Escherichia_coli_2.csv")


        count=0

        for i in range(len(name)):
            if i==0:
                data=dataA
            elif i==1:
                data=dataE
            elif i==2:
                data=dataH
            elif i==3:
                data=dataM
            elif i==5:
                data=dataD
            elif i==6:
                data=dataE1
            elif i==7:
                data=dataE2
            else:
                data=dataS
            
        

            prediction_dna_list = []

            # Calculate similarity scores for each pair of sequences
            is_valid = True
            # continue

            # if i==0 or i==1:
            #     continue
            for _, row in data.iterrows():
                AA_seq = row["protein"]
                input=AA_tokenize(AA_seq)
                # if not use_repo:
                if len(input)>2048*3:
                        #count+=1
                        AA_seq=AA_seq[:2048]
                        input=input[:2048*3]
                organism = ORGANISM2ID[row["organism"]]
                # input=torch.device(input)
                # organism=torch.device(organism)
                # #print(f"input:{input}")
                #print(f"organism:{organism}")
                # print(input)
                # print("len input",input)


                embedding,logits,prediction_dna = embed_sequence(model,input,organism,AA_seq)

                # Append results to lists
                prediction_dna_list.append(prediction_dna.replace('U','T'))

                is_valid = _split_into_codons(prediction_dna,AA_seq)

                if not is_valid:
                    print(f"prediction_dna:{prediction_dna} is not valid")
                    exit()
            

            # Create a new DataFrame with similarity scores
            predict_data = pd.DataFrame({
                "natural_dna": data["dna"],
                "prediction_dna": prediction_dna_list,
            
            })

            # Save to a new CSV file

            file_path = f"data/evaluate/predicet_csv/{weight_name}_{name[i]}_prediction_dna.csv"

            # 确保目录存在
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            predict_data.to_csv(file_path, index=False)
            
            
            print(f"{name[i]} organism predict has done!!!")
            print(f"{name[i]} organism predict has done!!!")
            print(f"{name[i]} organism predict has done!!!")




    
        #***********************************codon recover rate***********************************************

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


        name=['A','E','H','M','S','D','E1','E2']


        dataA = pd.read_csv(f"data/evaluate/predicet_csv/{weight_name}_{name[0]}_prediction_dna.csv")
        dataE = pd.read_csv(f"data/evaluate/predicet_csv/{weight_name}_{name[1]}_prediction_dna.csv")
        dataH = pd.read_csv(f"data/evaluate/predicet_csv/{weight_name}_{name[2]}_prediction_dna.csv")
        dataM = pd.read_csv(f"data/evaluate/predicet_csv/{weight_name}_{name[3]}_prediction_dna.csv")
        dataS = pd.read_csv(f"data/evaluate/predicet_csv/{weight_name}_{name[4]}_prediction_dna.csv")
        dataD = pd.read_csv(f"data/evaluate/predicet_csv/{weight_name}_{name[5]}_prediction_dna.csv")
        dataE1 = pd.read_csv(f"data/evaluate/predicet_csv/{weight_name}_{name[6]}_prediction_dna.csv")
        dataE2 = pd.read_csv(f"data/evaluate/predicet_csv/{weight_name}_{name[7]}_prediction_dna.csv")

        name_path=[]
        for i in range(len(name)):
            if i==0:
                data=dataA
            elif i==1:
                data=dataE
            elif i==2:
                data=dataH
            elif i==3:
                data=dataM
            elif i==5:
                data=dataD
            elif i==6:
                data=dataE1
            elif i==7:
                data=dataE2
            else:
                data=dataS

            
            
            natural_predict_similarities = []


            # Calculate similarity scores for each pair of sequences
            for _, row in data.iterrows():
                dna_natural = row["natural_dna"]
                dna_predict = row["prediction_dna"]


                # Calculate similarities
                natural_pre_similarity = calculate_similarity(dna_natural, dna_predict)
            

                # Append results to lists
                natural_predict_similarities.append(natural_pre_similarity)


            # Create a new DataFrame with similarity scores
            similarity_data = pd.DataFrame({
                "natural_predict_similarity": natural_predict_similarities,
            
            })

            # Save to a new CSV file
            save_path = f"data/evaluate/recovery/{weight_name}_{name[i]}_with_similarity.csv"
            name_path.append(save_path)
            # 确保目录存在
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            similarity_data.to_csv(save_path, index=False)

            print(f"Similarity calculations completed and saved to {name[i]}_with_similarity.csv.")


        all_column_means = []

        # 遍历每个文件，逐列计算平均值
        for file in name_path:
            df = pd.read_csv(file)
            column_means = df.mean(axis=0)  # 逐列计算平均值
            #print(column_means)
            all_column_means.append(column_means)

        # 转换为 DataFrame，计算总体平均值
        all_means_df = pd.DataFrame(all_column_means)
        save_path=f"data/evaluate/recovery/ALL/{weight_name}_all_similarity.csv"
        if new_data:
          save_path=f"data/infer_test/my_predict/seq_smilar/ALL/new_data/{weight_name}_all_similarity.csv"
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)


        # Calculate overall mean and add it to the DataFrame
        overall_mean = all_means_df.mean(axis=0)  # 对每列的平均值再求平均
        all_means_df.loc['overall_mean'] = overall_mean
        print("save_path",save_path)

        # Save the DataFrame to a CSV file
        all_means_df.to_csv(save_path, index=True)

        # 输出结果
        print("A-E-H-M-S文件逐列平均值：")
        print(all_means_df)
        print("\n总体平均值：")
        print(overall_mean)

