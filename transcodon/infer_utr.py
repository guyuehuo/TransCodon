"""Module to use transcodon as a pretrained model."""

import os
import pickle
import requests
from typing import Optional, Union, List
import numpy as np
import torch
from .alphabet import Alphabet,ORGANISM2ID,AMINO_ACID_TO_INDEX
from .sequence import CodonSequence,WithUtr_CodonSequence_infer,WithUtr_CodonSequence
from .model import ProteinBertModel
import numpy as np
import random



def embed_sequence(model,utr_seq,sequence: Union[str, WithUtr_CodonSequence_infer], species_type_ids: int = 0,AA_seq:str="",average: bool = True) -> torch.Tensor:
        """Embeds an individual sequence using transcodon. If the ``average''
        flag is True, then the representation is averaged over all
        possible odons, providing a vector representation of the
        sequence."""
        #print(sequence.type)
        if isinstance(sequence, str):
            #seq = WithUtr_CodonSequence(utr_seq,sequence)
            seq = WithUtr_CodonSequence_infer(utr_seq,sequence)
            #print("1")
        elif isinstance(sequence, WithUtr_CodonSequence_infer):
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
        #exit()
        #output.to("cpu")
        logits = output['logits']
        #print("logits",logits)
        repr_ =output['representations']
        logits=logits[:,1:-1,:]
        logits=logits[:,:100,:]
        
        #base_to_index = {'A': 4, 'U': 5, 'C': 6, 'G': 7}
        base_to_index = {'A': 5, 'U': 6, 'C': 7, 'G': 8}

        # 屏蔽无效索引（只保留5-8）
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[..., 5:9] = False  # 保留5-8的索引
        logits = logits.masked_fill(mask, -float('inf'))

        predicted_indices = logits.argmax(dim=-1).squeeze().tolist()
        #print("predicted_indices",predicted_indices)
        All_toks_list = ["<cls>", "<pad>", "<eos>", "<unk>","<sep>",'A','U','C','G',"<mask>"]
        #print("predicted_indices",predicted_indices)

        All_toks_list = np.array(All_toks_list)

        # 使用 predicted_indices 作为索引获取对应的预测结果
        predicted_dna = [All_toks_list[idx] for idx in predicted_indices]
        #print("predicted_dna",predicted_dna)
        predicted_dna=(
            "".join([token for token in predicted_dna]).strip().upper()
        )

        if average:
            #return repr_.mean(axis=1),logits,predicted_dna
            return repr_,logits,predicted_dna
        else:
            return repr_,logits,predicted_dna

def embed_sequences( sequences: List[Union[str, WithUtr_CodonSequence_infer]]) -> torch.Tensor:
        """Embeds a set of sequences using transcodon."""
        return torch.cat([embed_sequence(seq, average=True) for seq in sequences], dim=0)

def tokenize( seq: WithUtr_CodonSequence_infer) -> torch.Tensor:
        assert isinstance(seq, WithUtr_CodonSequence_infer), 'seq must be CodonSequence'
        alphabet = Alphabet.from_architecture('CodonModel')
        bc = alphabet.get_batch_converter()
        _, _, tokens = bc([('', seq.seq)])
        return tokens

