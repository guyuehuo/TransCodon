"""Module to use CaLM as a pretrained model."""

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
All_toks_list = ["<cls>", "<pad>", "<eos>", "<unk>",'KAAA', 'NAAU', 'NAAC', 'KAAG', 'IAUA', 'IAUU', 'IAUC', 'MAUG', 'TACA', 'TACU', 
             'TACC', 'TACG', 'RAGA', 'SAGU', 'SAGC', 'RAGG', 'YUAU', 'YUAC', '#UAA','#UGA',
             'LUUA', 'FUUU', 'FUUC', 'LUUG', 'SUCA', 'SUCU', 'SUCC', 'SUCG', 'CUGU','#UAG',
             'CUGC', 'WUGG', 'QCAA', 'HCAU', 'HCAC', 'QCAG', 'LCUA', 'LCUU', 'LCUC', 'LCUG', 
             'PCCA', 'PCCU', 'PCCC', 'PCCG', 'RCGA', 'RCGU', 'RCGC', 'RCGG', 'EGAA', 'DGAU', 
             'DGAC', 'EGAG', 'VGUA', 'VGUU', 'VGUC', 'VGUG', 'AGCA', 'AGCU', 'AGCC', 'AGCG', 
             'GGGA', 'GGGU', 'GGGC', 'GGGG','PUNK', 'VUNK', 'WUNK', 'DUNK', 'RUNK', 'SUNK', 'FUNK', 'YUNK', 
             'IUNK', 'EUNK', 'TUNK', 'MUNK', 'AUNK', 'KUNK', 'HUNK', 'CUNK', 'LUNK', 'GUNK', 'QUNK', 'NUNK','#UNK',
             "<mask>"]
class ArgDict:
    def __init__(self, d):
        self.__dict__ = d

_ARGS = {
    'max_positions': 1024,
    'batch_size': 46,
    'accumulate_gradients': 1,
    'mask_proportion': 0.25,
    'leave_percent': 0.10,
    'mask_percent': 0.80,
    'warmup_steps': 1000,
    'weight_decay': 0.1,
    'lr_scheduler': 'warmup_cosine',
    'learning_rate': 4e-4,
    'num_steps': 121000,
    'num_layers': 12,
    'embed_dim': 768,
    'attention_dropout': 0.,
    'logit_bias': False,
    'rope_embedding': True,
    'ffn_embed_dim': 768*4,
    'attention_heads': 12
}
ARGS = ArgDict(_ARGS)

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed(42)  # 设置随机种子

class CaLM:
    """Module to use the Codon adaptation Language Model (CaLM)
    as published in C. Outeiral and C. M. Deane, "Codon language
    embeddings provide strong signals for protein engineering",
    bioRxiv (2022), doi: 10.1101/2022.12.15.519894."""

    def __init__(self, args: dict=ARGS, weights_file: Optional[str] = None) -> None:
        if weights_file is None:
            model_folder = os.path.join(os.path.dirname(__file__), 'calm_weights')
            weights_file = os.path.join(model_folder, 'calm_weights.ckpt')
            if not os.path.exists(weights_file):
                print('Downloading model weights...')
                os.makedirs(model_folder, exist_ok=True)
                url = 'http://opig.stats.ox.ac.uk/data/downloads/calm_weights.pkl'
                with open(weights_file, 'wb') as handle:
                    handle.write(requests.get(url).content)

        self.alphabet = Alphabet.from_architecture('CodonModel')
        self.model = ProteinBertModel(args, self.alphabet)
        self.bc = self.alphabet.get_batch_converter()
        self.All_toks_list = All_toks_list

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(weights_file, 'rb') as handle:
            # 加载模型权重文件到相应设备
            checkpoint = torch.load(handle, map_location=device, weights_only=False)
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            # 加载权重到模型
            self.model.load_state_dict(state_dict, strict=False)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
      
        # with open(weights_file, 'rb') as handle:
        #     state_dict = pickle.load(handle)
        #     self.model.load_state_dict(state_dict)



    def __call__(self, x,y):
        return self.model(x,y)

    def embed_sequence(self, sequence: Union[str, CodonSequence_infer], species_type_ids: int = 0,average: bool = True) -> torch.Tensor:
        """Embeds an individual sequence using CaLM. If the ``average''
        flag is True, then the representation is averaged over all
        possible odons, providing a vector representation of the
        sequence."""
        if isinstance(sequence, str):
            seq = CodonSequence_infer(sequence)
        elif isinstance(sequence, CodonSequence_infer):
            seq = sequence
        else:
            raise ValueError('Input sequence must be string or CodonSequence.')
        
        if isinstance(species_type_ids,str):
            species_type_ids = ORGANISM2ID[species_type_ids]
        elif isinstance(species_type_ids,int):
            pass
        else:
            raise ValueError('species_type_ids must be str or int')
    
        #print("seq",seq,seq.seq)
        tokens = self.tokenize(seq)
        print("tokens",tokens)
        print("species_type_ids",species_type_ids)  
        output = self.model(tokens,species_type_ids, repr_layers=[12])
        logits = output['logits']
        #print("logits",logits)
        repr_ =output['representations'][12]
        All_toks_list=self.All_toks_list



        if True:
            #**************************************************************
            # print(" ")
            # print("tokens",tokens)
            #print("AMINO_ACID_TO_INDEX",AMINO_ACID_TO_INDEX)
            #print("All_toks",All_toks_list)
            # print(" All_toks[token][0]",All_toks_list[8][0],AMINO_ACID_TO_INDEX[All_toks_list[8][0]])
            possible_tokens_per_position = [
                    AMINO_ACID_TO_INDEX[All_toks_list[token][0]] for token in tokens[0]]
            
            # print("possible_tokens_per_position",possible_tokens_per_position)
            # print(" ")
            # #*
            mask = torch.full_like(logits, float("-1e9"))
            # print("mask.shape",mask.shape)
            # print("mask",mask)

            for pos, possible_tokens in enumerate(possible_tokens_per_position):
                # print("pos",pos)
                # print("possible_tokens",possible_tokens)
                mask[:, pos, possible_tokens] = 0

            #print("mask",mask)
            # print("mask[1][2]",mask[0][1][11])

            logits=logits+mask

        logits=logits[:,1:-1,:]
        #predictions = []
       
        # Decode the predicted DNA sequence from the model output
        
        predicted_indices = logits.argmax(dim=-1).squeeze().tolist()
        #print("predicted_indices",predicted_indices)

        All_toks_list = np.array(All_toks_list)

        # 使用 predicted_indices 作为索引获取对应的预测结果
        predicted_dna = [All_toks_list[idx] for idx in predicted_indices]
        

        #predicted_dna = list(All_toks_list[predicted_indices])
        predicted_dna = (
            "".join([token[-3:] for token in predicted_dna]).strip().upper()
        )


        if average:
            return repr_.mean(axis=1),logits,predicted_dna
        else:
            return repr_,logits,predicted_dna

    def embed_sequences(self, sequences: List[Union[str, CodonSequence_infer]]) -> torch.Tensor:
        """Embeds a set of sequences using CaLM."""
        return torch.cat([self.embed_sequence(seq, average=True) for seq in sequences], dim=0)

    def tokenize(self, seq: CodonSequence_infer) -> torch.Tensor:
        assert isinstance(seq, CodonSequence_infer), 'seq must be CodonSequence'
        _, _, tokens = self.bc([('', seq.seq)])
        return tokens

