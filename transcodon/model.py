"""Implementation of the ProteinBERT model.

This code has been modified from the original implementation
by Facebook Research, describing its ESM-1b paper."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .alphabet import AMINO_ACID_TO_INDEX
from .modules import (
    TransformerLayer,
    LearnedPositionalEmbedding,
    RobertaLMHead,
    RNA_2D_RobertaLMHead,
    RNA_2D_Structure_Connector,
    ESM1bLayerNorm
)

All_toks_list = ["<cls>", "<pad>", "<eos>", "<unk>",'A','U','C','G','KA', 'NU', 'NC', 'KG', 'IA', 'IU', 'IC', 'MG', 'TA', 'TU', 
             'TC', 'TG', 'RA', 'SU', 'SC', 'RG', 'YU', 'YC', '#A',
             'LA', 'FU', 'FC', 'LG', 'SA', 'SG', 'CU','#G',
             'CC', 'WG', 'QA', 'HU', 'HC', 'QG', 'LA', 'LU', 'LC', 'LG', 
             'PA', 'PU', 'PC', 'PG', 'RA', 'RU', 'RC', 'RG', 'EA', 'DU', 
             'DC', 'EG', 'VA', 'VU', 'VC', 'VG', 'AA', 'AU', 'AC', 'AG', 
             'GA', 'GU', 'GC', 'GG','PUNK', 'VUNK', 'WUNK', 'DUNK', 'RUNK', 'SUNK', 'FUNK', 'YUNK', 
             'IUNK', 'EUNK', 'TUNK', 'MUNK', 'AUNK', 'KUNK', 'HUNK', 'CUNK', 'LUNK', 'GUNK', 'QUNK', 'NUNK','#UNK',
             "<mask>"]
class ProteinBertModel(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--num_layers", default=12, type=int, metavar="N", help="number of layers"
        )
        parser.add_argument(
            "--embed_dim", default=768, type=int, metavar="N", help="embedding dimension"
        )
        parser.add_argument(
            "--attention_dropout", default=0., type=float, help="dropout on attention"
        )
        parser.add_argument(
            "--logit_bias", action="store_true", help="whether to apply bias to logits"
        )
        parser.add_argument(
            "--rope_embedding",
            default=True,
            #default=False,
            type=bool,
            help="whether to use Rotary Positional Embeddings"
        )
        parser.add_argument(
            "--ffn_embed_dim",
            default=768*4,
            type=int,
            metavar="N",
            help="embedding dimension for FFN",
        )
        parser.add_argument(
            "--attention_heads",
            default=12,
            type=int,
            metavar="N",
            help="number of attention heads",
        )
        parser.add_argument(
            "--species_embedding",
            default=True,
            type=bool,
            help="whether to add species_embedding"
        )
        parser.add_argument(
            "--struct_label",
            default=True,
            type=bool,
            help="whether to add struct_label"
        )
        parser.add_argument(
            "--species_vocab_size",
            default=1438,
            type=int,
            metavar="N",
            help="species_vocab_size"
        )

    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.emb_layer_norm_before = getattr(self.args, "emb_layer_norm_before", False)
        self.model_version = "ESM-1b"
        self.add_species_embedding = getattr(self.args, "species_embedding", False)
        #self.valid_codons = AMINO_ACID_TO_INDEX
        self._init_submodules_esm1b()

    def _init_submodules_common(self):
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
        )
        self.species_embedding = (nn.Embedding(self.args.species_vocab_size, self.args.embed_dim)  
                                  if self.add_species_embedding else None)
        
        # self.struct_label_output = nn.Sequential(
        #     nn.Linear(self.args.embed_dim, self.args.max_positions),  # 将隐向量映射为 L*L 的矩阵
        #     #nn.Unflatten(1, (self.args.max_positions, self.args.max_positions)),       # 将一维向量重塑为 L*L 矩阵
        #     nn.Sigmoid()  # 输出连接矩阵的概率
        # )
        self.struct_linear1 = nn.Linear(self.args.embed_dim, self.args.embed_dim)  #映射到4种碱基类型
        self.struct_linear2 = nn.Linear(self.args.embed_dim, self.args.embed_dim)
        self.struct_linear3 = nn.Linear(self.args.embed_dim, self.args.embed_dim)
        # 查询向量生成层
        #self.query = nn.Linear(self.args.embed_dim, self.args.embed_dim)
        # 键向量生成层
        #self.key = nn.Linear(self.args.embed_dim, self.args.embed_dim)
        # 无连接类别偏置项
        #self.null_bias = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
        #self.activation = nn.ReLU()
        self.activation= nn.ReLU()
        self.connector = RNA_2D_Structure_Connector(self.args.embed_dim)
        self.norm = ESM1bLayerNorm(self.args.embed_dim)

         # 存在性预测头
        self.existence_head = nn.Sequential(
            nn.Linear(self.args.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    self.args.attention_heads,
                    attention_dropout=self.args.attention_dropout,
                    add_bias_kv=(self.model_version != "ESM-1b"),
                    use_esm1b_layer_norm=(self.model_version == "ESM-1b"),
                    rope_embedding=self.args.rope_embedding,
                )
                for _ in range(self.args.num_layers)
            ]
        )

    def _init_submodules_esm1b(self):
        self._init_submodules_common()
        self.embed_scale = 1
        if not self.args.rope_embedding:
            print("Using learned positional embeddings")
            self.embed_positions = LearnedPositionalEmbedding(
                self.args.max_positions, self.args.embed_dim, self.padding_idx
            )
        self.emb_layer_norm_before = (
            ESM1bLayerNorm(self.args.embed_dim) if self.emb_layer_norm_before else None
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.args.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )
        #self.structure_linear = nn.Linear(self.args.embed_dim, 3)
        self.structure_linear = RNA_2D_RobertaLMHead(
            embed_dim=self.args.embed_dim,
            output_dim=3
        )

    def forward(self, tokens,species_type_ids, repr_layers=[12], need_head_weights=False):
        
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T
        #print(tokens)
        x = self.embed_scale * self.embed_tokens(tokens)
        #print("0",x)
        #print("x.shape",x.shape)    
        if getattr(self.args, "token_dropout", False):
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).float() / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

            #print("process mask")
            # print("x",x)
        #print("1",x.shape)
        if not self.args.rope_embedding:
            x = x + self.embed_positions(tokens)

        if self.add_species_embedding:
            # print("x.shape",x.shape)
            #print("species_type_ids.shape",species_type_ids.shape)
            # print("self.species_embedding(species_type_ids).shape",self.species_embedding(species_type_ids).unsqueeze(1).shape)
            x=x+self.species_embedding(species_type_ids).unsqueeze(1)
        #print("2",x.shape)
        if self.emb_layer_norm_before:
            x = self.emb_layer_norm_before(x)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x
        #print("3",x.shape)
        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        struct_logits=self.structure_linear(x)
        x = self.lm_head(x)
        
           
        result = {"logits": x,"struct_label_output":struct_logits ,"representations": hidden_representations}
        #result = {"logits": x,"representations": hidden_representations}
   
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if self.model_version == "ESM-1":
                # ESM-1 models have an additional null-token for attention, which we remove
                attentions = attentions[..., :-1]
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions

        
        return result

    @property
    def num_layers(self):
        return self.args.num_layers
