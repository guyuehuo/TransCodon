"""Implementation of the Alphabet and BatchConverter classes.

This code has been modified from the original implementation
by Facebook Research, describing its ESM-1b paper."""

import itertools
import os
from typing import Sequence, Tuple, List, Union,Dict
import pickle
import re
import shutil
import torch
from pathlib import Path
import json

# rna_2d_toks ={
#             '...': 0,
#             '..(': 1, '..)': 2,
#             '.(.': 3, '.).': 4, '(..': 5,
#             ')..': 6, '.))': 7, '.)(': 8,
#             '.()': 9, '.((': 10, '(.)': 11,
#             '(.(': 12, ').)': 13, ').(': 14,
#             '().': 15, '((.': 16, ')).': 17,
#             ')(.': 18, '(()': 19, '(((': 20,
#             '())': 21, '()(': 22, ')()': 23,
#             ')((': 24, ')))': 25, '))(': 26,
#             '***': -100  # 特殊符号，表示填充或无效值
#         }
rna_2d_toks ={
            '.':0,'(':1,')':2,
            '*': -100  # 特殊符号，表示填充或无效值
        }

proteinseq_toks = {
    'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
}
# codonseq_toks = {
#     'toks': ['AAA', 'AAU', 'AAC', 'AAG', 'AUA', 'AUU', 'AUC', 'AUG', 'ACA', 'ACU', 'ACC', 'ACG', 'AGA', 'AGU', 'AGC', 'AGG', 'UAA', 'UAU', 'UAC', 'UAG', 'UUA', 'UUU', 'UUC', 'UUG', 'UCA', 'UCU', 'UCC', 'UCG', 'UGA', 'UGU', 'UGC', 'UGG', 'CAA', 'CAU', 'CAC', 'CAG', 'CUA', 'CUU', 'CUC', 'CUG', 'CCA', 'CCU', 'CCC', 'CCG', 'CGA', 'CGU', 'CGC', 'CGG', 'GAA', 'GAU', 'GAC', 'GAG', 'GUA', 'GUU', 'GUC', 'GUG', 'GCA', 'GCU', 'GCC', 'GCG', 'GGA', 'GGU', 'GGC', 'GGG']
# }


# AA_codonseq_toks = {
#     'toks': ['KAAA', 'NAAU', 'NAAC', 'KAAG', 'IAUA', 'IAUU', 'IAUC', 'MAUG', 'TACA', 'TACU', 
#              'TACC', 'TACG', 'RAGA', 'SAGU', 'SAGC', 'RAGG', 'YUAU', 'YUAC', '#UAA','#UGA',
#              'LUUA', 'FUUU', 'FUUC', 'LUUG', 'SUCA', 'SUCU', 'SUCC', 'SUCG', 'CUGU','#UAG',
#              'CUGC', 'WUGG', 'QCAA', 'HCAU', 'HCAC', 'QCAG', 'LCUA', 'LCUU', 'LCUC', 'LCUG', 
#              'PCCA', 'PCCU', 'PCCC', 'PCCG', 'RCGA', 'RCGU', 'RCGC', 'RCGG', 'EGAA', 'DGAU', 
#              'DGAC', 'EGAG', 'VGUA', 'VGUU', 'VGUC', 'VGUG', 'AGCA', 'AGCU', 'AGCC', 'AGCG', 
#              'GGGA', 'GGGU', 'GGGC', 'GGGG','PUNK', 'VUNK', 'WUNK', 'DUNK', 'RUNK', 'SUNK', 'FUNK', 'YUNK', 
#              'IUNK', 'EUNK', 'TUNK', 'MUNK', 'AUNK', 'KUNK', 'HUNK', 'CUNK', 'LUNK', 'GUNK', 'QUNK', 'NUNK','#UNK']
#              }
AA_codonseq_toks = {
    'toks': ['A','U','C','G']
             }
AA_codonseq_toks_replace = {
    'toks':  ['A','U','C','G']
             }
# AA_codonseq_toks_replace = {
#     'toks': ['KA', 'NU', 'NC', 'KG', 'IA', 'IU', 'IC', 'MG', 'TA', 'TU', 
#              'TC', 'TG', 'RA', 'SU', 'SC', 'RG', 'YU', 'YC', '#A',
#              'LA', 'FU', 'FC', 'LG', 'SA', 'SG', 'CU','#G',
#              'CC', 'WG', 'QA', 'HU', 'HC', 'QG', 'LA', 'LU', 'LC', 'LG', 
#              'PA', 'PU', 'PC', 'PG', 'RA', 'RU', 'RC', 'RG', 'EA', 'DU', 
#              'DC', 'EG', 'VA', 'VU', 'VC', 'VG', 'AA', 'AU', 'AC', 'AG', 
#              'GA', 'GU', 'GC', 'GG']
#              }
# All_toks = {
#     'toks': ["<cls>", "<pad>", "<eos>", "<unk>",'A','U','C','G','KA', 'NU', 'NC', 'KG', 'IA', 'IU', 'IC', 'MG', 'TA', 'TU', 
#              'TC', 'TG', 'RA', 'SU', 'SC', 'RG', 'YU', 'YC', '#A',
#              'LA', 'FU', 'FC', 'LG', 'SA', 'SG', 'CU','#G',
#              'CC', 'WG', 'QA', 'HU', 'HC', 'QG', 'LA', 'LU', 'LC', 'LG', 
#              'PA', 'PU', 'PC', 'PG', 'RA', 'RU', 'RC', 'RG', 'EA', 'DU', 
#              'DC', 'EG', 'VA', 'VU', 'VC', 'VG', 'AA', 'AU', 'AC', 'AG', 
#              'GA', 'GU', 'GC', 'GG','PUNK', 'VUNK', 'WUNK', 'DUNK', 'RUNK', 'SUNK', 'FUNK', 'YUNK', 
#              'IUNK', 'EUNK', 'TUNK', 'MUNK', 'AUNK', 'KUNK', 'HUNK', 'CUNK', 'LUNK', 'GUNK', 'QUNK', 'NUNK','#UNK',
#              "<mask>"]
#              }
All_toks = {
    'toks': ["<cls>", "<pad>", "<eos>", "<unk>","<sep>",'A','U','C','G',"<mask>"]
             }
All_toks_list = ["<cls>", "<pad>", "<eos>", "<unk>","<sep>",'A','U','C','G',"<mask>"]

AMINO_ACIDS: List[str] = [
    "A",  # Alanine
    "C",  # Cysteine
    "D",  # Aspartic acid
    "E",  # Glutamic acid
    "F",  # Phenylalanine
    "G",  # Glycine
    "H",  # Histidine
    "I",  # Isoleucine
    "K",  # Lysine
    "L",  # Leucine
    "M",  # Methionine
    "N",  # Asparagine
    "P",  # Proline
    "Q",  # Glutamine
    "R",  # Arginine
    "S",  # Serine
    "T",  # Threonine
    "V",  # Valine
    "W",  # Tryptophan
    "Y",  # Tyrosine
]
STOP_SYMBOLS = ["#"]  # Stop codon symbols

AMINO_ACID_TO_INDEX = {
    aa: sorted(
        [i for i,t in enumerate(All_toks['toks']) if t[0] == aa and t[1:] != "UNK"]
    )
    for aa in (AMINO_ACIDS + STOP_SYMBOLS)
}
AMINO_ACID_TO_INDEX["<"] = [0, 1, 2, 3, 89]

# Dictionary mapping each organism name to respective organism id

# 初始化一个空字典
# ORGANISM2ID = {}
# with open('/sugon_store/huqiuyue/transcodon/new_alphabet/transcodon/organism_to_id.json', 'r') as json_file:
#     ORGANISM2ID = json.load(json_file)

ORGANISM2ID = {}
with open('transcodon/organism_to_id.json', 'r') as json_file:
    ORGANISM2ID = json.load(json_file)
# print(ORGANISM2ID)



class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_codons: bool = True,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_codons = use_codons

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self):
        return BatchConverter(self)

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        if name in ("ESM-1b", "roberta_large"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>","<sep>")
            append_toks = ("<mask>",)
            prepend_bos = False
            append_eos = False
            use_codons = False
        elif name in ("CodonModel"):
            standard_toks = AA_codonseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>","<sep>")
            append_toks = ("<mask>",)
            prepend_bos = False
            append_eos = False
            use_codons = True
        else:
            raise ValueError("Unknown architecture selected")
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_codons)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return labels, strs, tokens
