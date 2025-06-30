"""Classes to deal with codon sequences."""

import abc
from Bio.Seq import Seq
from typing import Union, List


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
    'UAA': '#', 'UAG': '#', 'UGA': '#',  #1
    'CAU': 'H', 'CAC': 'H',
    'CAA': 'Q', 'CAG': 'Q',
    'AAU': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K',
    'GAU': 'D', 'GAC': 'D',
    'GAA': 'E', 'GAG': 'E',
    'UGU': 'C', 'UGC': 'C',
    'UGG': 'W',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGU': 'S', 'AGC': 'S','UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', #2
    'AGA': 'R', 'AGG': 'R',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

def _split_into_codons(seq: str):
    """Yield successive 3-letter chunks of a string/sequence."""
    for i in range(0, len(seq), 1):
            #if codon_to_amino_acid[seq[i:i + 3]] == 'Stop':
            #     #continue
            #     yield codon_to_amino_acid[seq[i:i + 3]]+seq[i:i + 3]
            # else:
            # if codon_to_amino_acid[seq[i:i + 3]]=='S':
            #      yield codon_to_amino_acid[seq[i:i + 3]]+seq[i],codon_to_amino_acid[seq[i:i + 3]]+seq[i+1],codon_to_amino_acid[seq[i:i + 3]]+seq[i+2]
            
            # if codon_to_amino_acid[seq[i:i + 3]]=='#':    
            #      yield seq[i],codon_to_amino_acid[seq[i:i + 3]]+seq[i+1],codon_to_amino_acid[seq[i:i + 3]]+seq[i+2]
            # else:
            #      yield seq[i],seq[i+1],codon_to_amino_acid[seq[i:i + 3]]+seq[i+2]
            yield seq[i]


def _split_into_rna2d(seq: str):
    """Yield successive 3-letter chunks of a string/sequence."""
    for i in range(0, len(seq), 1):
            #if codon_to_amino_acid[seq[i:i + 3]] == 'Stop':
            #     #continue
            #     yield codon_to_amino_acid[seq[i:i + 3]]+seq[i:i + 3]
            # else:
            yield seq[i]

def _split_infer(seq: str):
    """Yield successive 3-letter chunks of a string/sequence."""
    for i in range(0, len(seq), 1):
            #if codon_to_amino_acid[seq[i:i + 3]] == 'Stop':
            #     #continue
            #     yield codon_to_amino_acid[seq[i:i + 3]]+seq[i:i + 3]
            # else:
            if seq[i]=='*':
                 yield '<mask>'
            else:
                 yield seq[i]

def codons_convert_AA(seq: str):
    """Yield successive 3-letter chunks of a string/sequence."""
    for i in range(0, len(seq), 3):
            #continue
            #     yield codon_to_amino_acid[seq[i:i + 3]]+seq[i:i + 3]
            # else:
            yield codon_to_amino_acid[seq[i:i + 3]]+seq[i:i + 3]
        
class Sequence(abc.ABC):
    """Abstract base class for sequence data."""

    @property
    def seq(self):
        return self._seq 

    @property
    def tokens(self):
        return self._seq.split()

    def _sanitize(self, tokens: List[str]):
        return [x.strip() for x in tokens
            if x.strip() != '']


class CodonSequence(Sequence):
    """Class containing a sequence of codons.

    >>> seq = CodonSequence('ATGGCGCTAAAGCGGATC')
    >>> seq.tokens
    ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']

    >>> seq = CodonSequence('ATG GCG CTA AAG CGG ATC')
    >>> seq.tokens
    ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']
    """

    def __init__(self, seq_utr_: Union[str, Seq],seq_: Union[str, Seq]):
        super().__init__()
        seq_utr = str(seq_utr_)
        seq = str(seq_)
        _tokens = ['<cls>'] \
            + list(_split_into_codons(seq.replace('T', 'U').replace(' ', ''))) \
            + ['<eos>']
        _tokens = self._sanitize(_tokens)
        #("codon _tokens:",_tokens)
        self._seq = ' '.join(_tokens)

class WithUtr_CodonSequence(Sequence):
    """Class containing a sequence of codons.

    >>> seq = CodonSequence('ATGGCGCTAAAGCGGATC')
    >>> seq.tokens
    ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']

    >>> seq = CodonSequence('ATG GCG CTA AAG CGG ATC')
    >>> seq.tokens
    ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']
    """

    def __init__(self, utr_seq_: Union[str, Seq],cds_seq_: Union[str, Seq]):
        super().__init__()
        utr_seq = str(utr_seq_)
        cds_seq = str(cds_seq_)
        _tokens = ['<cls>'] \
            + list(_split_into_codons(utr_seq.replace('T', 'U').replace(' ', ''))) \
            + ['<sep>'] \
            + list(_split_into_codons(cds_seq.replace('T', 'U').replace(' ', ''))) \
            + ['<eos>']
        _tokens = self._sanitize(_tokens)
        #("codon _tokens:",_tokens)
        self._seq = ' '.join(_tokens)



class RNA2DSequence(Sequence):
    """Class containing a sequence of codons.

    >>> seq = CodonSequence('ATGGCGCTAAAGCGGATC')
    >>> seq.tokens
    ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']

    >>> seq = CodonSequence('ATG GCG CTA AAG CGG ATC')
    >>> seq.tokens
    ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']
    """

    def __init__(self, seq_: Union[str, Seq]):
        super().__init__()
        seq = str(seq_)
        _tokens = ['<cls>'] \
            + list(_split_into_rna2d(seq.replace(' ', ''))) \
            + ['<eos>']
        _tokens = self._sanitize(_tokens)
        #("rna token",_tokens)
        self._seq = ' '.join(_tokens)

class CodonSequence_infer(Sequence):
    """Class containing a sequence of codons.

    >>> seq = CodonSequence('ATGGCGCTAAAGCGGATC')
    >>> seq.tokens
    ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']

    >>> seq = CodonSequence('ATG GCG CTA AAG CGG ATC')
    >>> seq.tokens
    ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']
    """

    def __init__(self, seq_: Union[str, Seq]):
        super().__init__()
        seq = str(seq_)
        _tokens = ['<cls>'] \
            + list(_split_infer(seq.replace(' ', ''))) \
            + ['<eos>']
        _tokens = self._sanitize(_tokens)
        self._seq = ' '.join(_tokens)

class WithUtr_CodonSequence_infer(Sequence):
    """Class containing a sequence of codons.

    >>> seq = CodonSequence('ATGGCGCTAAAGCGGATC')
    >>> seq.tokens
    ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']

    >>> seq = CodonSequence('ATG GCG CTA AAG CGG ATC')
    >>> seq.tokens
    ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']
    """

    def __init__(self, utr_seq_: Union[str, Seq],cds_seq_: Union[str, Seq]):
        super().__init__()
        utr_seq = str(utr_seq_)
        cds_seq = str(cds_seq_)
        _tokens = ['<cls>'] \
            + list(_split_infer(utr_seq.replace('T', 'U').replace(' ', ''))) \
            + ['<sep>'] \
            + list(_split_infer(cds_seq.replace('T', 'U').replace(' ', ''))) \
            + ['<eos>']
        _tokens = self._sanitize(_tokens)
        #("codon _tokens:",_tokens)
        self._seq = ' '.join(_tokens)
# class CodonSequence_infer_utr(Sequence):
#     """Class containing a sequence of codons.

#     >>> seq = CodonSequence('ATGGCGCTAAAGCGGATC')
#     >>> seq.tokens
#     ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']

#     >>> seq = CodonSequence('ATG GCG CTA AAG CGG ATC')
#     >>> seq.tokens
#     ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']
#     """

#     def __init__(self, seq_: Union[str, Seq]):
#         super().__init__()
#         seq = str(seq_)
#         seq = str(seq_)
#         _tokens = ['<cls>'] \
#             + list(_split_infer(seq.replace(' ', ''))) \
#             + ['<eos>']
#         _tokens = self._sanitize(_tokens)
#         self._seq = ' '.join(_tokens)


class AminoAcidSequence(Sequence):
     def __init__(self, seq_: Union[str, Seq]):
        super().__init__()
        seq = str(seq_)
        _tokens = ['<cls>'] \
            + list(codons_convert_AA(seq.replace('T', 'U').replace(' ', ''))) \
            + ['<eos>']
        _tokens = self._sanitize(_tokens)
        self._seq = ' '.join(_tokens)

