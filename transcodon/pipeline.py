"""Utilities to preprocess data for training."""

import abc
import itertools
from copy import deepcopy
from typing import List, Tuple
from collections import namedtuple

import torch
import numpy as np
from Bio.Data.CodonTable import standard_dna_table

from .sequence import Sequence
from .alphabet import Alphabet,AA_codonseq_toks_replace,rna_2d_toks 


def _split_array(array: np.ndarray, chunks: List[int]):
    """Split an array into N chunks of defined size."""
    assert np.sum(chunks) == len(array)
    # Randomly shuffle the array
    shuffled_array = np.random.permutation(array)
    #shuffled_array = array
    acc = 0
    arrays = []
    for chunk in chunks:
        arrays.append(shuffled_array[acc:acc+chunk])
        acc += chunk
    return arrays


PipelineInput = namedtuple('PipelineInput', ['sequence','organism','struct_label'])
PipelineOutput = namedtuple('PipelineOutput', ['input','labels','struct_label','ground_truth','organism'])
_PipelineData = namedtuple('PipelineData',
    ['ground_truth', 'sequence', 'target_mask','organism','struct_label'])

class PipelineData(_PipelineData):
    """Data structure for inner pipeline data."""

    @property
    def size(self):
        """Number of sequences in the data, equivalent
        to batch size."""
        assert len(self.ground_truth) == len(self.sequence)
        assert len(self.ground_truth) == len(self.target_mask)
        return len(self.ground_truth)

    def iterate(self):
        """Iterate over the data."""
        for i in range(self.size):
            yield self.ground_truth[i], self.sequence[i], self.target_mask[i],self.struct_label[i]


class PipelineBlock(abc.ABC):
    """Base class for data preprocessing pipeline blocks."""

    @abc.abstractmethod
    def __call__(self, input_: PipelineData) -> PipelineData:
        """Apply the block to a sequence."""
        raise NotImplementedError


class PipelineEntrypoint(PipelineBlock):
    """Starting point for a pipeline."""

    @abc.abstractmethod
    def __call__(self, input_: PipelineInput) -> PipelineData:
        """Apply the block to a sequence."""
        raise NotImplementedError


class PipelineEndpoint(PipelineBlock):
    """Final point for a pipeline."""

    @abc.abstractmethod
    def __call__(self, input_: PipelineData) -> PipelineOutput:
        """Apply the block to the data."""
        raise NotImplementedError


class Pipeline:
    """Class to preprocess data for training.

    This class is used to preprocess data for training. It is a pipeline of
    transformations that are applied to the data. The pipeline is defined by a
    list of callables that are applied in order.
    """

    def __init__(self, pipeline: List[PipelineBlock]):
        """Initialize the pipeline.

        Args:
            pipeline: List of callables that are applied in order.
        """

        if not issubclass(type(pipeline[0]), PipelineEntrypoint):
            raise ValueError('First block in a pipeline must be PipelineEntrypoint.')
        for block in pipeline[1:-1]:
            if issubclass(type(block), PipelineEntrypoint) or issubclass(type(block), PipelineEndpoint):
                raise ValueError('Intermediate blocks cannot be PipelineEntrypoint or PipelineEndpoint.')
        self.pipeline = pipeline

    #def __call__(self, data_: List[Sequence],organism_: List[int]) -> PipelineEndpoint:
    def __call__(self, data_: dict) -> PipelineEndpoint:
        """Apply the pipeline to the data.

        Args:
            data: Data to apply the pipeline to.

        Returns:
            Data after the pipeline has been applied.
        """
        #print("data_",data_)
        seq=[item["sequence"] for item in data_]
        organism_=[ item["organism"] for item in data_]
        struct_label_=[ item["struct_label"] for item in data_]
        # print("organism_",organism_)
        # print("seq",seq)
        # print("struct_label_",struct_label_)
        #exit()

        data = PipelineInput(sequence=seq,organism=organism_,struct_label=struct_label_)
        for transform in self.pipeline:
            data = transform(data)
        return data._asdict()


class DataCollator(PipelineEntrypoint):
    """Class to process sequences and apply random masking. The output
    of a call to DataCollator are strings of tokens, separated by spaces,
    and arrays which are zero except where a token change has occurred."""

    def __init__(self, params, alphabet):
        self.params = params
        self.alphabet = alphabet

        if self.alphabet.use_codons:
            # self.coding_toks = [''.join(letters)
            #     for letters in itertools.product(['A', 'U', 'C', 'G'], repeat=3)]
            self.coding_toks=AA_codonseq_toks_replace["toks"]
        else:
            self.coding_toks = list('ARNDCQEGHILKMFPSTWYV')

    def __call__(self, input_: PipelineInput) -> PipelineData:
        output = PipelineData(ground_truth=[],
            sequence=[], target_mask=[],organism=[],struct_label=[])

        for seq in input_.sequence:
            tokens, mask = self._mask_seq(seq.tokens)
            output.target_mask.append(mask)
            output.sequence.append(' '.join(tokens))
            output.ground_truth.append(' '.join(seq.tokens))
            # print("colltor target_mask",mask)
            # print("colltor sequence",' '.join(tokens))
            # print("colltor ground_truth",' '.join(seq.tokens))
        for organism in input_.organism:
            output.organism.append(organism)
        for struct_label in input_.struct_label:
            output.struct_label.append(' '.join(struct_label.tokens))
            #print("colltor struct_label",' '.join(struct_label.tokens))
        #exit()

        return output

    def _mask_seq(self, tokens_: List[str]):
        tokens = deepcopy(tokens_)
        num_tokens = len(tokens)
        num_changed_tokens = int(num_tokens * self.params.mask_proportion)
        num_to_mask = int(num_changed_tokens * self.params.mask_percent)
        num_to_leave = int(num_changed_tokens * self.params.leave_percent)
        num_to_change = num_changed_tokens - num_to_mask - num_to_leave

        #print("num_tokens",tokens,num_tokens)

        # Apply masking
        idxs = np.random.choice(
            np.arange(1, num_tokens-1), # avoid <cls> and <eos>
            size=num_changed_tokens, replace=False)
 
        #print("idxs",idxs)
        idxs_mask, _, idxs_change = _split_array(idxs,
            [num_to_mask, num_to_leave, num_to_change])
        
        for idx_mask in idxs_mask:
            # if  tokens[idx_mask][0]=='#':
            #     idxs.append(idx_mask-1)
            #     tokens[idx_mask-1] =tokens[idx_mask][0] + 'UNK'

            # if tokens[idx_mask][0]=='S':
            #      idxs.append(idx_mask-1)
            #      idxs.append(idx_mask-2)
            #      tokens[idx_mask-1] =tokens[idx_mask][0] + 'UNK'
            #      tokens[idx_mask-2] =tokens[idx_mask][0] + 'UNK'
            tokens[idx_mask] = "<mask>"
        for idx_change in idxs_change:
            tokens[idx_change] = np.random.choice(self.coding_toks)

        # print("idxs_mask",idxs_mask)
        # print("idxs_change",idxs_change)
        # print("changed_tokens",tokens)

        # Generate masks
        mask = np.zeros(num_tokens)
        mask[idxs] = 1.

        return tokens, mask


class DataTrimmer(PipelineBlock):
    """Class to trim sequences. Returns sequences and masks that have
    been trimmed to the maximum number of positions of the model."""

    def __init__(self, params, alphabet):
        self.params = params
        self.alphabet = alphabet

    def __call__(self, input_: PipelineData) -> PipelineData:
        output = PipelineData(ground_truth=[],
            sequence=[], target_mask=[],organism=[],struct_label=[])
        #print("input_",input_)
        for ground_truth, sequence, target_mask,struct_label in input_.iterate():
            #print("struct_label",struct_label)
            ground_truth, sequence, target_mask,struct_label = self._trim_seq(
                ground_truth, sequence, target_mask,struct_label)
            output.ground_truth.append(ground_truth)
            output.sequence.append(sequence)
            output.target_mask.append(target_mask)
            output.struct_label.append(struct_label)
        for organism in input_.organism:
            output.organism.append(organism)
        # for struct_label in input_.struct_label:
        #     output.struct_label.append(struct_label)

        return output

    def _trim_seq(
        self,
        original_seq: str,
        masked_seq: str,
        mask: np.ndarray,
        struct_label: str
    ) -> Tuple[str, str, np.ndarray,str]:
        # print("original_seq",original_seq)
        # print("masked_seq",masked_seq)

        original_tokens = original_seq.split()
        masked_tokens = masked_seq.split()
        struct_tokens = struct_label.split()
        #print("masked_tokens",masked_tokens)
        n_tokens = len(original_tokens)
        if n_tokens <= self.params.max_positions:
            # print("trim original_seq",original_seq)
            # print("trim masked_seq",masked_seq) 
            # print("trim mask",mask[:30])
            # print("trim struct_label",struct_label)
            return original_seq, masked_seq, mask,struct_label
        else:
            start = np.random.randint(0, n_tokens-self.params.max_positions)
            end = start+self.params.max_positions
            new_original_seq = ' '.join(original_tokens[start:end])
            new_masked_seq = ' '.join(masked_tokens[start:end])
            new_mask = mask[start:end]
            new_struct_label=' '.join(struct_tokens[start:end])
            return new_original_seq, new_masked_seq, new_mask,new_struct_label


class DataPadder(PipelineBlock):
    """Class to pad sequences."""

    def __init__(self, params, alphabet):
        self.params = params
        self.alphabet = alphabet

    def __call__(self, input_: PipelineData) -> PipelineData:
        output = PipelineData(ground_truth=[],
            sequence=[], target_mask=[],organism=[],struct_label=[])

        max_positions = max(len(seq.split()) for seq in input_.sequence)

        for ground_truth, sequence, target_mask,struct_label in input_.iterate():
            ground_truth, sequence, target_mask,struct_label= self._pad_seq(
                ground_truth, sequence, target_mask,struct_label,
                max_positions=max_positions)
            output.ground_truth.append(ground_truth)
            output.sequence.append(sequence)
            output.target_mask.append(target_mask)
            output.struct_label.append(struct_label)
        for organism in input_.organism:
            output.organism.append(organism)

        return output
  
    def _pad_seq(
        self,
        original_seq: str,
        masked_seq: str,
        mask: np.ndarray,
        struct_label:str,
        max_positions: int,
    ) -> Tuple[str, str, np.ndarray,str]:
        n_tokens = len(original_seq.split())
        if len(masked_seq.split()) < max_positions:
            original_seq_ = ' '.join(original_seq.split() \
                + ['<pad>'] * (max_positions - n_tokens))
            masked_seq_ = ' '.join(masked_seq.split() \
                + ['<pad>'] * (max_positions - n_tokens))
            
            struct_label = ' '.join(struct_label.split() \
                + ['*'] * (max_positions - n_tokens))
            mask_ = np.concatenate([mask, np.zeros(max_positions - n_tokens)])
            #struct_label=np.concatenate([struct_label, np.zeros(max_positions - n_tokens)])

            # print("pad original_seq",original_seq)
            # print("pad masked_seq",masked_seq)
            # print("pad mask",mask)
            # print("pad struct_label",struct_label)
          
            return original_seq_, masked_seq_, mask_,struct_label
        else:
            return original_seq, masked_seq, mask,struct_label


class DataPreprocessor(PipelineEndpoint):
    """Class to transform tokens into PyTorch Tensors."""

    def __init__(self, params, alphabet):
        self.params = params
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

    def __call__(self, input_: PipelineData) -> PipelineOutput:
        #print("origin",input_.sequence)
        new_input = self._compute_input(input_.sequence)
        # print("new_input",new_input)
        # print("input_.target_mask",input_.target_mask)
        # print("input_.ground_truth",input_.ground_truth)
        labels = self._compute_input(input_.ground_truth)
        mask = self._compute_mask(input_.target_mask)
        #print("mask",mask[:30])  
        #print("labels",labels)

        labels[~mask.bool()] = -100
        #print("new labels",labels)

        # print("input_.struct_label",input_.struct_label)
        # print(len(input_.struct_label))
        #exit()

        struct_label = self._compute_2d_label(input_.struct_label)
        # print("mask shape",mask.shape)
        #print("struct_label",struct_label)
        #exit()
        struct_label[~mask.bool()]=-100
        #print("new struct_label",struct_label)
        #exit()
        # print("input_.struct_label 0",len(input_.struct_label))
        # #print("input_.struct_label 1",input_.struct_label[0].shape)
        # print("input_.organism",input_.organism)
        # print("labels",labels)
        # print("input_.struct_label",input_.struct_label)
        # exit()
        #print("input_.struct_label 2",input_.struct_label[1].shape)
        #exit()
        return PipelineOutput(input=new_input, labels=labels,struct_label=struct_label,
            ground_truth=input_.ground_truth,organism=torch.tensor(input_.organism))

    def _compute_input(self, seq_list: List[str]) -> torch.Tensor:
        #print(seq_list)
        _, _, input_ = self.batch_converter([
            ('', seq) for seq in seq_list])
        return input_.to(dtype=torch.int32)

    def _compute_mask(self, mask_list: List[np.ndarray]) -> torch.Tensor:
        return torch.tensor(np.stack([
            mask for mask in mask_list], axis=0))
    
    def _compute_2d_label(self, label_list: List[np.ndarray]) -> torch.Tensor:
        converted_labels = []
        
        for label_seq in label_list:
            # 将每个符号转换为索引
            label_seq=label_seq.split()
            converted_seq = [rna_2d_toks.get(tok, -100) for tok in label_seq]
            converted_labels.append(converted_seq)

        # 将列表转换为 numpy 数组
        converted_labels = np.array(converted_labels, dtype=np.int32)

        # 将 numpy 数组转换为 torch.Tensor
        return torch.tensor(converted_labels, dtype=torch.long)


