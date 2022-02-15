import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import Subset as _Subset

from typing import List, Optional, Union

__all__ = [
    "SequenceDataset",
    "Subset",
    "pad_and_sort_batch",
    "sort_batch",
]


class SequenceDataset(Dataset):

    def __init__(self, features:np.array, 
                 labels:Optional[np.array]=None, 
                 indices:Optional[List[Union[str,int]]]=None, 
                 input_dim:Optional[int]=None, 
                 output_dim:Optional[int]=None):
        
        if type(features[0]) == list and input_dim is None:
            raise ValueError("Cannot infer input dim")
        self.input_dim = input_dim or features[0].shape[-1]
        
        if labels is not None and labels.ndim == 1 and output_dim is None:
            raise ValueError("Cannot infer output dim")
        self.output_dim = output_dim or (None if labels is None else labels.shape[1])
        
        if labels is not None and len(labels) != len(features):
            raise ValueError(f"Size mismatch: {len(features)} and {len(labels)} examples")

        if indices is not None and len(indices) != len(features):
            raise ValueError(f"Size mismatch: {len(features)} and {len(indices)} examples")        
        
        self.feature_lengths = [len(feats) for feats in features]
        self.features = features
        self.labels = labels
        self.indices = list(indices if indices is not None else np.arange(len(features)))
        
    def __getitem__(self, idx:Union[slice,str,int]) -> List[torch.Tensor]:
        
        if isinstance(idx, slice):
            return self.data_collator([self[i] for i in range(*idx.indices(len(self)))])
        
        if isinstance(idx, str):
            if idx not in self.indices:
                raise KeyError(f"{idx}")
            idx = self.indices.index(idx)

        tensors = (torch.tensor(self.features[idx]), torch.tensor(self.feature_lengths[idx]))
        if self.labels is not None:
            tensors += (torch.tensor(self.labels[idx]),)

        return tensors

    def __len__(self) -> int:
        return len(self.features)

    @staticmethod
    def data_collator(batch:List[List[torch.Tensor]]) -> List[torch.Tensor]:
        return pad_batch(batch)


class Subset(_Subset):

    def __init__(self, dataset:SequenceDataset, indices:List[str]):
        super(Subset, self).__init__(dataset, indices)
        if hasattr(dataset, "input_dim"):
            self.input_dim = dataset.input_dim
        if hasattr(dataset, "output_dim"):
            self.output_dim = dataset.output_dim
            
    def data_collator(self, batch:List[List[torch.Tensor]]) -> List[torch.Tensor]:
        return self.dataset.data_collator(batch)

    def save(self, filename:str) -> None:
        with open(filename, "w") as f:
            index = self.dataset.indices[self.indices]
            f.writelines(map("{}\n".format, index))


def pad_batch(batch:List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    features, lengths, *tensors = list(zip(*batch))
    features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.)
    lengths = torch.stack(lengths, 0)
    tensors = tuple(torch.stack(tensor, 0) for tensor in tensors)
    return (features, lengths, *tensors)


def sort_batch(lengths:torch.LongTensor, *tensors:List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Sort a minibatch by the length of the sequences with the longest sequences first
    return the sorted batch targes and sequence lengths.
    This way the output can be used by pack_padded_sequences(...)
    """
    lengths, sort_order = lengths.sort(0, descending=True)
    return (lengths,) + tuple(tensor[sort_order] for tensor in tensors)


def pad_and_sort_batch(batch:List[List[torch.Tensor]]) -> List[torch.Tensor]:
    features, lengths, *tensors = pad_batch(batch)
    lengths, features, *tensors = sort_batch(lengths, features, *tensors)
    return (features, lengths, *tensors)
