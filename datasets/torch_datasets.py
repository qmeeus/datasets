import warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset as _IterableDataset, Subset as _Subset, TensorDataset

from typing import Any, Dict, Iterable, List, Optional, Union

__all__ = [
    "SequenceDataset",
    "Subset",
    "pad_and_sort_batch",
    "sort_batch",
]

Json = Dict[str,Any]


class MultiModalDataset(Dataset):
    
    def __init__(self, *datasets:Dataset, indices:Optional[List[str]]=None, sort_batch=False):
        
        if not all(len(dataset) == len(datasets[0]) for dataset in datasets):
            raise ValueError("Not all datasets have the same number of examples")

        self.datasets = datasets
        self.indices = np.array(indices) if indices is not None else np.arange(len(self.datasets[0]))
        self.sort_batch = sort_batch
        self.dims = tuple([dataset.dim for dataset in self.datasets])

    @property
    def data_collator(self):
        return pad_and_sort_batch if self.sort_batch else pad_batch

    def __len__(self):
        return len(self.datasets[0])
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            batch = [self[i] for i in range(*idx.indices(len(self)))]
            return self.data_collator(batch)
        
        if isinstance(idx, str):
            idx = np.where(self.indices == idx)[0]
            if not len(idx):
                raise KeyError(f"{idx} not in index")
            idx = idx[0]

        return tuple(flatten(*[dataset[idx] for dataset in self.datasets]))


class SequenceDataset2(Dataset):
    
    def __init__(self, sequences:np.ndarray, dim=None):
        if not isinstance(sequences, np.ndarray):
            raise ValueError(f"Got type {type(sequences)}. Expected numpy array")
        
        self.dim = sequences[0].shape[-1]
        if dim and self.dim != dim:
            raise ValueError(f"Mismatching dimension {dim}. Expected {self.dim}")
        
        self.sequences = sequences
        self.lengths = list(map(len, sequences))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        arrays = self.sequences[idx], self.lengths[idx]
        return to_tensor(*arrays)


class ArrayDataset(TensorDataset):
    
    def __init__(self, array, dim=None):
        super(ArrayDataset, self).__init__(*to_tensor(array))
        if array.ndim == 1 and array.dtype == int and set(array) != {0, 1} and not dim:
            raise ValueError("Data dimension could not be inferred and must be explicit")
        self.dim = array.shape[1] if array.ndim == 2 else dim or 1
        

class SequenceDataset(MultiModalDataset):

    """
    A pytorch dataset build from any number of arrays that can be sequences
    The difference between sequences and labels is made by checking whether the dtype
    of the array is object or not
    """

    def __init__(self, features:np.ndarray,
                 labels:Optional[np.ndarray]=None,
                 indices:Optional[List[Union[str,int]]]=None,
                 input_dim:Optional[int]=None,
                 output_dim:Optional[int]=None):

        warnings.warn("Use MultiModalDataset instead", DeprecationWarning)
        if input_dim or output_dim:
            warnings.warn("input_dim and output_dim should be inferred", DeprecationWarning)

        datasets = (SequenceDataset2(features),)
        if labels is not None:
            datasets += (TensorDataset(torch.tensor(labels)),)
            
        super(SequenceDataset, self).__init__(*datasets, indices=indices)
        self.input_dim, self.output_dim = self.dims
        self.feature_lengths = self.datasets[0].lengths

    @property
    def features(self):
        return self.datasets[0].sequences
    
    @property
    def labels(self):
        if len(self.datasets) > 1:
            return self.datasets[1].tensors[0]

        # if type(features[0]) == list and input_dim is None:
        #     raise ValueError("Cannot infer input dim")
        # self.input_dim = input_dim or features[0].shape[-1]

        # if labels is not None and labels.ndim == 1 and output_dim is None:
        #     raise ValueError("Cannot infer output dim")
        # self.output_dim = output_dim or (None if labels is None else labels.shape[1])

        # if labels is not None and len(labels) != len(features):
        #     raise ValueError(f"Size mismatch: {len(features)} and {len(labels)} examples")

        # if indices is not None and len(indices) != len(features):
        #     raise ValueError(f"Size mismatch: {len(features)} and {len(indices)} examples")

        # self.feature_lengths = [len(feats) for feats in features]
        # self.features = features
        # self.labels = labels
        # self.indices = np.array(indices) if indices is not None else np.arange(len(features))
        # if len(np.unique(self.indices)) != len(self.indices):
        #     raise ValueError("Index contains duplicates")

    # def __getitem__(self, idx:Union[slice,str,int]) -> List[torch.Tensor]:

    #     if isinstance(idx, slice):
    #         return self.data_collator([self[i] for i in range(*idx.indices(len(self)))])

    #     if isinstance(idx, str):
    #         idx = np.where(self.indices == idx)[0]
    #         if not len(idx):
    #             raise KeyError(f"{idx}")
    #         idx = idx[0]            

    #     tensors = self.features[idx], self.feature_lengths[idx]
        
    #     if self.labels is not None:
    #         tensors += (self.labels[idx],)

    #     return to_tensor(*tensors)

    # def __len__(self) -> int:
    #     return len(self.features)

    # @staticmethod
    # def data_collator(batch:List[List[torch.Tensor]]) -> List[torch.Tensor]:
    #     return pad_batch(batch)


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


def to_tensor(*arrays):
    tensors = ()
    for array in arrays:
        if isinstance(array, torch.Tensor):
            tensor = array.clone()
        else:
            tensor = torch.tensor(array)
        tensors += (tensor,)
    return tensors


def flatten(*arrays):
    for array in arrays:
        if isinstance(array, (tuple, list)):
            yield from array
        else:
            yield array
