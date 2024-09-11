import numpy as np
import torch
from torch.utils.data import TensorDataset


def test_dataset():
    from datasets import Dataset
    grabo = Dataset("config/grabo.2.json")
    pp10 = grabo("pp10")
    print(len(pp10))
    print(pp10.input_dim, pp10.output_dim)


def test_sequence_dataset():
    from datasets import Dataset
    from datasets.torch_datasets import SequenceDataset2, MultiModalDataset
    from datasets.torch_datasets import ArrayDataset
    
    grabo = Dataset("config/grabo.json", ["fbank", "tasks"])

    feats_dict = grabo.load(grabo._data["fbank"]["pp10"], "scp")
    tasks_dict = grabo.load(grabo._data["tasks"]["pp10"], "txt")
    indices = sorted(set(feats_dict).intersection(set(tasks_dict)))
    feats, _ = grabo.process([feats_dict[index] for index in indices], "feats")
    tasks, _ = grabo.process([tasks_dict[index] for index in indices], "tasks")

    feats = SequenceDataset2(feats)
    tasks = ArrayDataset(tasks)
    print(feats[0], tasks[0])

    dataset = MultiModalDataset(feats, tasks, indices=indices)
    assert len(feats) == len(tasks) == len(dataset)
    print(dataset[9])
    
    import ipdb; ipdb.set_trace()
    print(dataset[:3])    
    

if __name__ == "__main__":
    test_sequence_dataset()
