import json
import h5py
import kaldiio
import numpy as np
import os
import sys
from math import floor
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

sys.path.append("/esat/spchtemp/scratch/qmeeus/repos/assist")
from assist.tasks import Structure, coder_factory, read_task
from assist.tools import parse_line, logger
            
from .torch_datasets import SequenceDataset
from .torch_datasets import Subset

__all__ = [
    "Dataset",
    "Grabo",
    "FluentSpeechCommands",
    "VaccinChat",
    "SmartLights",
    "SequenceDataset",
    "Subset",
]

CONFIG_DIR = Path(__file__).parents[1]/"config"


class Dataset:

    @staticmethod
    def parse_args(parser):
        parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset config")
        parser.add_argument("--input-key", type=str, default="fbank", help="Input key in dataset config")
        parser.add_argument("--output-key", type=str, default="tasks", help="Output key in dataset config")
        return parser

    @classmethod
    def from_args(cls, args):
        return cls(args.dataset, [args.input_key, args.output_key])

    def __init__(self, config, load_keys=None):

        if isinstance(config, (str, Path)):
            config = self.load_config(config)
        
        self.attributes = config["dataset"]
        self.data_keys = load_keys or list(config["files"])
        
        self._converters = config["converters"]
        self._data = config["files"]
        
    def __call__(self, subsets, sample=1.):
        
        if isinstance(subsets, str):
            subsets = [subsets]
        
        data = ()
        for data_key in self.data_keys:
            filenames = [fn for subset, fn in self._data[data_key].items() if subset in subsets]
            if not filenames:
                raise ValueError(f"Invalid subsets: {subsets}. Available: {self.data_keys}")
            filetype = self._data[data_key]["_meta"]["format"]
            data += (self.merge(self.load, filenames, filetype),)
            
        data = self.validate_subset(*data)
        index = list(data[0])
        if 0 < sample < 1:
            index = np.random.choice(index, int(len(index) * sample), replace=False)
        
        arrays = ()
        dims = ()
        for data_key, table in zip(self.data_keys, data):
            table = [table[idx] for idx in index]
            array, dim = self.process(table, self._data[data_key]["_meta"]["type"])
            arrays += (array,)
            dims += (dim,)
        
        # TODO: Implement for more than 1 input & 1 output
        inputs, outputs = arrays[:2]
        idim, odim = dims[:2]
        return SequenceDataset(inputs, outputs, index, idim, odim)
                
    @property
    def classes(self):
        if "classes" not in self._converters:
            raise ValueError("Classes not listed in config")
        return self._converters["classes"]
        
    @property
    def tokenizer(self):
        if "tokenizer" not in self._converters:
            raise ValueError("Tokenizer not set in config")
        tokenizer = self._converters["tokenizer"]
        if type(tokenizer) is str:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self._converters["tokenizer"] = tokenizer
        return tokenizer

    @property
    def coder(self):
        if "coder" not in self._converters:
            raise ValueError("Tokenizer not set in config")
        
        coder = self._converters["coder"]
        if type(coder) is dict:    
            Coder = coder_factory(coder["type"])
            structure = Structure(coder["structure"])
            coder = self._converters["coder"] = Coder(structure, coder["conf"])
        return coder
        
    def validate_subset(self, *inputs):
        on_error = self.attributes.get("on_error", "raise")
        uttids = [set(data) for data in inputs]
        errors = set.union(*uttids) - set.intersection(*uttids)
        msg = f"{len(errors)} mismatches for {len(inputs)} arrays of lengths {list(map(len, inputs))}"
        if errors:
            if on_error == "raise":
                logger.error(msg)
                raise ValueError(msg)
            if on_error == "warn":
                logger.warn(msg)

        outputs = [
            {uttid: sample for uttid, sample in data.items() if uttid not in errors}
            for data in inputs
        ]

        if not all(outputs):
            msg = "No examples left after removing errors"
            logger.error(msg)
            raise ValueError(msg)

        return outputs
        
    def load(self, filename, filetype):
        
        if filetype == "scp":
            return dict(kaldiio.load_scp(filename))
        if filetype == "npz":
            return dict(np.load(filename))
        if filetype == "h5":
            with h5py.File(filename, "r") as h5f:
                return {uttid: h5f[uttid][()] for uttid in h5f}
        if filetype == "txt":
            with open(filename) as f:
                return dict(map(parse_line, f))

        raise NotImplementedError(f"Unknown filetype {filetype} for {filename}")

    def process(self, data, datatype):

        if datatype == "feats":
            data = np.array(data, dtype="object")
            return data, data[0].shape[-1]
        if datatype == "labels":
            label2class = {label: index for index, label in enumerate(self.classes)}
            to_class = label2class.__getitem__
            return np.array(list(map(to_class, data))), len(label2class)
        if datatype == "tasks":
            encode = lambda task: self.coder.encode(read_task(task))
            data = np.array(list(map(encode, data)))
            return data, self.coder.numlabels
        if datatype == "text":
            return self.tokenizer(data)["input_ids"], self.tokenizer.vocab_size

        raise NotImplementedError(f"Unknown datatype {datatype} for data of type {type(data)}")

    @staticmethod
    def split(dataset, sizes):
        if sum(sizes) != 1:
            raise ValueError(f"Total size requested is not equal to 1.")
        
        indices = dataset.indices

        if type(sizes[0]) is float:
            sizes = [floor(p * len(indices)) for p in sizes]
            sizes[0] = len(indices) - sum(sizes[1:])
            
        subsets = []
        for size in sizes[:-1]:
            subset, indices = train_test_split(indices, train_size=size)
            subsets.append(Subset(dataset, subset))

        assert sizes[-1] == len(indices)
        subsets.append(Subset(dataset, indices))
        return subsets

    @staticmethod
    def save_splits(outdir, splits):
        os.makedirs(outdir, exist_ok=True)
        for name, subset in splits.items():
            subset.save(f"{outdir}/{name}.txt")
        
    @staticmethod
    def merge(load, files, *args):
        out = {}
        for filename in files:
            out.update(load(filename, *args))
        return out
    
    @staticmethod
    def load_config(config):
        with open(config, "r") as f:
            return json.load(f)

    # def __call__(self, subset=None, indices=None, p=1.):
    #     if not(subset or indices):
    #         raise ValueError("No argument supplied.")
        
    #     ikey, okey = self.input_key, self.output_key
    #     subsets = list(self.config[ikey]) if subset is None else [subset]

    #     if not all(self.has_subset(subset) for subset in subsets):
    #         raise KeyError(f"Invalid subset: {subset}")

    #     inputs, outputs = self.load_inputs_and_outputs(subsets)
    #     index = np.array(list(inputs))

    #     if indices:
    #         if type(indices) is str:
    #             with open(indices) as f:
    #                 indices = list(map(str.strip, f))
    #         index = [idx for idx in index if idx in indices]
    #         if not len(index):
    #             raise ValueError("No element left. Look for mismatches in indices.")

    #     if 0 < p < 1:
    #         index = np.random.choice(index, int(len(index) * p), replace=False)

    #     if any(key in ("text", "asr") for key in (ikey, okey)) and self.tokenizer is None:
    #         raise ValueError("No tokenizer supplied.")

    #     if ikey in ("text", "asr"):
    #         inputs = self.tokenizer([inputs[idx] for idx in index])["input_ids"]
    #         input_dim = self.tokenizer.vocab_size
    #     else:
    #         inputs = np.array([inputs[idx] for idx in index], dtype="object")
    #         input_dim = inputs.shape[-1]

    #     if okey in ("text", "asr"):
    #         outputs = self.tokenizer([outputs[idx] for idx in index])["input_ids"]
    #         output_dim = self.tokenizer.vocab_size
    #     else:
    #         outputs = np.array([outputs[idx] for idx in index])
    #         output_dim = self.output_dim

    #     return SequenceDataset(inputs, outputs, index, input_dim=input_dim, output_dim=output_dim)


class Grabo(Dataset):

    CONFIG_FILE = CONFIG_DIR/"grabo.json"

    def __init__(self, inputs="fbank", outputs="tasks"):
        super(Grabo, self).__init__(self.CONFIG_FILE, [inputs, outputs])


class Patience(Dataset):

    CONFIG_FILE = CONFIG_DIR/"patience.json"

    def __init__(self, inputs="fbank", outputs="tasks"):
        super(Patience, self).__init__(self.CONFIG_FILE, [inputs, outputs])


class FluentSpeechCommands(Dataset):

    CONFIG_FILE = CONFIG_DIR/"fluent.json"

    def __init__(self, inputs="fbank", outputs="tasks"):
        super(FluentSpeechCommands, self).__init__(self.CONFIG_FILE, [inputs, outputs])


class VaccinChat(Dataset):

    CONFIG_FILE = CONFIG_DIR/"vaccinchat.json"

    def __init__(self, inputs="fbank", outputs="labels"):
        super(VaccinChat, self).__init__(self.CONFIG_FILE, [inputs, outputs])


class SmartLights(Dataset):

    CONFIG_FILE = CONFIG_DIR/"smartlights.json"

    def __init__(self, inputs="fbank", outputs="tasks"):
        super(SmartLights, self).__init__(self.CONFIG_FILE, [inputs, outputs])

