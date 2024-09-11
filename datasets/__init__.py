import json
from operator import itemgetter
import warnings
import h5py
import kaldiio
import numpy as np
import os
import sys

from argparse import ArgumentParser, Namespace
from functools import partial
from logzero import logger
from math import floor
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# FIXME: should not use sys and ideally remove dependency to assist
sys.path.append("/esat/spchtemp/scratch/qmeeus/repos/assist")
from assist.tasks.coder import Coder as CoderBase
from assist.tasks import Structure, coder_factory, read_task
from assist.tools import parse_line, logger

from .torch_datasets import ArrayDataset, MultiModalDataset, SequenceDataset, SequenceDataset2, Subset

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

Json = Dict[str,Any]
PathLike = Union[Path,str]
TorchDataset = Union[SequenceDataset,Subset]

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

    @property
    def input_key(self):
        # TODO: backward compatibility with assist. Remove when possible
        warnings.warn("Deprecated use of input_key", DeprecationWarning)
        return self.data_keys[0]
    
    @property
    def output_key(self):
        # TODO: backward compatibility with assist. Remove when possible
        warnings.warn("Deprecated use of output_key", DeprecationWarning)
        return self.data_keys[1]

    @staticmethod
    def parse_args(parser:ArgumentParser) -> ArgumentParser:
        # TODO: For compatibility
        warnings.warn("Deprecated use of parse_args. Replace with add_arguments", DeprecationWarning)
        return Dataset.add_arguments(parser)
        
    @staticmethod
    def add_arguments(parser:ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group("Dataset")
        group.add_argument("--dataset", type=str, required=True, help="Path to dataset config")
        group.add_argument("--input-key", type=str, default=None, help="Input key in dataset config")
        group.add_argument("--output-key", type=str, default=None, help="Output key in dataset config")
        group.add_argument("--data-keys", type=lambda s: s.split(","), default=None, help="Comma-separated list of keys to load")
        return parser        

    @classmethod
    def from_args(cls, args:Namespace) -> 'Dataset':
        if args.data_keys is None:
            args.data_keys = [args.input_key, args.output_key]
        return cls(args.dataset, args.data_keys)

    def __init__(self, config:Json, load_keys:Optional[List[str]]=None):

        if isinstance(config, (str, Path)):
            config = self.load_config(config)

        self.attributes = config["dataset"]
        self.data_keys = load_keys or list(config["files"])

        errors = list(filter(lambda k: k not in config["files"], self.data_keys))
        if errors:
            raise ValueError(f"Unknown data key(s): {errors}")

        self._converters = config["converters"]
        self._data = config["files"]
        
    def __call__(self, subsets:Optional[Union[str,List[str]]]=None, p:Optional[Union[int,float]]=None) -> TorchDataset:

        if subsets is None:
            subsets = self.get_available_subsets()

        if isinstance(subsets, str):
            subsets = [subsets]

        data = ()
        for data_key in self.data_keys:
            filenames = [fn for subset, fn in self._data[data_key].items() if subset in subsets]
            if not filenames:
                raise ValueError(f"Invalid subsets: {subsets}. Available: {self.get_available_subsets()}")
            filetype = self._data[data_key]["_meta"]["format"]
            data += (self.merge_dict(*map(partial(self.load, filetype=filetype), filenames)),)

        data = self.validate_subset(*data)
        index = sorted(data[0])

        if isinstance(p, (int, float)) and p > 0:
            if type(p) is float:
                p = int(len(index) * p)
            index = np.random.choice(index, p, replace=False)

        datasets = ()
        for data_key, table in zip(self.data_keys, data):
            table = [table[idx] for idx in index]
            datatype = self._data[data_key]["_meta"]["type"]
            array, dim = self.process(table, datatype)
            DatasetClass = SequenceDataset2 if datatype in ("feats", "text", "tokens", *self._converters) else ArrayDataset
            opts = dict(dim=dim, name=data_key)
            if datatype in ("text", "tokens", *self._converters):
                opts["length"] = False
            if DatasetClass == SequenceDataset2:
                opts["pad_value"] = {"feats": 0., "text": 1, "tokens": -1}.get(datatype, 0) 
            datasets += (DatasetClass(array, **opts),)

        logger.info(f"Loaded {self.attributes['name']}/{subsets} ({len(index)} examples)")
        return MultiModalDataset(*datasets, indices=index)

    def load_converter(self, key):
        if type(self._converters[key]) is str:
            with open(self._converters[key]) as f:
                self._converters[key] = list(map(str.strip, f))

    @property
    def vocabulary(self) -> List[str]:
        # ESPnet's char_list
        if "tokens" not in self._converters:
            raise ValueError("Vocabulary not listed in config")
        vocab = self._converters["tokens"]
        if type(vocab) is str:
            with open(vocab) as f:
                vocab = dict(map(str.split, f))
                vocab = dict(zip(vocab.keys(), map(int, vocab.values())))
                vocab.update({"<blank>": 0, "<eos>": len(vocab) + 1})
                vocab = dict(sorted(vocab.items(), key=itemgetter(1)))
                self._converters["tokens"] = vocab
        return vocab

    @property
    def classes(self) -> List[str]:
        if "classes" not in self._converters:
            raise ValueError("Classes not listed in config")
        classes = self._converters["classes"]
        if type(classes) is str:
            with open(classes) as f:
                self._converters["classes"] = list(filter(bool, map(str.strip, f)))
        return self._converters["classes"]

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if "tokenizer" not in self._converters:
            raise ValueError("Tokenizer not set in config")
        tokenizer = self._converters["tokenizer"]
        if type(tokenizer) is str:
            self._converters["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer)
        return self._converters["tokenizer"]

    @property
    def coder(self) -> CoderBase:
        if "coder" not in self._converters:
            raise ValueError("Coder not set in config")

        coder = self._converters["coder"]
        if type(coder) is dict:
            Coder = coder_factory(coder["type"])
            structure = Structure(coder["structure"])
            coder = self._converters["coder"] = Coder(structure, coder["conf"])
        return coder

    def get_available_subsets(self, data_keys:Optional[Union[List[str],str]]=None) -> List[str]:
        data_keys = data_keys or self.data_keys

        if type(data_keys) is str:
            return [key for key in self._data[data_keys] if key != "_meta"]

        if not all(type(key) is str for key in data_keys):
            raise TypeError(f"Wrong type for {data_keys} (expected list of strings)")

        return sorted(set.intersection(*map(set, map(self.get_available_subsets, set(data_keys)))))

    def validate_subset(self, *inputs:List[Json]) -> List[Json]:
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

    def load(self, filename:PathLike, filetype:str) -> Json:

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

    def process(self, data:List[Union[str,float,int]], datatype:str) -> Tuple[np.array,int]:

        if datatype == "feats":
            data = np.array(data, dtype="object")
            return data, data[0].shape[-1]
        if datatype == "labels":
            label2class = {label: index for index, label in enumerate(self.classes)}
            to_class = label2class.__getitem__
            return np.array(list(map(to_class, data)), dtype=int), len(label2class)
        if datatype == "tasks":
            encode = lambda task: self.coder.encode(read_task(task))
            encoded = np.array(list(map(encode, data)), dtype=int)
            return encoded, self.coder.numlabels
        if datatype == "text":
            tokens = self.tokenizer(data, add_special_tokens=False, return_tensors="np")["input_ids"]
            tokens = np.array(list(map(np.array, tokens)), dtype=object)
            return tokens, self.tokenizer.vocab_size
        if datatype == "tokens":
            unk = self.vocabulary["<unk>"]
            encode = lambda tokens: np.array([self.vocabulary.get(token, unk) for token in tokens.split()], dtype=int)
            tokens = np.array(list(map(encode, data)), dtype=object)
            output_dim = len(self.vocabulary)
            return tokens, output_dim
        if datatype in self._converters:
            self.load_converter(datatype)
            vocab = self._converters[datatype]
            encode = lambda tokens: np.array(list(map(vocab.index, tokens.split())), dtype=int)
            tokens = np.array(list(map(encode, data)), dtype=object)
            output_dim = len(vocab)
            return tokens, output_dim

        raise NotImplementedError(f"Unknown datatype {datatype} for data of type {type(data)}")

    @staticmethod
    def split(dataset:TorchDataset, sizes:List[Union[int,float]], stratify:Optional[List[Any]]=None) -> List[Subset]:
        if sum(sizes) != 1:
            raise ValueError(f"Total size requested is not equal to 1.")

        indices = np.arange(len(dataset))

        if type(sizes[0]) is float:
            sizes = [floor(p * len(indices)) for p in sizes]
            sizes[0] = len(indices) - sum(sizes[1:])

        if type(stratify) is list:
            stratify = np.array(stratify)

        subsets = []
        for size in sizes[:-1]:
            labels = stratify[indices] if stratify is not None else None
            subset, indices = train_test_split(indices, train_size=size, stratify=labels)
            subsets.append(Subset(dataset, subset))

        assert sizes[-1] == len(indices)
        subsets.append(Subset(dataset, indices))
        return subsets

    @staticmethod
    def save_splits(outdir:PathLike, splits:Dict[str,Subset]) -> None:
        os.makedirs(outdir, exist_ok=True)
        for name, subset in splits.items():
            subset.save(f"{outdir}/{name}.txt")

    @staticmethod
    def merge_dict(*dictionaries):
        out = {}
        for dictionary in dictionaries:
            out.update(dictionary)
        return out

    @staticmethod
    def load_config(config:PathLike) -> Json:
        with open(config, "r") as f:
            return json.load(f)


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

