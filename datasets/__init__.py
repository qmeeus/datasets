import json
import h5py
import kaldiio
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append("/esat/spchtemp/scratch/qmeeus/repos/assist")
from assist.tasks import Structure
from assist.tasks import coder_factory
from assist.tasks import read_task
from assist.tools import parse_line
from assist.tools import logger

from .torch_datasets import SequenceDataset
from .torch_datasets import Subset

__all__ = [
    "Dataset",
    "Grabo",
    "FluentSpeechCommands",
    "ChatBot",
    "SmartLights",
    "SequenceDataset",
    "Subset",
]

SUBSETS = ["train", "valid", "test"]
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
        return cls(args.dataset, args.input_key, args.output_key)

    def __init__(self, config, inputs="fbank", outputs="tasks", tokenizer=None):

        self.config = self.load_config(config) if isinstance(config, (str, Path)) else config
        assert inputs in self.config, f"Unknown inputs given: {inputs}"
        assert outputs in self.config, f"Unknown outputs given: {outputs}"
        self.input_key = inputs
        self.output_key = outputs
        self.splits = {}

        if outputs == "tasks":
            assert "coder" in self.config
            self.coder = self.load_coder()
            self.output_dim = self.coder.numlabels
        elif outputs == "labels":
            assert "classes" in self.config
            self.output_dim = len(self.config["classes"])
        else:
            raise NotImplementedError(f"Unknown output type: {outputs}")

        if any(key in ("asr", "text") for key in (inputs, outputs)):
            self.tokenizer = tokenizer
            if not tokenizer:
                logger.warn("No tokenizer supplied.")

    def has_subset(self, subset):
        ikey, okey = self.input_key, self.output_key
        return all(subset in self.config[key] for key in (ikey, okey))

    def __call__(self, subset=None, indices=None, p=1.):
        if not(subset or indices):
            raise ValueError("No argument supplied.")
        
        ikey, okey = self.input_key, self.output_key
        subsets = list(self.config[ikey]) if subset is None else [subset]

        if not all(self.has_subset(subset) for subset in subsets):
            raise KeyError(f"Invalid subset: {subset}")

        inputs, outputs = self.load_inputs_and_outputs(subsets)
        index = np.array(list(inputs))

        if indices:
            if type(indices) is str:
                with open(indices) as f:
                    indices = list(map(str.strip, f))
            index = [idx for idx in index if idx in indices]
            if not len(index):
                raise ValueError("No element left. Look for mismatches in indices.")

        if 0 < p < 1:
            index = np.random.choice(index, int(len(index) * p), replace=False)

        if any(key in ("text", "asr") for key in (ikey, okey)) and self.tokenizer is None:
            raise ValueError("No tokenizer supplied.")

        if ikey in ("text", "asr"):
            inputs = self.tokenizer([inputs[idx] for idx in index])["input_ids"]
            input_dim = self.tokenizer.vocab_size
        else:
            inputs = np.array([inputs[idx] for idx in index], dtype="object")
            input_dim = inputs.shape[-1]

        if okey in ("text", "asr"):
            outputs = self.tokenizer([outputs[idx] for idx in index])["input_ids"]
            output_dim = self.tokenizer.vocab_size
        else:
            outputs = np.array([outputs[idx] for idx in index])
            output_dim = self.output_dim

        return SequenceDataset(inputs, outputs, index, input_dim=input_dim, output_dim=output_dim)

    def load_inputs_and_outputs(self, subsets, ikey=None, okey=None):
        if type(subsets) is str:
            subsets = [subsets]

        ikey = ikey or self.input_key
        okey = okey or self.output_key

        all_inputs = {}
        all_outputs = {}
        for subset in subsets:

            input_file = self.config[ikey][subset]
            if input_file.endswith(".scp"):
                inputs = dict(kaldiio.load_scp(input_file))
            elif input_file.endswith(".npz"):
                inputs = dict(np.load(input_file))
            elif input_file.endswith(".h5"):
                with h5py.File(input_file, "r") as h5f:
                    inputs = {uttid: h5f[uttid][()] for uttid in h5f}
            elif ikey in ("text", "asr"):
                # if self.tokenizer is None:
                #     raise ValueError("Tokenizer not set.")
                with open(input_file) as f:
                    inputs = dict(map(parse_line, f))
            else:
                raise NotImplementedError(f"I don't know how to read this file: {input_file}")

            if okey == "tasks":
                taskfile = self.config["tasks"][subset]
                outputs = self.load_and_encode_tasks(taskfile)
            elif okey == "labels":
                labelfile = self.config["labels"][subset]
                outputs = self.load_and_encode_labels(labelfile)
            else:
                raise NotImplementedError(f"Unknown output {okey}")

            all_inputs.update(inputs)
            all_outputs.update(outputs)

        return self.validate_inputs_and_outputs(all_inputs, all_outputs)

    def load_coder(self):
        Coder = coder_factory(self.config["coder"]["type"])
        structure = Structure(self.config["coder"]["structure"])
        return Coder(structure, self.config["coder"]["conf"])

    def load_and_encode_labels(self, labelfile):
        class2index = {label: idx for idx, label in enumerate(self.config["classes"])}
        num_classes = len(self.config["classes"])
        labels = dict()
        with open(labelfile) as f:
            for uttid, label in map(parse_line, f):
                # array = np.zeros(num_classes)
                # array[class2index[label]] = 1
                # labels[uttid] = array
                labels[uttid] = class2index[label]
        return labels

    def load_and_encode_tasks(self, taskfile):
        tasks = dict()
        with open(taskfile) as f:
            for uttid, taskstring in map(parse_line, f):
                task = read_task(taskstring)
                encoded = self.coder.encode(task)
                tasks[uttid] = encoded
        return tasks

    def validate_inputs_and_outputs(self, inputs, labels):
        on_error = self.config.get("on_error", "raise")
        uttids = [set(data) for data in (inputs, labels)]
        errors = set.union(*uttids) - set.intersection(*uttids)
        error_message = f"{len(errors)} mismatches for {len(inputs)} inputs and {len(labels)} outputs"
        if errors and on_error == "raise":
            raise ValueError(error_message)
        elif errors:
            if on_error == "warn":
                logger.warn(error_message)

            inputs, labels = (
                {uttid: sample for uttid, sample in data.items() if uttid not in errors}
                for data in (inputs, labels)
            )

        if not(inputs and labels):
            raise ValueError("No examples left after removing errors")

        return inputs, labels

    def split(self, target, new, train_size=.7, test_size=.3):
        if train_size + test_size != 1:
            raise ValueError(f"Total size requested is not equal to 1...")
        data = self(data_key)
        train, test = train_test_split(np.arange(len(data)), test_size=test_size)
        self.splits[data_key] = (train, test)
        return Subset(data, train), Subset(data, test)

    @staticmethod
    def save_splits(outdir, splits):
        os.makedirs(outdir, exist_ok=True)
        for name, subset in splits.items():
            subset.save(f"{outdir}/{name}.txt")

    @staticmethod
    def load_config(config):
        with open(config, "r") as f:
            return json.load(f)


class Grabo(Dataset):

    CONFIG_FILE = CONFIG_DIR/"grabo.json"

    def __init__(self, inputs="fbank", outputs="tasks", tokenizer=None):
        super(Grabo, self).__init__(self.CONFIG_FILE, inputs, outputs, tokenizer)


class FluentSpeechCommands(Dataset):

    CONFIG_FILE = CONFIG_DIR/"fluent.json"

    def __init__(self, inputs="fbank", outputs="tasks", tokenizer=None):
        super(FluentSpeechCommands, self).__init__(self.CONFIG_FILE, inputs, outputs, tokenizer)


class ChatBot(Dataset):

    CONFIG_FILE = CONFIG_DIR/"chatbot.json"

    def __init__(self, inputs="fbank", outputs="labels", tokenizer=None):
        super(ChatBot, self).__init__(self.CONFIG_FILE, inputs, outputs, tokenizer)


class SmartLights(Dataset):

    CONFIG_FILE = CONFIG_DIR/"smartlights.json"

    def __init__(self, inputs="fbank", outputs="tasks", tokenizer=None):
        super(SmartLights, self).__init__(self.CONFIG_FILE, inputs, outputs, tokenizer)

