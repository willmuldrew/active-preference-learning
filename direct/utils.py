import gc
import json
import random
import collections
import time
from contextlib import contextmanager

import numpy as np
import torch

import pynvml


@contextmanager
def timeit(name, timing_dict):
    """Context manager to help us capture timing information."""
    start_time = time.time()
    yield
    timing_dict[name] = time.time() - start_time


class LengthSampler:
    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value + 1))

    def __call__(self, n=None):
        if n is None:
            return np.random.choice(self.values)
        else:
            return np.random.choice(self.values, n).tolist()


def get_default_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def ordered_sub_dict(d, keys):
    return collections.OrderedDict((k, d[k]) for k in keys if k in d)


def dict_to_str(d, dps=3):
    def fmt(v):
        if torch.is_tensor(v) and torch.numel(v) == 1:
            v = v.item()

        try:
            v = float(v)
        except (TypeError, ValueError):
            pass

        if type(v) == float:
            return f"{v:.{dps}f}"
        else:
            return str(v)

    return ", ".join(f"{k}: {fmt(v)}" for k, v in d.items())


def print_stats(logs: dict, prefix: str, fields: list[str]):
    print(prefix, dict_to_str(ordered_sub_dict(logs, fields)))


def cols_to_records(data):
    keys = list(data.keys())
    res = []
    for n in range(len(data[keys[0]])):
        res.append({k: data[k][n] for k in keys})
    return res


def tensor_cols_to(data, device):
    res = dict()
    for k, v in data.items():
        if torch.is_tensor(v):
            v = v.to(device)
        if isinstance(v, list) and len(v) > 0 and torch.is_tensor(v[0]):
            v = [t.to(device) for t in v]
        res[k] = v
    return res


def records_to_cols(data):
    keys = next(iter(data)).keys()
    res = {k: list() for k in keys}
    for r in data:
        for k in keys:
            res[k].append(r[k])
    return res


def print_gpu_utilization(msg=""):
    pynvml.nvmlInit()
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(msg, f"GPU memory occupied: {info.used//1024**2} MB.")


def argsort(items, key=None) -> list[int]:
    indexes = list(range(len(items)))
    if key is None:
        return list(sorted(indexes, key=lambda i: items[i]))
    else:
        return list(sorted(indexes, key=lambda i: key(items[i])))


def shuffled(pairs):
    res = list(pairs)
    random.shuffle(res)
    return res


class JSONFileLogger:
    def __init__(self, path):
        self.path = path

    def log(self, obj):
        with open(self.path, "a") as f:
            f.write(json.dumps(obj))
            f.write("\n")


def always_assert(msg):
    if random.random() < 100:
        assert False, msg


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.01):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.delta = delta
        self.stop = False

    def update(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter = 0

    def should_stop(self):
        return self.stop


class LossMAEarlyStopper:
    def __init__(self, threshold=0.1):
        self.hwm = -1e10
        self.alpha = 0.9
        self._ma = None
        self.stop = False
        self.threshold = threshold

    def update(self, stats):
        value = stats["direct/loss/total"]

        if self.ma is None:
            self._ma = value
        else:
            self._ma = self.ma * self.alpha + (1 - self.alpha) * value

        if self.ma > self.hwm:
            self.hwm = self.ma

        if self.ma < self.threshold * self.hwm:
            self.stop = True

    @property
    def ma(self):
        return self._ma

    @property
    def target(self):
        return self.threshold * self.hwm

    def should_stop(self):
        return self.stop


def seed_everything(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def tensor_serialize(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError("Type not serializable")

