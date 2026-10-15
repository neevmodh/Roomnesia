"""Load SceneDiff annotation pickles safely: plain data only, every class/function reference is refused,
so a malicious pickle cannot execute code."""

import pickle


class _PlainData(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}: only plain data is allowed")


def load(path):
    with open(path, "rb") as f:
        return _PlainData(f).load()
