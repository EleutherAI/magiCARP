from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, Tuple

# specifies a dictionary of converters
_CONVERTERS: Dict[str, Dict[str, any]] = {}  # registry

def register_converter(origin_name_or_cls, destination_name_or_cls):
    """Decorator used register a CARP converter 

        Args:
            origin_name_or_cls: Name of the origin type
            destination_name_or_cls: Name of the destination type
    """

    def register_class(cls, origin_name, destination_name):
        if origin_name not in _CONVERTERS.keys():
            _CONVERTERS[origin_name] = {}

        _CONVERTERS[origin_name][destination_name] = cls
        setattr(sys.modules[__name__], origin_name+"_"+destination_name, cls)
        return cls

    if isinstance(origin_name_or_cls, str):
        origin_name = origin_name_or_cls.lower()
        destination_name = destination_name_or_cls.lower()
        return lambda c: register_class(c, origin_name, destination_name)

class Converter:
    def convert(self, path_orig : str, path_dest : str, **kwargs):
        raise NotImplementedError("Must be overridden.")

from carp.pytorch.checkpointing.converters import *
from carp.pytorch.checkpointing.convert_v1_to_v2 import *


def get_converter(origin_name, destination_name):
    return _CONVERTERS[origin_name.lower()][destination_name.lower()]

def get_converters_names():
    converters = list()
    for k,v in _CONVERTERS.items():
        for u in v.keys():
            converters.append(k + ", " + u)
    return converters
