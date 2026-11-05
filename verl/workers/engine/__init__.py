# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from importlib import import_module

from .base import BaseEngine, EngineRegistry
from .fsdp import FSDPEngine, FSDPEngineWithLMHead

__all__ = [
    "BaseEngine",
    "EngineRegistry",
    "FSDPEngine",
    "FSDPEngineWithLMHead",
]

_OPTIONAL_EXPORTS = {
    "torchtitan": ("TorchTitanEngine", "TorchTitanEngineWithLMHead"),
    "veomni": ("VeOmniEngine", "VeOmniEngineWithLMHead"),
    "automodel": ("AutomodelEngine", "AutomodelEngineWithLMHead"),
    "mindspeed": ("MindspeedEngineWithLMHead", "MindspeedEngineWithValueHead", "MindSpeedMegatronEngineWithLMHead"),
    "megatron": ("MegatronEngine", "MegatronEngineWithLMHead", "MegatronEngineWithValueHead"),
}
_BACKEND_ERRORS = {}


def _load_backend(backend):
    module_name = "mindspeed" if backend == "mindspeed_megatron" else backend
    if module_name not in _OPTIONAL_EXPORTS:
        return
    if module_name in _BACKEND_ERRORS:
        raise _BACKEND_ERRORS[module_name]
    if all(name in globals() for name in _OPTIONAL_EXPORTS[module_name]):
        return
    try:
        if module_name == "megatron":
            # Preserve adaptor/monkey-patch ordering; Mindspeed is optional for Megatron.
            try:
                import_module(".mindspeed", __name__)
            except ImportError:
                pass
        module = import_module(f".{module_name}", __name__)
        exports = {name: getattr(module, name) for name in _OPTIONAL_EXPORTS[module_name]}
        globals().update(exports)
        __all__.extend(name for name in exports if name not in __all__)
    except Exception as error:
        # ponytail: failures are frozen for this process; restart after fixing dependencies.
        # Failed imports can leave decorated classes registered. Never return those classes.
        _BACKEND_ERRORS[module_name] = error
        raise


def __getattr__(name):
    for backend, names in _OPTIONAL_EXPORTS.items():
        if name in names:
            try:
                _load_backend(backend)
            except ImportError:
                # Compatibility for optional-class availability checks only.
                # Selected registry backends use _load_backend directly and fail visibly.
                return None
            return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
