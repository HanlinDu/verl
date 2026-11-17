# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Check WORLD and subgroup full-state loading without allocating a GPU model."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from packaging import version


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("subgroup", [False, True])
def test_full_state_load_preserves_world_broadcast_contract(rank, subgroup, monkeypatch):
    import torch.distributed.checkpoint.state_dict as state_dict

    setter = Mock()
    monkeypatch.setattr(state_dict, "set_model_state_dict", setter)
    path = Path(__file__).resolve().parents[2] / "verl/utils/fsdp_utils.py"
    function = next(
        n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == "fsdp2_load_full_state_dict"
    )
    namespace = {
        "torch": torch,
        "version": version,
        "get_device_id": lambda: "cpu",
        "dist": SimpleNamespace(get_rank=lambda: rank, get_global_rank=lambda group, index: 3, broadcast=Mock()),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
    model = SimpleNamespace(named_buffers=lambda: [("buffer", object())])
    model.to = Mock(return_value=model)
    model.to_empty = Mock(return_value=model)
    group = object() if subgroup else None
    namespace["fsdp2_load_full_state_dict"](model, {}, process_group=group)
    assert setter.call_args.kwargs["options"].broadcast_from_rank0 is (not subgroup)
    assert model.to_empty.called is (not subgroup and rank != 0)
    assert model.to.called is (subgroup or rank == 0)
    assert namespace["dist"].broadcast.call_args.kwargs == {"src": 3 if subgroup else 0, "group": group}
