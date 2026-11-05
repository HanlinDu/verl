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

"""HCCL teardown has the same repeated-resize contract as NCCL."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("rank", [0, None, -1, "missing"])
@pytest.mark.parametrize("rebuild", [True, False])
def test_hccl_finalize_is_safe_twice(rank, rebuild):
    path = Path(__file__).resolve().parents[2] / "verl/checkpoint_engine/hccl_checkpoint_engine.py"
    cls = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == "HCCLCheckpointEngine")
    method = next(n for n in cls.body if getattr(n, "name", None) == "finalize")
    namespace = {"torch": SimpleNamespace(npu=SimpleNamespace(empty_cache=Mock()))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    destroy = Mock()
    communicator = object()
    engine = SimpleNamespace(
        rebuild_group=rebuild,
        world_size=2,
        pyhccl=SimpleNamespace(comm=communicator, destroyComm=destroy),
        send_buf=object(),
        recv_buf=object(),
    )
    if rank != "missing":
        engine.rank = rank
    namespace["finalize"](engine)
    namespace["finalize"](engine)
    assert destroy.call_count == int(rebuild and rank == 0)
    assert engine.send_buf is None and engine.recv_buf is None
    if rebuild:
        assert engine.rank is None and engine.world_size is None
