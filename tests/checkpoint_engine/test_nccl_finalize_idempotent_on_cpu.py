"""Idempotent-finalize regression for the NCCL checkpoint engine.

With ``rebuild_group=True`` the engine finalizes after every weight sync and
sets ``rank`` to None. The next resize calls ``finalize()`` again on the retired
engine, and ``getattr(self, "rank", -1) >= 0`` then compares None with int:

    TypeError: '>=' not supported between instances of 'NoneType' and 'int'

Observed in a real 8-GPU resize run at
``verl/checkpoint_engine/nccl_checkpoint_engine.py:149`` via
``TrainingWorker.reconfigure_checkpoint_engine``.
"""

import ast
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_finalize(namespace):
    path = ROOT / "verl/checkpoint_engine/nccl_checkpoint_engine.py"
    tree = ast.parse(path.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NCCLCheckpointEngine")
    method = next(node for node in owner.body if getattr(node, "name", None) == "finalize")
    method = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    module = types.ModuleType("_nccl_finalize_under_test")
    module.__dict__.update(namespace)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"),
        module.__dict__,
    )
    return module.finalize


def _engine(*, rank, world_size, rebuild_group=True):
    return types.SimpleNamespace(
        rank=rank,
        world_size=world_size,
        rebuild_group=rebuild_group,
        group_name="default",
        send_buf=object(),
        recv_buf=object(),
    )


def test_finalize_after_rank_becomes_none_is_idempotent():
    destroyed = []
    finalize = _load_finalize(
        {
            "collective": types.SimpleNamespace(
                destroy_collective_group=lambda name: destroyed.append(name),
            ),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
        }
    )
    engine = _engine(rank=3, world_size=6)

    finalize(engine)
    assert destroyed == ["default"]
    assert engine.rank is None and engine.world_size is None

    # The retired engine is finalized again by the next resize.
    finalize(engine)

    assert destroyed == ["default"], "group destroyed twice"
    assert engine.send_buf is None and engine.recv_buf is None


def test_finalize_without_prior_join_is_safe():
    """rank was never assigned: no communicator exists, so nothing to destroy."""
    destroyed = []
    finalize = _load_finalize(
        {
            "collective": types.SimpleNamespace(
                destroy_collective_group=lambda name: destroyed.append(name),
            ),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
        }
    )
    engine = _engine(rank=None, world_size=None)

    finalize(engine)

    assert destroyed == []
    assert engine.send_buf is None


def test_finalize_skips_destroy_when_rank_negative():
    """A non-member rank reports -1; that is also not a joined communicator."""
    destroyed = []
    finalize = _load_finalize(
        {
            "collective": types.SimpleNamespace(
                destroy_collective_group=lambda name: destroyed.append(name),
            ),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
        }
    )
    engine = _engine(rank=-1, world_size=6)

    finalize(engine)

    assert destroyed == []


def test_finalize_without_rebuild_group_never_destroys():
    destroyed = []
    finalize = _load_finalize(
        {
            "collective": types.SimpleNamespace(
                destroy_collective_group=lambda name: destroyed.append(name),
            ),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
        }
    )
    engine = _engine(rank=0, world_size=6, rebuild_group=False)

    finalize(engine)

    assert destroyed == []
    assert engine.rank == 0 and engine.world_size == 6
    assert engine.send_buf is None


def test_finalize_missing_rank_attribute_is_safe():
    """Engines that never joined have no ``rank`` attribute at all."""
    destroyed = []
    finalize = _load_finalize(
        {
            "collective": types.SimpleNamespace(
                destroy_collective_group=lambda name: destroyed.append(name),
            ),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
        }
    )
    engine = types.SimpleNamespace(rebuild_group=True, group_name="default", send_buf=object(), recv_buf=object())

    finalize(engine)

    assert destroyed == []
