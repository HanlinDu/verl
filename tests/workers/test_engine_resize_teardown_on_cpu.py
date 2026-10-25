"""The engine subgroup teardown must invalidate DTensor caches.

Stale DTensor specs survive a destroyed subgroup because ``DeviceMesh`` hashes
by content. This guards the wiring: if the teardown stops calling the
invalidation, a later resize back to an equal-content mesh can reuse a spec
whose process group no longer resolves.
"""

import ast
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_destroy_process_group(namespace):
    path = ROOT / "verl/workers/engine_workers.py"
    tree = ast.parse(path.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainingWorker")
    method = next(node for node in owner.body if getattr(node, "name", None) == "destroy_process_group")
    method = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    module = types.ModuleType("_resize_teardown_under_test")
    module.__dict__.update(namespace)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"),
        module.__dict__,
    )
    return module.destroy_process_group


def test_destroy_process_group_invalidates_dtensor_caches():
    events = []

    class Engine:
        process_group = object()

        def set_process_group(self, value):
            events.append(("set_process_group", value))

    fake_dist = types.SimpleNamespace(
        barrier=lambda *args, **kwargs: events.append(("barrier", None)),
        destroy_process_group=lambda group: events.append(("destroy", group)),
    )
    engine = Engine()
    destroy = _load_destroy_process_group(
        {
            "torch": types.SimpleNamespace(distributed=fake_dist),
            "clear_dtensor_dispatch_caches": lambda: events.append(("clear_caches", None)),
        }
    )
    self = types.SimpleNamespace(engine=engine)

    destroy(self)

    kinds = [event[0] for event in events]
    assert "destroy" in kinds
    assert "clear_caches" in kinds
    assert kinds.index("clear_caches") > kinds.index("destroy")
    assert ("set_process_group", None) in events


def test_teardown_survives_missing_process_group():
    """No subgroup yet: teardown must still invalidate and not raise."""
    events = []

    class Engine:
        process_group = None

        def set_process_group(self, value):
            events.append(("set_process_group", value))

    fake_dist = types.SimpleNamespace(
        barrier=lambda *args, **kwargs: None,
        destroy_process_group=lambda group: events.append(("destroy", group)),
    )
    destroy = _load_destroy_process_group(
        {
            "torch": types.SimpleNamespace(distributed=fake_dist),
            "clear_dtensor_dispatch_caches": lambda: events.append(("clear_caches", None)),
        }
    )
    self = types.SimpleNamespace(engine=Engine())

    destroy(self)

    assert ("clear_caches", None) in events
    assert not any(kind == "destroy" for kind, _ in events)
