# Copyright 2026 Individual Contributor: LeonardW-sl
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
"""CPU regression for the fused startup first batch on the ordinary trainer flow.

The test executes the real ``fit``, ``_async_gen_next_batch``,
``_restore_startup_fused_batch``, ``fit_step`` and ``_fit_generate`` bodies
extracted from the trainer source, plus the checkpoint helpers used by the
opt-in first step. Generation, the hard switch and the step consumers are
lightweight substitutes: no GPU, Ray, torch, tensordict or training run is
involved.

Contract under test (flag defaults off):

* ``_async_gen_next_batch`` keeps returning the ordinary five-tuple
  ``(metrics, timing_raw, epoch, batch, future_reward)`` and ``_fit_generate``
  keeps returning ``(batch, batch_data_future)``; the fused path adds no ready
  flag and no measurement event.
* ``_restore_startup_fused_batch(batch, metrics, timing_raw)`` runs after the
  first generation is joined and before ``_balance_batch``. It requires step
  one, an active 2/6 actor-rollout split, and the complete first batch
  (responses and rollout_log_probs); it then asks the shared-pool hard switch
  for 4/4 and clears ``_startup_fused_first_batch_pending``.
* The hard switch on that first step writes a scratch checkpoint instead of a
  normal one, so the previous checkpoint, ``latest_checkpointed_iteration.txt``
  and any remote copy stay untouched.

Run with: python3 -m pytest tests/experimental/one_step_off_policy/test_startup_fused_on_cpu.py
"""

import ast
import asyncio
import itertools
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "verl/experimental/one_step_off_policy/ray_trainer.py"
METHODS = (
    "fit",
    "_async_gen_next_batch",
    "_restore_startup_fused_batch",
    "fit_step",
    "_fit_generate",
)
CHECKPOINT_METHODS = (
    "_dynamic_resize_checkpoint_step_folder",
    "_dynamic_resize_transient_checkpoint_folder",
    "_save_dynamic_resize_transient_checkpoint",
    "_remove_dynamic_resize_transient_checkpoint",
    "_apply_dynamic_resize_hard_switch",
)
BATCH_SIZE = 96
SAMPLES = 5
TRAIN_STEPS = 6
SCHEDULE = {2: [6, 2], 4: [2, 6]}
FUSED_FLAG = "VERL_STARTUP_FUSED_FIRST_BATCH"
_UUID_COUNTER = itertools.count(1)


class _Tensor:
    """Minimal tensor stand-in supporting the operations the real methods use."""

    def __init__(self, values):
        self.values = [list(row) if isinstance(row, list) else row for row in values]

    def tolist(self):
        return self.values

    def sum(self, dim=-1):
        assert dim == -1
        return _Tensor([sum(row) for row in self.values])


def _flat(tensor):
    return [row[0] for row in tensor.values]


def _reorder(value, order):
    if isinstance(value, _Tensor):
        return _Tensor([value.values[index] for index in order])
    return [value[index] for index in order]


class _TensorDict(dict):
    """TensorDict stand-in: ``len`` is the batch dim, not the key count."""

    def __len__(self):
        for tensor in self.values():
            return len(tensor.values)
        return 0


class _Batch:
    """Small DataProto-shaped container that keeps rows aligned across reorders.

    ``repeat``/``union``/``reorder`` follow the real ``DataProto`` contract:
    repeating keeps the key set, and union refuses mismatched batch sizes or
    disagreeing values instead of silently overwriting them.
    """

    def __init__(self, rows, responses, *, probabilities=False, attention=True):
        self.rows = list(rows)
        self.batch = _TensorDict({"responses": _Tensor([[value] for value in responses])})
        if attention:
            self.batch["attention_mask"] = _Tensor([[1, 2] for _ in responses])
        if probabilities:
            self.batch["rollout_log_probs"] = _Tensor([[float(value)] for value in responses])
        self.non_tensor_batch = {"index": list(rows), "uid": [f"uid-{row}" for row in rows]}
        self.meta_info = {"timing": {}}

    def __len__(self):
        return len(self.rows)

    def repeat(self, repeat_times, interleave):
        assert interleave is True
        rows = [row for row in self.rows for _ in range(repeat_times)]
        responses = [value for value in _flat(self.batch["responses"]) for _ in range(repeat_times)]
        output = _Batch(rows, responses, attention=False)
        for key, tensor in self.batch.items():
            if key != "responses":
                output.batch[key] = _Tensor([row for row in tensor.values for _ in range(repeat_times)])
        output.non_tensor_batch = {
            key: [value for value in values for _ in range(repeat_times)]
            for key, values in self.non_tensor_batch.items()
        }
        return output

    def union(self, other):
        assert isinstance(other, _Batch)
        assert other.rows == self.rows, "DataProto.union requires identical batch sizes"
        for key, tensor in other.batch.items():
            if key in self.batch:
                assert self.batch[key].values == tensor.values, f"conflicting batch key {key!r} in union"
            else:
                self.batch[key] = tensor
        for key, values in other.non_tensor_batch.items():
            if key in self.non_tensor_batch:
                assert self.non_tensor_batch[key] == list(values), f"conflicting non-tensor key {key!r} in union"
            else:
                self.non_tensor_batch[key] = list(values)
        return self

    def reorder(self, order):
        order = list(order)
        assert sorted(order) == list(range(len(self)))
        for container in (self.batch, self.non_tensor_batch):
            for key, value in list(container.items()):
                container[key] = _reorder(value, order)
        self.rows = [self.rows[index] for index in order]

    def digest(self):
        probabilities = _flat(self.batch["rollout_log_probs"]) if "rollout_log_probs" in self.batch else []
        return list(zip(self.rows, _flat(self.batch["responses"]), probabilities, strict=True))


def _strip_imports(node):
    """Drop the method-level imports the harness stubs and reject nested ones.

    ``fit`` imports its logger lazily; the namespace provides that name. A new
    import anywhere else would pull a heavy module at execution time, so it is
    reported instead of silently resolved.
    """
    node.body = [stmt for stmt in node.body if not isinstance(stmt, ast.Import | ast.ImportFrom)]
    nested = [stmt for stmt in ast.walk(node) if isinstance(stmt, ast.Import | ast.ImportFrom)]
    if nested:
        raise AssertionError(f"{node.name} imports outside its body; extend the harness namespace instead")
    return node


def _select(config, path, default=None):
    node = config
    for part in path.split("."):
        if node is None:
            return default
        node = node.get(part, default) if isinstance(node, dict) else getattr(node, part, default)
    return node


def _load_methods():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OneStepOffRayTrainer")
    required = METHODS + CHECKPOINT_METHODS
    found = {node.name: node for node in cls.body if getattr(node, "name", None) in required}
    missing = sorted(set(required) - set(found))
    if missing:
        raise AssertionError(f"trainer is missing fused methods: {missing}")
    namespace = {
        "Any": object,
        "asyncio": asyncio,
        "os": os,
        "shutil": shutil,
        "DataProto": SimpleNamespace(from_single_dict=lambda batch: batch),
        "compute_response_mask": lambda batch: _Tensor([[1] for _ in range(len(batch))]),
        "marked_timer": lambda *_args, **_kwargs: nullcontext(),
        "np": SimpleNamespace(array=lambda values, dtype=None: list(values)),
        "OmegaConf": SimpleNamespace(select=_select, to_container=lambda config, resolve=False: config),
        "torch": SimpleNamespace(sum=lambda tensor, dim=-1: tensor.sum(dim)),
        "tqdm": lambda total, initial, desc: SimpleNamespace(update=lambda value: None, close=lambda: None),
        "Tracking": lambda **_kwargs: SimpleNamespace(log=lambda **_kwargs: None),
        "uuid": SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=f"{next(_UUID_COUNTER):032x}")),
        "pprint": lambda _value: None,
        "Role": SimpleNamespace(Critic="critic"),
        "logger": SimpleNamespace(warning=lambda *_args, **_kwargs: None),
    }
    body = [_strip_imports(node) for node in found.values()]
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    namespace["_run_fit"] = lambda trainer: asyncio.run(trainer.fit())
    return namespace


def _config():
    return SimpleNamespace(
        data=SimpleNamespace(train_batch_size=BATCH_SIZE),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=SAMPLES, temperature=1.0)),
        trainer=SimpleNamespace(
            balance_batch=True,
            project_name="cpu-regression",
            experiment_name="fused-startup",
            logger="console",
            get=lambda _key, default=None: default,
        ),
        global_profiler=SimpleNamespace(steps=None),
    )


def _batches(count, size=BATCH_SIZE):
    return [
        _Batch(
            [row for row in range(batch * size, (batch + 1) * size)],
            [row for row in range(batch * size, (batch + 1) * size)],
        )
        for batch in range(count)
    ]


def _expected_rows(batch_index, size=BATCH_SIZE, samples=SAMPLES):
    return tuple(row for row in range(batch_index * size, (batch_index + 1) * size) for _ in range(samples))


def _get_gen_batch(batch):
    """Prompt-side batch: the dataset rows, without generation-only keys."""
    gen_batch = _Batch(list(batch.rows), _flat(batch.batch["responses"]))
    gen_batch.non_tensor_batch = {key: list(values) for key, values in batch.non_tensor_batch.items()}
    return gen_batch


def _install_trainer(namespace, *, fused, faults=(), apply_port=None):
    trainer = SimpleNamespace(
        global_steps=1,
        epoch=0,
        is_last_step=False,
        total_training_steps=TRAIN_STEPS,
        metrics={},
        timing_raw={},
        config=_config(),
        _startup_fused_first_batch_pending=False,
        _get_gen_batch=_get_gen_batch,
        async_rollout_manager=SimpleNamespace(generate_sequences=None),
        _load_checkpoint=lambda: None,
        _validate=lambda: {"score": 0.0},
        _create_continuous_iterator=lambda: iter((0, batch) for batch in _batches(TRAIN_STEPS)),
        _fit_prepare_step=lambda: setattr(
            trainer, "is_last_step", trainer.global_steps == trainer.total_training_steps
        ),
        _fit_start_profile=lambda: None,
        _fit_stop_profile=lambda: None,
        _fit_validate=lambda: None,
        _fit_save_checkpoint=lambda: None,
        _fit_collect_metrics=lambda batch: None,
        _fit_experimental=lambda batch: None,
        _fit_dump_data=lambda batch: None,
        _record_resize_control_observation=lambda **kwargs: {},
        _fit_update_weights=lambda: None,
        _should_defer_next_rollout_for_resize=lambda: trainer.global_steps in SCHEDULE,
        _fit_postprocess_step=lambda: setattr(trainer, "global_steps", trainer.global_steps + 1),
    )
    for name in (
        "_fit_compute_reward",
        "_fit_compute_log_prob",
        "_fit_compute_ref_log_prob",
        "_fit_compute_critic",
        "_fit_compute_advantage",
        "_fit_update_critic",
    ):
        setattr(trainer, name, lambda batch: batch)
    for name in METHODS:
        setattr(trainer, name, namespace[name].__get__(trainer))

    arity = {"generate": set(), "fit_generate": set()}
    raw_generate = trainer._async_gen_next_batch
    raw_fit_generate = trainer._fit_generate

    async def generate_with_arity(continuous_iterator):
        result = await raw_generate(continuous_iterator)
        arity["generate"].add(None if result is None else len(result))
        return result

    async def fit_generate_with_arity(batch_data_future, continuous_iterator):
        result = await raw_fit_generate(batch_data_future, continuous_iterator)
        arity["fit_generate"].add(len(result))
        return result

    trainer._async_gen_next_batch = generate_with_arity
    trainer._fit_generate = fit_generate_with_arity

    active_topology = [2, 6] if fused else [4, 4]
    trainer._active_actor_size = lambda: active_topology[0]
    trainer._active_rollout_size = lambda: active_topology[1]

    generated, trained, switches, order = [], [], [], []
    trainer._fit_update_actor = (
        lambda batch: trained.append((trainer.global_steps, tuple(active_topology), batch.digest())) or batch
    )

    async def generate(batch):
        index = len(generated) + 1
        rows = list(batch.non_tensor_batch["index"])
        responses = _flat(batch.batch["responses"])
        generated.append((index, tuple(active_topology), tuple(rows)))
        if "partial_outputs" in faults and index == 1:
            rows = rows[:-1]
            responses = responses[:-1]
        probabilities = not ("missing_probability" in faults and index == 1)
        output = _Batch(rows, responses, probabilities=probabilities)
        # The rollout engine echoes the request's non-tensor fields (uid/index).
        output.non_tensor_batch = {key: list(values)[: len(rows)] for key, values in batch.non_tensor_batch.items()}
        return output

    async def apply(item, required_action):
        if "restore_failed" in faults and required_action == "expand_train":
            return {"resize/hard_switch_success": 0.0}
        return {"resize/hard_switch_success": 1.0}

    async def scheduled_apply_port(item, required_action):
        order.append(("switch", trainer.global_steps, required_action))
        switches.append((trainer.global_steps, tuple(active_topology), item, required_action))
        if apply_port is not None:
            metrics = await apply_port(item, required_action)
        else:
            metrics = await apply(item, required_action)
        if item is not None and "wrong_topology" not in faults and metrics.get("resize/hard_switch_success") == 1.0:
            active_topology[:] = item["actor_pool"]["size"]
        return metrics

    async def later_resize():
        target = SCHEDULE.get(trainer.global_steps)
        if target is not None:
            await scheduled_apply_port(None, "scheduled")
            active_topology[:] = target
        return {}

    def balance(batch, metrics):
        order.append(("balance", trainer.global_steps))
        batch.reorder(list(reversed(range(len(batch)))))

    trainer.async_rollout_manager.generate_sequences = generate
    trainer._apply_dynamic_resize_hard_switch = scheduled_apply_port
    trainer._maybe_dynamic_resize = later_resize
    trainer._balance_batch = balance
    observed = SimpleNamespace(
        generated=generated,
        trained=trained,
        switches=switches,
        order=order,
        topology=active_topology,
        arity=arity,
    )
    return trainer, observed


def _run(namespace, *, fused, faults=(), apply_port=None):
    old_flag = os.environ.get(FUSED_FLAG)
    if fused:
        os.environ[FUSED_FLAG] = "1"
    else:
        os.environ.pop(FUSED_FLAG, None)
    try:
        trainer, observed = _install_trainer(namespace, fused=fused, faults=faults, apply_port=apply_port)
        error = None
        try:
            namespace["_run_fit"](trainer)
        except Exception as exc:
            error = exc
        return trainer, observed, error
    finally:
        if old_flag is None:
            os.environ.pop(FUSED_FLAG, None)
        else:
            os.environ[FUSED_FLAG] = old_flag


def _checkpoint_trainer(namespace, tmp_path, *, pending, strategy="fsdp", use_critic=False):
    """Minimal trainer for the checkpoint helpers: no flow, no workers."""
    config = SimpleNamespace(
        trainer=SimpleNamespace(default_local_dir=str(tmp_path)),
        actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(strategy=strategy)),
    )
    trainer = SimpleNamespace(
        config=config,
        global_steps=1,
        use_critic=use_critic,
        timing_raw={},
        _startup_fused_first_batch_pending=pending,
    )
    calls = []

    class _WorkerGroup:
        def __init__(self, role):
            self.role = role

        def save_checkpoint(self, path, remote_path, step, **kwargs):
            calls.append((self.role, path, remote_path, step, kwargs))

    trainer.actor_rollout_wg = _WorkerGroup("actor")
    trainer.critic_wg = _WorkerGroup("critic")
    trainer._save_checkpoint = _forbidden_save
    for name in CHECKPOINT_METHODS:
        setattr(trainer, name, namespace[name].__get__(trainer))
    return trainer, calls


def _forbidden_save(*_args, **_kwargs):
    raise AssertionError("transient checkpointing must not use the normal save path")


def test_both_modes_align_and_first_batch_is_consumed_once():
    generated_expected = {
        True: [(2, 6), (4, 4), (6, 2), (6, 2), (2, 6), (2, 6)],
        False: [(4, 4), (4, 4), (6, 2), (6, 2), (2, 6), (2, 6)],
    }
    trained_expected = [(4, 4), (4, 4), (6, 2), (6, 2), (2, 6), (2, 6)]
    for fused in (True, False):
        namespace = _load_methods()
        trainer, observed, error = _run(namespace, fused=fused)
        assert error is None
        assert len(observed.generated) == len(observed.trained) == TRAIN_STEPS
        assert observed.arity == {"generate": {5}, "fit_generate": {2}}
        assert [index for index, *_ in observed.generated] == list(range(1, TRAIN_STEPS + 1))
        expected = [_expected_rows(index) for index in range(TRAIN_STEPS)]
        assert [rows for _, _, rows in observed.generated] == expected
        trained_rows = [tuple(sorted(row[0] for row in digest)) for _, _, digest in observed.trained]
        assert trained_rows == [tuple(sorted(rows)) for rows in expected]
        assert len(set(trained_rows)) == TRAIN_STEPS
        actions = [entry[2] for entry in observed.order if entry[0] == "switch"]
        assert actions.count("expand_train") == (1 if fused else 0)
        if fused:
            assert observed.order.index(("switch", 1, "expand_train")) < observed.order.index(("balance", 1))
            item = next(item for step, _, item, action in observed.switches if action == "expand_train")
            assert item["actor_pool"]["size"] == [4, 4]
            assert item["rollout_pool"]["size"] == [4, 4]
            assert item["actor_pool"]["from_pool"] == "shared_pool"
            assert item["rollout_pool"]["from_pool"] == "shared_pool"
            assert item["release_old"] is False
            assert trainer._startup_fused_first_batch_pending is False
        assert [step for step, _, _, action in observed.switches if action == "scheduled"] == sorted(SCHEDULE)
        assert [topology for _, topology, _ in observed.generated] == generated_expected[fused]
        assert [topology for _, topology, _ in observed.trained] == trained_expected
        for _, _, digest in observed.trained:
            for row, response, probability in digest:
                assert response == row
                assert probability == float(row)


@pytest.mark.parametrize(
    ("fault", "error_types"),
    [
        # A short generation cannot be joined to the repeated batch at all.
        ("partial_outputs", (AssertionError, ValueError)),
        # Generation without log probabilities is rejected by the fused guard.
        ("missing_probability", RuntimeError),
        ("restore_failed", RuntimeError),
        ("wrong_topology", RuntimeError),
    ],
)
def test_failure_before_training(fault, error_types):
    namespace = _load_methods()
    _trainer, observed, error = _run(namespace, fused=True, faults=(fault,))
    assert isinstance(error, error_types)
    assert observed.trained == []
    assert not any(entry[0] == "balance" for entry in observed.order)
    assert len(observed.generated) == 1
    if fault in {"restore_failed", "wrong_topology"}:
        assert [action for _, _, _, action in observed.switches].count("expand_train") == 1


def test_apply_port_direct_switch_and_later_scheduled_switches():
    namespace = _load_methods()
    calls = []

    async def apply_port(item, required_action):
        calls.append((required_action, item is None))
        return {"resize/hard_switch_success": 1.0}

    _trainer, observed, error = _run(namespace, fused=True, apply_port=apply_port)
    assert error is None
    assert calls == [("expand_train", False), ("scheduled", True), ("scheduled", True)]
    assert [step for step, _, _, action in observed.switches if action == "scheduled"] == sorted(SCHEDULE)


def test_transient_checkpoint_folder_stays_out_of_the_step_namespace(tmp_path):
    namespace = _load_methods()
    trainer, _calls = _checkpoint_trainer(namespace, tmp_path, pending=True)

    normal = trainer._dynamic_resize_checkpoint_step_folder()
    transient = trainer._dynamic_resize_transient_checkpoint_folder()

    assert normal == str(tmp_path / "global_step_1")
    assert os.path.dirname(transient) == str(tmp_path / ".dynamic_resize_transient")
    assert os.path.basename(transient).startswith("global_step_1_")
    assert not transient.startswith(normal + os.sep)


def test_transient_save_keeps_previous_checkpoint_and_latest_pointer(tmp_path):
    namespace = _load_methods()
    previous = tmp_path / "global_step_0"
    previous.mkdir()
    (previous / "actor").mkdir()
    latest = tmp_path / "latest_checkpointed_iteration.txt"
    latest.write_text("0", encoding="utf-8")

    trainer, calls = _checkpoint_trainer(namespace, tmp_path, pending=True, use_critic=True)
    folder = trainer._dynamic_resize_transient_checkpoint_folder()
    trainer._save_dynamic_resize_transient_checkpoint(folder)

    assert [role for role, *_ in calls] == ["actor", "critic"]
    for _role, path, remote_path, step, kwargs in calls:
        assert path.startswith(folder + os.sep)
        assert remote_path is None
        assert step == 1
        assert kwargs["transient"] is True
        assert kwargs["max_ckpt_to_keep"] is None
        assert kwargs["save_full_model_for_dynamic_resize"] is True
    assert latest.read_text(encoding="utf-8") == "0"
    assert (previous / "actor").is_dir()
    assert not (tmp_path / "global_step_1").exists()


def test_transient_cleanup_keeps_previous_checkpoint_and_latest_pointer(tmp_path):
    namespace = _load_methods()
    previous = tmp_path / "global_step_0"
    previous.mkdir()
    latest = tmp_path / "latest_checkpointed_iteration.txt"
    latest.write_text("0", encoding="utf-8")

    trainer, _calls = _checkpoint_trainer(namespace, tmp_path, pending=True)
    folder = Path(trainer._dynamic_resize_transient_checkpoint_folder())
    (folder / "actor").mkdir(parents=True)

    trainer._remove_dynamic_resize_transient_checkpoint(str(folder))
    assert not folder.exists()
    assert previous.is_dir()
    assert latest.read_text(encoding="utf-8") == "0"

    trainer._remove_dynamic_resize_transient_checkpoint(str(folder))
    assert latest.read_text(encoding="utf-8") == "0"


@pytest.mark.parametrize(
    ("pending", "required_action", "transient"),
    [(True, "expand_train", True), (False, "scheduled", False)],
)
def test_hard_switch_uses_the_scratch_checkpoint_only_on_the_first_step(tmp_path, pending, required_action, transient):
    namespace = _load_methods()
    previous = tmp_path / "global_step_0"
    previous.mkdir()
    latest = tmp_path / "latest_checkpointed_iteration.txt"
    latest.write_text("0", encoding="utf-8")
    seen = []

    trainer, _calls = _checkpoint_trainer(namespace, tmp_path, pending=pending)

    async def impl(*, item, required_action, checkpoint_step_folder=None, transient=False):
        # The real implementation falls back to the step folder when the wrapper passes None.
        folder = checkpoint_step_folder or trainer._dynamic_resize_checkpoint_step_folder()
        seen.append((required_action, checkpoint_step_folder, transient))
        Path(folder).mkdir(parents=True, exist_ok=True)
        return {"resize/hard_switch_success": 1.0}

    trainer._apply_dynamic_resize_hard_switch_impl = impl
    metrics = asyncio.run(trainer._apply_dynamic_resize_hard_switch(item={"step": 1}, required_action=required_action))

    assert metrics == {"resize/hard_switch_success": 1.0}
    [(seen_action, folder, seen_transient)] = seen
    assert (seen_action, seen_transient) == (required_action, transient)
    if transient:
        assert folder is not None
        assert os.path.dirname(folder) == str(tmp_path / ".dynamic_resize_transient")
        assert not Path(folder).exists()
    else:
        assert folder is None
        assert (tmp_path / "global_step_1").is_dir()
    assert previous.is_dir()
    assert latest.read_text(encoding="utf-8") == "0"
