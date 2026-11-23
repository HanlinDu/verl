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
extracted from the trainer source, plus the real
``_apply_dynamic_resize_hard_switch`` wrapper/implementation and the
checkpoint helpers used by the opt-in first step. Generation and the step
consumers are lightweight substitutes; the worker rebuild, checkpoint-engine
weight push and rollout runtime are also substituted because they need Ray and
accelerators. No real GPU, Ray, torch, tensordict or training run is involved.

Contract under test (flag defaults off):

* The retained first batch is generated exactly once on the 2/6 split and
  trained once after the restore to 4/4, with responses and rollout log
  probabilities preserved through balancing.
* ``_restore_startup_fused_batch(batch, metrics, timing_raw)`` runs after the
  first generation is joined and before ``_balance_batch``. It requires step
  one, an active 2/6 actor-rollout split, and the complete first batch
  (responses and rollout_log_probs); it then asks the shared-pool hard switch
  for 4/4 and clears ``_startup_fused_first_batch_pending`` only after success.
* The real hard switch chooses its checkpoint path from the pending flag and
  the configured handoff backend. Disk-backed first steps use a scratch
  checkpoint outside the step namespace, so the previous checkpoint,
  ``latest_checkpointed_iteration.txt`` and any remote copy stay untouched.
  GPU-direct handoff performs no save, load or handoff-staging operation at
  all, with the flag on or off. Disk-backed later switches keep using the
  normal resumable checkpoint.
* The flag only enables a fresh run; a resumed run is rejected before
  generation. A failure while restoring the retained first batch leaves the
  flag pending and stops before balance, actor update or checkpoint cleanup of
  the prior step.

Run with: python3 -m pytest tests/experimental/one_step_off_policy/test_startup_fused_on_cpu.py
"""

import ast
import asyncio
import itertools
import os
import shutil
import time
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
    "_gpu_direct_actor_model_resize_enabled",
    "_apply_dynamic_resize_hard_switch",
    "_apply_dynamic_resize_hard_switch_impl",
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
        "time": time,
        "DataProto": SimpleNamespace(from_single_dict=lambda batch: batch),
        "compute_response_mask": lambda batch: _Tensor([[1] for _ in range(len(batch))]),
        "dynamic_resize_shared_pool_enabled": lambda config: bool(getattr(config, "dynamic_resize_shared_pool", False)),
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


def _install_trainer(namespace, *, fused, faults=(), initial_step=1):
    trainer = SimpleNamespace(
        global_steps=initial_step,
        epoch=0,
        is_last_step=False,
        total_training_steps=TRAIN_STEPS,
        metrics={},
        timing_raw={},
        config=_config(),
        _startup_fused_first_batch_pending=False,
        _get_gen_batch=_get_gen_batch,
        async_rollout_manager=SimpleNamespace(generate_sequences=None),
        _load_checkpoint=lambda: setattr(trainer, "global_steps", initial_step - 1),
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

    async def scheduled_apply(item, required_action):
        order.append(("switch", trainer.global_steps, required_action))
        switches.append((trainer.global_steps, tuple(active_topology), item, required_action))
        metrics = await apply(item, required_action)
        if item is not None and "wrong_topology" not in faults and metrics.get("resize/hard_switch_success") == 1.0:
            active_topology[:] = item["actor_pool"]["size"]
        return metrics

    async def later_resize():
        target = SCHEDULE.get(trainer.global_steps)
        if target is not None:
            await scheduled_apply(None, "scheduled")
            active_topology[:] = target
        return {}

    def balance(batch, metrics):
        order.append(("balance", trainer.global_steps))
        batch.reorder(list(reversed(range(len(batch)))))

    trainer.async_rollout_manager.generate_sequences = generate
    trainer._apply_dynamic_resize_hard_switch = scheduled_apply
    trainer._maybe_dynamic_resize = later_resize
    trainer._balance_batch = balance
    observed = SimpleNamespace(
        generated=generated,
        trained=trained,
        switches=switches,
        order=order,
        topology=active_topology,
    )
    return trainer, observed


def _run(namespace, *, fused, faults=(), initial_step=1):
    old_flag = os.environ.get(FUSED_FLAG)
    if fused:
        os.environ[FUSED_FLAG] = "1"
    else:
        os.environ.pop(FUSED_FLAG, None)
    try:
        trainer, observed = _install_trainer(
            namespace,
            fused=fused,
            faults=faults,
            initial_step=initial_step,
        )
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


def _checkpoint_trainer(
    namespace,
    tmp_path,
    *,
    pending,
    strategy="fsdp",
    use_critic=False,
    gpu_direct=False,
    fail_at=None,
):
    """Real hard-switch methods with worker/runtime operations replaced by CPU spies.

    The only substitutions are cross-process or accelerator operations: worker
    checkpoint I/O, the checkpoint-engine weight push, and runtime destruction
    / reconstruction. The pending flag, backend gate, path selection and hard
    switch body are the real extracted methods.
    """
    config = SimpleNamespace(
        trainer=SimpleNamespace(default_local_dir=str(tmp_path)),
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(strategy=strategy),
            rollout=SimpleNamespace(backend="gpu_direct" if gpu_direct else "other"),
        ),
        dynamic_resize_shared_pool=True,
    )
    calls = []
    checkpoints = []
    trainer = SimpleNamespace(
        config=config,
        global_steps=1,
        use_critic=use_critic,
        hybrid_engine=True,
        timing_raw={},
        _startup_fused_first_batch_pending=pending,
        _active_dynamic_resize_topology={"actor_size": 2, "rollout_size": 6},
        _host_staging_config=SimpleNamespace(
            enable=True,
            effective_backend=lambda: "gpu_direct" if gpu_direct else "other",
            gpu_direct_low_memory=False,
        ),
        _dynamic_resize_target_sizes=lambda item: (
            int(item["actor_pool"]["size"][0]),
            int(item["rollout_pool"]["size"][0]),
        ),
    )

    def record(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        if fail_at == name:
            raise RuntimeError(f"injected {name} failure")

    def save_checkpoint(path, remote_path, step, **kwargs):
        checkpoints.append(("actor", path, remote_path, step, kwargs))
        Path(path).mkdir(parents=True, exist_ok=True)
        return None

    trainer.actor_rollout_wg = SimpleNamespace(save_checkpoint=save_checkpoint)
    trainer.critic_wg = SimpleNamespace(
        save_checkpoint=lambda path, remote_path, step, **kwargs: checkpoints.append(
            ("critic", path, remote_path, step, kwargs)
        )
    )
    trainer._save_checkpoint = lambda **kwargs: (
        checkpoints.append(("normal", None, None, 1, kwargs)) or record("normal_save", **kwargs)
    )
    trainer._prepare_dynamic_resize_handoff_session = lambda folder: record("prepare_handoff", folder) or {}
    trainer._stage_dynamic_resize_optimizer_handoff = lambda folder: record("stage_optimizer", folder) or {}
    trainer._load_dynamic_resize_checkpoint = lambda folder: record("load_checkpoint", folder)
    trainer._complete_dynamic_resize_handoff_session = lambda folder: (record("complete_handoff", folder) or {})

    async def destroy_runtime(**kwargs):
        record("destroy_runtime", **kwargs)
        if fail_at == "destroy_runtime":
            raise RuntimeError("injected destroy_runtime failure")

    async def update_weights(step):
        record("update_weights", step)
        if fail_at == "update_weights":
            raise RuntimeError("injected update_weights failure")

    async def rebuild_runtime(**kwargs):
        record("rebuild_runtime", **kwargs)
        if fail_at == "rebuild_runtime":
            raise RuntimeError("injected rebuild_runtime failure")

        trainer._active_dynamic_resize_topology = {
            "actor_size": kwargs["actor_size"],
            "rollout_size": kwargs["rollout_size"],
        }
        return {}

    trainer._latest_gpu_direct_resize_metrics = {}
    trainer._latest_rollout_rebuild_metrics = {}
    trainer._resize_controller = None
    trainer._destroy_dynamic_resize_runtime = destroy_runtime
    trainer.checkpoint_manager = SimpleNamespace(update_weights=update_weights)
    trainer._rebuild_hybrid_dynamic_resize_runtime = rebuild_runtime

    for name in CHECKPOINT_METHODS:
        setattr(trainer, name, namespace[name].__get__(trainer))
    return trainer, calls, checkpoints


def _hard_switch_request():
    return {
        "step": 1,
        "actor_pool": {"mode": "split", "from_pool": "shared_pool", "index": 0, "size": [4, 4]},
        "rollout_pool": {"mode": "split", "from_pool": "shared_pool", "index": 1, "size": [4, 4]},
        "release_old": False,
    }


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
    trainer, observed, error = _run(namespace, fused=True, faults=(fault,))
    assert isinstance(error, error_types)
    assert observed.trained == []
    assert not any(entry[0] == "balance" for entry in observed.order)
    assert len(observed.generated) == 1
    assert trainer._startup_fused_first_batch_pending is True
    if fault in {"restore_failed", "wrong_topology"}:
        assert [action for _, _, _, action in observed.switches].count("expand_train") == 1


def test_transient_checkpoint_folder_stays_out_of_the_step_namespace(tmp_path):
    namespace = _load_methods()
    trainer, _calls, _checkpoints = _checkpoint_trainer(namespace, tmp_path, pending=True)

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

    trainer, calls, _checkpoints = _checkpoint_trainer(namespace, tmp_path, pending=True, use_critic=True)
    folder = trainer._dynamic_resize_transient_checkpoint_folder()
    trainer._save_dynamic_resize_transient_checkpoint(folder)

    assert [role for role, *_ in _checkpoints] == ["actor", "critic"]
    for _role, path, remote_path, step, kwargs in _checkpoints:
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

    trainer, _calls, _checkpoints = _checkpoint_trainer(namespace, tmp_path, pending=True)
    folder = Path(trainer._dynamic_resize_transient_checkpoint_folder())
    (folder / "actor").mkdir(parents=True)

    trainer._remove_dynamic_resize_transient_checkpoint(str(folder))
    assert not folder.exists()
    assert previous.is_dir()
    assert latest.read_text(encoding="utf-8") == "0"

    trainer._remove_dynamic_resize_transient_checkpoint(str(folder))
    assert latest.read_text(encoding="utf-8") == "0"


@pytest.mark.parametrize(
    ("pending", "gpu_direct", "expected_path"),
    [
        (True, True, "gpu_direct_none"),
        (True, False, "transient"),
        (False, True, "gpu_direct_none"),
        (False, False, "normal"),
    ],
)
def test_hard_switch_implementation_selects_checkpoint_path(tmp_path, pending, gpu_direct, expected_path):
    namespace = _load_methods()
    previous = tmp_path / "global_step_0"
    previous.mkdir()
    latest = tmp_path / "latest_checkpointed_iteration.txt"
    latest.write_text("0", encoding="utf-8")

    trainer, calls, checkpoints = _checkpoint_trainer(namespace, tmp_path, pending=pending, gpu_direct=gpu_direct)
    metrics = asyncio.run(
        trainer._apply_dynamic_resize_hard_switch(item=_hard_switch_request(), required_action="expand_train")
    )

    assert metrics["resize/hard_switch_success"] == 1.0
    assert metrics["resize/schedule_applied"] == 1.0
    assert metrics["resize/active_actor_size"] == 4.0
    assert metrics["resize/active_rollout_size"] == 4.0
    assert trainer._active_dynamic_resize_topology == {"actor_size": 4, "rollout_size": 4}

    if expected_path == "gpu_direct_none":
        assert checkpoints == []
        assert not (tmp_path / ".dynamic_resize_transient").exists()
        assert not (tmp_path / "global_step_1").exists()
        assert [name for name, *_ in calls] == [
            "destroy_runtime",
            "rebuild_runtime",
            "update_weights",
        ]
    elif expected_path == "transient":
        assert len(checkpoints) == 1
        role, path, remote_path, step, kwargs = checkpoints[0]
        assert role == "actor"
        assert os.path.dirname(path).startswith(str(tmp_path / ".dynamic_resize_transient"))
        assert Path(path).name == "actor"
        assert remote_path is None
        assert step == 1
        assert kwargs["transient"] is True
        assert kwargs["max_ckpt_to_keep"] is None
        assert kwargs["save_full_model_for_dynamic_resize"] is True
        assert [name for name, *_ in calls] == [
            "prepare_handoff",
            "stage_optimizer",
            "destroy_runtime",
            "rebuild_runtime",
            "load_checkpoint",
            "complete_handoff",
            "update_weights",
        ]
        assert not any((tmp_path / ".dynamic_resize_transient").iterdir())
        assert not (tmp_path / "global_step_1").exists()
    else:
        assert checkpoints == [("normal", None, None, 1, {"save_full_model_for_dynamic_resize": True})]
        assert [name for name, *_ in calls] == [
            "normal_save",
            "prepare_handoff",
            "stage_optimizer",
            "destroy_runtime",
            "rebuild_runtime",
            "load_checkpoint",
            "complete_handoff",
            "update_weights",
        ]
        assert not (tmp_path / ".dynamic_resize_transient").exists()
    assert previous.is_dir()
    assert latest.read_text(encoding="utf-8") == "0"


@pytest.mark.parametrize("fail_at", ["destroy_runtime", "rebuild_runtime", "update_weights"])
def test_hard_switch_failure_cleans_scratch_checkpoint(tmp_path, fail_at):
    namespace = _load_methods()
    previous = tmp_path / "global_step_0"
    previous.mkdir()
    latest = tmp_path / "latest_checkpointed_iteration.txt"
    latest.write_text("0", encoding="utf-8")

    trainer, calls, checkpoints = _checkpoint_trainer(namespace, tmp_path, pending=True, fail_at=fail_at)
    with pytest.raises(RuntimeError, match=f"injected {fail_at} failure"):
        asyncio.run(
            trainer._apply_dynamic_resize_hard_switch(item=_hard_switch_request(), required_action="expand_train")
        )

    assert len(checkpoints) == 1
    assert checkpoints[0][1].startswith(str(tmp_path / ".dynamic_resize_transient"))
    assert not any((tmp_path / ".dynamic_resize_transient").iterdir())
    assert [name for name, *_ in calls][-1] == fail_at
    assert previous.is_dir()
    assert latest.read_text(encoding="utf-8") == "0"


def test_resumed_fused_run_is_rejected_before_generation(namespace=None):
    namespace = namespace or _load_methods()
    _trainer, observed, error = _run(namespace, fused=True, initial_step=4)
    assert isinstance(error, RuntimeError)
    assert str(error) == "startup-fused first batch requires a fresh training run"
    assert observed.generated == []
    assert observed.trained == []


@pytest.mark.parametrize("fault", ["restore_failed", "wrong_topology"])
def test_fused_restore_failure_keeps_flag_pending(fault):
    namespace = _load_methods()
    trainer, _observed, error = _run(namespace, fused=True, faults=(fault,))
    assert isinstance(error, RuntimeError)
    assert trainer._startup_fused_first_batch_pending is True
