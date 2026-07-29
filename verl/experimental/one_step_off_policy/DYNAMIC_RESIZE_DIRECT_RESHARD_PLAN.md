# Multi-Node Dynamic Resize with Direct Reshard

## 1. Background

The current experimental dynamic-resize path changes the effective GPU split
between training and rollout at runtime. Its state handoff gathers training
state into host-side storage, then recreates the training group and restores
the state under the new world size.

That design has a central-host assumption: a single host-side staging service
must be able to observe or hold the model-wide state needed by the new
topology. It is therefore not a suitable final reshard path for multi-node
training.

The proposed multi-node path removes the full-model aggregation step:

```text
old training shards
    -> compute intersections between old and new shard ownership
    -> send only the required slices from each old owner to each new owner
    -> assemble each destination shard directly
```

The design is inspired by sharded delta weight sync: ownership, coordinates,
and routing are determined before transport, and the data plane never needs to
materialize the whole model on a coordinator. Unlike delta weight sync, this
path transfers complete destination training shards rather than only
step-to-step changes, and it must eventually cover optimizer state in addition
to model parameters.

## 2. Goals and Scope

### Goals

- Change the logical training/rollout GPU ratio across multiple nodes.
- Avoid gathering complete model or optimizer state on one host.
- Support both training scale-up and scale-down.
- Prefer persistent physical workers that change logical roles.
- Transfer every destination shard directly from its old shard owners.
- Bound staging memory independently of total model size.
- Preserve training state exactly in full-state mode.
- Fail before committing a partially completed topology change.

### Initial scope

- Fixed physical worker/GPU pool.
- Resize at a full optimizer-step boundary.
- FSDP2 with a one-dimensional `Shard(0)` mesh.
- No simultaneous TP, PP, or EP topology change.
- Fixed model structure and dtype.
- No elastic node join/removal.
- Prefer TransferQueue, with a stable NCCL group as fallback.

### Deferred scope

- Overlap with forward/backward/optimizer update or rollout generation.
- Arbitrary FSDP1 FlatParameter layouts.
- Fine-grained NCCL failure recovery.
- Simultaneous rollout TP resize.

## 3. Training-State Contract

The implementation must explicitly state which training state is preserved:

| State | Required action |
|---|---|
| Model parameters | Reshard |
| Adam moments and other optimizer tensors | Reshard for exact continuation |
| FP32 master parameters, if present | Reshard for exact continuation |
| Gradients | Do not migrate; resize at a zero-gradient safe point |
| Scheduler/global step | Replicate or broadcast |
| AMP loss scaler | Replicate or broadcast |
| RNG state | Preserve according to the chosen reproducibility contract |

Two development modes are useful:

- `weights_only`: preserve model weights and rebuild optimizer state. This
  validates topology and transport but is not exact training continuation.
- `full_training_state`: preserve model, optimizer, master weights, and
  required replicated state. This is the production target.

## 4. Logical Reshard Algorithm

### Stable identities

The plan must distinguish:

```text
physical_worker_id
node_id
local_gpu_id
old_role / new_role
old_train_rank / new_train_rank
```

`physical_worker_id` remains stable across role transitions.

### State specification

Every migrated tensor needs a world-size-independent description:

```python
StateSpec(
    key,
    global_shape,
    dtype,
    global_numel,
    placement,
    alignment,
    padding,
)
```

The key must remain stable across runtime reconstruction and must not depend
only on Python object identity or the old logical rank.

### Range intersection

For a flat `Shard(0)` state:

```text
OldShard(source) = [old_begin, old_end)
NewShard(destination) = [new_begin, new_end)

TransferSlice(source, destination)
    = OldShard(source) intersection NewShard(destination)
```

Each non-empty intersection becomes:

```python
TransferTask(
    task_id,
    state_key,
    source_worker_id,
    destination_worker_id,
    source_offset,
    destination_offset,
    numel,
    dtype,
)
```

If source and destination are the same persistent worker, use a local
copy/view rather than network transfer.

### Plan invariants

For every destination state:

- destination ranges completely cover its local shard;
- ranges do not overlap;
- source and destination ranges are in bounds;
- both ranges describe the same global coordinates;
- dtype and logical state key match;
- padding is explicit.

The plan must be deterministic from the old topology, new topology, and state
manifest.

## 5. Reshard Manager

Introduce a control-plane component such as `DynamicReshardManager`. It
computes and coordinates transfers but never carries model payloads.

```python
ReshardRequest(
    resize_id,
    global_step,
    old_topology,
    new_topology,
    state_manifest,
    transport_kind,
)

ReshardPlan(
    resize_id,
    plan_version,
    state_specs,
    local_moves,
    transfer_rounds,
    expected_receives,
    plan_checksum,
)
```

Responsibilities:

- collect old-shard metadata;
- compute new ownership and range intersections;
- validate coverage, overlap, bounds, and key consistency;
- select local, intra-node, and inter-node routes;
- schedule bounded transfer rounds;
- track completion and verification;
- keep old sources alive until all dependent destinations acknowledge;
- atomically publish the new topology only after successful verification.

The manager must not gather, concatenate, or retain full model tensors.

## 6. Resize Transaction

Use an explicit transaction:

```text
RUNNING
  -> QUIESCE_REQUESTED
  -> QUIESCED
  -> PLAN_VALIDATED
  -> DESTINATIONS_PREPARED
  -> TRANSFERRING
  -> VERIFYING
  -> COMMITTING
  -> RUNNING_WITH_NEW_TOPOLOGY
```

### Quiesce

- Finish the current optimizer step.
- Ensure no model/optimizer collective remains active.
- Enter a known zero-gradient state.
- Stop accepting new rollout requests.
- Drain, save, or abort in-flight generation.
- Record `global_step`, `resize_id`, and old topology version.

### Plan

- Collect old training-state manifests.
- Select new train workers and logical ranks.
- Generate and validate transfer tasks.
- Split large states into bounded chunks.
- Schedule non-conflicting transfer rounds.

### Prepare destinations

Workers changing from rollout to training must:

- release KV cache;
- release rollout weights if required;
- initialize an empty/meta training runtime where supported;
- allocate destination parameter and optimizer buffers;
- prepare transport endpoints.

Old training sources retain their state until commit.

### Transfer and verify

Sources send only planned views. Destinations write into preallocated buffers
using destination offsets, independent of arrival order. Verification checks
task identity, resize version, byte counts, dtype, bounds, complete coverage,
overlap, and checksums.

### Commit

Only after every destination verifies successfully:

- activate new training process groups;
- bind received shards to FSDP and optimizer runtime;
- atomically publish the new topology version;
- release state on workers leaving training;
- transition those workers to rollout;
- synchronize rollout weights and resume generation.

## 7. Worker Lifecycle Requirements

The desirable long-term model is:

```text
Persistent physical worker
  + TrainRoleController
  + RolloutRoleController
  + ReshardTransportEndpoint
```

Code inspection must determine whether the current runtime supports:

- retaining Ray actors while rebuilding logical worker groups;
- destroying and recreating process groups safely;
- fully releasing SGLang KV cache and weights;
- constructing FSDP under a new device mesh in an existing process;
- populating local DTensor shards without full all-gather;
- rebuilding optimizer parameter references;
- keeping a transport endpoint alive across role transitions.

The required ordering is:

```text
freeze old roles
  -> transfer and verify
  -> global commit
  -> release old state
  -> switch roles
```

## 8. Transport

Keep the reshard planner independent of the data plane:

```python
class ReshardTransport:
    def prepare(self, plan): ...
    def send(self, task, tensor_view): ...
    def receive_into(self, task, destination_view): ...
    def flush(self): ...
    def close(self): ...
```

### TransferQueue candidate

TransferQueue is preferred only if it supports:

- cross-node GPU payloads;
- explicit source/destination routing;
- variable sizes and bounded backpressure;
- stable task identity and completion acknowledgement;
- endpoint reuse across role changes;
- safe tensor-lifetime and CUDA-stream semantics;
- failure reporting that prevents premature commit.

The code inspection must establish whether the current TransferQueue is a
general tensor transport or primarily a rollout-data/replay-buffer system.

### Stable NCCL fallback

A fallback is a long-lived NCCL group spanning all persistent physical
workers. Logical roles change while transport ranks remain stable. Candidate
operations are `batch_isend_irecv`, `isend`/`irecv`, or
`all_to_all_single` with explicit split sizes.

Collective ordering must remain identical. Idle ranks must participate with a
defined zero-length/no-op path whenever the primitive requires all ranks.

### Local and staged fallback

- Same worker: local storage/copy.
- Same node: CUDA IPC/NVLink.
- Cross node: NCCL/RDMA.
- Last resort: source-local host staging to destination-local host staging.

The last path is still multi-node capable because no central host assembles
the complete model.

## 9. Transfer Scheduling

Initially schedule rounds so every source and destination has at most one
active peer in a round. This is a bipartite edge-scheduling problem and avoids
an unconstrained all-to-all burst.

Later optimizations can add:

- multiple streams per worker;
- small-state coalescing;
- large-embedding chunking;
- local/intra-node priority;
- NIC-aware routing;
- parameter/optimizer pipelining;
- double-buffered receive and assembly.

Chunk sizes must bound both source and destination staging memory.

## 10. FSDP Integration

### FSDP2 first

FSDP2 DTensor metadata provides explicit global shape and placement. The first
implementation should support:

```text
one-dimensional device mesh
Shard(0)
fixed parameter order and model structure
```

Train-to-train reshard should use training-native state keys and coordinates,
not Hugging Face export coordinates.

The code inspection must answer:

- Can a new module be built on `meta` and populated from local shards?
- Can a DTensor local shard be replaced directly?
- Does state-dict load trigger full gather?
- When must the new process group/device mesh exist?
- Does replacing parameters invalidate optimizer references?
- Can distributed checkpoint load directly across changed world sizes?

### FSDP1 later

FSDP1 FlatParameter boundaries can cross original parameters. Its adapter must
plan against a stable global FlatParameter layout, not per-HF-parameter ranges.

## 11. Optimizer State

Exact continuation needs placement specifications for:

- parameters;
- `exp_avg`;
- `exp_avg_sq`;
- FP32 master parameters;
- per-parameter or global step counters.

If optimizer tensors share parameter placement, routing geometry can be
reused with different keys. Otherwise, they need independent plans. State
keys must not rely on old Python parameter IDs.

## 12. Failure Semantics

The minimum safe policy is fail-stop:

```text
any task failure
  -> abort resize
  -> do not publish new topology
  -> do not release old training state
```

If the old runtime remains intact it may resume; otherwise restart from a
checkpoint. Fine-grained retries require transport-level idempotency and
reliable acknowledgement and should not be assumed.

Record the resize/plan versions, old/new topology, plan checksum, completed
task bitmap, every worker's last state, and the failed task endpoints.

## 13. Verification

### Planner tests

- `2 -> 4`, `4 -> 2`, `3 -> 5`;
- uneven lengths, padding, empty shards;
- tensors smaller than world size;
- local moves and rank permutations;
- multiple tensors/dtypes;
- gap, overlap, and bounds failures.

### GPU/two-node tests

- chunking and backpressure;
- P2P ordering and CUDA stream lifetime;
- acknowledgement before source release;
- checksum failure;
- a topology that changes from node-local training to cross-node training;
- confirmation that payload does not pass through the driver/full-model host.

### Numerical/stress tests

- initialize values from global coordinates and verify every destination;
- compare next-step loss/update with a no-resize baseline;
- verify optimizer, scheduler, scaler, global step, and RNG contract;
- repeatedly resize through `2 -> 6 -> 3 -> 5 -> 2`;
- test CPU offload, large embeddings, in-flight rollout, and failures.

## 14. Metrics

```text
resize/quiesce_seconds
resize/plan_seconds
resize/prepare_seconds
resize/transfer_seconds
resize/verify_seconds
resize/commit_seconds

resize/model_bytes
resize/optimizer_bytes
resize/local_copy_bytes
resize/intra_node_bytes
resize/inter_node_bytes
resize/peak_staging_bytes

resize/task_count
resize/round_count
resize/max_source_fanout
resize/max_destination_fanin
resize/effective_bandwidth_gbps
```

Acceptance requires lower latency and removal of the single-host full-model
memory peak.

## 15. Delivery Stages

1. Confirm current code and lifecycle constraints.
2. Implement and unit-test deterministic `ReshardPlan`.
3. Prototype transport with synthetic tensors on two nodes.
4. Integrate weights-only FSDP2 resize at a safe boundary.
5. Add optimizer and complete state migration.
6. Enable persistent-worker train/rollout role transitions.
7. Optimize locality, scheduling, bucketing, and overlap.

Repository guidance requires the hard-switch path to be stable before pinned
staging, communicator caching, or direct-reshard optimization. Direct reshard
should therefore be a separate disabled-by-default backend after the
hard-switch completion criteria are met.

## 16. Code-Inspection Checklist

1. Dynamic-resize entry point and safe-point timing.
2. Whether worker actors are destroyed, reused, or role-switched.
3. Current host gather/reshard data structures.
4. Whether parameters, optimizer state, and master weights migrate.
5. FSDP1/FSDP2 local-shard export and restore APIs.
6. New training-worker destination-buffer construction.
7. Training, rollout, and checkpoint process-group lifetimes.
8. TransferQueue topology, payload, ordering, completion, and failure semantics.
9. Whether an existing NCCL group can span future logical roles.
10. Rollout KV-cache/weight release and runtime reconstruction.
11. Ray worker-group support for logical membership changes.
12. Components caching rank, world size, or device mesh.
13. Checkpoint, scheduler, and global-step interaction.
14. Reusable distributed-test infrastructure.

## 17. Code-Inspection Findings

The following results are confirmed against branch `dynamic-resize-migrate`
at commit `f171382b`. They describe the implementation as it exists in this
checkout; statements under "Recommended" are design proposals rather than
current behavior.

### 17.1 Executive conclusion

The existing implementation provides a useful hard-switch control plane and
several staging prototypes, but it does **not** yet provide distributed direct
reshard or persistent worker role conversion.

The current resize sequence is:

```text
optimizer step completes
  -> save actor checkpoint
  -> optionally page each old rank's raw optimizer state
  -> kill rollout and training actors
  -> create new actor groups under the target world size
  -> load exact-size shards, or broadcast a rank-0 full model
  -> synchronize actor weights to rollout
```

Consequently:

- model resize still depends on a rank-0 full-model artifact when the target
  world size has not previously been checkpointed;
- optimizer state is not restored when world size changes;
- staged optimizer pages are not consumed by the production restore path;
- old sources and new destinations never coexist, so source-to-destination
  direct transfer cannot be inserted without a worker-lifecycle change;
- TransferQueue can be used as a distributed staging/control abstraction, but
  its current backend is not a `receive_into` GPU-direct reshard transport;
- the current NCCL checkpoint engine is a rank-0 broadcast topology, not an
  all-training-ranks reshard topology.

### 17.2 Resize entry point and safe point

Confirmed in
`verl/experimental/one_step_off_policy/ray_trainer.py`:

- hard switch is enabled only when dynamic resize, shared resource pools, and
  `hard_switch.enable` are all enabled;
- a schedule entry is selected by exact `global_steps` equality;
- generation of the next rollout batch is deferred when that step has a
  resize scheduled;
- `_maybe_dynamic_resize()` runs after the actor update and the step's
  validation/checkpoint/profiling work;
- the training engine zeros gradients around `train_batch`, so this boundary
  is normally after an optimizer step and outside backward.

This is a reasonable initial quiescence point, but it is not yet an explicit
distributed transaction barrier. The rollout abort/sleep call is best-effort:
exceptions are logged and actor destruction continues.

Recommended:

1. preserve this boundary for the first direct-reshard implementation;
2. add a train-side `quiesce(resize_id)` RPC and require acknowledgements from
   every source and destination;
3. assert no active forward/backward, optimizer collective, or in-flight
   rollout before accepting the plan;
4. publish a topology version only after destination verification.

### 17.3 Current topology and Ray worker lifecycle

`verl/experimental/separation/utils.py` constructs one fixed shared placement
pool by summing the configured train and rollout stores per node. Every resize
schedule is required to keep the combined GPU count constant.

`SubRayResourcePool` in `verl/single_controller/ray/base.py` selects contiguous
placement-group bundles. This gives a deterministic physical allocation, but
`RayWorkerGroup._init_with_subresource_pool()` calls `_create_worker()` for
each target rank. It does not rebind existing Ray actor handles.

`_destroy_dynamic_resize_runtime()` in the trainer kills:

- rollout replica workers and server handles;
- the global load balancer;
- every current training worker.

`_rebuild_dynamic_resize_runtime()` then constructs new worker groups, models,
rollout manager, and checkpoint manager. Therefore current logical ranks,
actor identities, Python objects, CUDA contexts, and default process groups
are all ephemeral.

`TrainingWorker` captures `RANK` and `WORLD_SIZE` from the actor environment
and initializes the default process group in its constructor. Its `reset()`
only calls engine initialization; it does not destroy and rebuild the
communicator for a different world size. The helper
`initialize_global_process_group_ray()` also returns early when the default
group is already initialized.

`WorkerCommInitPlan` records desired rank/world-size/mesh metadata, but search
shows that it is currently consumed only by CPU unit tests. It is not an
executable role-transition mechanism.

**Conclusion:** persistent train/rollout role conversion is a prerequisite,
not a capability already hidden behind the existing pool abstraction.

### 17.4 Current model and optimizer handoff

`FSDPCheckpointManager.save_checkpoint()` writes one model and optimizer file
per old world-size/rank. For dynamic resize it additionally gathers an FSDP2
full state dict on rank 0 and writes `dynamic_resize_full_model.pt`.

On load:

- if a shard file for the exact target world size and target rank exists, it
  is loaded;
- otherwise rank 0 loads the dynamic full-model file and
  `fsdp2_load_full_state_dict()` broadcasts/reshares it;
- the optimizer loader only accepts exact target-world-size optimizer files;
- when those files are absent, it records `skipped_missing_target_shards` and
  leaves a freshly initialized optimizer in place.

`stage_optimizer_for_dynamic_resize()` serializes each old rank's raw
`optimizer.state_dict()` into pages named by old rank. This is useful staging
work, but:

- the keys and partitioning still describe the old runtime;
- `load_paged_optimizer_state_dict()` has no production caller;
- no mapping from stable parameter/state keys to new destination shards is
  generated;
- the hard-switch path marks the resize completed even when optimizer restore
  is skipped.

This means current cross-world-size semantics are effectively
`weights_only`, not exact training continuation.

Additional state issues:

- scheduler state is restored as replicated state;
- fallback extra-state loading can use an old rank-0 file on every new rank,
  so the per-rank RNG contract is currently undefined;
- the actor checkpoint gets the dynamic full-model flag, while the critic
  checkpoint does not. If a separately sharded critic is enabled and its
  world size changes, the critic has no equivalent fallback;
- all paths are local filesystem paths. A multi-node run also requires either
  a shared filesystem or explicit movement of manifests/artifacts.

### 17.5 FSDP integration constraints

Confirmed in `verl/workers/engine/fsdp/transformer_impl.py` and
`verl/utils/fsdp_utils.py`:

- FSDP2 builds its device mesh from the current default distributed world;
- the module can be initialized with meta-aware construction;
- current initialization captures a full state before FSDP application and
  uses `fsdp2_load_full_state_dict()` to distribute it from rank 0;
- optimizer construction happens after the sharded module is created;
- `fsdp2_sharded_save_to_cpu()` exposes each DTensor's local tensor and
  DTensor spec;
- `fsdp2_sharded_load_from_cpu()` requires the current mesh to equal the saved
  mesh, so it cannot restore those local shards under a changed world size.

`DetachActorWorker` wraps these local save/load helpers, but inherits the same
equal-mesh restriction and does not cover optimizer state.

The direct-reshard integration therefore needs a new FSDP2 adapter that:

1. derives stable global coordinates and keys from the old local DTensors;
2. allocates the target DTensor layout under the new mesh;
3. receives directly into, or copies into, the target local tensors;
4. validates complete coverage before exposing the module;
5. creates or rebinds the optimizer only after parameter object identity is
   final;
6. builds optimizer state from stable parameter keys rather than old
   optimizer parameter IDs.

The first implementation should explicitly reject FSDP1, TP/PP/EP changes,
and non-`Shard(0)` placements.

### 17.6 Pinned CPU staging

`PinnedCPUStagingService` is a named detached Ray actor with a process-local
in-memory session dictionary. It deep-copies/pins objects in that actor.
Production dynamic-resize control does not currently call
`get_or_create_pinned_cpu_staging_service()`; its callers are tests.

Even when enabled, a single named service is a central staging owner rather
than a distributed direct-reshard data plane. A multi-node fallback should
instead instantiate source-local and destination-local services and retain
the same range-based plan; no service should assemble the whole model.

### 17.7 TransferQueue assessment

The repository uses TransferQueue as asynchronous key/value data storage for
rollout and replay data. Its bridge translates batch metadata and performs
`put_data`/`get_data`; the trainer initializes it globally from configuration.

In the installed `transfer_queue` package in this environment:

- `SimpleStorage` stores payloads in CPU-backed storage actors;
- the Mooncake client converts CUDA tensors to CPU before putting them and
  contains a TODO for GPU-direct RDMA;
- the API returns stored objects rather than receiving into a preallocated
  destination GPU view;
- a successful put confirms storage completion, not destination application
  and verification;
- routing is key/storage based, not an explicit stable
  source-worker-to-destination-worker task.

There is also a version skew: the installed package reports `0.1.6`, while
the repository's mock/import guidance requests `0.1.8`.

**Conclusion:** TransferQueue is suitable for:

- plan and manifest publication;
- task metadata;
- distributed host-staging fallback;
- backpressure experiments.

It is not currently sufficient by itself for the desired direct GPU reshard.
Using it for the first multi-node correctness prototype is possible, but the
implementation should label that backend `distributed_staging`, not
`direct_gpu`.

Required TransferQueue proof before promoting it:

1. verify behavior against the repository-supported package version;
2. measure whether Mooncake can avoid device-to-host copies in that version;
3. add explicit task ACK after destination copy and checksum;
4. add idempotent task/version keys and bounded object lifetime;
5. demonstrate that no storage actor holds a full-model aggregate.

### 17.8 NCCL and checkpoint-engine assessment

`NCCLCheckpointEngine` creates a Ray collective topology consisting of:

- actor/training rank 0 as source rank 0;
- every other trainer excluded with rank `-1`;
- rollout replicas as receiver ranks.

The payload operation is broadcast. The hard-switch path also gives rebuilt
actors a fresh unique checkpoint group name. This is appropriate for rollout
weight synchronization but cannot be reused unchanged for train-shard
resharding.

A direct NCCL backend needs all old owners and new owners in a separate,
stable transport group and needs P2P or split all-to-all semantics. Reusing the
training default process group is unsafe because its membership and world size
are precisely what resize changes.

Recommended transport hierarchy:

```text
same persistent worker   -> local copy
same host                -> initially NCCL P2P; optimize to CUDA IPC later
different host           -> NCCL P2P or all-to-all on a stable transport group
fallback                 -> distributed TransferQueue/host staging
```

The stable transport group should be independent of the FSDP default process
group. The current code has no such group across all physical workers.

### 17.9 Rollout-runtime role conversion

The rollout manager supports abort, sleep, KV-cache release, resume, replica
addition, and replica removal. Those operations are useful quiescence
primitives, but they do not implement conversion into a training worker.

For the inspected standalone SGLang path, `sleep()` releases KV cache but
does not guarantee release of model weights. The current hard switch ultimately
kills the actors to reclaim all state.

Before a rollout GPU can become a training destination in-place, the runtime
needs an explicit contract:

```text
abort requests
  -> drain/confirm no active kernels
  -> release KV cache
  -> release model weights and communicators
  -> report available memory
  -> construct target training role
```

The reverse train-to-rollout transition needs symmetrical teardown and must
retain source training shards until all destination ACKs arrive.

### 17.10 Failure and transaction gaps

The current hard-switch path has no rollback transaction. Once old actors are
killed, failure during reconstruction or load cannot resume the old runtime.
The control path catches some quiescence errors and continues. Completion
manifests also do not make optimizer migration a commit prerequisite.

Direct reshard must change the ordering:

```text
old state remains live
  -> new destinations prepared
  -> every range transferred
  -> destination coverage/checksum verified
  -> atomic topology commit
  -> old state released
```

Until old and new logical roles can overlap, the system can only use a
checkpoint/staging handoff and cannot provide this failure property.

### 17.11 Reuse matrix

| Component | Reuse | Required change |
|---|---|---|
| Resize schedule and target-size validation | Yes | Add topology/version IDs |
| Shared placement pool | Partial | Add stable physical-worker identities |
| `SubRayResourcePool` | Partial | Do not recreate actors for role changes |
| Hard-switch safe-point selection | Yes | Add explicit distributed quiescence |
| Handoff manifests and page format | Partial | Use stable tensor/state keys |
| `WorkerCommInitPlan` | Partial | Add an execution/teardown path |
| FSDP2 local DTensor extraction | Yes | Add changed-mesh destination restore |
| FSDP full-state fallback | Baseline only | Remove from direct backend |
| Pinned staging service | Fallback only | Make it per-node/per-worker |
| TransferQueue | Control/staging | Add destination ACK and GPU-direct proof |
| NCCL checkpoint engine | Pattern only | New all-owner reshard topology/backend |
| Rollout abort/KV release | Partial | Add full weight/runtime teardown |

### 17.12 Recommended implementation sequence

#### Phase 0: make hard switch semantically explicit

- Introduce `weights_only` and `full_training_state` policies.
- Do not report exact success when optimizer state was skipped.
- Define critic behavior and RNG semantics.
- Validate shared-filesystem assumptions at startup.
- Add a transaction ID and failure-state manifest.

This establishes a trustworthy baseline before optimizing data movement.

#### Phase 1: deterministic pure planner

Add CPU-only modules with no Ray/NCCL dependency:

```text
topology.py
state_spec.py
reshard_plan.py
```

Inputs are stable physical-worker metadata, old/new logical ranks, and global
tensor specifications. Outputs are local moves, transfer tasks, rounds,
coverage bitmaps, and a checksum. Exhaustively test uneven and empty ranges.

#### Phase 2: persistent-worker lifecycle spike

Choose and validate one lifecycle architecture before integrating real model
state:

1. **Persistent physical actor:** all shared-pool workers remain alive; role
   runtimes and FSDP groups are replaceable subcomponents.
2. **Persistent carrier plus role subprocess:** the carrier owns the stable
   transport endpoint and spawns/destroys train or rollout subprocesses.
3. **Transient overlap:** old and new actors temporarily colocate on bundles.
   This is the smallest code change but has the highest GPU-memory risk.

The preferred long-term choice is the persistent physical actor, but current
use of the default process group makes it a material refactor. A role
subprocess/carrier prototype may isolate that risk more cleanly.

The spike is successful only if a worker can complete:

```text
rollout -> empty destination -> train(new rank/world)
train(old rank/world) -> retained source -> rollout
```

without losing the independent transport endpoint.

#### Phase 3: synthetic two-node transport

Implement `ReshardTransport` with two backends:

- NCCL P2P as the direct backend;
- TransferQueue distributed CPU staging as the correctness fallback.

Use coordinate-initialized tensors, bounded chunks, explicit destination ACK,
checksums, timeout, and fail-stop behavior. Do not involve FSDP yet.

#### Phase 4: FSDP2 weights-only integration

- export old local parameter DTensors as stable global ranges;
- construct the target module/mesh;
- receive target local shards;
- verify and activate the new FSDP runtime;
- compare the next forward result with a no-resize baseline.

Keep the rank-0 full-state path available as a separately selected fallback,
not as an implicit part of the direct backend.

#### Phase 5: exact optimizer continuation

- define stable keys for every parameter and optimizer state;
- reuse parameter geometry for Adam moments where layouts match;
- handle scalar replicated counters separately;
- create/rebind optimizer references after target parameters are final;
- compare the next optimizer step bitwise or within a documented tolerance.

#### Phase 6: production hardening

- transaction log and topology-version fencing;
- repeated-resize stress tests;
- two-node failure injection;
- memory and bandwidth metrics;
- optional locality-aware scheduling and overlap.

### 17.13 Questions that still require environment validation

Code inspection alone cannot confirm:

1. whether the deployment's Ray placement and GPU memory permit transient old
   and new actor overlap;
2. whether the production filesystem is shared across nodes;
3. which TransferQueue version/backend will be deployed and whether it has a
   newer GPU-direct path;
4. whether InfiniBand/RoCE/NCCL P2P is available between every intended pair;
5. whether rollout TP is fixed during train/rollout-ratio changes;
6. whether exact optimizer/RNG continuation is a product requirement or
   weights-only continuation is acceptable for an initial milestone;
7. whether critic and reference-policy roles participate in resize.

These should become startup capability checks and explicit configuration, not
implicit assumptions.

### 17.14 Existing test baseline and missing coverage

The following CPU tests pass in this checkout:

```text
test_dynamic_resize_control_on_cpu.py
test_dynamic_resize_shared_pool_on_cpu.py
test_staging_backend_on_cpu.py
test_worker_init_plan_on_cpu.py
test_training_worker_resize_snapshot_on_cpu.py

34 passed
```

They validate control decisions, shared-pool calculations, paging, metadata
plans, and raw optimizer snapshot round trips. They do not validate:

- cross-world-size optimizer restore;
- persistent actor role conversion;
- changed-mesh FSDP local-shard restore;
- multi-node direct payload routing;
- destination ACK/transaction commit;
- a full dynamic-resize hard switch with critic enabled;
- numerical equivalence of the next training step.

Those gaps map directly to Phases 1 through 6 above.

## 18. Source Evidence Index

The principal implementation anchors in this checkout are:

| Topic | Source anchor |
|---|---|
| Resize enable/defer | `ray_trainer.py:290`, `ray_trainer.py:306` |
| Optimizer staging control | `ray_trainer.py:645-711` |
| Destroy and rebuild | `ray_trainer.py:868`, `ray_trainer.py:925` |
| Hard-switch transaction | `ray_trainer.py:940` |
| Resize dispatch | `ray_trainer.py:997` |
| Shared-pool split | `single_controller/ray/base.py:272` |
| New Ray actor construction | `single_controller/ray/base.py:583`, `:623` |
| Default PG initialization | `utils/distributed.py:82` |
| Worker optimizer snapshot | `workers/engine_workers.py:205` |
| Full-model fallback load | `utils/checkpoint/fsdp_checkpoint_manager.py:175` |
| Dynamic full-model save | `utils/checkpoint/fsdp_checkpoint_manager.py:312`, `:437` |
| Rank-0 full-state broadcast | `utils/fsdp_utils.py:476` |
| Same-mesh local save/load | `utils/fsdp_utils.py:1093`, `:1135` |
| Detach actor wrapper | `experimental/separation/engine_workers.py:36` |
| Pinned central actor | `pinned_cpu_staging.py:78`, `:239` |
| Paged optimizer loader | `staging_backend.py:443` |
| Metadata-only comm plan | `worker_init_plan.py:11` |
| NCCL rank-0 topology | `checkpoint_engine/nccl_checkpoint_engine.py:103`, `:160-171` |
| SGLang sleep/KV release | `async_sglang_server.py:465`, `:492` |
