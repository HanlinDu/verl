"""Conservative communicator cache for one-step-off weight sync groups.

The first round of caching only reused live communicator handles when the
trainer revisited the exact same actor/rollout worker-group pair. This file now
also keeps a topology-scoped metadata registry so future steps can prewarm
communicators for newly spawned workers without pretending that dead-process
handles themselves are reusable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _normalize_spec(spec: dict[str, Any] | None) -> Any:
    if spec is None:
        return None
    return json.loads(json.dumps(spec, sort_keys=True))


def build_topology_key(*, actor_spec: dict[str, Any] | None, rollout_spec: dict[str, Any] | None) -> str:
    """Build a stable topology key from the current actor/rollout pool specs."""

    normalized = {
        "actor_pool": _normalize_spec(actor_spec),
        "rollout_pool": _normalize_spec(rollout_spec),
    }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


@dataclass(slots=True)
class CommunicatorCacheConfig:
    enable: bool = False
    reserve_schedule_topologies: bool = True

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "CommunicatorCacheConfig":
        cfg = cfg or {}
        return cls(
            enable=bool(cfg.get("enable", False)),
            reserve_schedule_topologies=bool(cfg.get("reserve_schedule_topologies", True)),
        )


@dataclass(slots=True)
class WeightSyncGroupCacheEntry:
    topology_key: str
    group_name: str
    actor_group_id: int
    rollout_group_id: int


@dataclass(slots=True)
class TopologyCommunicatorRegistryEntry:
    topology_key: str
    actor_spec: dict[str, Any] | None
    rollout_spec: dict[str, Any] | None
    group_name: str | None = None
    actor_world_size: int | None = None
    rollout_world_size: int | None = None
    world_size: int | None = None


class WeightSyncCommunicatorCache:
    """Cache only communicator metadata that is safe to reuse.

    The first implementation deliberately reuses a cached communicator only when
    the exact actor/rollout worker-group objects are still alive. This avoids
    hiding implicit lifecycle bugs behind aggressive reuse.
    """

    def __init__(self, config: CommunicatorCacheConfig):
        self.config = config
        self._entries: dict[str, WeightSyncGroupCacheEntry] = {}
        self._topology_registry: dict[str, TopologyCommunicatorRegistryEntry] = {}
        self._reserved_group_names: dict[str, str] = {}
        self._sequence = 0

    def register_topology(
        self,
        topology_key: str,
        *,
        actor_spec: dict[str, Any] | None = None,
        rollout_spec: dict[str, Any] | None = None,
        group_name: str | None = None,
        actor_world_size: int | None = None,
        rollout_world_size: int | None = None,
        world_size: int | None = None,
    ) -> TopologyCommunicatorRegistryEntry | None:
        """Store topology-scoped metadata independently from live worker handles.

        This is the minimum registry needed for later prewarm work: the entry is
        stable across worker-group lifecycles, while live handle reuse remains
        guarded by exact worker identity checks in `get()`.
        """
        if not self.config.enable:
            return None

        entry = self._topology_registry.get(topology_key)
        if entry is None:
            entry = TopologyCommunicatorRegistryEntry(
                topology_key=topology_key,
                actor_spec=_normalize_spec(actor_spec),
                rollout_spec=_normalize_spec(rollout_spec),
                group_name=group_name,
                actor_world_size=actor_world_size,
                rollout_world_size=rollout_world_size,
                world_size=world_size,
            )
            self._topology_registry[topology_key] = entry
        else:
            if actor_spec is not None:
                entry.actor_spec = _normalize_spec(actor_spec)
            if rollout_spec is not None:
                entry.rollout_spec = _normalize_spec(rollout_spec)
            if group_name is not None:
                entry.group_name = group_name
            if actor_world_size is not None:
                entry.actor_world_size = int(actor_world_size)
            if rollout_world_size is not None:
                entry.rollout_world_size = int(rollout_world_size)
            if world_size is not None:
                entry.world_size = int(world_size)

        if entry.group_name is not None:
            self._reserved_group_names[topology_key] = entry.group_name
        return entry

    def get_topology(self, topology_key: str) -> TopologyCommunicatorRegistryEntry | None:
        if not self.config.enable:
            return None
        return self._topology_registry.get(topology_key)

    def reserve(self, topology_key: str) -> str | None:
        # Reserve a stable name for a topology even before the communicator is
        # created. This gives the trainer deterministic topology-to-group naming
        # without claiming that the communicator has already been built.
        if not self.config.enable:
            return None
        self.register_topology(topology_key)
        if topology_key not in self._reserved_group_names:
            self._sequence += 1
            self._reserved_group_names[topology_key] = f"actor_rollout_topology_{self._sequence}"
            entry = self._topology_registry.get(topology_key)
            if entry is not None:
                entry.group_name = self._reserved_group_names[topology_key]
        return self._reserved_group_names[topology_key]

    def get(self, *, topology_key: str, actor_wg, rollout_wg) -> WeightSyncGroupCacheEntry | None:
        # A cache hit is valid only when the exact worker-group objects are the
        # same. Re-entering a topology with newly spawned workers is treated as
        # a cache miss on purpose.
        if not self.config.enable:
            return None
        entry = self._entries.get(topology_key)
        if entry is None:
            return None
        if entry.actor_group_id != id(actor_wg) or entry.rollout_group_id != id(rollout_wg):
            return None
        return entry

    def put(
        self,
        *,
        topology_key: str,
        group_name: str,
        actor_wg,
        rollout_wg,
        actor_spec: dict[str, Any] | None = None,
        rollout_spec: dict[str, Any] | None = None,
        actor_world_size: int | None = None,
        rollout_world_size: int | None = None,
        world_size: int | None = None,
    ) -> WeightSyncGroupCacheEntry | None:
        if not self.config.enable:
            return None
        self.register_topology(
            topology_key,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
            group_name=group_name,
            actor_world_size=actor_world_size,
            rollout_world_size=rollout_world_size,
            world_size=world_size,
        )
        self._reserved_group_names[topology_key] = group_name
        entry = WeightSyncGroupCacheEntry(
            topology_key=topology_key,
            group_name=group_name,
            actor_group_id=id(actor_wg),
            rollout_group_id=id(rollout_wg),
        )
        self._entries[topology_key] = entry
        return entry

    def discard_by_group_name(self, group_name: str) -> None:
        stale_keys = [key for key, entry in self._entries.items() if entry.group_name == group_name]
        for key in stale_keys:
            self._entries.pop(key, None)
        stale_reserved_keys = [key for key, reserved_name in self._reserved_group_names.items() if reserved_name == group_name]
        for key in stale_reserved_keys:
            self._reserved_group_names.pop(key, None)
        for entry in self._topology_registry.values():
            if entry.group_name == group_name:
                # Keep topology metadata so a new worker lifecycle can still hit
                # the registry later, but force a fresh reserved group name.
                entry.group_name = None

    def reserved_group_name(self, topology_key: str) -> str | None:
        return self._reserved_group_names.get(topology_key)