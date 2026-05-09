from verl.experimental.one_step_off_policy.communicator_cache import (
    CommunicatorCacheConfig,
    TopologyCommunicatorRegistryEntry,
    WeightSyncCommunicatorCache,
    build_topology_key,
)


class _DummyGroup:
    pass


def test_topology_key_is_stable_under_dict_order():
    key_a = build_topology_key(
        actor_spec={"mode": "split", "from_pool": "shared_pool", "size": [6, 2], "index": 0},
        rollout_spec={"index": 1, "size": [6, 2], "from_pool": "shared_pool", "mode": "split"},
    )
    key_b = build_topology_key(
        actor_spec={"index": 0, "size": [6, 2], "mode": "split", "from_pool": "shared_pool"},
        rollout_spec={"mode": "split", "from_pool": "shared_pool", "size": [6, 2], "index": 1},
    )

    assert key_a == key_b


def test_cache_hits_only_for_same_worker_group_pair():
    cache = WeightSyncCommunicatorCache(CommunicatorCacheConfig(enable=True))
    topology_key = build_topology_key(actor_spec={"size": [6, 2]}, rollout_spec={"size": [6, 2]})
    actor_wg = _DummyGroup()
    rollout_wg = _DummyGroup()

    cache.put(
        topology_key=topology_key,
        group_name="actor_rollout_topology_1",
        actor_wg=actor_wg,
        rollout_wg=rollout_wg,
        is_prewarmed=True,
    )

    entry = cache.get(topology_key=topology_key, actor_wg=actor_wg, rollout_wg=rollout_wg)
    assert entry is not None
    assert entry.is_prewarmed is True
    assert cache.get(topology_key=topology_key, actor_wg=_DummyGroup(), rollout_wg=rollout_wg) is None
    topology_entry = cache.get_topology(topology_key)
    assert isinstance(topology_entry, TopologyCommunicatorRegistryEntry)
    assert topology_entry.group_name == "actor_rollout_topology_1"


def test_discard_by_group_name_invalidates_entry():
    cache = WeightSyncCommunicatorCache(CommunicatorCacheConfig(enable=True))
    topology_key = build_topology_key(actor_spec={"size": [2, 6]}, rollout_spec={"size": [2, 6]})
    actor_wg = _DummyGroup()
    rollout_wg = _DummyGroup()

    cache.put(topology_key=topology_key, group_name="actor_rollout_topology_2", actor_wg=actor_wg, rollout_wg=rollout_wg)
    cache.discard_by_group_name("actor_rollout_topology_2")

    assert cache.get(topology_key=topology_key, actor_wg=actor_wg, rollout_wg=rollout_wg) is None


def test_discard_by_group_name_releases_reserved_name_for_future_rebuild():
    cache = WeightSyncCommunicatorCache(CommunicatorCacheConfig(enable=True))
    topology_key = build_topology_key(actor_spec={"size": [4, 4]}, rollout_spec={"size": [4, 4]})
    actor_wg = _DummyGroup()
    rollout_wg = _DummyGroup()

    reserved_name = cache.reserve(topology_key)
    cache.put(topology_key=topology_key, group_name=reserved_name, actor_wg=actor_wg, rollout_wg=rollout_wg)

    cache.discard_by_group_name(reserved_name)

    new_reserved_name = cache.reserve(topology_key)

    assert new_reserved_name != reserved_name
    assert cache.get_topology(topology_key) is not None


def test_register_topology_persists_metadata_across_worker_lifecycle_changes():
    cache = WeightSyncCommunicatorCache(CommunicatorCacheConfig(enable=True))
    topology_key = build_topology_key(
        actor_spec={"mode": "split", "from_pool": "shared_pool", "size": [6, 2], "index": 0},
        rollout_spec={"mode": "split", "from_pool": "shared_pool", "size": [6, 2], "index": 1},
    )

    entry = cache.register_topology(
        topology_key,
        actor_spec={"from_pool": "shared_pool", "index": 0, "size": [6, 2], "mode": "split"},
        rollout_spec={"index": 1, "size": [6, 2], "mode": "split", "from_pool": "shared_pool"},
        actor_world_size=6,
        rollout_world_size=2,
        world_size=8,
    )

    assert entry is not None
    assert entry.actor_world_size == 6
    assert entry.rollout_world_size == 2
    assert entry.world_size == 8
    assert entry.actor_spec == {"from_pool": "shared_pool", "index": 0, "mode": "split", "size": [6, 2]}
    assert entry.rollout_spec == {"from_pool": "shared_pool", "index": 1, "mode": "split", "size": [6, 2]}


def test_discard_keeps_topology_registry_but_drops_live_binding():
    cache = WeightSyncCommunicatorCache(CommunicatorCacheConfig(enable=True))
    topology_key = build_topology_key(actor_spec={"size": [2, 6]}, rollout_spec={"size": [2, 6]})
    actor_wg = _DummyGroup()
    rollout_wg = _DummyGroup()
    group_name = cache.reserve(topology_key)

    cache.put(topology_key=topology_key, group_name=group_name, actor_wg=actor_wg, rollout_wg=rollout_wg)
    cache.discard_by_group_name(group_name)

    assert cache.get(topology_key=topology_key, actor_wg=actor_wg, rollout_wg=rollout_wg) is None
    topology_entry = cache.get_topology(topology_key)
    assert topology_entry is not None
    assert topology_entry.group_name is None


def test_put_can_record_non_prewarmed_live_binding():
    cache = WeightSyncCommunicatorCache(CommunicatorCacheConfig(enable=True))
    topology_key = build_topology_key(actor_spec={"size": [4, 4]}, rollout_spec={"size": [4, 4]})
    actor_wg = _DummyGroup()
    rollout_wg = _DummyGroup()

    cache.put(
        topology_key=topology_key,
        group_name="actor_rollout_topology_3",
        actor_wg=actor_wg,
        rollout_wg=rollout_wg,
        is_prewarmed=False,
    )

    entry = cache.get(topology_key=topology_key, actor_wg=actor_wg, rollout_wg=rollout_wg)
    assert entry is not None
    assert entry.is_prewarmed is False


def test_mark_activated_clears_prewarm_flag_for_live_binding():
    cache = WeightSyncCommunicatorCache(CommunicatorCacheConfig(enable=True))
    topology_key = build_topology_key(actor_spec={"size": [6, 2]}, rollout_spec={"size": [6, 2]})
    actor_wg = _DummyGroup()
    rollout_wg = _DummyGroup()

    cache.put(
        topology_key=topology_key,
        group_name="actor_rollout_topology_4",
        actor_wg=actor_wg,
        rollout_wg=rollout_wg,
        is_prewarmed=True,
    )

    updated = cache.mark_activated(topology_key=topology_key, actor_wg=actor_wg, rollout_wg=rollout_wg)

    assert updated is not None
    assert updated.is_prewarmed is False