"""DTensor dispatch-cache invalidation for dynamic-resize process-group teardown.

``DeviceMesh.__eq__`` compares mesh content (ranks, shape, dim names, device
type, thread id) and ignores the underlying process groups, while each
``new_group`` call gets a distinct group name. DTensor's sharding propagator
caches specs that embed the mesh of the first call. After a resize destroys a
subgroup and a later resize recreates an equal-content mesh, a cache hit
returns a spec whose group can no longer be resolved:

    RuntimeError: Could not resolve the process group registered under the name N

Call :func:`clear_dtensor_dispatch_caches` on every rank right after destroying
a resize subgroup. Kept in its own module so the invalidation is importable and
testable without Ray.
"""

from __future__ import annotations


def clear_dtensor_dispatch_caches() -> None:
    """Drop DTensor caches that may reference destroyed process groups."""
    try:
        from torch.distributed.tensor import DTensor

        propagator = DTensor._op_dispatcher.sharding_propagator
    except (ImportError, AttributeError):
        return

    op_cache = getattr(propagator, "propagate_op_sharding", None)
    # torch >= 2.9 wraps the lru_cache in a thread-local LocalLRUCache.
    op_cache = getattr(op_cache, "cache", op_cache)
    if hasattr(op_cache, "cache_clear"):
        op_cache.cache_clear()

    meta_cache = getattr(propagator, "_propagate_tensor_meta", None)
    if hasattr(meta_cache, "cache_clear"):
        meta_cache.cache_clear()

    try:
        from torch.distributed.tensor._redistribute import _gen_transform_infos

        _gen_transform_infos.cache_clear()
    except (ImportError, AttributeError):
        pass
