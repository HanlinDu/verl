"""Topology-aware NCCL route classification for multi-node resize acceptance.

The single-node acceptance treats any ``NET`` transport as failure, because a
correct single-node FSDP2 shard transfer must stay on P2P/SHM links. Multi-node
pools make that rule wrong: cross-node traffic legitimately uses ``NET``, and the
distinction that matters becomes *which* NET path was taken.

Pure stdlib: no torch, no NCCL, no GPU. Classification is exercised against
literal NCCL channel tokens so the rules stay testable without a run.
"""

from __future__ import annotations

import re

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

# NCCL prints transports such as:
#   Channel 00/0 : 1[3] -> 2[4] via P2P/CUMEM
#   Channel 01/0 : 0[0] -> 4[0] via NET/IB/GDRDMA
#   Channel 01/0 : 0[0] -> 4[0] via NET/Socket
CHANNEL_RE = re.compile(r"Channel\s+\S+\s*:.*\svia\s+(\S+)")

# Intra-node expectations: P2P paths are GPU-direct; SHM is a host round trip.
# NCCL prints these tokens with a fixed case, but the exact spelling of a given
# fabric varies by build, so matching is case-insensitive and an unmatched NET/*
# token is reported instead of silently accepted.
P2P_RE = re.compile(r"P2P/(?:CUMEM|IPC|direct)(?:/read)?(?:/CE)?", re.IGNORECASE)
# Cross-node expectations: RDMA-capable fabrics carry the payload without a
# host bounce, while NET/Socket falls back to a host socket.
NET_RDMA_RE = re.compile(r"NET/(?:IB|ROCE|RDMA)(?:/GDRDMA)?", re.IGNORECASE)
NET_SOCKET_RE = re.compile(r"NET/Socket", re.IGNORECASE)


def classify_transport(token: str) -> str:
    """Classify a NCCL channel transport token."""
    if token.startswith("SHM"):
        return "shm"
    if P2P_RE.fullmatch(token):
        return "p2p"
    if NET_RDMA_RE.fullmatch(token):
        return "net_rdma"
    if NET_SOCKET_RE.fullmatch(token):
        return "net_socket"
    if token.startswith("NET"):
        return "net_other"
    return "other"


def route_verdict(transports: list[str], *, cross_node: bool) -> tuple[str, str]:
    """Return ``(verdict, reason)`` for one transfer's observed channel transports.

    ``cross_node`` states whether the pool spans hosts, which decides whether
    ``NET`` is a legitimate path or a fallback that must be reported.
    """
    kinds = [classify_transport(token) for token in transports]
    if not kinds:
        return UNKNOWN, "no NCCL Channel transport lines found"
    if "other" in kinds:
        return UNKNOWN, "unrecognized channel transports present; cannot confirm route"
    bad_shm = [token for token, kind in zip(transports, kinds, strict=True) if kind == "shm"]
    # Cross-node: RDMA is the goal; only host-bounce paths are failures.
    cross_node_bad_net = [
        token for token, kind in zip(transports, kinds, strict=True) if kind in ("net_socket", "net_other")
    ]
    # Single-node: any NET means the payload left the host at all.
    single_node_bad_net = [token for token, kind in zip(transports, kinds, strict=True) if kind.startswith("net")]

    if cross_node:
        # Cross-node P2P is impossible over a fabric; RDMA is the goal and a
        # socket is a correct but slower fallback that must not be called RDMA.
        if cross_node_bad_net:
            return FAIL, f"cross-node transfer fell back to host transport: {sorted(set(cross_node_bad_net))}"
        if bad_shm:
            return FAIL, f"cross-node transfer used shared memory: {sorted(set(bad_shm))}"
        rdma = [token for token, kind in zip(transports, kinds, strict=True) if kind == "net_rdma"]
        p2p = [token for token, kind in zip(transports, kinds, strict=True) if kind == "p2p"]
        if rdma and p2p:
            return PASS, "mixed intra-node P2P and cross-node RDMA transport"
        if rdma:
            return PASS, "cross-node transfer used RDMA transport"
        # A cross-node pool that only ever shows intra-node links gives no
        # evidence that payload crossed hosts; do not call that a pass.
        return UNKNOWN, "cross-node pool but only intra-node P2P observed; no inter-node payload evidence"
    if bad_shm:
        return FAIL, f"single-node transfer used shared memory: {sorted(set(bad_shm))}"
    if single_node_bad_net:
        return FAIL, f"single-node transfer left the host: {sorted(set(single_node_bad_net))}"
    return PASS, "single-node transfer stayed on GPU-direct P2P"
