import pytest

from verl.utils.resize_route import FAIL, PASS, UNKNOWN, classify_transport, route_verdict


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("P2P/CUMEM", "p2p"),
        ("P2P/IPC", "p2p"),
        ("P2P/direct", "p2p"),
        ("P2P/CUMEM/read", "p2p"),
        ("NET/IB/GDRDMA", "net_rdma"),
        ("NET/IB", "net_rdma"),
        ("NET/RoCE", "net_rdma"),
        ("NET/Socket", "net_socket"),
        ("NET", "net_other"),
        ("NET/CollNetDirect", "net_other"),
        ("SHM/nvls", "shm"),
        ("NVLS", "other"),
    ],
)
def test_classify_transport_separates_rdma_from_host_fallback(token, expected):
    assert classify_transport(token) == expected


def test_single_node_rejects_net_transport():
    """A single-node pool must not silently accept the old FAIL case."""
    verdict, reason = route_verdict(["P2P/CUMEM", "NET/IB/GDRDMA"], cross_node=False)
    assert verdict == FAIL
    assert "NET/IB/GDRDMA" in reason


def test_cross_node_accepts_rdma_transport():
    """The single-node rule would FAIL this; multi-node requires PASS."""
    verdict, reason = route_verdict(["NET/IB/GDRDMA", "NET/IB"], cross_node=True)
    assert verdict == PASS
    assert "RDMA" in reason


def test_cross_node_accepts_mixed_intra_node_p2p_and_rdma():
    verdict, reason = route_verdict(["P2P/CUMEM", "NET/IB/GDRDMA"], cross_node=True)
    assert verdict == PASS
    assert "mixed" in reason


def test_cross_node_rejects_host_socket_fallback():
    verdict, reason = route_verdict(["NET/Socket"], cross_node=True)
    assert verdict == FAIL
    assert "NET/Socket" in reason


def test_cross_node_rejects_shared_memory():
    verdict, reason = route_verdict(["SHM/nvls"], cross_node=True)
    assert verdict == FAIL
    assert "shared memory" in reason


def test_cross_node_rejects_unknown_net_transport():
    verdict, reason = route_verdict(["NET/CollNetDirect"], cross_node=True)
    assert verdict == FAIL


def test_empty_transports_is_unknown_not_pass():
    assert route_verdict([], cross_node=True)[0] == UNKNOWN
    assert route_verdict([], cross_node=False)[0] == UNKNOWN


def test_unknown_token_is_unknown_in_both_topologies():
    assert route_verdict(["NVLS"], cross_node=True)[0] == UNKNOWN
    assert route_verdict(["NVLS"], cross_node=False)[0] == UNKNOWN


def test_single_node_all_p2p_passes():
    verdict, reason = route_verdict(["P2P/CUMEM", "P2P/IPC"], cross_node=False)
    assert verdict == PASS
    assert "P2P" in reason


def test_single_node_rejects_shared_memory():
    verdict, reason = route_verdict(["P2P/CUMEM", "SHM/nvls"], cross_node=False)
    assert verdict == FAIL
    assert "shared memory" in reason
