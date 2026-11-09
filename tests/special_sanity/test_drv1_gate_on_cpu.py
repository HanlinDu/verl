# Copyright 2026 Individual Contributor: HanlinDu
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
"""Regression tests for rejecting stale, incomplete, or altered acceptance evidence."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("drv1_verify", Path(__file__).parents[2] / "scripts/drv1/verify.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


@pytest.fixture
def evidence(tmp_path):
    """Build actual temporary logs/JUnit, without representing them as GPU execution."""
    candidate = {
        "repository": "HanlinDu/verl",
        "pr": 99,
        "merge_sha": "a" * 40,
        "tree": "b" * 40,
        "base_sha": "c" * 40,
        "head_sha": "d" * 40,
    }
    (tmp_path / "pytest.log").write_text("one test passed\n")
    (tmp_path / "junit.xml").write_text('<testsuites><testsuite tests="1" failures="0" errors="0"/></testsuites>')
    cpu = {
        "kind": "cpu",
        "status": "passed",
        "clean": True,
        "sha": candidate["merge_sha"],
        "tree": candidate["tree"],
        "checks": [{"exit_code": 0, "log": "pytest.log", "sha256": verify.digest(tmp_path / "pytest.log")}],
        "junit_sha256": verify.digest(tmp_path / "junit.xml"),
    }
    return candidate, cpu, tmp_path


def test_cpu_artifacts_must_match_their_recorded_hash(evidence):
    candidate, report, directory = evidence
    assert not verify.report_errors(report, candidate, "cpu", directory)
    (directory / "pytest.log").write_text("changed after validation")
    assert "missing or modified command log" in verify.report_errors(report, candidate, "cpu", directory)


@pytest.mark.parametrize("field,value", [("sha", "e" * 40), ("tree", "f" * 40), ("clean", False), ("status", "failed")])
def test_stale_or_failed_cpu_evidence_is_blocked(evidence, field, value):
    candidate, report, directory = evidence
    report[field] = value
    assert verify.report_errors(report, candidate, "cpu", directory)


def test_empty_test_run_is_not_a_pass(evidence):
    candidate, report, directory = evidence
    (directory / "junit.xml").write_text('<testsuites><testsuite tests="0"/></testsuites>')
    report["junit_sha256"] = verify.digest(directory / "junit.xml")
    assert "no CPU tests executed" in verify.report_errors(report, candidate, "cpu", directory)


def test_new_skips_are_not_silently_accepted(evidence):
    candidate, report, directory = evidence
    (directory / "junit.xml").write_text(
        '<testsuites><testsuite tests="1"><testcase name="new"><skipped/></testcase></testsuite></testsuites>'
    )
    report["junit_sha256"] = verify.digest(directory / "junit.xml")
    assert "unapproved CPU test skip" in verify.report_errors(report, candidate, "cpu", directory)


def test_only_the_known_accelerator_pin_memory_skip_is_allowed(evidence):
    candidate, report, directory = evidence
    (directory / "junit.xml").write_text(
        '<testsuites><testsuite tests="1">'
        '<testcase name="test_paged_state_dict_can_be_read_back_as_local_pinned_pages">'
        "<skipped/></testcase></testsuite></testsuites>"
    )
    report["junit_sha256"] = verify.digest(directory / "junit.xml")
    assert not verify.report_errors(report, candidate, "cpu", directory)


@pytest.mark.parametrize(
    "paths,expected",
    [
        (["DRV1_INTEGRATION.md", "scripts/drv1/verify.py"], True),
        (["docs/start/install.rst"], True),
        (["docs/conf.py"], False),
        (["verl/experimental/one_step_off_policy/ray_trainer.py"], False),
        (["requirements.txt"], False),
        (["examples/train.sh"], False),
        (["DRV1_INTEGRATION.md", "verl/workers/engine_workers.py"], False),
    ],
)
def test_runtime_change_cannot_use_documentation_exemption(paths, expected):
    assert verify.governance_only(paths) is expected


def gpu_fixture(evidence):
    """Create synthetic evidence solely for validator unit tests."""
    candidate, report, directory = copy.deepcopy(evidence)
    report.update({"kind": "gpu", "profile": "h20-16", "candidate": candidate, "checks": []})
    for name in verify.CASES:
        path = directory / (name + ".log")
        path.write_text("synthetic validator fixture; not a real GPU run")
        report["checks"].append({"case": name, "exit_code": 0, "log": path.name, "sha256": verify.digest(path)})
    report["probe"] = {
        "profile": "h20-16",
        "status": "passed",
        "tree": candidate["tree"],
        "sha": candidate["merge_sha"],
        "checks": ["all_reduce", "broadcast", "all_gather", "ring_p2p"],
        "ranks": [
            {
                "rank": i,
                "local_rank": i % 8,
                "host": "node" + str(i // 8),
                "gpu": "NVIDIA H20",
                "tree": candidate["tree"],
                "sha": candidate["merge_sha"],
                "clean": True,
            }
            for i in range(16)
        ],
    }
    report["observations"] = dict.fromkeys(verify.OBSERVATIONS, True)
    report["observations"].update({"reviewed_by": "fixture", "notes": "synthetic unit-test data"})
    return candidate, report, directory


def test_complete_synthetic_gpu_evidence_passes_validator_only(evidence):
    candidate, report, directory = gpu_fixture(evidence)
    assert not verify.gpu_errors(report, candidate, "h20-16", directory)


@pytest.mark.parametrize(
    "problem",
    [
        "single_host",
        "wrong_model",
        "duplicate_rank",
        "wrong_checkout",
        "no_training",
        "missing_observation",
        "missing_probe_check",
    ],
)
def test_incomplete_gpu_evidence_is_blocked(evidence, problem):
    candidate, report, directory = gpu_fixture(evidence)
    if problem == "single_host":
        for rank in report["probe"]["ranks"]:
            rank["host"] = "one-node"
    elif problem == "wrong_model":
        report["probe"]["ranks"][0]["gpu"] = "RTX 5090"
    elif problem == "duplicate_rank":
        report["probe"]["ranks"][0]["rank"] = 1
    elif problem == "wrong_checkout":
        report["probe"]["ranks"][0]["tree"] = "e" * 40
    elif problem == "no_training":
        report["checks"] = []
    elif problem == "missing_observation":
        report["observations"]["checkpoint_resume"] = False
    else:
        report["probe"]["checks"] = []
    assert verify.gpu_errors(report, candidate, "h20-16", directory)


def test_nonzero_command_is_kept_in_evidence(tmp_path):
    result = verify.run_command(
        [sys.executable, "-c", "print('failed command'); raise SystemExit(7)"], tmp_path / "failure.log", 10
    )
    assert result["exit_code"] == 7
    assert "failed command" in (tmp_path / "failure.log").read_text()


def test_training_child_does_not_inherit_github_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "synthetic-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "synthetic-secret")
    log = tmp_path / "environment.log"
    result = verify.run_command(
        [
            sys.executable,
            "-c",
            "import os; assert 'GH_TOKEN' not in os.environ; assert 'GITHUB_TOKEN' not in os.environ",
        ],
        log,
        10,
    )
    assert result["exit_code"] == 0
    assert "synthetic-secret" not in log.read_text()


def test_runtime_gate_blocks_missing_gpu_profiles(evidence, monkeypatch, capsys):
    candidate, cpu, directory = evidence
    verify.save(directory / "candidate.json", candidate)
    verify.save(directory / "report.json", cpu)
    monkeypatch.setattr(
        verify, "identity", lambda: {"sha": candidate["merge_sha"], "tree": candidate["tree"], "clean": True}
    )
    monkeypatch.setattr(verify, "check_candidate", lambda *args: None)
    monkeypatch.setattr(verify, "git", lambda *args: "verl/workers/engine_workers.py")
    args = SimpleNamespace(candidate=directory / "candidate.json", cpu=directory / "report.json", gpu=[], publish=False)
    assert verify.gate(args) == 1
    result = json.loads(capsys.readouterr().out)
    assert len(result["errors"]) == 4
    assert result["required_gpu_profiles"] == list(verify.PROFILES)


def test_metadata_gate_does_not_claim_gpu_execution(evidence, monkeypatch, capsys):
    candidate, cpu, directory = evidence
    verify.save(directory / "candidate.json", candidate)
    verify.save(directory / "report.json", cpu)
    monkeypatch.setattr(
        verify, "identity", lambda: {"sha": candidate["merge_sha"], "tree": candidate["tree"], "clean": True}
    )
    monkeypatch.setattr(verify, "check_candidate", lambda *args: None)
    monkeypatch.setattr(verify, "git", lambda *args: "DRV1_INTEGRATION.md")
    args = SimpleNamespace(candidate=directory / "candidate.json", cpu=directory / "report.json", gpu=[], publish=False)
    assert verify.gate(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["required_gpu_profiles"] == []
    assert "separately required" in result["human_pr_approval"]
