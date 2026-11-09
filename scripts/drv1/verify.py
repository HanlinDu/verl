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
"""Prepare exact PR merge trees and collect evidence without merging v1."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPOSITORY = "HanlinDu/verl"
PROFILES = {"h20-8": ("H20", 1), "h20-16": ("H20", 2), "5090-8": ("5090", 1), "5090-16": ("5090", 2)}
CASES = ("disabled", "resize", "checkpoint-resume")
OBSERVATIONS = ("disabled_regression", "finite_loss", "two_resizes", "post_resize_training", "checkpoint_resume")
GOVERNANCE_FILES = {
    "AGENTS.md",
    "DRV1_INTEGRATION.md",
    ".github/CODEOWNERS",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/workflows/drv1.yml",
    ".github/workflows/drv1-acceptance.yml",
    "tests/special_sanity/test_drv1_gate_on_cpu.py",
    "tests/special_sanity/check_license.py",
}


def git(*args):
    """Read or update Git using argument arrays, never shell interpolation."""
    return subprocess.check_output(["git", *args], text=True).strip()


def identity():
    """Identify the entire checkout, including untracked non-ignored files."""
    return {
        "sha": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "clean": not bool(git("status", "--porcelain")),
    }


def save(path, value):
    """Write JSON evidence."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load(path):
    """Read JSON evidence."""
    return json.loads(Path(path).read_text())


def digest(path):
    """Hash logs incrementally, including large training logs."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def api(path, payload=None, token=None):
    """Use fixed GitHub endpoints; never forward credentials to artifact URLs."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "drv1-verifier"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        "https://api.github.com/" + path,
        headers=headers,
        data=json.dumps(payload).encode() if payload is not None else None,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def credential():
    """Obtain a credential from the configured Git helper, only for explicit publish."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_TRACE") and k != "GIT_CURL_VERBOSE"}
    env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        ["git", "-c", "credential.interactive=false", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode or not fields.get("password"):
        raise ValueError("GitHub authentication unavailable; configure Git credentials locally")
    return fields["password"]


def output_dir(value):
    """Require fresh evidence directories outside the checkout."""
    path = Path(value).resolve()
    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    if path == root or root in path.parents:
        raise ValueError("Evidence must be outside the checkout")
    path.mkdir(parents=True, exist_ok=False)
    return path


def governance_only(paths):
    """Conservatively require GPUs for anything outside documentation/governance."""
    return all(
        p in GOVERNANCE_FILES
        or p.startswith("scripts/drv1/")
        or (p.startswith("docs/") and p.endswith((".md", ".rst")))
        for p in paths
    )


def prepare(args):
    """Fetch GitHub's merge candidate for either a same-repository or fork PR."""
    pr = api(f"repos/{REPOSITORY}/pulls/{args.pr}")
    if pr["base"]["repo"]["full_name"] != REPOSITORY or pr["base"]["ref"] != "v1" or pr["state"] != "open":
        raise ValueError("An open PR targeting HanlinDu/verl:v1 is required")
    remote = git("remote", "get-url", "origin")
    if remote not in ("https://github.com/HanlinDu/verl.git", "git@github.com:HanlinDu/verl.git"):
        raise ValueError("origin must be HanlinDu/verl")
    out = output_dir(args.output)
    git("fetch", "--no-tags", "origin", f"refs/pull/{args.pr}/merge")
    sha = git("rev-parse", "FETCH_HEAD")
    parents = git("rev-list", "--parents", "-n", "1", sha).split()[1:]
    latest = git("ls-remote", "origin", "refs/heads/v1").split()[0]
    if parents != [latest, pr["head"]["sha"]] or latest != pr["base"]["sha"]:
        raise ValueError("Merge candidate is stale or conflicted; update/resolve the PR and retry")
    manifest = {
        "schema": 1,
        "repository": REPOSITORY,
        "pr": args.pr,
        "base_sha": latest,
        "head_sha": pr["head"]["sha"],
        "merge_sha": sha,
        "tree": git("rev-parse", sha + "^{tree}"),
    }
    save(out / "candidate.json", manifest)
    git("worktree", "add", "--detach", str(out / "checkout"), sha)
    print(json.dumps({"candidate": str(out / "candidate.json"), "checkout": str(out / "checkout"), **manifest}))


def run_command(argv, log, timeout):
    """Run an operator-supplied argv, preserving exit status and its full log."""
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise ValueError("Commands must be nonempty JSON arrays of strings")
    started = time.time()
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"GH_TOKEN", "GITHUB_TOKEN", "GIT_ASKPASS", "SSH_ASKPASS", "SSH_AUTH_SOCK"}
        and not k.startswith(("VSCODE_GIT_", "GIT_TRACE"))
        and k != "GIT_CURL_VERBOSE"
    }
    env["GIT_TERMINAL_PROMPT"] = "0"
    # The trusted tool may live in another checkout with an editable install.
    # Put the candidate first so tests and driver imports use the tested tree.
    candidate_root = git("rev-parse", "--show-toplevel")
    env["PYTHONPATH"] = candidate_root + os.pathsep + env.get("PYTHONPATH", "")
    with log.open("w") as stream:
        try:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=env)
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    return {
        "argv": argv,
        "exit_code": code,
        "seconds": round(time.time() - started, 3),
        "log": log.name,
        "sha256": digest(log),
    }


def cpu(args):
    """Run the research baseline and gate regression tests; do not suppress failures."""
    before = identity()
    out = output_dir(args.output)
    suites = [
        "tests/experimental/one_step_off_policy",
        "tests/workers/test_training_worker_resize_snapshot_on_cpu.py",
        "tests/special_sanity/test_drv1_gate_on_cpu.py",
    ]
    check = run_command(
        [sys.executable, "-m", "pytest", "-q", "-rs", *suites, "--junitxml=" + str(out / "junit.xml")],
        out / "pytest.log",
        900,
    )
    after = identity()
    status = "passed" if check["exit_code"] == 0 and after == before else "failed"
    report = {"schema": 1, "kind": "cpu", **before, "status": status, "checks": [check]}
    if (out / "junit.xml").is_file():
        report["junit_sha256"] = digest(out / "junit.xml")
    save(out / "report.json", report)
    print(json.dumps(report))
    return 0 if status == "passed" else 1


def check_candidate(candidate, current):
    """Reject dirty or substituted checkouts."""
    if candidate.get("repository") != REPOSITORY or not current["clean"]:
        raise ValueError("A clean checkout of the recorded research candidate is required")
    if current["sha"] != candidate["merge_sha"] or current["tree"] != candidate["tree"]:
        raise ValueError("Checkout differs from the recorded merge candidate")
    parents = git("rev-list", "--parents", "-n", "1", current["sha"]).split()[1:]
    if parents != [candidate["base_sha"], candidate["head_sha"]]:
        raise ValueError("Candidate parent commits do not match its manifest")


def gpu(args):
    """Collect three training cases separately from the NCCL infrastructure probe."""
    candidate, before = load(args.candidate), identity()
    check_candidate(candidate, before)
    out = output_dir(args.output)
    probe, plan = load(args.probe), load(args.plan)
    checks = []
    for name in CASES:
        checks.append({"case": name, **run_command(plan[name], out / (name + ".log"), args.timeout)})
        if checks[-1]["exit_code"]:
            break
    status = "passed" if len(checks) == len(CASES) and all(c["exit_code"] == 0 for c in checks) else "failed"
    if identity() != before:
        status = "failed"
    report = {
        "schema": 1,
        "kind": "gpu",
        **before,
        "profile": args.profile,
        "status": status,
        "candidate": candidate,
        "probe": probe,
        "plan": plan,
        "checks": checks,
        "observations": load(args.observations) if args.observations else {},
    }
    save(out / "report.json", report)
    print(
        json.dumps(
            {
                "report": str(out / "report.json"),
                "status": status,
                "note": "Exit codes alone are not training acceptance; operator observations are required.",
            }
        )
    )
    return 0 if status == "passed" else 1


def report_errors(report, candidate, kind, directory):
    """Validate source binding and on-disk artifacts, including pytest skips."""
    errors = []
    if report.get("kind") != kind or report.get("status") != "passed" or report.get("clean") is not True:
        errors.append("missing successful, clean " + kind + " report")
    if report.get("tree") != candidate["tree"] or report.get("sha") != candidate["merge_sha"]:
        errors.append("report describes another candidate")
    checks = report.get("checks", [])
    if not checks or any(c.get("exit_code") != 0 for c in checks):
        errors.append("commands failed or absent")
    for check in checks:
        log = directory / check.get("log", "")
        if log.parent != directory or not log.is_file() or digest(log) != check.get("sha256"):
            errors.append("missing or modified command log")
    if kind == "cpu":
        junit = directory / "junit.xml"
        if not junit.is_file() or digest(junit) != report.get("junit_sha256"):
            errors.append("missing or modified JUnit report")
        else:
            suites = list(ET.parse(junit).getroot().iter("testsuite"))
            if not suites or sum(int(s.get("tests", "0")) for s in suites) == 0:
                errors.append("no CPU tests executed")
            for suite in suites:
                if int(suite.get("failures", "0")) or int(suite.get("errors", "0")):
                    errors.append("JUnit contains failures")
            for skipped in ET.parse(junit).getroot().iter("testcase"):
                if (
                    skipped.find("skipped") is not None
                    and skipped.get("name") != "test_paged_state_dict_can_be_read_back_as_local_pinned_pages"
                ):
                    errors.append("unapproved CPU test skip")
    return errors


def gpu_errors(report, candidate, profile, directory):
    """Require real topology evidence and explicit human interpretation of training logs."""
    errors = report_errors(report, candidate, "gpu", directory)
    probe = report.get("probe", {})
    model, nodes = PROFILES[profile]
    ranks = probe.get("ranks", [])
    if (
        report.get("profile") != profile
        or report.get("candidate") != candidate
        or probe.get("profile") != profile
        or probe.get("status") != "passed"
        or probe.get("tree") != candidate["tree"]
        or probe.get("sha") != candidate["merge_sha"]
    ):
        errors.append("probe or report belongs to another profile/candidate")
    if set(probe.get("checks", [])) != {"all_reduce", "broadcast", "all_gather", "ring_p2p"}:
        errors.append("NCCL payload checks are incomplete")
    if (
        len(ranks) != nodes * 8
        or {r.get("rank") for r in ranks} != set(range(nodes * 8))
        or len({r.get("host") for r in ranks}) != nodes
    ):
        errors.append("expected 8 GPUs per node on the requested number of distinct nodes")
    for host in {r.get("host") for r in ranks}:
        local = [r for r in ranks if r.get("host") == host]
        if (
            {r.get("local_rank") for r in local} != set(range(8))
            or any(model.lower() not in r.get("gpu", "").lower() for r in local)
            or any(
                r.get("tree") != candidate["tree"]
                or r.get("sha") != candidate["merge_sha"]
                or r.get("clean") is not True
                for r in local
            )
        ):
            errors.append("GPU identity, per-node placement, or checkout mismatch")
    if [c.get("case") for c in report.get("checks", [])] != list(CASES):
        errors.append("all three training cases must run")
    observation = report.get("observations", {})
    if not observation.get("reviewed_by") or not observation.get("notes"):
        errors.append("operator log review is missing")
    if any(observation.get(key) is not True for key in OBSERVATIONS):
        errors.append("training/resize/checkpoint observations are incomplete")
    return errors


def gate(args):
    """Fail closed, optionally publishing acceptance only as the repository owner."""
    candidate, current = load(args.candidate), identity()
    check_candidate(candidate, current)
    paths = git("diff", "--name-only", candidate["base_sha"], current["sha"]).splitlines()
    errors = report_errors(load(args.cpu), candidate, "cpu", Path(args.cpu).resolve().parent)
    reports = {}
    for filename in args.gpu:
        report = load(filename)
        profile = report.get("profile")
        if profile in reports or profile not in PROFILES:
            errors.append("duplicate or unknown GPU profile")
        reports[profile] = (report, Path(filename).resolve().parent)
    required = [] if governance_only(paths) else list(PROFILES)
    for profile in required:
        if profile not in reports:
            errors.append("missing GPU evidence: " + profile)
        else:
            report, directory = reports[profile]
            errors.extend(profile + ": " + error for error in gpu_errors(report, candidate, profile, directory))
    result = {
        "status": "blocked" if errors else "passed",
        "candidate": candidate,
        "required_gpu_profiles": required,
        "errors": errors,
        "human_pr_approval": "separately required; this tool never approves or merges a PR",
    }
    print(json.dumps(result, indent=2))
    if errors:
        return 1
    if args.publish:
        if not args.evidence_url or not args.evidence_url.startswith("https://"):
            raise ValueError("A durable HTTPS evidence URL is required for publishing")
        token = credential()
        if api("user", token=token)["login"].lower() != "hanlindu":
            raise ValueError("Only HanlinDu publishes drv1/acceptance")
        pr = api(f"repos/{REPOSITORY}/pulls/{candidate['pr']}", token=token)
        if (
            pr["state"] != "open"
            or pr["base"]["ref"] != "v1"
            or pr["base"]["sha"] != candidate["base_sha"]
            or pr["head"]["sha"] != candidate["head_sha"]
            or git("ls-remote", "origin", "refs/heads/v1").split()[0] != candidate["base_sha"]
        ):
            raise ValueError("PR changed or v1 advanced; prepare and validate again")
        api(
            f"repos/{REPOSITORY}/statuses/{candidate['head_sha']}",
            {
                "state": "success",
                "context": "drv1/acceptance",
                "target_url": args.evidence_url,
                "description": "Verified candidate tree " + candidate["tree"][:12],
            },
            token=token,
        )
    return 0


def post_merge(args):
    """Verify that GitHub integrated precisely the tested tree and parents."""
    candidate, current = load(args.candidate), identity()
    if not current["clean"] or current["tree"] != candidate["tree"]:
        raise ValueError("Integrated tree differs from the tested tree")
    if git("rev-list", "--parents", "-n", "1", "HEAD").split()[1:] != [candidate["base_sha"], candidate["head_sha"]]:
        raise ValueError("Expected a merge commit with the validated parents")
    if git("ls-remote", "origin", "refs/heads/v1").split()[0] != current["sha"]:
        raise ValueError("Checkout is not the current remote v1")
    print(
        json.dumps(
            {
                "status": "tree_verified",
                **current,
                "next": "Check push CI and perform the documented post-merge training smoke before known-good tagging.",
            }
        )
    )


def main():
    """Expose explicit operations; no subcommand merges, pushes, or approves a PR."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--pr", type=int, required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=prepare)
    command = sub.add_parser("cpu")
    command.add_argument("--output", required=True)
    command.set_defaults(func=cpu)
    command = sub.add_parser("gpu")
    command.add_argument("--candidate", required=True)
    command.add_argument("--profile", choices=PROFILES, required=True)
    command.add_argument("--probe", required=True)
    command.add_argument("--plan", required=True)
    command.add_argument("--observations")
    command.add_argument("--timeout", type=int, default=1800)
    command.add_argument("--output", required=True)
    command.set_defaults(func=gpu)
    command = sub.add_parser("gate")
    command.add_argument("--candidate", required=True)
    command.add_argument("--cpu", required=True)
    command.add_argument("--gpu", nargs="*", default=[])
    command.add_argument("--publish", action="store_true")
    command.add_argument("--evidence-url")
    command.set_defaults(func=gate)
    command = sub.add_parser("post-merge")
    command.add_argument("--candidate", required=True)
    command.set_defaults(func=post_merge)
    args = parser.parse_args()
    try:
        return args.func(args) or 0
    except (ValueError, OSError, KeyError, subprocess.SubprocessError, ET.ParseError) as error:
        print(f"BLOCKED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
