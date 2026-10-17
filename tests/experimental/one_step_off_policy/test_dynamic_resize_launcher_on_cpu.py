# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""``shell/run_dynamic_resize.sh``: what the launcher runs, without running it.

A stand-in ``python3`` at the front of ``PATH`` records every command the launcher starts, and where and
in which environment it starts it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "verl/experimental/one_step_off_policy/shell/run_dynamic_resize.sh"

STAND_IN_PYTHON3 = f"""#!{sys.executable}
import json, os, sys

if sys.argv[1:] == ["-"]:  # the data conversion script on stdin: hand the launcher its two paths back
    print("TRAIN_FILE=train.parquet")
    print("VAL_FILE=val.parquet")
    sys.exit(0)
with open(os.environ["RECORD"], "a") as record:
    record.write(json.dumps({{"argv": sys.argv[1:], "env": dict(os.environ), "cwd": os.getcwd()}}) + "\\n")
"""


@pytest.fixture
def launch(tmp_path):
    """Run the launcher with ``env`` on top of a bare environment; its exit code and the commands it started."""
    stand_in = tmp_path / "bin" / "python3"
    stand_in.parent.mkdir()
    stand_in.write_text(STAND_IN_PYTHON3)
    stand_in.chmod(0o755)
    record = tmp_path / "record.jsonl"

    def run(*args, **env):
        record.unlink(missing_ok=True)
        env = {
            "PATH": f"{stand_in.parent}:{os.environ['PATH']}",
            "RECORD": str(record),
            "CKPT_DIR": str(tmp_path / "ckpt"),
            "MODEL_PATH": "/models/some-lm",
            **env,
        }
        result = subprocess.run(["bash", str(LAUNCHER), *args], env=env, capture_output=True, text=True, timeout=120)
        commands = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
        return result.returncode, commands

    return run


def _is_training(command):
    return command["argv"][:2] == ["-m", "verl.experimental.one_step_off_policy.main_ppo"]


def test_launcher_starts_training_with_the_arguments_of_the_caller_last(launch):
    code, commands = launch("trainer.total_training_steps=3")

    assert code == 0
    (training,) = commands
    assert _is_training(training)
    assert training["argv"][-1] == "trainer.total_training_steps=3"
