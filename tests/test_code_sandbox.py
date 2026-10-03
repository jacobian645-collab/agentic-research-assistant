# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

import pytest

from research_assistant.mcp_servers.code_exec_server import PolicyViolation, check_policy, execute


async def test_runs_simple_code():
    r = await execute("import math\nprint(round(math.sqrt(2), 3))")
    assert r["exit_code"] == 0 and r["stdout"].strip() == "1.414"


@pytest.mark.parametrize(
    "code",
    [
        "import os\nprint(os.environ)",
        "import subprocess",
        "from socket import socket",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "eval('1+1')",
        "__import__('os')",
    ],
)
def test_policy_rejects_dangerous_code(code):
    with pytest.raises(PolicyViolation):
        check_policy(code)


async def test_policy_violation_returned_as_structured_error():
    r = await execute("import os")
    assert r["exit_code"] == 2 and "PolicyViolation" in r["stderr"]


async def test_cpu_limit_kills_infinite_loop():
    r = await execute("while True:\n    pass", timeout_s=1)
    assert r["timed_out"] is True
    assert r["duration_s"] < 5
