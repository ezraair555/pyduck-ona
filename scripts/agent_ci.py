#!/usr/bin/env python3
"""Agent-friendly local CI runner for pyduck-ona.

Produces machine-parseable output so an agent can decide whether the
repo is in a safe state to commit/push.

Usage:
    python scripts/agent_ci.py
    python scripts/agent_ci.py --format json
    python scripts/agent_ci.py --skip-mypy
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    command: str
    passed: bool
    stdout: str
    stderr: str
    returncode: int
    required: bool


def _run(cmd: list[str], *, timeout: int = 300) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=Path(__file__).resolve().parent.parent,
    )
    return proc.returncode, proc.stdout, proc.stderr


def ruff_check() -> CheckResult:
    rc, out, err = _run(["ruff", "check", "."])
    return CheckResult(
        name="ruff",
        command="ruff check .",
        passed=rc == 0,
        stdout=out,
        stderr=err,
        returncode=rc,
        required=True,
    )


def pytest_check() -> CheckResult:
    rc, out, err = _run(["pytest", "-q"])
    # Pytest returns 0 on pass, 1 on failures, 2 on errors.
    passed = rc == 0
    return CheckResult(
        name="pytest",
        command="pytest -q",
        stdout=out,
        stderr=err,
        returncode=rc,
        passed=passed,
        required=True,
    )


def mypy_check() -> CheckResult:
    rc, out, err = _run(["mypy", "src/pyduck_ona"])
    return CheckResult(
        name="mypy",
        command="mypy src/pyduck_ona",
        stdout=out,
        stderr=err,
        returncode=rc,
        passed=rc == 0,
        required=False,  # strict typing is a goal; not gating yet
    )


def _summary(results: list[CheckResult]) -> str:
    required_passed = all(r.passed for r in results if r.required)
    all_passed = all(r.passed for r in results)
    if required_passed and not all_passed:
        return "required checks passed (non-gating mypy still has errors)"
    if all_passed:
        return "all checks passed"
    return "required checks failed"


def _render_markdown(results: list[CheckResult]) -> str:
    lines = ["# Agent CI Results", ""]
    lines.append(f"**Summary:** {_summary(results)}")
    lines.append("")
    lines.append("| Check | Required | Status | Return |")
    lines.append("|-------|----------|--------|--------|")
    for r in results:
        status = "✅ pass" if r.passed else "❌ fail"
        lines.append(f"| {r.name} | {'yes' if r.required else 'no'} | {status} | {r.returncode} |")
    lines.append("")
    for r in results:
        if not r.passed:
            lines.append(f"## {r.name} output")
            if r.stdout.strip():
                lines.append("```")
                lines.append(r.stdout.strip()[:2000])
                lines.append("```")
            if r.stderr.strip():
                lines.append("```")
                lines.append(r.stderr.strip()[:2000])
                lines.append("```")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent-friendly CI runner")
    parser.add_argument(
        "--format",
        choices=["json", "md", "markdown"],
        default="md",
        help="Output format",
    )
    parser.add_argument(
        "--skip-mypy",
        action="store_true",
        help="Skip the non-gating mypy check",
    )
    args = parser.parse_args(argv)

    checks = [ruff_check, pytest_check]
    if not args.skip_mypy:
        checks.append(mypy_check)

    results = [check() for check in checks]

    if args.format == "json":
        payload = {
            "summary": _summary(results),
            "exit_code": 0 if all(r.passed for r in results if r.required) else 1,
            "checks": [asdict(r) for r in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_render_markdown(results))

    return 0 if all(r.passed for r in results if r.required) else 1


if __name__ == "__main__":
    sys.exit(main())
