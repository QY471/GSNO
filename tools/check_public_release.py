"""Check tracked working-tree files for private artifacts and local paths.

This check does not inspect Git history or establish redistribution rights.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_NAME_PARTS = (
    "checkpoint",
    "private",
    "internal",
    "screen",
    "analysis",
    "archive",
)
BLOCKED_SUFFIXES = {
    ".h5",
    ".mat",
    ".pth",
    ".pth.tar",
    ".pt",
    ".ckpt",
    ".zip",
    ".tar",
    ".log",
    ".out",
    ".csv",
}
SECRET_PATTERNS = (
    re.compile(r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"),
    re.compile(r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
)
LOCAL_PATH_PATTERNS = (
    re.compile(r"\b[A-Za-z]:\\"),
    re.compile(r"(?<![A-Za-z])/(?:home|root|mnt|Users)/"),
)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [ROOT / item for item in result.stdout.decode().split("\0") if item]


def main() -> int:
    failures: list[str] = []
    paths = tracked_files()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        lowered = relative.lower()
        is_release_readme = lowered == "checkpoints/readme.md"
        if not is_release_readme and any(
            part in lowered for part in BLOCKED_NAME_PARTS
        ):
            failures.append(f"private artifact name: {relative}")
        if not is_release_readme and any(
            lowered.endswith(suffix) for suffix in BLOCKED_SUFFIXES
        ):
            failures.append(f"private artifact suffix: {relative}")
        if not path.is_file():
            failures.append(f"missing tracked file: {relative}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except OSError:
            failures.append(f"unreadable tracked file: {relative}")
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                failures.append(f"secret-like content: {relative}")
        for pattern in LOCAL_PATH_PATTERNS:
            if pattern.search(text):
                failures.append(f"machine-local path: {relative}")

    if failures:
        print("Public-release safety check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"Working-tree safety check passed for {len(paths)} tracked files (history not checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
