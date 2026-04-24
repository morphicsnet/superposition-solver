#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


WORKSPACE_ROOT = Path("/Volumes/128/superposition-demo")
SOURCE_MATH = Path("/Users/mini/Library/Mobile Documents/com~apple~CloudDocs/MATH.md")

EXCLUDED_BASENAMES = {
    "__pycache__",
    ".pytest_cache",
    "target",
    "outputs",
    "pytest-of-mini",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_excluded(path: Path, workspace_root: Path) -> bool:
    if path == workspace_root:
        return False
    if any(part in EXCLUDED_BASENAMES for part in path.relative_to(workspace_root).parts):
        return True
    return path.name.startswith(".")


def included_directories(workspace_root: Path) -> list[Path]:
    included: list[Path] = []
    for current_root, dirnames, _filenames in __import__("os").walk(workspace_root):
        current = Path(current_root)
        if is_excluded(current, workspace_root):
            dirnames[:] = []
            continue
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not is_excluded(current / dirname, workspace_root)
        ]
        included.append(current)
    included.sort()
    return included


def excluded_top_level_directories(workspace_root: Path) -> list[Path]:
    excluded = []
    for candidate in sorted(path for path in workspace_root.iterdir() if path.is_dir()):
        if is_excluded(candidate, workspace_root):
            excluded.append(candidate)
    return excluded


def verify(
    workspace_root: Path,
    included: list[Path],
    source_hash: str,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for directory in included:
        target = directory / "MATH.md"
        if not target.exists():
            errors.append(f"missing MATH.md: {target}")
            continue
        if sha256_file(target) != source_hash:
            errors.append(f"hash mismatch: {target}")

    for excluded in excluded_top_level_directories(workspace_root):
        target = excluded / "MATH.md"
        if target.exists():
            errors.append(f"excluded directory was touched: {target}")

    return not errors, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill MATH.md into every included project directory."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write files. Default is dry-run.",
    )
    args = parser.parse_args()

    if not WORKSPACE_ROOT.is_dir():
        print(f"workspace root not found: {WORKSPACE_ROOT}", file=sys.stderr)
        return 1
    if not SOURCE_MATH.is_file():
        print(f"source MATH.md not found: {SOURCE_MATH}", file=sys.stderr)
        return 1

    source_hash = sha256_file(SOURCE_MATH)
    included = included_directories(WORKSPACE_ROOT)
    excluded_top = excluded_top_level_directories(WORKSPACE_ROOT)

    print("workspace_root:", WORKSPACE_ROOT)
    print("source_math:", SOURCE_MATH)
    print("source_sha256:", source_hash)
    print("mode:", "apply" if args.apply else "dry-run")
    print("included_directories:")
    for directory in included:
        print(directory)
    print("included_directory_count:", len(included))
    print("excluded_top_level_directories:")
    for directory in excluded_top:
        print(directory)
    print("excluded_top_level_directory_count:", len(excluded_top))

    if not args.apply:
        return 0

    written_count = 0
    overwritten_count = 0
    for directory in included:
        target = directory / "MATH.md"
        if target.exists():
            overwritten_count += 1
        shutil.copyfile(SOURCE_MATH, target)
        written_count += 1

    ok, errors = verify(WORKSPACE_ROOT, included, source_hash)
    print("written_file_count:", written_count)
    print("overwritten_file_count:", overwritten_count)
    if ok:
        print("verification: ok")
        return 0

    print("verification: failed", file=sys.stderr)
    for error in errors:
        print(error, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
