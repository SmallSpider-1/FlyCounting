#!/usr/bin/env python3
"""Auditably swap the two YOLO class IDs in the confirmed-bad R2 test labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_and_swap(path: Path) -> tuple[str, dict[int, int]]:
    source = path.read_text(encoding="utf-8")
    output: list[str] = []
    counts = {0: 0, 1: 0}
    for line_number, line in enumerate(source.splitlines(keepends=True), 1):
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        if not body.strip():
            output.append(line)
            continue
        fields = body.split()
        if len(fields) != 5 or fields[0] not in {"0", "1"}:
            raise ValueError(f"{path}:{line_number}: expected canonical five-field YOLO row with class 0/1")
        for value in fields[1:]:
            float(value)
        old_class = int(fields[0])
        counts[old_class] += 1
        prefix_length = len(body) - len(body.lstrip())
        prefix = body[:prefix_length]
        remainder = body[prefix_length + 1 :]
        output.append(f"{prefix}{1 - old_class}{remainder}{ending}")
    return "".join(output), counts


def atomic_write(path: Path, content: str) -> None:
    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--expected-files", type=int, default=405)
    parser.add_argument("--expected-class-0", type=int, default=1065)
    parser.add_argument("--expected-class-1", type=int, default=335)
    args = parser.parse_args()

    labels_dir = args.labels_dir.resolve()
    backup_dir = args.backup_dir.resolve()
    files = sorted(labels_dir.glob("*.txt"))
    if len(files) != args.expected_files:
        raise RuntimeError(f"Refusing repair: expected {args.expected_files} labels, found {len(files)}")
    if backup_dir.exists():
        raise FileExistsError(f"Refusing to overwrite recovery backup: {backup_dir}")

    swapped: dict[Path, str] = {}
    before_counts = {0: 0, 1: 0}
    before_hashes: dict[str, str] = {}
    for path in files:
        content, counts = parse_and_swap(path)
        swapped[path] = content
        before_counts[0] += counts[0]
        before_counts[1] += counts[1]
        before_hashes[path.name] = sha256(path)
    expected = {0: args.expected_class_0, 1: args.expected_class_1}
    if before_counts != expected:
        raise RuntimeError(f"Refusing repair: expected pre-repair counts {expected}, found {before_counts}")

    backup_dir.mkdir(parents=True)
    for path in files:
        shutil.copy2(path, backup_dir / path.name)

    for path in files:
        atomic_write(path, swapped[path])

    after_counts = {0: 0, 1: 0}
    after_hashes: dict[str, str] = {}
    for path in files:
        _, counts = parse_and_swap(path)
        after_counts[0] += counts[0]
        after_counts[1] += counts[1]
        after_hashes[path.name] = sha256(path)
        if sha256(backup_dir / path.name) != before_hashes[path.name]:
            raise RuntimeError(f"Backup verification failed for {path.name}")
    if after_counts != {0: expected[1], 1: expected[0]}:
        raise RuntimeError(f"Post-repair counts are invalid: {after_counts}")

    record = {
        "schema_version": 1,
        "repaired_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "R2/test YOLO labels only",
        "operation": "swap class_id 0<->1; preserve every bbox coordinate",
        "labels_dir": str(labels_dir),
        "backup_dir": str(backup_dir),
        "files": len(files),
        "before_counts": {str(key): value for key, value in before_counts.items()},
        "after_counts": {str(key): value for key, value in after_counts.items()},
        "before_sha256": before_hashes,
        "after_sha256": after_hashes,
    }
    (backup_dir / "repair_manifest.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: record[key] for key in ("repaired_at", "scope", "files", "before_counts", "after_counts", "backup_dir")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
