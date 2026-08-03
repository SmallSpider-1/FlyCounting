#!/usr/bin/env python3
"""Finalize fruitfly_mot_v1 into a clean, self-contained dataset root.

The operation is atomic and recoverable: a sibling staging tree is built with
regular hard-linked image/video files, the original mixed import tree is renamed
to a timestamped backup, and the staged tree is promoted. The caller must run the
strict validator and TrackEval smoke test before deleting the returned backup.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


TIMEZONE = ZoneInfo("Asia/Shanghai")
MARKER = ".fruitfly_mot_v1_generated.json"
SEQUENCE_IDS = tuple(f"FFMOT-{index:02d}" for index in range(1, 16))


class FinalizeError(RuntimeError):
    """Raised when finalization cannot be completed safely."""


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source.resolve(), target)


def copy_generated_metadata(source_root: Path, staging_root: Path) -> None:
    for directory in ("annotations", "manifests", "metadata", "quality_reports", "seqmaps"):
        shutil.copytree(source_root / directory, staging_root / directory)
    for sequence_id in SEQUENCE_IDS:
        source_sequence = source_root / "MOTChallenge" / "all" / sequence_id
        target_sequence = staging_root / "MOTChallenge" / "all" / sequence_id
        shutil.copytree(source_sequence / "gt", target_sequence / "gt")
        target_sequence.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_sequence / "seqinfo.ini", target_sequence / "seqinfo.ini")
        for source_image in sorted((source_sequence / "img1").glob("*.jpg")):
            hardlink(source_image.resolve(), target_sequence / "img1" / source_image.name)


def archive_source_annotations(
    dataset_root: Path,
    staging_root: Path,
    frame_rows: list[dict[str, str]],
) -> tuple[Path, list[dict[str, str]]]:
    archive_path = staging_root / "archives" / "source_annotations_v1.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_rows: list[dict[str, str]] = []
    with tarfile.open(archive_path, "w:gz", compresslevel=9) as archive:
        for row in frame_rows:
            source_json = dataset_root / row["source_json"]
            if not source_json.is_file():
                raise FinalizeError(f"missing source JSON: {source_json}")
            member = f"json/{int(row['sequence_id'].split('-')[1])}/{source_json.name}"
            archive.add(source_json, arcname=member, recursive=False)
            archive_rows.append(
                {
                    "original_path": row["source_json"],
                    "archive_member": member,
                    "sha256": row["json_sha256"],
                }
            )
        decisions_path = dataset_root / "source_decisions.json"
        if not decisions_path.is_file():
            raise FinalizeError(f"missing source decisions: {decisions_path}")
        archive.add(decisions_path, arcname="source_decisions.json", recursive=False)

    expected = {row["archive_member"]: row["sha256"] for row in archive_rows}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        if set(expected) - set(members):
            raise FinalizeError("source annotation archive is incomplete")
        for name, digest in expected.items():
            extracted = archive.extractfile(members[name])
            if extracted is None or hashlib.sha256(extracted.read()).hexdigest() != digest:
                raise FinalizeError(f"source annotation archive hash mismatch: {name}")
    return archive_path, archive_rows


def rewrite_manifests(
    dataset_root: Path,
    source_standardized: Path,
    staging_root: Path,
    archive_rows: list[dict[str, str]],
) -> None:
    archive_member_by_source = {row["original_path"]: row["archive_member"] for row in archive_rows}
    frame_path = staging_root / "manifests" / "frames.csv"
    frames = read_csv(frame_path)
    for row in frames:
        row["source_image"] = row["mot_image"]
        row["source_json"] = (
            "archives/source_annotations_v1.tar.gz::" + archive_member_by_source[row["source_json"]]
        )
    write_csv(frame_path, list(frames[0]), frames)

    sequence_path = staging_root / "manifests" / "sequences.csv"
    sequences = read_csv(sequence_path)
    for row in sequences:
        sequence_id = row["sequence_id"]
        row["source_video"] = f"videos/{sequence_id}.mp4"
        row["source_frame_dir"] = f"MOTChallenge/all/{sequence_id}/img1"
    write_csv(sequence_path, list(sequences[0]), sequences)

    shutil.copy2(dataset_root / "source_decisions.json", staging_root / "metadata" / "source_decisions.json")
    (staging_root / "metadata" / "dataset.yaml").write_text(
        "dataset_name: fruitfly_mot_v1\n"
        "dataset_version: 1.0.0-build1\n"
        f"finalized_at: '{now_iso()}'\n"
        "layout_status: finalized\n"
        "source_frame_indexing: zero_based\n"
        "mot_frame_indexing: one_based\n"
        "primary_evaluation: class_agnostic\n"
        "trackeval_do_preproc: false\n"
        "split_status: unassigned\n"
        "image_materialization: regular_files\n"
        "source_annotation_archive: archives/source_annotations_v1.tar.gz\n",
        encoding="utf-8",
    )
    (staging_root / "metadata" / "README.md").write_text(
        "# fruitfly_mot_v1\n\n"
        "This is the finalized laboratory MOT dataset. `MOTChallenge/all/` contains the 15 complete "
        "sequences with regular JPG files, GT and `seqinfo.ini`; `videos/` contains the corresponding MP4s.\n\n"
        "The authoritative species-aware table is `annotations/mot_annotations.csv`. The primary TrackEval "
        "view is class-agnostic and writes class 1 in `gt.txt`. Source LabelMe/X-AnyLabeling JSON is retained "
        "compactly in `archives/source_annotations_v1.tar.gz`.\n\n"
        "The split is still `unassigned`; `all` is an engineering view, not an independent test set.\n",
        encoding="utf-8",
    )
    annotation_policy = staging_root / "metadata" / "annotation_policy.md"
    annotation_policy.write_text(
        annotation_policy.read_text(encoding="utf-8")
        + "- Original per-frame JSON is retained in `archives/source_annotations_v1.tar.gz`; it is not "
        "required by TrackEval.\n",
        encoding="utf-8",
    )


def checksum_candidates(staging_root: Path) -> list[Path]:
    excluded = {
        staging_root / MARKER,
        staging_root / "quality_reports" / "checksums.sha256",
        staging_root / "quality_reports" / "dataset_fingerprint.txt",
        staging_root / "quality_reports" / "validation_report.json",
        staging_root / "quality_reports" / "trackeval_smoke_test.json",
        staging_root / "quality_reports" / "source_audit.json",
        staging_root / "quality_reports" / "finalization_report.json",
    }
    return sorted(
        (path for path in staging_root.rglob("*") if path.is_file() and path not in excluded),
        key=lambda path: path.relative_to(staging_root).as_posix(),
    )


def regenerate_provenance(
    staging_root: Path,
    archive_path: Path,
    archive_rows: list[dict[str, str]],
) -> dict[str, Any]:
    checksum_rows = []
    for path in checksum_candidates(staging_root):
        checksum_rows.append((path.relative_to(staging_root).as_posix(), sha256_file(path)))
    checksum_text = "".join(f"{digest}  {path}\n" for path, digest in checksum_rows)
    reports = staging_root / "quality_reports"
    (reports / "checksums.sha256").write_text(checksum_text, encoding="utf-8")
    canonical = (staging_root / "annotations" / "mot_annotations.csv").read_bytes()
    fingerprint = hashlib.sha256(checksum_text.encode("utf-8") + canonical).hexdigest()
    (reports / "dataset_fingerprint.txt").write_text(fingerprint + "\n", encoding="utf-8")

    prior_audit = json.loads((reports / "source_audit.json").read_text(encoding="utf-8"))
    prior_audit.update(
        {
            "status": "finalized",
            "dataset_root": str(staging_root),
            "output": str(staging_root),
            "layout_status": "finalized",
            "checksum_files": len(checksum_rows),
            "dataset_fingerprint": fingerprint,
        }
    )
    (reports / "source_audit.json").write_text(
        json.dumps(prior_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "status": "staged",
        "finalized_at": now_iso(),
        "layout": "dataset_root_is_standardized_root",
        "regular_images": 5567,
        "videos": 15,
        "archived_source_json": len(archive_rows),
        "archive": archive_path.relative_to(staging_root).as_posix(),
        "archive_sha256": sha256_file(archive_path),
        "checksum_files": len(checksum_rows),
        "dataset_fingerprint": fingerprint,
    }
    (reports / "finalization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (staging_root / MARKER).write_text(
        json.dumps(
            {
                "dataset_name": "fruitfly_mot_v1",
                "generated_by": "scripts/finalize_fruitfly_mot_v1_layout.py",
                "finalized_at": report["finalized_at"],
                "dataset_fingerprint": fingerprint,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def finalize(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    source_standardized = dataset_root / "standardized"
    if not (source_standardized / MARKER).is_file():
        raise FinalizeError(f"missing nested standardized dataset: {source_standardized}")
    for sequence in range(1, 16):
        if not (dataset_root / str(sequence)).is_dir() or not (dataset_root / f"{sequence}.mp4").is_file():
            raise FinalizeError(f"missing numbered source for sequence {sequence}")

    staging_root = dataset_root.parent / f".{dataset_root.name}.finalizing-{os.getpid()}"
    stamp = datetime.now(TIMEZONE).strftime("%Y%m%dT%H%M%S")
    backup_root = dataset_root.parent / f"{dataset_root.name}.pre-finalize-{stamp}"
    if staging_root.exists() or backup_root.exists():
        raise FinalizeError("staging or backup destination already exists")
    staging_root.mkdir()

    copy_generated_metadata(source_standardized, staging_root)
    for sequence, sequence_id in enumerate(SEQUENCE_IDS, start=1):
        hardlink(dataset_root / f"{sequence}.mp4", staging_root / "videos" / f"{sequence_id}.mp4")

    frames = read_csv(source_standardized / "manifests" / "frames.csv")
    archive_path, archive_rows = archive_source_annotations(dataset_root, staging_root, frames)
    write_csv(
        staging_root / "manifests" / "source_annotation_archive.csv",
        ["original_path", "archive_member", "sha256"],
        archive_rows,
    )
    rewrite_manifests(dataset_root, source_standardized, staging_root, archive_rows)
    report = regenerate_provenance(staging_root, archive_path, archive_rows)

    if len(list((staging_root / "MOTChallenge" / "all").glob("*/img1/*.jpg"))) != 5567:
        raise FinalizeError("staged image count is not 5567")
    if any(path.is_symlink() for path in staging_root.rglob("*")):
        raise FinalizeError("staged dataset unexpectedly contains symbolic links")
    if len(list((staging_root / "videos").glob("*.mp4"))) != 15:
        raise FinalizeError("staged video count is not 15")

    dataset_root.rename(backup_root)
    staging_root.rename(dataset_root)
    source_audit_path = dataset_root / "quality_reports" / "source_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    source_audit["dataset_root"] = str(dataset_root)
    source_audit["output"] = str(dataset_root)
    source_audit_path.write_text(
        json.dumps(source_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report.update(
        {
            "status": "promoted_pending_external_validation",
            "dataset_root": str(dataset_root),
            "backup_root": str(backup_root),
        }
    )
    (dataset_root / "quality_reports" / "finalization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/fruitfly_mot_v1"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        finalize(args.dataset_root)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
