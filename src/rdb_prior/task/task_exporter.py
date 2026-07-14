"""Persistence utilities for relational task bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping
import json

from ..generators.base import json_ready
from .task_sampler import TaskBundle


def save_task_bundle(output_dir: Any, bundle: TaskBundle) -> Dict[str, str]:
    """Save a task bundle as JSON metadata plus parquet labels and splits."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    task_path = root / "task.json"
    labels_path = root / "labels.parquet"
    manifest_path = root / "feature_manifest.json"

    _write_json(
        task_path,
        {
            "spec": bundle.spec.to_dict(),
            "metadata": dict(bundle.metadata),
        },
    )
    bundle.labels.to_parquet(labels_path, index=False)
    _write_json(manifest_path, bundle.feature_manifest)

    paths: Dict[str, str] = {
        "task": str(task_path),
        "labels": str(labels_path),
        "feature_manifest": str(manifest_path),
    }
    for split_name, split_df in bundle.splits.items():
        split_path = splits_dir / f"{split_name}.parquet"
        split_df.to_parquet(split_path, index=False)
        paths[f"split_{split_name}"] = str(split_path)
    return paths


def save_task_spec(output_dir: Any, spec: Mapping[str, Any]) -> str:
    """Save a standalone task specification JSON."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "task.json"
    _write_json(path, spec)
    return str(path)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(data), indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["save_task_bundle", "save_task_spec"]
