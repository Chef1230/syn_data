"""Persistence helpers for generated relational databases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping
import json

import pandas as pd

from .base import json_ready


def save_tables_as_parquet(tables: Mapping[str, pd.DataFrame], output_dir: Any) -> Dict[str, str]:
    """Save generated tables under output_dir/tables/{table_id}.parquet."""
    root = Path(output_dir)
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}
    for table_id, df in tables.items():
        path = table_dir / f"{table_id}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")
        paths[str(table_id)] = str(path)
    return paths


def save_metadata(metadata: Mapping[str, Any], output_dir: Any) -> str:
    """Save generation metadata as generation_metadata.json."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "generation_metadata.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(dict(metadata)), f, indent=2, ensure_ascii=False)
    return str(path)


def save_generation_report(report: Mapping[str, Any], output_dir: Any) -> str:
    """Save a lightweight generation report as generation_report.json."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "generation_report.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(dict(report)), f, indent=2, ensure_ascii=False)
    return str(path)