"""Export syn_data databases to the DBInfer/RDB_PFN DFS input layout.

The generated syn_data layout is:

    db_xxxxxx/
      tables/*.parquet
      schema/schema.json
      task/{task.json,labels.parquet,feature_manifest.json,splits/*.parquet}

RDB_PFN's DFS preprocessing expects a DBInfer-style dataset:

    dataset/
      metadata.yaml
      data/*.parquet
      tasks/<task_name>/{train,validation,test}.parquet

This script bridges those layouts without importing RDB_PFN code.
"""

from __future__ import annotations

import argparse
import json
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar

import pandas as pd
import yaml

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None


T = TypeVar("T")


SYN_TO_DBINFER_SPLIT = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}

METRIC_MAP = {
    "roc_auc": "auroc",
    "auc": "auroc",
    "auroc": "auroc",
    "accuracy": "accuracy",
    "f1": "f1",
    "mae": "mae",
    "mse": "mse",
    "rmse": "rmse",
    "r2": "r2",
    "logloss": "logloss",
}


@dataclass(frozen=True)
class ExportResult:
    database_dir: str
    output_dir: str | None
    status: str
    reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert syn_data generated databases to RDB_PFN DFS input directories."
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        default=Path("outputs/databases/sample_50"),
        help="Root containing generated db_* directories, or one generated db directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/dbinfer_for_dfs"),
        help="Directory where DBInfer/RDB_PFN DFS input datasets will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing converted dataset directories.",
    )
    parser.add_argument(
        "--include-hidden-columns",
        action="store_true",
        help="Ignore feature_manifest column filtering. Mostly useful for debugging leakage checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database_dirs = discover_database_dirs(args.database_root)
    if not database_dirs:
        raise RuntimeError(f"No syn_data generated DBs found under {args.database_root}.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[ExportResult] = []
    for db_dir in _progress(database_dirs, "Exporting DBInfer DFS inputs", "db"):
        out_dir = args.output_root / db_dir.name
        try:
            export_one_database(
                db_dir=db_dir,
                out_dir=out_dir,
                overwrite=args.overwrite,
                include_hidden_columns=args.include_hidden_columns,
            )
        except Exception as exc:  # keep batch conversion moving
            results.append(
                ExportResult(str(db_dir), None, "skipped", f"{type(exc).__name__}: {exc}")
            )
            continue
        results.append(ExportResult(str(db_dir), str(out_dir), "exported"))

    report = {
        "database_root": str(args.database_root),
        "output_root": str(args.output_root),
        "num_exported": sum(item.status == "exported" for item in results),
        "num_skipped": sum(item.status != "exported" for item in results),
        "items": [item.__dict__ for item in results],
    }
    (args.output_root / "export_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _progress(iterable: Iterable[T], desc: str, unit: str) -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, ascii=True, file=sys.stdout, dynamic_ncols=True)


def discover_database_dirs(root: Path) -> list[Path]:
    root = root.resolve()
    if is_syn_database_dir(root):
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    return [path for path in sorted(root.iterdir()) if path.is_dir() and is_syn_database_dir(path)]


def is_syn_database_dir(path: Path) -> bool:
    return (
        (path / "tables").is_dir()
        and (path / "schema" / "schema.json").exists()
        and (path / "task" / "task.json").exists()
        and (path / "task" / "labels.parquet").exists()
        and (path / "task" / "feature_manifest.json").exists()
    )


def export_one_database(
    db_dir: Path,
    out_dir: Path,
    overwrite: bool,
    include_hidden_columns: bool,
) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    schema = read_json(db_dir / "schema" / "schema.json")
    task_json = read_json(db_dir / "task" / "task.json")
    manifest = read_json(db_dir / "task" / "feature_manifest.json")
    spec = task_json.get("spec", task_json)
    if not isinstance(spec, Mapping):
        raise ValueError("task.json does not contain a mapping spec.")

    nodes = schema.get("nodes", {})
    if not isinstance(nodes, Mapping):
        raise ValueError("schema.json is missing nodes.")

    visible_tables = table_names(schema, manifest, include_hidden_columns)
    table_schemas: list[dict[str, Any]] = []
    table_frames: dict[str, pd.DataFrame] = {}
    for table_name in visible_tables:
        table_path = db_dir / "tables" / f"{table_name}.parquet"
        if not table_path.exists():
            raise FileNotFoundError(table_path)
        df = pd.read_parquet(table_path)
        visible_cols = column_names(table_name, df, manifest, include_hidden_columns)
        df = df[[col for col in visible_cols if col in df.columns]].copy()
        if df.empty and len(df.columns) == 0:
            raise ValueError(f"No visible columns left for table {table_name}.")
        df.to_parquet(out_dir / "data" / f"{table_name}.parquet", index=False)
        table_frames[table_name] = df
        node = nodes.get(table_name, {})
        table_schemas.append(
            {
                "name": table_name,
                "source": f"data/{table_name}.parquet",
                "format": "parquet",
                "columns": [
                    column_schema(table_name, col, df[col], node)
                    for col in df.columns
                ],
                "time_column": time_column_for_table(node, df.columns),
            }
        )

    task_meta = export_task(
        db_dir=db_dir,
        out_dir=out_dir,
        spec=spec,
        manifest=manifest,
        table_schemas=table_schemas,
        table_frames=table_frames,
    )
    metadata = {
        "dataset_name": db_dir.name,
        "tables": table_schemas,
        "tasks": [task_meta],
    }
    (out_dir / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def table_names(
    schema: Mapping[str, Any],
    manifest: Mapping[str, Any],
    include_hidden_columns: bool,
) -> list[str]:
    if include_hidden_columns:
        nodes = schema.get("nodes", {})
        return [str(name) for name in sorted(nodes)]
    visible = manifest.get("visible_tables", [])
    return [str(name) for name in visible]


def column_names(
    table_name: str,
    df: pd.DataFrame,
    manifest: Mapping[str, Any],
    include_hidden_columns: bool,
) -> list[str]:
    if include_hidden_columns:
        return [str(col) for col in df.columns]
    visible_columns = manifest.get("visible_columns", {})
    if isinstance(visible_columns, Mapping) and table_name in visible_columns:
        return [str(col) for col in visible_columns[table_name]]
    return [str(col) for col in df.columns]


def export_task(
    db_dir: Path,
    out_dir: Path,
    spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    table_schemas: Sequence[Mapping[str, Any]],
    table_frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    task_id = str(spec.get("task_id") or db_dir.name)
    prediction_table = str(
        spec.get("prediction_unit_table")
        or manifest.get("prediction_unit_table")
        or spec.get("target_source_table")
    )
    prediction_pk = str(spec.get("prediction_unit_pk") or "prediction_id")
    label_col = str(spec.get("label_col") or "label")
    if prediction_table not in table_frames:
        raise ValueError(f"Prediction table {prediction_table!r} is not exported as a visible table.")
    prediction_df = table_frames[prediction_table]
    if prediction_pk not in prediction_df.columns:
        raise ValueError(f"Prediction table {prediction_table!r} is missing pk {prediction_pk!r}.")

    labels = pd.read_parquet(db_dir / "task" / "labels.parquet")
    required_labels = {"prediction_id", label_col, "cutoff_time"}
    missing = required_labels - set(labels.columns)
    if missing:
        raise ValueError(f"labels.parquet is missing required columns: {sorted(missing)}")

    task_feature_cols = task_feature_columns(
        prediction_table=prediction_table,
        prediction_pk=prediction_pk,
        prediction_df=prediction_df,
        manifest=manifest,
    )
    feature_lookup = prediction_df[[prediction_pk, *task_feature_cols]].drop_duplicates(prediction_pk)
    feature_lookup = feature_lookup.set_index(prediction_pk, drop=True)

    split_dir = out_dir / "tasks" / task_id
    split_dir.mkdir(parents=True, exist_ok=True)
    task_column_order = [prediction_pk, *task_feature_cols, "cutoff_time", label_col]
    for syn_split, dbinfer_split in SYN_TO_DBINFER_SPLIT.items():
        split_path = db_dir / "task" / "splits" / f"{syn_split}.parquet"
        if not split_path.exists():
            continue
        split_df = pd.read_parquet(split_path)
        merged = split_df[["prediction_id"]].merge(labels, on="prediction_id", how="left")
        if merged[label_col].isna().any():
            raise ValueError(f"Split {syn_split!r} contains prediction ids missing from labels.")
        task_df = pd.DataFrame({prediction_pk: merged["prediction_id"].to_numpy()})
        if task_feature_cols:
            joined_features = feature_lookup.loc[task_df[prediction_pk]].reset_index(drop=True)
            for col in task_feature_cols:
                task_df[col] = joined_features[col].to_numpy()
        task_df["cutoff_time"] = pd.to_datetime(merged["cutoff_time"])
        task_df[label_col] = pd.to_numeric(merged[label_col], errors="coerce").fillna(0).astype("int8")
        task_df = task_df[task_column_order]
        task_df.to_parquet(split_dir / f"{dbinfer_split}.parquet", index=False)

    table_schema_lookup = {tbl["name"]: tbl for tbl in table_schemas}
    prediction_schema = table_schema_lookup[prediction_table]
    prediction_columns = {
        col["name"]: col
        for col in prediction_schema["columns"]
    }
    task_columns = [dict(prediction_columns[prediction_pk])]
    task_columns.extend(dict(prediction_columns[col]) for col in task_feature_cols)
    task_columns.extend(
        [
            {"name": "cutoff_time", "dtype": "datetime"},
            {"name": label_col, "dtype": "category", "num_categories": 2},
        ]
    )
    return {
        "name": task_id,
        "source": f"tasks/{task_id}/{{split}}.parquet",
        "format": "parquet",
        "columns": task_columns,
        "time_column": "cutoff_time",
        "evaluation_metric": map_metric(str(spec.get("metric") or "roc_auc")),
        "target_column": label_col,
        "target_table": prediction_table,
        "task_type": map_task_type(str(spec.get("task_type") or "binary_classification")),
        "num_classes": 2,
    }


def task_feature_columns(
    prediction_table: str,
    prediction_pk: str,
    prediction_df: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> list[str]:
    visible_columns = manifest.get("visible_columns", {})
    if isinstance(visible_columns, Mapping) and prediction_table in visible_columns:
        candidates = [str(col) for col in visible_columns[prediction_table]]
    else:
        candidates = [str(col) for col in prediction_df.columns]
    excluded = {prediction_pk, "label", "outcome_time", "split"}
    return [
        col
        for col in candidates
        if col in prediction_df.columns
        and col not in excluded
        and not col.startswith("__")
    ]


def column_schema(
    table_name: str,
    column_name: str,
    series: pd.Series,
    node: Mapping[str, Any],
) -> dict[str, Any]:
    schema_col = schema_column(node, column_name)
    semantic_type = str(schema_col.get("semantic_type") or "").lower()
    raw_dtype = str(schema_col.get("dtype") or "").lower()

    if bool(schema_col.get("is_primary_key")) or column_name == node.get("primary_key"):
        return {
            "name": column_name,
            "dtype": "primary_key",
            "capacity": int(series.nunique(dropna=True)),
        }
    if bool(schema_col.get("is_foreign_key")) or semantic_type == "foreign_key":
        refs = schema_col.get("references") or {}
        if not refs:
            raise ValueError(f"{table_name}.{column_name} is a foreign key without references.")
        return {
            "name": column_name,
            "dtype": "foreign_key",
            "link_to": f"{refs['table']}.{refs['column']}",
            "capacity": int(series.nunique(dropna=True)),
        }
    if bool(schema_col.get("is_time")) or semantic_type == "timestamp" or is_datetime_series(series):
        return {"name": column_name, "dtype": "datetime"}
    if semantic_type == "categorical" or raw_dtype == "category" or is_categorical_series(series):
        return {
            "name": column_name,
            "dtype": "category",
            "num_categories": max(1, int(series.nunique(dropna=True))),
        }
    if pd.api.types.is_numeric_dtype(series):
        return {"name": column_name, "dtype": "float", "in_size": 1}
    return {
        "name": column_name,
        "dtype": "category",
        "num_categories": max(1, int(series.nunique(dropna=True))),
    }


def schema_column(node: Mapping[str, Any], column_name: str) -> Mapping[str, Any]:
    for col in node.get("columns", []):
        if str(col.get("name")) == column_name:
            return col
    return {"name": column_name}


def time_column_for_table(node: Mapping[str, Any], columns: Iterable[str]) -> str | None:
    time_col = node.get("time_col")
    if time_col and time_col in set(columns):
        return str(time_col)
    for col in node.get("columns", []):
        name = str(col.get("name"))
        if bool(col.get("is_time")) and name in set(columns):
            return name
    return None


def is_datetime_series(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def is_categorical_series(series: pd.Series) -> bool:
    return (
        isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    )


def map_metric(metric: str) -> str:
    return METRIC_MAP.get(metric.lower(), metric)


def map_task_type(task_type: str) -> str:
    lowered = task_type.lower()
    if "classification" in lowered or lowered in {"binary", "multiclass"}:
        return "classification"
    if "regression" in lowered:
        return "regression"
    return lowered


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
