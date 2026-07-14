"""Export generated task bundles into RDBPFN HDF5 prior dumps.

RDBPFN pretraining consumes in-context tabular episodes stored in HDF5. This
module adapts synthetic relational task bundles into that format. It keeps the
relational database/task generation code independent from RDBPFN training code:
the only contract is the on-disk task bundle produced by ``task_exporter.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar

try:
    import h5py
except ImportError:  # pragma: no cover - exercised only in environments without h5py.
    h5py = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for CLI progress.
    tqdm = None

import numpy as np
import pandas as pd


T = TypeVar("T")


@dataclass(frozen=True)
class RDBPFNExportConfig:
    """Configuration for converting task bundles to RDBPFN HDF5 episodes."""

    total_rows: int = 256
    max_columns: int = 128
    seed: int = 42
    include_val_as_query: bool = True
    compression: str = "lzf"

    def __post_init__(self) -> None:
        if self.total_rows < 2:
            raise ValueError("total_rows must be at least 2.")
        if self.max_columns <= 0:
            raise ValueError("max_columns must be positive.")
        if self.compression not in {"lzf", "gzip", None}:
            raise ValueError("compression must be 'lzf', 'gzip', or None.")


@dataclass
class RDBPFNSample:
    """One RDBPFN training episode."""

    dataset_id: str
    task_id: str
    x: np.ndarray
    y: np.ndarray
    num_features: int
    num_available_features: int
    split_idx: int
    category_mask: np.ndarray


def export_database_root_to_h5(
    database_root: Any,
    output_path: Any,
    config: Optional[RDBPFNExportConfig] = None,
) -> Dict[str, Any]:
    """Discover database directories under ``database_root`` and write HDF5."""
    root = Path(database_root)
    if not root.exists():
        raise FileNotFoundError(f"database_root does not exist: {root}")
    database_dirs = discover_task_database_dirs(root)
    return export_task_directories_to_h5(database_dirs, output_path, config=config)


def export_task_directories_to_h5(
    database_dirs: Sequence[Any],
    output_path: Any,
    config: Optional[RDBPFNExportConfig] = None,
) -> Dict[str, Any]:
    """Convert one or more generated DB task directories into one HDF5 file."""
    cfg = config or RDBPFNExportConfig()
    dirs = [Path(path) for path in database_dirs]
    if not dirs:
        raise ValueError("database_dirs must contain at least one directory.")

    samples: List[RDBPFNSample] = []
    skipped: List[Dict[str, str]] = []
    for idx, db_dir in enumerate(_progress(dirs, desc="Building RDBPFN samples", unit="db")):
        try:
            sample = build_rdbpfn_sample(db_dir, cfg, sample_seed=cfg.seed + idx)
        except Exception as exc:
            skipped.append({"database_dir": str(db_dir), "reason": str(exc)})
            continue
        samples.append(sample)

    if not samples:
        preview = skipped[:5]
        raise RuntimeError(f"No RDBPFN samples were built. Skipped preview: {preview}")

    write_rdbpfn_h5(samples=samples, output_path=output_path, config=cfg)
    return {
        "output_path": str(Path(output_path)),
        "num_samples": len(samples),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "total_rows": cfg.total_rows,
        "max_columns": cfg.max_columns,
    }


def discover_task_database_dirs(database_root: Any) -> List[Path]:
    """Return generated DB directories that contain task and table artifacts."""
    root = Path(database_root)
    candidates: List[Path] = []
    if _is_task_database_dir(root):
        candidates.append(root)
    for path in sorted(root.iterdir()) if root.exists() else []:
        if path.is_dir() and _is_task_database_dir(path):
            candidates.append(path)
    return candidates


def build_rdbpfn_sample(
    database_dir: Any,
    config: RDBPFNExportConfig,
    sample_seed: Optional[int] = None,
) -> RDBPFNSample:
    """Build one RDBPFN episode from a generated database directory."""
    db_dir = Path(database_dir)
    task_dir = db_dir / "task"
    tables_dir = db_dir / "tables"
    task_json = _read_json(task_dir / "task.json")
    manifest = _read_json(task_dir / "feature_manifest.json")
    labels = pd.read_parquet(task_dir / "labels.parquet")

    spec = _task_spec(task_json)
    task_id = str(spec.get("task_id", db_dir.name))
    prediction_table = str(spec.get("prediction_unit_table") or manifest.get("prediction_unit_table"))
    prediction_pk = str(spec.get("prediction_unit_pk") or "prediction_id")
    if not prediction_table:
        raise ValueError("Task spec is missing prediction_unit_table.")

    table_path = tables_dir / f"{prediction_table}.parquet"
    if not table_path.exists():
        raise FileNotFoundError(f"Prediction unit table parquet not found: {table_path}")
    table_df = pd.read_parquet(table_path)
    if prediction_pk not in table_df.columns:
        raise ValueError(f"Prediction table {prediction_table!r} is missing pk {prediction_pk!r}.")

    visible_columns = _visible_columns_for_table(manifest, prediction_table, table_df)
    feature_columns = [
        col
        for col in visible_columns
        if col in table_df.columns
        and col != prediction_pk
        and col not in {"label", "outcome_time", "split"}
        and not str(col).startswith("__")
    ]
    if not feature_columns:
        raise ValueError(f"No usable visible feature columns for table {prediction_table!r}.")

    encoded, category_mask, total_available_features = _encode_feature_frame(
        table_df[[prediction_pk, *feature_columns]],
        pk_col=prediction_pk,
        feature_columns=feature_columns,
        max_columns=config.max_columns,
    )
    feature_lookup = encoded.set_index(prediction_pk, drop=True)
    labels = _prepare_labels(labels)
    labels = labels[labels["prediction_id"].isin(feature_lookup.index)].copy()
    if labels.empty:
        raise ValueError("No labels can be joined to prediction table features.")

    rng = np.random.default_rng(config.seed if sample_seed is None else int(sample_seed))
    train_labels = labels[labels["split"] == "train"]
    query_splits = ["test"]
    if config.include_val_as_query:
        query_splits.insert(0, "val")
    query_labels = labels[labels["split"].isin(query_splits)]
    if train_labels.empty or query_labels.empty:
        train_labels, query_labels = _fallback_train_query_split(labels, rng)

    selected = _sample_episode_rows(
        train_labels=train_labels,
        query_labels=query_labels,
        total_rows=config.total_rows,
        rng=rng,
    )
    split_idx = int((selected["split"] == "train").sum())
    if split_idx <= 0 or split_idx >= config.total_rows:
        raise ValueError("RDBPFN episode requires at least one train row and one query row.")

    x = feature_lookup.loc[selected["prediction_id"]].to_numpy(dtype=np.float32, copy=True)
    y = selected["label"].to_numpy(dtype=np.int32, copy=True)
    _ensure_binary_labels(y, rng)

    return RDBPFNSample(
        dataset_id=db_dir.name,
        task_id=task_id,
        x=x,
        y=y,
        num_features=x.shape[1],
        num_available_features=total_available_features,
        split_idx=split_idx,
        category_mask=category_mask[: x.shape[1]].astype(np.uint8, copy=False),
    )


def write_rdbpfn_h5(
    samples: Sequence[RDBPFNSample],
    output_path: Any,
    config: RDBPFNExportConfig,
) -> None:
    """Write samples using the HDF5 fields consumed by RDBPFN dataloaders."""
    if h5py is None:
        raise ImportError("rdbpfn_exporter requires h5py to write RDBPFN HDF5 files.")
    if not samples:
        raise ValueError("samples must not be empty.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    total = len(samples)
    chunk_rows = min(4, total)

    with h5py.File(output, "w") as h5:
        dset_x = h5.create_dataset(
            "X",
            shape=(total, config.total_rows, config.max_columns),
            dtype="float32",
            compression=config.compression,
            chunks=(chunk_rows, config.total_rows, config.max_columns),
        )
        dset_y = h5.create_dataset(
            "y",
            shape=(total, config.total_rows),
            dtype="int32",
            compression=config.compression,
            chunks=(chunk_rows, config.total_rows),
        )
        dset_num_features = h5.create_dataset("num_features", shape=(total,), dtype="int32")
        dset_num_available = h5.create_dataset(
            "num_available_features",
            shape=(total,),
            dtype="int32",
        )
        dset_num_rows = h5.create_dataset("num_datapoints", shape=(total,), dtype="int32")
        dset_split = h5.create_dataset("single_eval_pos", shape=(total,), dtype="int32")
        dset_category = h5.create_dataset(
            "feature_is_categorical",
            shape=(total, config.max_columns),
            dtype="uint8",
            compression=config.compression,
        )
        h5.create_dataset("max_num_classes", data=np.array([2], dtype=np.int32))

        dataset_ids = []
        task_ids = []
        for idx, sample in enumerate(_progress(samples, desc="Writing RDBPFN H5", unit="sample")):
            num_features = int(sample.num_features)
            if num_features <= 0:
                raise ValueError(f"Sample {sample.task_id!r} has no features.")
            if sample.x.shape != (config.total_rows, num_features):
                raise ValueError(f"Sample {sample.task_id!r} has invalid X shape {sample.x.shape}.")
            if sample.y.shape != (config.total_rows,):
                raise ValueError(f"Sample {sample.task_id!r} has invalid y shape {sample.y.shape}.")
            dset_x[idx, :, :num_features] = sample.x.astype(np.float32, copy=False)
            dset_y[idx, :] = sample.y.astype(np.int32, copy=False)
            dset_num_features[idx] = num_features
            dset_num_available[idx] = int(sample.num_available_features)
            dset_num_rows[idx] = config.total_rows
            dset_split[idx] = int(sample.split_idx)
            dset_category[idx, :num_features] = sample.category_mask[:num_features]
            dataset_ids.append(sample.dataset_id)
            task_ids.append(sample.task_id)

        string_dtype = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("dataset_id", data=np.asarray(dataset_ids, dtype=object), dtype=string_dtype)
        h5.create_dataset("task_id", data=np.asarray(task_ids, dtype=object), dtype=string_dtype)


def _progress(iterable: Iterable[T], desc: str, unit: str) -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, ascii=True, file=sys.stdout, dynamic_ncols=True)


def _is_task_database_dir(path: Path) -> bool:
    return (
        (path / "task" / "task.json").exists()
        and (path / "task" / "labels.parquet").exists()
        and (path / "task" / "feature_manifest.json").exists()
        and (path / "tables").is_dir()
    )


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _task_spec(task_json: Mapping[str, Any]) -> Dict[str, Any]:
    spec = task_json.get("spec", task_json)
    if not isinstance(spec, Mapping):
        raise ValueError("task.json does not contain a mapping spec.")
    return {str(k): v for k, v in spec.items()}


def _visible_columns_for_table(
    manifest: Mapping[str, Any],
    table_id: str,
    table_df: pd.DataFrame,
) -> List[str]:
    visible_columns = manifest.get("visible_columns", {})
    if isinstance(visible_columns, Mapping) and table_id in visible_columns:
        return [str(col) for col in visible_columns[table_id]]
    visible_tables = set(str(t) for t in manifest.get("visible_tables", []))
    if table_id in visible_tables:
        return [str(col) for col in table_df.columns]
    raise ValueError(f"Prediction table {table_id!r} is not visible in feature_manifest.")


def _encode_feature_frame(
    df: pd.DataFrame,
    pk_col: str,
    feature_columns: Sequence[str],
    max_columns: int,
) -> Tuple[pd.DataFrame, np.ndarray, int]:
    selected = list(feature_columns)[:max_columns]
    result = pd.DataFrame({pk_col: df[pk_col].to_numpy()})
    category_mask = np.zeros(len(selected), dtype=np.uint8)

    for idx, col in enumerate(selected):
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            values = pd.to_datetime(series, errors="coerce").astype("int64").astype("float64")
            values = _standardize(values)
            result[col] = values.astype("float32")
        elif pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
            result[col] = _standardize_with_nan(values).astype("float32")
        else:
            category_mask[idx] = 1
            result[col] = _encode_categorical(series).astype("float32")

    return result, category_mask, len(feature_columns)


def _standardize_with_nan(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float64, copy=True)
    finite = np.isfinite(arr)
    fill = float(np.nanmean(arr[finite])) if finite.any() else 0.0
    arr[~finite] = fill
    return _standardize(arr)


def _standardize(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float64, copy=True)
    std = float(np.std(arr))
    if std <= 1e-8:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - float(np.mean(arr))) / std


def _encode_categorical(series: pd.Series) -> np.ndarray:
    values = series.astype("object").where(series.notna(), "__missing__")
    codes = pd.Categorical(values).codes.astype(np.float64)
    if codes.size == 0:
        return codes
    return codes


def _prepare_labels(labels: pd.DataFrame) -> pd.DataFrame:
    required = {"prediction_id", "label", "split"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels.parquet is missing required columns: {sorted(missing)}")
    result = labels.copy()
    result["label"] = pd.to_numeric(result["label"], errors="coerce").fillna(0).astype("int32")
    result["label"] = (result["label"] > 0).astype("int32")
    result["split"] = result["split"].astype(str)
    return result


def _fallback_train_query_split(
    labels: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = labels.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
    split = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * 0.7))))
    train = shuffled.iloc[:split].copy()
    query = shuffled.iloc[split:].copy()
    train["split"] = "train"
    query["split"] = "test"
    return train, query


def _sample_episode_rows(
    train_labels: pd.DataFrame,
    query_labels: pd.DataFrame,
    total_rows: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    train_rows = max(1, int(round(total_rows * 0.7)))
    train_rows = min(train_rows, total_rows - 1)
    query_rows = total_rows - train_rows
    train_sample = _sample_rows(train_labels, train_rows, rng)
    query_sample = _sample_rows(query_labels, query_rows, rng)
    train_sample = train_sample.copy()
    query_sample = query_sample.copy()
    train_sample["split"] = "train"
    query_sample["split"] = "test"
    return pd.concat([train_sample, query_sample], axis=0, ignore_index=True)


def _sample_rows(df: pd.DataFrame, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    replace = len(df) < n_rows
    indices = rng.choice(df.index.to_numpy(), size=n_rows, replace=replace)
    return df.loc[indices].reset_index(drop=True)


def _ensure_binary_labels(y: np.ndarray, rng: np.random.Generator) -> None:
    if y.size < 2:
        return
    if int(y.max()) == int(y.min()):
        y[int(rng.integers(0, y.size))] = 1 - int(y[0])


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export synthetic task bundles to RDBPFN HDF5.")
    parser.add_argument("--database-root", type=Path, required=True, help="Root containing db_* directories.")
    parser.add_argument("--output", type=Path, required=True, help="Output .h5 path.")
    parser.add_argument("--total-rows", type=int, default=256, help="Rows per RDBPFN episode.")
    parser.add_argument("--max-columns", type=int, default=128, help="Maximum feature columns per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--exclude-val-query",
        action="store_true",
        help="Use only test split as query rows instead of val+test.",
    )
    parser.add_argument(
        "--compression",
        default="lzf",
        choices=["lzf", "gzip", "none"],
        help="HDF5 dataset compression.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    compression = None if args.compression == "none" else args.compression
    config = RDBPFNExportConfig(
        total_rows=args.total_rows,
        max_columns=args.max_columns,
        seed=args.seed,
        include_val_as_query=not args.exclude_val_query,
        compression=compression,
    )
    report = export_database_root_to_h5(
        database_root=args.database_root,
        output_path=args.output,
        config=config,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "RDBPFNExportConfig",
    "RDBPFNSample",
    "discover_task_database_dirs",
    "build_rdbpfn_sample",
    "export_database_root_to_h5",
    "export_task_directories_to_h5",
    "write_rdbpfn_h5",
]
