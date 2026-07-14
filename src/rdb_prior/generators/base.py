"""Base classes and utilities for synthetic relational table generation.

The generators consume an already sampled relational schema. FK edges are treated
as schema-level join support only; feature generation may use parent tables as
context, but this module does not interpret FK edges as causal edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import random

import numpy as np
import pandas as pd


SchemaLike = Any
NodeLike = Any
FKLike = Any


def read_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read either a mapping key or an object attribute."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def node_id(node: NodeLike) -> str:
    value = read_attr(node, "node_id")
    if value is None:
        raise ValueError("Schema node is missing node_id.")
    return str(value)


def node_role(node: NodeLike) -> str:
    value = read_attr(node, "role")
    if value is None:
        raise ValueError(f"Schema node {node_id(node)!r} is missing role.")
    return str(value)


def node_primary_key(node: NodeLike) -> str:
    value = read_attr(node, "primary_key")
    if value is None:
        return f"{node_id(node).lower()}_id"
    return str(value)


def node_time_col(node: NodeLike) -> Optional[str]:
    value = read_attr(node, "time_col")
    return None if value in (None, "") else str(value)


def node_num_rows(node: NodeLike) -> int:
    value = read_attr(node, "num_rows", 0)
    return max(0, int(value))


def node_columns(node: NodeLike) -> List[Any]:
    value = read_attr(node, "columns", [])
    return list(value or [])


def column_name(column: Any) -> str:
    value = read_attr(column, "name")
    if value is None:
        raise ValueError("Column spec is missing name.")
    return str(value)


def column_dtype(column: Any) -> str:
    return str(read_attr(column, "dtype", ""))


def column_semantic_type(column: Any) -> str:
    return str(read_attr(column, "semantic_type", ""))


def column_is_primary_key(column: Any) -> bool:
    return bool(read_attr(column, "is_primary_key", False))


def column_is_foreign_key(column: Any) -> bool:
    return bool(read_attr(column, "is_foreign_key", False))


def column_is_time(column: Any) -> bool:
    return bool(read_attr(column, "is_time", False))


def column_is_label(column: Any) -> bool:
    return bool(read_attr(column, "is_label_candidate", False))


def schema_nodes(schema: SchemaLike) -> Dict[str, NodeLike]:
    nodes = read_attr(schema, "nodes", {})
    if isinstance(nodes, Mapping):
        return {str(k): v for k, v in nodes.items()}
    result: Dict[str, NodeLike] = {}
    for node in nodes or []:
        result[node_id(node)] = node
    return result


def schema_foreign_keys(schema: SchemaLike) -> List[FKLike]:
    return list(read_attr(schema, "foreign_keys", []) or [])


def fk_parent_table(fk: FKLike) -> str:
    value = read_attr(fk, "parent_table")
    if value is None:
        raise ValueError("Foreign key spec is missing parent_table.")
    return str(value)


def fk_child_table(fk: FKLike) -> str:
    value = read_attr(fk, "child_table")
    if value is None:
        raise ValueError("Foreign key spec is missing child_table.")
    return str(value)


def fk_parent_col(fk: FKLike) -> str:
    value = read_attr(fk, "parent_col")
    if value is None:
        raise ValueError("Foreign key spec is missing parent_col.")
    return str(value)


def fk_child_col(fk: FKLike) -> str:
    value = read_attr(fk, "child_col")
    if value is None:
        raise ValueError("Foreign key spec is missing child_col.")
    return str(value)


def foreign_keys_for_child(schema: SchemaLike, child_table: str) -> List[FKLike]:
    return [fk for fk in schema_foreign_keys(schema) if fk_child_table(fk) == child_table]


def choose_time_column(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return str(col)
    for col in df.columns:
        name = str(col).lower()
        if name.endswith("_time") or name in {"event_time", "measurement_time", "as_of_time", "outcome_time"}:
            return str(col)
    return None


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        name = str(col)
        if name.startswith("__") or name.endswith("_id") or name == "label":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(name)
    return cols


def json_ready(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-serializable structures."""
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class GenerationContext:
    """Shared state and temporal settings for table generation."""

    seed: int = 42
    start_time: Any = "2020-01-01"
    end_time: Any = "2022-01-01"
    cutoff_time: Optional[Any] = None
    future_window_days: int = 30
    history_window_days: int = 365
    enable_leakage_guard: bool = True
    row_scale: float = 1.0
    max_rows_per_table: Optional[int] = None
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    table_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.start_time = pd.Timestamp(self.start_time)
        self.end_time = pd.Timestamp(self.end_time)
        if self.cutoff_time is None:
            self.cutoff_time = self.end_time - pd.Timedelta(days=int(self.future_window_days))
        else:
            self.cutoff_time = pd.Timestamp(self.cutoff_time)
        if self.start_time >= self.end_time:
            raise ValueError("GenerationContext requires start_time < end_time.")
        if not (self.start_time <= self.cutoff_time <= self.end_time):
            raise ValueError("cutoff_time must be within [start_time, end_time].")
        if self.row_scale <= 0:
            raise ValueError("row_scale must be positive.")
        if self.max_rows_per_table is not None and self.max_rows_per_table <= 0:
            raise ValueError("max_rows_per_table must be positive when provided.")

    def planned_rows(self, requested_rows: int) -> int:
        requested = max(0, int(requested_rows))
        value = int(round(requested * float(self.row_scale)))
        if requested > 0:
            value = max(1, value)
        if self.max_rows_per_table is not None:
            value = min(value, int(self.max_rows_per_table))
        return max(0, value)

    def register_table(self, generated: "GeneratedTable") -> None:
        self.tables[generated.table_id] = generated.dataframe
        self.table_metadata[generated.table_id] = dict(generated.metadata)


@dataclass
class GeneratedTable:
    """A generated DataFrame and its table-level metadata."""

    table_id: str
    role: str
    dataframe: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTableGenerator:
    """Base class for role-specific table generators.

    Subclasses implement role-specific table construction. The base class only
    provides schema access and generic sampling utilities.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None, seed: Optional[int] = None):
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.seed = seed
        self.py_random = random.Random(seed)

    def generate(
        self,
        node: NodeLike,
        schema: SchemaLike,
        parent_tables: Mapping[str, pd.DataFrame],
        fk_specs: Sequence[FKLike],
        context: GenerationContext,
        scm_assignment: Optional[Mapping[str, Any]] = None,
    ) -> GeneratedTable:
        raise NotImplementedError

    def make_primary_key(self, table_id: str, n_rows: int, pk_col: str) -> pd.Series:
        del table_id
        return pd.Series(np.arange(int(n_rows), dtype=np.int64), name=pk_col)

    def sample_foreign_key(
        self,
        parent_df: pd.DataFrame,
        parent_pk: str,
        n_rows: int,
        long_tail: bool = True,
    ) -> np.ndarray:
        if parent_pk not in parent_df.columns:
            raise ValueError(f"Parent table is missing primary key column {parent_pk!r}.")
        if len(parent_df) == 0:
            raise ValueError(f"Cannot sample foreign key from empty parent table for {parent_pk!r}.")

        values = parent_df[parent_pk].to_numpy()
        weights = None
        if long_tail:
            weights = self._activity_weights(parent_df)
        sampled = self.rng.choice(values, size=int(n_rows), replace=True, p=weights)
        return sampled

    def _activity_weights(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if "__activity_score" not in df.columns or len(df) == 0:
            return None
        weights = pd.to_numeric(df["__activity_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        if total <= 0:
            return None
        return weights / total

    def sample_numeric(self, n_rows: int, distribution: str = "normal") -> np.ndarray:
        n = int(n_rows)
        if distribution == "normal":
            values = self.rng.normal(loc=0.0, scale=1.0, size=n)
        elif distribution == "lognormal":
            values = self.rng.lognormal(mean=0.0, sigma=0.8, size=n)
        elif distribution == "uniform":
            values = self.rng.uniform(0.0, 1.0, size=n)
        elif distribution == "poisson":
            values = self.rng.poisson(lam=2.0, size=n).astype(float)
        else:
            raise ValueError(f"Unknown numeric distribution: {distribution!r}.")
        return values.astype("float32")

    def sample_categorical(
        self,
        n_rows: int,
        cardinality: int = 10,
        long_tail: bool = True,
    ) -> pd.Categorical:
        n = int(n_rows)
        k = max(1, int(cardinality))
        labels = np.array([f"c{i}" for i in range(k)], dtype=object)
        weights = None
        if long_tail:
            ranks = np.arange(1, k + 1, dtype=float)
            weights = 1.0 / ranks
            weights = weights / weights.sum()
        values = self.rng.choice(labels, size=n, replace=True, p=weights)
        return pd.Categorical(values, categories=list(labels))

    def sample_timestamps(
        self,
        n_rows: int,
        start: Any,
        end: Any,
        bursty: bool = False,
    ) -> pd.Series:
        n = int(n_rows)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if n <= 0:
            return pd.Series(pd.to_datetime([]))
        if start_ts >= end_ts:
            return pd.Series(pd.to_datetime([start_ts] * n))

        total_seconds = max(1, int((end_ts - start_ts).total_seconds()))
        if bursty and n > 0:
            centers = self.rng.integers(0, total_seconds, size=min(8, max(1, n)))
            chosen = self.rng.choice(centers, size=n, replace=True)
            jitter = self.rng.normal(loc=0.0, scale=max(1.0, total_seconds / 80.0), size=n).astype(int)
            offsets = np.clip(chosen + jitter, 0, total_seconds)
        else:
            offsets = self.rng.integers(0, total_seconds + 1, size=n)
        return pd.Series(start_ts + pd.to_timedelta(offsets, unit="s"))

    def apply_missingness(self, df: pd.DataFrame, missing_rate: float) -> pd.DataFrame:
        rate = float(missing_rate)
        if rate <= 0 or len(df) == 0:
            return df
        if rate >= 1:
            raise ValueError("missing_rate must be < 1.")
        result = df.copy()
        for col in result.columns:
            name = str(col)
            if name.startswith("__") or name.endswith("_id") or name == "label":
                continue
            if pd.api.types.is_datetime64_any_dtype(result[col]):
                continue
            mask = self.rng.random(len(result)) < rate
            result.loc[mask, col] = np.nan
        return result

    def safe_sort_by_time(self, df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
        if time_col and time_col in df.columns:
            return df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
        return df.reset_index(drop=True)

    def add_foreign_keys(
        self,
        df: pd.DataFrame,
        fk_specs: Sequence[FKLike],
        parent_tables: Mapping[str, pd.DataFrame],
        n_rows: int,
        long_tail: bool = True,
    ) -> pd.DataFrame:
        result = df.copy()
        for fk in fk_specs:
            parent_id = fk_parent_table(fk)
            parent_df = parent_tables.get(parent_id)
            if parent_df is None:
                raise ValueError(f"Parent table {parent_id!r} has not been generated.")
            result[fk_child_col(fk)] = self.sample_foreign_key(
                parent_df=parent_df,
                parent_pk=fk_parent_col(fk),
                n_rows=n_rows,
                long_tail=long_tail,
            )
        return result

    def add_schema_feature_columns(
        self,
        df: pd.DataFrame,
        node: NodeLike,
        context: GenerationContext,
        skip_labels: bool = True,
    ) -> pd.DataFrame:
        result = df.copy()
        n_rows = len(result)
        for column in node_columns(node):
            name = column_name(column)
            if name in result.columns:
                continue
            if column_is_primary_key(column) or column_is_foreign_key(column):
                continue
            if skip_labels and column_is_label(column):
                continue
            semantic = column_semantic_type(column).lower()
            dtype = column_dtype(column).lower()
            if column_is_time(column) or "datetime" in dtype or semantic == "timestamp":
                result[name] = self.sample_timestamps(n_rows, context.start_time, context.end_time)
            elif semantic == "categorical" or "category" in dtype:
                result[name] = self.sample_categorical(n_rows, cardinality=12, long_tail=True)
            elif semantic in {"numeric", "outcome_label"} or any(token in dtype for token in ("float", "int")):
                result[name] = self.sample_numeric(n_rows, distribution="normal")
            else:
                result[name] = self.sample_categorical(n_rows, cardinality=8, long_tail=False)
        return result

    def generated_columns_metadata(self, df: pd.DataFrame) -> List[str]:
        return [str(col) for col in df.columns]