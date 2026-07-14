"""Label and split generation for relational task bundles."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..generators.base import (
    SchemaLike,
    choose_time_column,
    json_ready,
    node_primary_key,
    node_role,
    schema_nodes,
)
from .task_sampler import TaskBundle, TaskSpec


class LabelGenerator:
    """Build labels, splits, and feature manifests from a task specification."""

    def __init__(self, seed: int = 42, positive_rate: float = 0.35):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.positive_rate = float(positive_rate)

    def build_task_bundle(
        self,
        spec: TaskSpec,
        schema: SchemaLike,
        tables: Mapping[str, pd.DataFrame],
        metadata: Mapping[str, Any],
    ) -> TaskBundle:
        if spec.label_source_mode == "existing_outcome_label":
            labels, derived_signal_columns = self._build_outcome_labels(spec, tables)
        elif spec.label_source_mode == "derived_table_label":
            labels, derived_signal_columns = self._build_derived_labels(spec, schema, tables, metadata)
        else:
            raise ValueError(f"Unknown label_source_mode: {spec.label_source_mode!r}.")

        labels = labels.sort_values("prediction_id", kind="mergesort").reset_index(drop=True)
        labels, splits = self._assign_splits(labels, spec)
        feature_manifest = self._build_feature_manifest(
            spec=spec,
            schema=schema,
            tables=tables,
            metadata=metadata,
            derived_signal_columns=derived_signal_columns,
        )
        bundle_metadata = json_ready(
            {
                "task_id": spec.task_id,
                "num_labels": len(labels),
                "positive_rate": float(labels["label"].mean()) if len(labels) else 0.0,
                "label_source_mode": spec.label_source_mode,
                "derived_from_table": (
                    spec.target_source_table if spec.label_source_mode == "derived_table_label" else None
                ),
                "derived_signal_columns": list(derived_signal_columns),
                "seed": self.seed,
            }
        )
        return TaskBundle(
            spec=spec,
            labels=labels,
            splits=splits,
            feature_manifest=feature_manifest,
            metadata=bundle_metadata,
        )

    def _build_outcome_labels(
        self,
        spec: TaskSpec,
        tables: Mapping[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, List[str]]:
        if spec.target_source_table not in tables:
            raise ValueError(f"Target source table {spec.target_source_table!r} is missing from tables.")
        outcome_df = tables[spec.target_source_table]
        if spec.label_col not in outcome_df.columns:
            raise ValueError(
                f"Outcome table {spec.target_source_table!r} is missing label column {spec.label_col!r}."
            )
        if not spec.target_fk_col or spec.target_fk_col not in outcome_df.columns:
            raise ValueError(
                f"Outcome task requires FK column {spec.target_fk_col!r} on {spec.target_source_table!r}."
            )

        time_col = spec.target_time_col or choose_time_column(outcome_df, preferred="outcome_time")
        if time_col is None or time_col not in outcome_df.columns:
            raise ValueError(f"Outcome table {spec.target_source_table!r} is missing outcome time column.")

        cutoff = pd.Timestamp(spec.cutoff_time)
        future_end = cutoff + pd.Timedelta(days=int(spec.future_window_days))
        times = pd.to_datetime(outcome_df[time_col])
        invalid = (times <= cutoff) | (times > future_end)
        if bool(invalid.any()):
            raise ValueError(
                f"Outcome table {spec.target_source_table!r} contains labels outside the future window."
            )

        labels = pd.DataFrame(
            {
                "prediction_id": outcome_df[spec.target_fk_col].to_numpy(),
                "prediction_unit_table": spec.prediction_unit_table,
                "target_source_table": spec.target_source_table,
                "label": self._coerce_binary_label(outcome_df[spec.label_col]),
                "cutoff_time": cutoff,
                "outcome_time": times,
            }
        )
        if len(labels) == 0:
            raise ValueError("Outcome task produced no labels.")

        labels = (
            labels.groupby("prediction_id", as_index=False, sort=True)
            .agg(
                {
                    "prediction_unit_table": "first",
                    "target_source_table": "first",
                    "label": "max",
                    "cutoff_time": "first",
                    "outcome_time": "min",
                }
            )
            .reset_index(drop=True)
        )
        return labels[self._label_columns_without_split()], []

    def _build_derived_labels(
        self,
        spec: TaskSpec,
        schema: SchemaLike,
        tables: Mapping[str, pd.DataFrame],
        metadata: Mapping[str, Any],
    ) -> Tuple[pd.DataFrame, List[str]]:
        if spec.target_source_table not in tables:
            raise ValueError(f"Target source table {spec.target_source_table!r} is missing from tables.")
        nodes = schema_nodes(schema)
        source_df = tables[spec.target_source_table]
        source_df = self._eligible_source_rows(spec, source_df, metadata)
        if len(source_df) == 0:
            raise ValueError(f"Target source table {spec.target_source_table!r} produced no eligible rows.")

        pk_col = spec.prediction_unit_pk
        if pk_col not in source_df.columns:
            if spec.target_source_table in nodes:
                pk_col = node_primary_key(nodes[spec.target_source_table])
            if pk_col not in source_df.columns:
                raise ValueError(
                    f"Target source table {spec.target_source_table!r} is missing primary key {pk_col!r}."
                )

        label_values, signal_columns = self._derive_binary_values(source_df, pk_col)
        cutoff = pd.Timestamp(spec.cutoff_time)
        labels = pd.DataFrame(
            {
                "prediction_id": source_df[pk_col].to_numpy(),
                "prediction_unit_table": spec.prediction_unit_table,
                "target_source_table": spec.target_source_table,
                "label": label_values.astype("int8"),
                "cutoff_time": cutoff,
                "outcome_time": cutoff,
            }
        )
        return labels[self._label_columns_without_split()], signal_columns

    def _eligible_source_rows(
        self,
        spec: TaskSpec,
        source_df: pd.DataFrame,
        metadata: Mapping[str, Any],
    ) -> pd.DataFrame:
        cutoff = pd.Timestamp(spec.cutoff_time)
        result = source_df.copy()

        if spec.target_source_role == "event":
            time_col = spec.target_time_col or choose_time_column(result)
            if time_col is None or time_col not in result.columns:
                raise ValueError(f"Event target source {spec.target_source_table!r} requires a time column.")
            return result.loc[pd.to_datetime(result[time_col]) < cutoff].reset_index(drop=True)

        if spec.target_source_role == "summary":
            table_meta = self._table_metadata(metadata).get(spec.target_source_table, {})
            if not bool(table_meta.get("used_only_history_before_cutoff", False)):
                raise ValueError(f"Summary target source {spec.target_source_table!r} is not cutoff-safe.")
            time_col = spec.target_time_col or choose_time_column(result, preferred="as_of_time")
            if time_col is not None and time_col in result.columns:
                if bool((pd.to_datetime(result[time_col]) > cutoff).any()):
                    raise ValueError(f"Summary target source {spec.target_source_table!r} is after cutoff_time.")
            return result.reset_index(drop=True)

        if spec.target_source_role == "entity":
            time_col = spec.target_time_col or choose_time_column(result)
            if time_col is not None and time_col in result.columns:
                result = result.loc[pd.to_datetime(result[time_col]) <= cutoff]
            return result.reset_index(drop=True)

        return result.reset_index(drop=True)

    def _derive_binary_values(self, df: pd.DataFrame, pk_col: str) -> Tuple[np.ndarray, List[str]]:
        candidate_columns = self._signal_candidate_columns(df, pk_col)
        arrays: List[np.ndarray] = []
        signal_columns: List[str] = []
        for col in candidate_columns:
            encoded = self._encode_signal_column(df[col])
            if encoded is None:
                continue
            arrays.append(encoded)
            signal_columns.append(col)

        n_rows = len(df)
        if arrays:
            weights = np.linspace(1.0, 0.5, num=len(arrays), dtype=float)
            signal = np.zeros(n_rows, dtype=float)
            for weight, values in zip(weights, arrays):
                signal += weight * values
        else:
            signal = self.rng.normal(0.0, 1.0, size=n_rows)

        signal += self.rng.normal(0.0, 0.25, size=n_rows)
        labels = np.zeros(n_rows, dtype="int8")
        if n_rows == 1:
            labels[0] = int(signal[0] >= float(np.median(signal)))
            return labels, signal_columns

        positive_count = int(round(n_rows * self.positive_rate))
        positive_count = max(1, min(n_rows - 1, positive_count))
        order = np.argsort(signal, kind="mergesort")
        labels[order[-positive_count:]] = 1
        return labels, signal_columns

    def _signal_candidate_columns(self, df: pd.DataFrame, pk_col: str) -> List[str]:
        hidden: List[str] = []
        public: List[str] = []
        for col in df.columns:
            name = str(col)
            if name == pk_col or name.endswith("_id") or name in {"label", "outcome_time"}:
                continue
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            if name.startswith("__latent_"):
                hidden.append(name)
                continue
            if name.startswith("__") or name == "__activity_score":
                continue
            if (
                pd.api.types.is_numeric_dtype(df[col])
                or isinstance(df[col].dtype, pd.CategoricalDtype)
                or df[col].dtype == object
            ):
                public.append(name)

        return hidden[:2] + public[:4]

    @staticmethod
    def _encode_signal_column(series: pd.Series) -> Optional[np.ndarray]:
        if len(series) == 0:
            return None
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            values = pd.Categorical(series.astype("object").where(series.notna(), "__missing__")).codes.astype(float)
        std = float(np.std(values))
        if std <= 1e-8:
            return np.zeros(len(series), dtype=float)
        return (values - float(np.mean(values))) / std

    @staticmethod
    def _coerce_binary_label(series: pd.Series) -> np.ndarray:
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            return (numeric >= 0.5).astype("int8")
        codes = pd.Categorical(series.astype("object").where(series.notna(), "__missing__")).codes
        return (codes > 0).astype("int8")

    def _assign_splits(
        self,
        labels: pd.DataFrame,
        spec: TaskSpec,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if len(labels) == 0:
            raise ValueError("Cannot split an empty labels frame.")

        ids = labels["prediction_id"].drop_duplicates().to_numpy()
        shuffled = self.rng.permutation(ids)
        split_ids = self._split_prediction_ids(shuffled, spec.split_fractions)
        split_map: Dict[Any, str] = {}
        for split_name, values in split_ids.items():
            for value in values:
                split_map[value] = split_name

        result = labels.copy()
        result["split"] = result["prediction_id"].map(split_map)
        if result["split"].isna().any():
            raise ValueError("Split assignment failed for one or more prediction ids.")

        splits = {
            split_name: pd.DataFrame({"prediction_id": values, "split": split_name})
            for split_name, values in split_ids.items()
        }
        return result[self._label_columns_with_split()], splits

    @staticmethod
    def _split_prediction_ids(
        ids: np.ndarray,
        fractions: Mapping[str, float],
    ) -> Dict[str, np.ndarray]:
        n = len(ids)
        if n == 0:
            raise ValueError("No prediction ids available for split generation.")
        if n == 1:
            return {"train": ids, "val": ids[:0], "test": ids[:0]}
        if n == 2:
            return {"train": ids[:1], "val": ids[:0], "test": ids[1:]}

        train_fraction = float(fractions.get("train", 0.70))
        val_fraction = float(fractions.get("val", 0.15))
        n_train = max(1, int(round(n * train_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            n_train = max(1, n - n_val - n_test)

        return {
            "train": ids[:n_train],
            "val": ids[n_train:n_train + n_val],
            "test": ids[n_train + n_val:],
        }

    def _build_feature_manifest(
        self,
        spec: TaskSpec,
        schema: SchemaLike,
        tables: Mapping[str, pd.DataFrame],
        metadata: Mapping[str, Any],
        derived_signal_columns: Sequence[str],
    ) -> Dict[str, Any]:
        nodes = schema_nodes(schema)
        table_metadata = self._table_metadata(metadata)
        derived_signal_set = set(derived_signal_columns)

        visible_tables: List[str] = []
        hidden_tables: List[str] = []
        visible_columns: Dict[str, List[str]] = {}
        excluded_columns: Dict[str, List[str]] = {}

        for table_id in sorted(tables):
            df = tables[table_id]
            role = self._table_role(table_id, nodes, table_metadata)
            table_meta = table_metadata.get(table_id, {})
            hidden = role == "outcome" or table_meta.get("visible_as_feature_by_default") is False
            if role == "summary" and table_meta.get("used_only_history_before_cutoff") is False:
                hidden = True

            if hidden:
                hidden_tables.append(table_id)
                excluded_columns[table_id] = [str(col) for col in df.columns]
                continue

            visible_tables.append(table_id)
            table_visible: List[str] = []
            table_excluded: List[str] = []
            for col in df.columns:
                name = str(col)
                if self._is_excluded_feature_column(
                    table_id=table_id,
                    column=name,
                    target_source_table=spec.target_source_table,
                    derived_signal_columns=derived_signal_set,
                ):
                    table_excluded.append(name)
                else:
                    table_visible.append(name)
            visible_columns[table_id] = table_visible
            excluded_columns[table_id] = table_excluded

        return json_ready(
            {
                "task_id": spec.task_id,
                "target_source_table": spec.target_source_table,
                "target_source_role": spec.target_source_role,
                "prediction_unit_table": spec.prediction_unit_table,
                "visible_tables": visible_tables,
                "hidden_tables": hidden_tables,
                "visible_columns": visible_columns,
                "excluded_columns": excluded_columns,
                "excluded_patterns": ["__latent_*"],
                "derived_signal_columns": list(derived_signal_columns),
            }
        )

    @staticmethod
    def _is_excluded_feature_column(
        table_id: str,
        column: str,
        target_source_table: str,
        derived_signal_columns: set[str],
    ) -> bool:
        if column in {"label", "outcome_time"}:
            return True
        if column.startswith("__latent_") or column == "__activity_score":
            return True
        if table_id == target_source_table and column in derived_signal_columns:
            return True
        return False

    @staticmethod
    def _table_metadata(metadata: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
        tables = metadata.get("tables", {})
        if not isinstance(tables, Mapping):
            return {}
        return {str(k): v for k, v in tables.items() if isinstance(v, Mapping)}

    @staticmethod
    def _table_role(
        table_id: str,
        nodes: Mapping[str, Any],
        table_metadata: Mapping[str, Mapping[str, Any]],
    ) -> str:
        if table_id in table_metadata and "role" in table_metadata[table_id]:
            return str(table_metadata[table_id]["role"])
        if table_id in nodes:
            return node_role(nodes[table_id])
        raise ValueError(f"Table {table_id!r} is not present in schema nodes.")

    @staticmethod
    def _label_columns_without_split() -> List[str]:
        return [
            "prediction_id",
            "prediction_unit_table",
            "target_source_table",
            "label",
            "cutoff_time",
            "outcome_time",
        ]

    @staticmethod
    def _label_columns_with_split() -> List[str]:
        return [
            "prediction_id",
            "prediction_unit_table",
            "target_source_table",
            "label",
            "cutoff_time",
            "outcome_time",
            "split",
        ]


__all__ = ["LabelGenerator"]
