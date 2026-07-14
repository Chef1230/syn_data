"""Outcome generator for future-window task labels."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .base import (
    BaseTableGenerator,
    FKLike,
    GeneratedTable,
    GenerationContext,
    NodeLike,
    SchemaLike,
    fk_child_col,
    fk_parent_col,
    fk_parent_table,
    node_columns,
    node_id,
    node_num_rows,
    node_primary_key,
    node_role,
    node_time_col,
    column_is_label,
    column_name,
    numeric_feature_columns,
)


class OutcomeGenerator(BaseTableGenerator):
    """Generate future-window labels and mark them as task targets."""

    def generate(
        self,
        node: NodeLike,
        schema: SchemaLike,
        parent_tables: Mapping[str, pd.DataFrame],
        fk_specs: Sequence[FKLike],
        context: GenerationContext,
        scm_assignment: Optional[Mapping[str, Any]] = None,
    ) -> GeneratedTable:
        del schema
        table_id = node_id(node)
        role = node_role(node)
        pk_col = node_primary_key(node)
        label_col = self._label_column(node)
        time_col = node_time_col(node) or "outcome_time"

        driver_fk = self._choose_driver_parent(fk_specs, parent_tables, context)
        if driver_fk is None:
            n_rows = context.planned_rows(node_num_rows(node))
            driver_rows = None
        else:
            driver_parent = parent_tables[fk_parent_table(driver_fk)]
            n_rows = len(driver_parent)
            driver_rows = driver_parent.reset_index(drop=True)

        df = pd.DataFrame({pk_col: self.make_primary_key(table_id, n_rows, pk_col)})
        for fk in fk_specs:
            parent_id = fk_parent_table(fk)
            parent_df = parent_tables.get(parent_id)
            if parent_df is None:
                raise ValueError(f"Parent table {parent_id!r} has not been generated.")
            if driver_fk is not None and fk is driver_fk and driver_rows is not None:
                df[fk_child_col(fk)] = driver_rows[fk_parent_col(fk)].to_numpy()
            else:
                df[fk_child_col(fk)] = self.sample_foreign_key(parent_df, fk_parent_col(fk), n_rows, long_tail=True)

        probabilities = self._label_probabilities(driver_rows, n_rows, scm_assignment)
        df[label_col] = (self.rng.random(n_rows) < probabilities).astype("int8")

        future_offsets = self.rng.integers(1, max(2, int(context.future_window_days) + 1), size=n_rows)
        df[time_col] = context.cutoff_time + pd.to_timedelta(future_offsets, unit="D")
        if time_col != "outcome_time":
            df["outcome_time"] = df[time_col]

        metadata: Dict[str, Any] = {
            "table_id": table_id,
            "role": role,
            "scm_family": (scm_assignment or {}).get("scm_family", "future_window_label"),
            "parent_tables": [fk_parent_table(fk) for fk in fk_specs],
            "num_rows": int(len(df)),
            "label_col": label_col,
            "time_col": time_col,
            "visible_as_feature_by_default": False,
            "is_task_target": True,
            "future_window_days": int(context.future_window_days),
            "cutoff_time": str(context.cutoff_time),
            "positive_rate": float(df[label_col].mean()) if len(df) else 0.0,
            "generated_columns": self.generated_columns_metadata(df),
        }
        return GeneratedTable(table_id=table_id, role=role, dataframe=df, metadata=metadata)

    def _choose_driver_parent(
        self,
        fk_specs: Sequence[FKLike],
        parent_tables: Mapping[str, pd.DataFrame],
        context: GenerationContext,
    ) -> Optional[FKLike]:
        for fk in fk_specs:
            parent_id = fk_parent_table(fk)
            if parent_id in parent_tables and context.table_metadata.get(parent_id, {}).get("role") == "summary":
                return fk
        for fk in fk_specs:
            if fk_parent_table(fk) in parent_tables:
                return fk
        return None

    def _label_column(self, node: NodeLike) -> str:
        for column in node_columns(node):
            if column_is_label(column):
                return column_name(column)
        return "label"

    def _label_probabilities(
        self,
        driver_rows: Optional[pd.DataFrame],
        n_rows: int,
        scm_assignment: Optional[Mapping[str, Any]],
    ) -> np.ndarray:
        base_rate = float((scm_assignment or {}).get("base_rate", 0.25))
        base_logit = np.log(base_rate / max(1.0 - base_rate, 1e-6))
        if driver_rows is None or len(driver_rows) == 0:
            logits = np.full(n_rows, base_logit, dtype=float)
        else:
            cols = numeric_feature_columns(driver_rows)
            if cols:
                signal = driver_rows[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
                signal = signal.mean(axis=1)
                if float(np.std(signal)) > 1e-6:
                    signal = (signal - float(np.mean(signal))) / float(np.std(signal))
                logits = base_logit + 0.8 * signal
            else:
                logits = np.full(len(driver_rows), base_logit, dtype=float)
        logits = logits + self.rng.normal(0.0, 0.5, size=len(logits))
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.clip(probs, 0.01, 0.99)