"""Measurement table generator for observed values around events or entities."""

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
    choose_time_column,
    fk_child_col,
    fk_parent_col,
    fk_parent_table,
    node_columns,
    node_id,
    node_num_rows,
    node_primary_key,
    node_role,
    node_time_col,
    column_is_foreign_key,
    column_is_label,
    column_is_primary_key,
    column_is_time,
    column_name,
    column_semantic_type,
)


class MeasurementGenerator(BaseTableGenerator):
    """Generate noisy continuous observations linked to event/entity parents."""

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
        n_rows = context.planned_rows(node_num_rows(node))
        time_col = node_time_col(node) or "measurement_time"
        noise_scale = float((scm_assignment or {}).get("noise_scale", 0.25))
        missing_rate = float((scm_assignment or {}).get("missing_rate", 0.03))

        df = pd.DataFrame({pk_col: self.make_primary_key(table_id, n_rows, pk_col)})

        temporal_fk = self._choose_temporal_parent(fk_specs, parent_tables)
        temporal_rows: Optional[pd.DataFrame] = None
        if temporal_fk is not None and n_rows > 0:
            parent_df = parent_tables[fk_parent_table(temporal_fk)]
            weights = self._activity_weights(parent_df)
            positions = self.rng.choice(parent_df.index.to_numpy(), size=n_rows, replace=True, p=weights)
            temporal_rows = parent_df.loc[positions].reset_index(drop=True)

        for fk in fk_specs:
            parent_id = fk_parent_table(fk)
            parent_df = parent_tables.get(parent_id)
            if parent_df is None:
                raise ValueError(f"Parent table {parent_id!r} has not been generated.")
            if temporal_fk is not None and fk is temporal_fk and temporal_rows is not None:
                df[fk_child_col(fk)] = temporal_rows[fk_parent_col(fk)].to_numpy()
            else:
                df[fk_child_col(fk)] = self.sample_foreign_key(
                    parent_df=parent_df,
                    parent_pk=fk_parent_col(fk),
                    n_rows=n_rows,
                    long_tail=True,
                )

        if temporal_rows is not None:
            parent_time_col = choose_time_column(temporal_rows)
            if parent_time_col is not None:
                offsets = self.rng.normal(loc=0.0, scale=6.0, size=n_rows)
                times = pd.to_datetime(temporal_rows[parent_time_col]) + pd.to_timedelta(offsets, unit="h")
                df[time_col] = times.clip(lower=context.start_time, upper=context.end_time)
            else:
                df[time_col] = self.sample_timestamps(n_rows, context.start_time, context.end_time)
        else:
            df[time_col] = self.sample_timestamps(n_rows, context.start_time, context.end_time, bursty=True)

        numeric_declared = False
        for column in node_columns(node):
            name = column_name(column)
            if name in df.columns:
                continue
            if column_is_primary_key(column) or column_is_foreign_key(column) or column_is_time(column):
                continue
            if column_is_label(column):
                continue
            semantic = column_semantic_type(column).lower()
            if semantic == "numeric":
                base = self.sample_numeric(n_rows, distribution="normal")
                noise = self.rng.normal(0.0, noise_scale, size=n_rows).astype("float32")
                values = base + noise
                outlier_mask = self.rng.random(n_rows) < 0.01
                values[outlier_mask] = values[outlier_mask] * 6.0
                df[name] = values.astype("float32")
                numeric_declared = True
            elif semantic == "categorical":
                df[name] = self.sample_categorical(n_rows, cardinality=8, long_tail=True)

        if not numeric_declared:
            values = self.sample_numeric(n_rows, distribution="normal")
            values += self.rng.normal(0.0, noise_scale, size=n_rows).astype("float32")
            df["value"] = values.astype("float32")

        df = self.apply_missingness(df, missing_rate=missing_rate)
        df = self.safe_sort_by_time(df, time_col)

        metadata: Dict[str, Any] = {
            "table_id": table_id,
            "role": role,
            "scm_family": (scm_assignment or {}).get("scm_family", "observation"),
            "parent_tables": [fk_parent_table(fk) for fk in fk_specs],
            "num_rows": int(len(df)),
            "time_col": time_col,
            "noise_scale": noise_scale,
            "missing_rate": missing_rate,
            "generated_columns": self.generated_columns_metadata(df),
        }
        return GeneratedTable(table_id=table_id, role=role, dataframe=df, metadata=metadata)

    def _choose_temporal_parent(
        self,
        fk_specs: Sequence[FKLike],
        parent_tables: Mapping[str, pd.DataFrame],
    ) -> Optional[FKLike]:
        for fk in fk_specs:
            parent_df = parent_tables.get(fk_parent_table(fk))
            if parent_df is not None and choose_time_column(parent_df) is not None:
                return fk
        return None