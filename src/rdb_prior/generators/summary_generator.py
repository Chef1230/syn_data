"""Cutoff-safe historical summary generator."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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
    node_id,
    node_primary_key,
    node_role,
    node_time_col,
    numeric_feature_columns,
)


class SummaryGenerator(BaseTableGenerator):
    """Generate as-of summary tables from cutoff-before historical rows."""

    AGGREGATION_OPS = ["count", "mean", "sum", "max", "recent_count", "has_history"]

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
        time_col = node_time_col(node) or "as_of_time"

        if not fk_specs:
            df = pd.DataFrame({pk_col: self.make_primary_key(table_id, 1, pk_col)})
            df[time_col] = context.cutoff_time
            if time_col != "as_of_time":
                df["as_of_time"] = context.cutoff_time
            df["has_history"] = False
            used_only_history = True
            leakage_violations = []
        else:
            aggregate_df, used_only_history, leakage_violations = self._build_aggregates(
                fk_specs=fk_specs,
                parent_tables=parent_tables,
                context=context,
            )
            aggregate_df.insert(0, pk_col, self.make_primary_key(table_id, len(aggregate_df), pk_col))
            aggregate_df[time_col] = context.cutoff_time
            if time_col != "as_of_time":
                aggregate_df["as_of_time"] = context.cutoff_time
            df = aggregate_df

        df = self.add_schema_feature_columns(df, node, context)
        df = self.safe_sort_by_time(df, time_col)

        metadata: Dict[str, Any] = {
            "table_id": table_id,
            "role": role,
            "scm_family": (scm_assignment or {}).get("scm_family", "window_aggregation"),
            "parent_tables": [fk_parent_table(fk) for fk in fk_specs],
            "num_rows": int(len(df)),
            "time_col": time_col,
            "requires_as_of_cutoff": True,
            "used_only_history_before_cutoff": bool(used_only_history),
            "aggregation_ops": list(self.AGGREGATION_OPS),
            "cutoff_time": str(context.cutoff_time),
            "history_window_days": int(context.history_window_days),
            "leakage_violations": leakage_violations,
            "generated_columns": self.generated_columns_metadata(df),
        }
        if context.enable_leakage_guard and leakage_violations:
            raise ValueError(f"Summary generation detected leakage violations: {leakage_violations}")

        return GeneratedTable(table_id=table_id, role=role, dataframe=df, metadata=metadata)

    def _build_aggregates(
        self,
        fk_specs: Sequence[FKLike],
        parent_tables: Mapping[str, pd.DataFrame],
        context: GenerationContext,
    ) -> Tuple[pd.DataFrame, bool, list]:
        primary_fk = fk_specs[0]
        parent_id = fk_parent_table(primary_fk)
        parent_df = parent_tables.get(parent_id)
        if parent_df is None:
            raise ValueError(f"Parent table {parent_id!r} has not been generated.")

        history_df, used_only_history, leakage_violations = self._history_only(parent_id, parent_df, context)
        group_col = self._choose_group_col(history_df, fk_parent_col(primary_fk))
        child_col = fk_child_col(primary_fk)

        if len(history_df) == 0:
            result = pd.DataFrame(columns=[child_col, group_col, "count", "has_history"])
            return result, used_only_history, leakage_violations

        numeric_cols = numeric_feature_columns(history_df)
        grouped = history_df.groupby(group_col, dropna=False)
        result = grouped.size().rename("count").reset_index()
        result["has_history"] = result["count"] > 0

        if numeric_cols:
            source_col = numeric_cols[0]
            stats = grouped[source_col].agg(["mean", "sum", "max"]).reset_index()
            stats = stats.rename(
                columns={
                    "mean": f"{parent_id}__{source_col}__mean",
                    "sum": f"{parent_id}__{source_col}__sum",
                    "max": f"{parent_id}__{source_col}__max",
                }
            )
            result = result.merge(stats, on=group_col, how="left")
        else:
            result[f"{parent_id}__numeric_mean"] = 0.0
            result[f"{parent_id}__numeric_sum"] = 0.0
            result[f"{parent_id}__numeric_max"] = 0.0

        time_col = choose_time_column(history_df)
        if time_col is not None:
            recent_start = context.cutoff_time - pd.Timedelta(days=int(context.history_window_days))
            recent = history_df[pd.to_datetime(history_df[time_col]) >= recent_start]
            recent_count = recent.groupby(group_col, dropna=False).size().rename("recent_count").reset_index()
            result = result.merge(recent_count, on=group_col, how="left")
            result["recent_count"] = result["recent_count"].fillna(0).astype("int64")
        else:
            result["recent_count"] = result["count"].astype("int64")

        if group_col == fk_parent_col(primary_fk):
            result[child_col] = result[group_col]
        else:
            latest_parent = history_df.groupby(group_col, dropna=False)[fk_parent_col(primary_fk)].last().reset_index()
            latest_parent = latest_parent.rename(columns={fk_parent_col(primary_fk): child_col})
            result = result.merge(latest_parent, on=group_col, how="left")

        for fk in fk_specs[1:]:
            col = fk_child_col(fk)
            if col in result.columns:
                continue
            other_parent = parent_tables.get(fk_parent_table(fk))
            if other_parent is None:
                raise ValueError(f"Parent table {fk_parent_table(fk)!r} has not been generated.")
            if len(result) == 0:
                result[col] = pd.Series(dtype=other_parent[fk_parent_col(fk)].dtype)
            else:
                result[col] = self.sample_foreign_key(other_parent, fk_parent_col(fk), len(result), long_tail=True)

        ordered = [child_col] + [c for c in result.columns if c != child_col]
        return result[ordered].reset_index(drop=True), used_only_history, leakage_violations

    def _history_only(
        self,
        parent_id: str,
        parent_df: pd.DataFrame,
        context: GenerationContext,
    ) -> Tuple[pd.DataFrame, bool, list]:
        time_col = choose_time_column(parent_df)
        if time_col is None:
            parent_role = str(context.table_metadata.get(parent_id, {}).get("role", ""))
            if context.enable_leakage_guard and parent_role in {"event", "measurement", "summary", "outcome"}:
                return parent_df.iloc[0:0].copy(), False, [
                    {
                        "parent_table": parent_id,
                        "reason": "temporal_parent_without_time_column",
                    }
                ]
            return parent_df.copy(), True, []

        times = pd.to_datetime(parent_df[time_col])
        future_mask = times >= context.cutoff_time
        history = parent_df.loc[~future_mask].copy()
        return history, True, []

    def _choose_group_col(self, parent_df: pd.DataFrame, parent_pk: str) -> str:
        for col in parent_df.columns:
            name = str(col)
            if name.endswith("_id") and name != parent_pk:
                return name
        return parent_pk