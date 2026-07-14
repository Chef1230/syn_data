"""Event table generator for timestamped interaction tables."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from .base import (
    BaseTableGenerator,
    FKLike,
    GeneratedTable,
    GenerationContext,
    NodeLike,
    SchemaLike,
    fk_parent_table,
    node_id,
    node_num_rows,
    node_primary_key,
    node_role,
    node_time_col,
)


class EventGenerator(BaseTableGenerator):
    """Generate timestamped event tables using parent tables as join context."""

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
        time_col = node_time_col(node) or "event_time"

        df = pd.DataFrame({pk_col: self.make_primary_key(table_id, n_rows, pk_col)})
        df = self.add_foreign_keys(df, fk_specs, parent_tables, n_rows, long_tail=True)
        df[time_col] = self.sample_timestamps(
            n_rows=n_rows,
            start=context.start_time,
            end=context.end_time,
            bursty=True,
        )
        df = self.add_schema_feature_columns(df, node, context)
        df = self.safe_sort_by_time(df, time_col)

        parent_activity_used = any(
            "__activity_score" in parent_tables[parent_id].columns
            for parent_id in [fk_parent_table(fk) for fk in fk_specs]
            if parent_id in parent_tables
        )
        metadata: Dict[str, Any] = {
            "table_id": table_id,
            "role": role,
            "scm_family": (scm_assignment or {}).get("scm_family", "temporal_interaction"),
            "parent_tables": [fk_parent_table(fk) for fk in fk_specs],
            "num_rows": int(len(df)),
            "time_col": time_col,
            "time_range": [str(context.start_time), str(context.end_time)],
            "event_intensity": {
                "bursty": True,
                "parent_activity_used": parent_activity_used,
            },
            "generated_columns": self.generated_columns_metadata(df),
        }
        return GeneratedTable(table_id=table_id, role=role, dataframe=df, metadata=metadata)