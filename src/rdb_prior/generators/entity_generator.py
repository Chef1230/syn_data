"""Entity table generator for anonymous relational schemas."""

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
    fk_parent_table,
    node_id,
    node_num_rows,
    node_primary_key,
    node_role,
    node_time_col,
)


class EntityGenerator(BaseTableGenerator):
    """Generate stable entity-like tables with latent profiles.

    Entity generation may use parent class/context/entity tables as join context
    through FK columns. The FK graph is not treated as a causal graph.
    """

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

        df = pd.DataFrame({pk_col: self.make_primary_key(table_id, n_rows, pk_col)})
        df = self.add_foreign_keys(df, fk_specs, parent_tables, n_rows, long_tail=True)

        activity = self.rng.lognormal(mean=0.0, sigma=0.9, size=n_rows).astype("float32")
        quality = self.rng.normal(loc=0.0, scale=1.0, size=n_rows).astype("float32")
        if n_rows > 0:
            activity_score = activity / max(float(np.mean(activity)), 1e-6)
        else:
            activity_score = activity

        df["__latent_activity"] = activity
        df["__latent_quality"] = quality
        df["__activity_score"] = np.clip(activity_score, 0.05, None).astype("float32")

        time_col = node_time_col(node)
        if time_col is not None and time_col not in df.columns:
            df[time_col] = self.sample_timestamps(n_rows, context.start_time, context.cutoff_time)

        df = self.add_schema_feature_columns(df, node, context)
        df = self.safe_sort_by_time(df, time_col)

        metadata: Dict[str, Any] = {
            "table_id": table_id,
            "role": role,
            "scm_family": (scm_assignment or {}).get("scm_family", "latent_profile"),
            "parent_tables": [fk_parent_table(fk) for fk in fk_specs],
            "num_rows": int(len(df)),
            "primary_key": pk_col,
            "time_col": time_col,
            "generated_columns": self.generated_columns_metadata(df),
            "latent_columns": ["__latent_activity", "__latent_quality", "__activity_score"],
        }
        return GeneratedTable(table_id=table_id, role=role, dataframe=df, metadata=metadata)