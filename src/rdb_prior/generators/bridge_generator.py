"""Bridge table generator for sparse many-to-many relations."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

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
    node_id,
    node_num_rows,
    node_primary_key,
    node_role,
)


class BridgeGenerator(BaseTableGenerator):
    """Generate sparse bridge/relation tables from parent key tuples."""

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
        requested_rows = context.planned_rows(node_num_rows(node))

        parent_ids = [fk_parent_table(fk) for fk in fk_specs]
        scm_family = "degree_based_matching"
        if not any("__activity_score" in parent_tables[p].columns for p in parent_ids if p in parent_tables):
            scm_family = "fk_pair_sampling"
        scm_family = (scm_assignment or {}).get("scm_family", scm_family)

        if not fk_specs:
            df = pd.DataFrame({pk_col: self.make_primary_key(table_id, requested_rows, pk_col)})
        else:
            rows = self._sample_unique_tuples(fk_specs, parent_tables, requested_rows)
            df = pd.DataFrame(rows)
            actual_rows = len(df)
            df.insert(0, pk_col, self.make_primary_key(table_id, actual_rows, pk_col))

        df = self.add_schema_feature_columns(df, node, context)

        fk_cols = [fk_child_col(fk) for fk in fk_specs]
        if fk_cols and len(df) > 0:
            unique_pair_ratio = float(df[fk_cols].drop_duplicates().shape[0] / len(df))
        else:
            unique_pair_ratio = 1.0

        parent_cardinality = 1
        for fk in fk_specs:
            parent_df = parent_tables.get(fk_parent_table(fk))
            if parent_df is not None:
                parent_cardinality *= max(1, len(parent_df))
        density = float(len(df) / parent_cardinality) if parent_cardinality > 0 else 0.0

        metadata: Dict[str, Any] = {
            "table_id": table_id,
            "role": role,
            "scm_family": scm_family,
            "parent_tables": parent_ids,
            "num_rows": int(len(df)),
            "density": density,
            "unique_pair_ratio": unique_pair_ratio,
            "generated_columns": self.generated_columns_metadata(df),
        }
        return GeneratedTable(table_id=table_id, role=role, dataframe=df, metadata=metadata)

    def _sample_unique_tuples(
        self,
        fk_specs: Sequence[FKLike],
        parent_tables: Mapping[str, pd.DataFrame],
        target_rows: int,
    ) -> List[Dict[str, Any]]:
        if target_rows <= 0:
            return []

        fk_cols = [fk_child_col(fk) for fk in fk_specs]
        rows: List[Dict[str, Any]] = []
        seen = set()
        max_attempts = max(10, target_rows * 8)

        for _ in range(max_attempts):
            row: Dict[str, Any] = {}
            key_values: List[Any] = []
            for fk in fk_specs:
                parent_id = fk_parent_table(fk)
                parent_df = parent_tables.get(parent_id)
                if parent_df is None:
                    raise ValueError(f"Parent table {parent_id!r} has not been generated.")
                sampled = self.sample_foreign_key(
                    parent_df=parent_df,
                    parent_pk=fk_parent_col(fk),
                    n_rows=1,
                    long_tail=True,
                )[0]
                row[fk_child_col(fk)] = sampled
                key_values.append(sampled)

            key = tuple(key_values)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(rows) >= target_rows:
                break

        if len(rows) < target_rows:
            # Sparse relation spaces can saturate quickly for small parents. Fill
            # remaining rows without forcing uniqueness so generation still closes.
            while len(rows) < target_rows:
                row = {}
                for fk in fk_specs:
                    parent_df = parent_tables[fk_parent_table(fk)]
                    row[fk_child_col(fk)] = self.sample_foreign_key(
                        parent_df=parent_df,
                        parent_pk=fk_parent_col(fk),
                        n_rows=1,
                        long_tail=True,
                    )[0]
                rows.append(row)

        return rows