"""Task specification sampling for generated relational databases.

The task layer consumes already generated tables. It does not sample schema
structure and it does not interpret FK edges as causal edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence
import uuid

import pandas as pd

from ..generators.base import (
    SchemaLike,
    choose_time_column,
    column_is_label,
    column_name,
    fk_child_col,
    fk_child_table,
    fk_parent_col,
    fk_parent_table,
    foreign_keys_for_child,
    json_ready,
    node_columns,
    node_primary_key,
    node_role,
    node_time_col,
    schema_nodes,
)


SUPPORTED_TARGET_ROLES = ("outcome", "summary", "entity", "event")
DEFAULT_TARGET_ROLE_PRIORITY = ("outcome", "summary", "entity", "event")


@dataclass(frozen=True)
class TaskSpec:
    """A decision-complete task definition for one generated database."""

    task_id: str
    task_type: str
    target_source_table: str
    target_source_role: str
    prediction_unit_table: str
    prediction_unit_pk: str
    label_source_mode: str
    label_col: str
    cutoff_time: str
    future_window_days: int
    metric: str = "roc_auc"
    split_strategy: str = "random_unit_split"
    target_source_pk: Optional[str] = None
    target_fk_col: Optional[str] = None
    target_time_col: Optional[str] = None
    split_fractions: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15}
    )

    def to_dict(self) -> Dict[str, Any]:
        return json_ready(
            {
                "task_id": self.task_id,
                "task_type": self.task_type,
                "target_source_table": self.target_source_table,
                "target_source_role": self.target_source_role,
                "prediction_unit_table": self.prediction_unit_table,
                "prediction_unit_pk": self.prediction_unit_pk,
                "label_source_mode": self.label_source_mode,
                "label_col": self.label_col,
                "cutoff_time": self.cutoff_time,
                "future_window_days": self.future_window_days,
                "metric": self.metric,
                "split_strategy": self.split_strategy,
                "target_source_pk": self.target_source_pk,
                "target_fk_col": self.target_fk_col,
                "target_time_col": self.target_time_col,
                "split_fractions": dict(self.split_fractions),
            }
        )


@dataclass
class TaskBundle:
    """Generated task artifacts before they are written to disk."""

    spec: TaskSpec
    labels: pd.DataFrame
    splits: Dict[str, pd.DataFrame]
    feature_manifest: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskSampler:
    """Select a target source table and derive a task specification."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def sample_task(
        self,
        schema: SchemaLike,
        tables: Mapping[str, pd.DataFrame],
        metadata: Mapping[str, Any],
        target_source_role: Optional[str] = None,
        target_source_table: Optional[str] = None,
    ) -> TaskSpec:
        nodes = schema_nodes(schema)
        table_metadata = self._table_metadata(metadata)

        target_table = self._choose_target_table(
            nodes=nodes,
            tables=tables,
            metadata=table_metadata,
            target_source_role=target_source_role,
            target_source_table=target_source_table,
        )
        target_node = nodes[target_table]
        target_role = self._table_role(target_table, nodes, table_metadata)
        target_pk = node_primary_key(target_node)
        target_time_col = self._time_column_for_table(target_table, target_node, tables, table_metadata)

        if target_role == "outcome":
            prediction_table, prediction_pk, target_fk_col = self._choose_outcome_prediction_unit(
                schema=schema,
                nodes=nodes,
                table_metadata=table_metadata,
                outcome_table=target_table,
            )
            label_col = self._label_column(target_node, tables[target_table], table_metadata.get(target_table, {}))
            label_source_mode = "existing_outcome_label"
        else:
            prediction_table = target_table
            prediction_pk = target_pk
            target_fk_col = None
            label_col = "label"
            label_source_mode = "derived_table_label"

        cutoff_time = str(self._cutoff_time(metadata))
        future_window_days = int(metadata.get("context", {}).get("future_window_days", 30))
        role_part = target_role.replace(" ", "_")
        task_id = f"{role_part}_{target_table}_{uuid.uuid5(uuid.NAMESPACE_URL, target_table + cutoff_time).hex[:8]}"

        return TaskSpec(
            task_id=task_id,
            task_type="binary_classification",
            target_source_table=target_table,
            target_source_role=target_role,
            prediction_unit_table=prediction_table,
            prediction_unit_pk=prediction_pk,
            label_source_mode=label_source_mode,
            label_col=label_col,
            cutoff_time=cutoff_time,
            future_window_days=future_window_days,
            target_source_pk=target_pk,
            target_fk_col=target_fk_col,
            target_time_col=target_time_col,
        )

    def _choose_target_table(
        self,
        nodes: Mapping[str, Any],
        tables: Mapping[str, pd.DataFrame],
        metadata: Mapping[str, Mapping[str, Any]],
        target_source_role: Optional[str],
        target_source_table: Optional[str],
    ) -> str:
        if target_source_table is not None:
            table_id = str(target_source_table)
            if table_id not in tables:
                raise ValueError(f"Requested target_source_table {table_id!r} is not in generated tables.")
            role = self._table_role(table_id, nodes, metadata)
            if target_source_role is not None and role != str(target_source_role):
                raise ValueError(
                    f"Requested table {table_id!r} has role {role!r}, not {target_source_role!r}."
                )
            if role not in SUPPORTED_TARGET_ROLES:
                raise ValueError(f"Role {role!r} is not supported as a task target source.")
            return table_id

        requested_role = str(target_source_role) if target_source_role is not None else None
        if requested_role is not None and requested_role not in SUPPORTED_TARGET_ROLES:
            raise ValueError(
                f"target_source_role must be one of {SUPPORTED_TARGET_ROLES}, got {requested_role!r}."
            )

        role_order: Sequence[str] = (requested_role,) if requested_role else DEFAULT_TARGET_ROLE_PRIORITY
        for role in role_order:
            candidates = [
                table_id
                for table_id in sorted(tables)
                if table_id in nodes and self._table_role(table_id, nodes, metadata) == role
            ]
            if candidates:
                return candidates[0]

        detail = f"role={requested_role!r}" if requested_role else f"roles={DEFAULT_TARGET_ROLE_PRIORITY!r}"
        raise ValueError(f"No generated table found for supported task target source {detail}.")

    def _choose_outcome_prediction_unit(
        self,
        schema: SchemaLike,
        nodes: Mapping[str, Any],
        table_metadata: Mapping[str, Mapping[str, Any]],
        outcome_table: str,
    ) -> tuple[str, str, str]:
        fks = foreign_keys_for_child(schema, outcome_table)
        if not fks:
            raise ValueError(
                f"Outcome table {outcome_table!r} has no parent FK; cannot infer prediction unit."
            )

        role_priority = {"summary": 0, "entity": 1, "event": 2, "bridge": 3, "context": 4, "class": 5}
        ranked = sorted(
            fks,
            key=lambda fk: (
                role_priority.get(self._table_role(fk_parent_table(fk), nodes, table_metadata), 99),
                fk_parent_table(fk),
            ),
        )
        chosen = ranked[0]
        parent_table = fk_parent_table(chosen)
        return parent_table, fk_parent_col(chosen), fk_child_col(chosen)

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
        if table_id not in nodes:
            raise ValueError(f"Table {table_id!r} is not present in schema nodes.")
        return node_role(nodes[table_id])

    @staticmethod
    def _cutoff_time(metadata: Mapping[str, Any]) -> str:
        value = metadata.get("context", {}).get("cutoff_time")
        if value is None:
            raise ValueError("Generation metadata is missing context.cutoff_time.")
        return str(value)

    @staticmethod
    def _time_column_for_table(
        table_id: str,
        node: Any,
        tables: Mapping[str, pd.DataFrame],
        table_metadata: Mapping[str, Mapping[str, Any]],
    ) -> Optional[str]:
        if table_id in table_metadata and table_metadata[table_id].get("time_col"):
            return str(table_metadata[table_id]["time_col"])
        preferred = node_time_col(node)
        if table_id in tables:
            return choose_time_column(tables[table_id], preferred=preferred)
        return preferred

    @staticmethod
    def _label_column(node: Any, df: pd.DataFrame, table_metadata: Mapping[str, Any]) -> str:
        if table_metadata.get("label_col") in df.columns:
            return str(table_metadata["label_col"])
        for column in node_columns(node):
            if column_is_label(column):
                name = column_name(column)
                if name in df.columns:
                    return name
        if "label" in df.columns:
            return "label"
        raise ValueError("Outcome task target table does not contain a label column.")


__all__ = [
    "SUPPORTED_TARGET_ROLES",
    "DEFAULT_TARGET_ROLE_PRIORITY",
    "TaskSpec",
    "TaskBundle",
    "TaskSampler",
]
