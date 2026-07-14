"""Database-level entry point for synthetic relational table generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import (
    BaseTableGenerator,
    FKLike,
    GenerationContext,
    NodeLike,
    SchemaLike,
    fk_child_table,
    fk_parent_table,
    foreign_keys_for_child,
    json_ready,
    node_id,
    node_role,
    schema_foreign_keys,
    schema_nodes,
)
from .bridge_generator import BridgeGenerator
from .entity_generator import EntityGenerator
from .event_generator import EventGenerator
from .measurement_generator import MeasurementGenerator
from .outcome_generator import OutcomeGenerator
from .summary_generator import SummaryGenerator
from .table_writer import save_generation_report, save_metadata, save_tables_as_parquet


class RelationalDatabaseGenerator:
    """Generate pandas DataFrames for every table in a sampled schema.

    This class consumes a SampledSchema-like object or schema dictionary. It does
    not sample schema structure. Tables are generated in FK topological order,
    with outcome tables forced to the end when possible.
    """

    ROLE_PRIORITY: Dict[str, int] = {
        "class": 0,
        "context": 0,
        "entity": 1,
        "bridge": 2,
        "event": 3,
        "measurement": 4,
        "summary": 5,
        "outcome": 6,
    }

    def __init__(self, seed: int = 42, generators: Optional[Mapping[str, BaseTableGenerator]] = None):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        default_generators: Dict[str, BaseTableGenerator] = {
            "class": EntityGenerator(rng=self.rng, seed=self.seed),
            "context": EntityGenerator(rng=self.rng, seed=self.seed),
            "entity": EntityGenerator(rng=self.rng, seed=self.seed),
            "bridge": BridgeGenerator(rng=self.rng, seed=self.seed),
            "event": EventGenerator(rng=self.rng, seed=self.seed),
            "measurement": MeasurementGenerator(rng=self.rng, seed=self.seed),
            "summary": SummaryGenerator(rng=self.rng, seed=self.seed),
            "outcome": OutcomeGenerator(rng=self.rng, seed=self.seed),
        }
        if generators:
            default_generators.update(dict(generators))
        self.generators = default_generators

    def generate_database(
        self,
        schema: SchemaLike,
        context: Optional[GenerationContext] = None,
        scm_assignments: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate all tables and return {'tables': ..., 'metadata': ...}."""
        context = context or GenerationContext(seed=self.seed)
        scm_assignments = scm_assignments or {}
        nodes = schema_nodes(schema)
        order = self._generation_order(schema, nodes)

        for table_id in order:
            node = nodes[table_id]
            role = node_role(node)
            generator = self.generators.get(role)
            if generator is None:
                raise ValueError(f"No table generator registered for role {role!r} on table {table_id!r}.")

            fk_specs = foreign_keys_for_child(schema, table_id)
            parent_tables: Dict[str, pd.DataFrame] = {}
            for fk in fk_specs:
                parent_id = fk_parent_table(fk)
                if parent_id not in context.tables:
                    raise ValueError(
                        f"Parent table {parent_id!r} for child {table_id!r} has not been generated. "
                        "Check FK topological order and schema acyclicity."
                    )
                parent_tables[parent_id] = context.tables[parent_id]

            generated = generator.generate(
                node=node,
                schema=schema,
                parent_tables=parent_tables,
                fk_specs=fk_specs,
                context=context,
                scm_assignment=scm_assignments.get(table_id),
            )
            context.register_table(generated)

        metadata = self._build_metadata(schema=schema, context=context, order=order)
        return {"tables": dict(context.tables), "metadata": metadata}

    def save_database(
        self,
        output_dir: Any,
        tables: Mapping[str, pd.DataFrame],
        metadata: Mapping[str, Any],
    ) -> Dict[str, str]:
        """Save tables and metadata to disk."""
        root = Path(output_dir)
        table_paths = save_tables_as_parquet(tables, root)
        metadata_path = save_metadata(metadata, root)
        report_path = save_generation_report(
            {
                "num_tables": len(tables),
                "tables": {table_id: {"shape": list(df.shape)} for table_id, df in tables.items()},
            },
            root,
        )
        return {"metadata": metadata_path, "report": report_path, **table_paths}

    def _generation_order(self, schema: SchemaLike, nodes: Mapping[str, NodeLike]) -> List[str]:
        edges = [(fk_parent_table(fk), fk_child_table(fk)) for fk in schema_foreign_keys(schema)]
        known = set(nodes)
        indeg = {table_id: 0 for table_id in nodes}
        children = {table_id: [] for table_id in nodes}

        for parent, child in edges:
            if parent not in known or child not in known:
                raise ValueError(f"Foreign key references unknown table: {(parent, child)!r}.")
            children[parent].append(child)
            indeg[child] += 1

        ready = sorted([table_id for table_id, deg in indeg.items() if deg == 0], key=self._node_sort_key(nodes))
        order: List[str] = []

        while ready:
            table_id = ready.pop(0)
            order.append(table_id)
            for child in sorted(children[table_id], key=self._node_sort_key(nodes)):
                indeg[child] -= 1
                if indeg[child] == 0:
                    ready.append(child)
                    ready.sort(key=self._node_sort_key(nodes))

        if len(order) != len(nodes):
            raise ValueError("Schema FK graph contains a cycle; table generation requires a DAG.")
        return order

    def _node_sort_key(self, nodes: Mapping[str, NodeLike]):
        def key(table_id: str) -> Tuple[int, str]:
            role = node_role(nodes[table_id])
            return self.ROLE_PRIORITY.get(role, 99), table_id
        return key

    def _build_metadata(
        self,
        schema: SchemaLike,
        context: GenerationContext,
        order: Sequence[str],
    ) -> Dict[str, Any]:
        schema_id = None
        if isinstance(schema, Mapping):
            schema_id = schema.get("schema_id")
        else:
            schema_id = getattr(schema, "schema_id", None)

        return json_ready(
            {
                "schema_id": schema_id,
                "generator": "RelationalDatabaseGenerator",
                "seed": self.seed,
                "generation_order": list(order),
                "context": {
                    "start_time": context.start_time,
                    "end_time": context.end_time,
                    "cutoff_time": context.cutoff_time,
                    "future_window_days": context.future_window_days,
                    "history_window_days": context.history_window_days,
                    "enable_leakage_guard": context.enable_leakage_guard,
                    "row_scale": context.row_scale,
                    "max_rows_per_table": context.max_rows_per_table,
                },
                "tables": context.table_metadata,
                "fk_graph_note": "FK edges are schema-level join support, not causal edges.",
            }
        )