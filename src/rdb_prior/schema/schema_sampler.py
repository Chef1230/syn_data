# src/rdb_prior/schema/schema_sampler.py
# -*- coding: utf-8 -*-
"""
Schema sampler for role/motif/time-aware synthetic relational database generation.

Design principles
-----------------
1. This module samples an anonymous FK-support schema graph.
2. FK graph is NOT treated as a causal graph.
3. Role is a latent structural variable used by the generator, not a table name.
4. Outcome should be downstream and should not point to visible input tables.
5. Summary should represent as-of historical aggregation, not future leakage.

Main output
-----------
SampledSchema:
    - nodes: anonymous tables T0, T1, ...
    - edges: FK-support edges parent -> child
    - role: latent structural role for each node
    - rank: DAG rank/depth position
    - columns: basic column metadata
    - motifs: sampled motif records used to bias the schema
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math
import os
import random
import uuid

from .motif_library import MotifLibrary, MotifPattern
from .role_sampler import (
    DEFAULT_ALLOWED_TRANSITIONS,
    DEFAULT_ROLE_PROFILES,
    RoleSampler,
    normalize_role,
    required_role_counts_from_motifs,
)


Role = str
NodeId = str
Edge = Tuple[NodeId, NodeId]

ROLE_ROW_COUNT_RANGES: Dict[Role, Tuple[int, int]] = {
    "class": (10, 1_000),
    "context": (10, 5_000),
    "entity": (10_000, 10_000),
    "event": (5_000, 10_000),
    "bridge": (10_000, 50_000),
    "measurement": (5_000, 20_000),
    "summary": (10_000, 50_000),
    "outcome": (10_000, 20_000),
}


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass
class ColumnSpec:
    name: str
    dtype: str
    semantic_type: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references: Optional[Dict[str, str]] = None
    is_time: bool = False
    is_label_candidate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "semantic_type": self.semantic_type,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "references": self.references,
            "is_time": self.is_time,
            "is_label_candidate": self.is_label_candidate,
        }


@dataclass
class SchemaNode:
    node_id: NodeId
    role: Role
    rank: int
    num_rows: int
    num_columns: int
    primary_key: str
    time_col: Optional[str]
    columns: List[ColumnSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_columns: bool = True) -> Dict[str, Any]:
        data = {
            "node_id": self.node_id,
            "role": self.role,
            "rank": self.rank,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "primary_key": self.primary_key,
            "time_col": self.time_col,
            "metadata": dict(self.metadata),
        }
        if include_columns:
            data["columns"] = [c.to_dict() for c in self.columns]
        return data


@dataclass
class ForeignKeySpec:
    parent_table: NodeId
    child_table: NodeId
    parent_col: str
    child_col: str
    cardinality: str = "one_to_many"
    optional: bool = False
    fanout_prior: Dict[str, Any] = field(default_factory=dict)

    @property
    def edge(self) -> Edge:
        return self.parent_table, self.child_table

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_table": self.parent_table,
            "child_table": self.child_table,
            "parent_col": self.parent_col,
            "child_col": self.child_col,
            "cardinality": self.cardinality,
            "optional": self.optional,
            "fanout_prior": dict(self.fanout_prior),
        }


@dataclass
class SampledSchema:
    schema_id: str
    nodes: Dict[NodeId, SchemaNode]
    foreign_keys: List[ForeignKeySpec]
    motifs: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    violations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def edges(self) -> List[Edge]:
        return [fk.edge for fk in self.foreign_keys]

    @property
    def role_counts(self) -> Counter:
        return Counter(node.role for node in self.nodes.values())

    @property
    def num_tables(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.foreign_keys)

    def to_dict(self, include_columns: bool = True) -> Dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "num_tables": self.num_tables,
            "num_edges": self.num_edges,
            "role_counts": dict(self.role_counts),
            "nodes": {
                node_id: node.to_dict(include_columns=include_columns)
                for node_id, node in self.nodes.items()
            },
            "foreign_keys": [fk.to_dict() for fk in self.foreign_keys],
            "motifs": list(self.motifs),
            "metadata": dict(self.metadata),
            "violations": list(self.violations),
        }

    def save_json(self, path: str, include_columns: bool = True) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(include_columns=include_columns), f, indent=2, ensure_ascii=False)


@dataclass
class SchemaSamplingConfig:
    """
    Initial config for schema sampling.

    edge_density:
        Target FK edge count is sampled approximately from:
            num_tables * Uniform(min_edge_density, max_edge_density)

    max_depth:
        Maximum rank. If max_depth=3, ranks are 0, 1, 2, 3.

    motif_count:
        Number of local motifs sampled before graph construction.

    require_connected:
        If True, try to make the FK graph weakly connected.

    allow_rank_skip_edges:
        If False, only rank r -> rank r+1 edges are allowed.
        If True, rank r -> rank r+k edges are allowed for k >= 1.
    """

    min_tables: int = 3
    max_tables: int = 8
    table_count_values: Tuple[int, ...] = ()
    table_count_weights: Tuple[float, ...] = ()
    feature_columns_by_table_count: Tuple[Dict[str, int], ...] = field(default_factory=tuple)
    max_depth: int = 3
    schema_depth_values: Tuple[int, ...] = (1, 2, 3)
    schema_depth_weights: Tuple[float, ...] = (0.25, 0.50, 0.25)

    min_edge_density: float = 0.8
    max_edge_density: float = 1.4

    min_motifs: int = 1
    max_motifs: int = 4
    max_motif_rank_span: int = 3

    require_connected: bool = True
    allow_rank_skip_edges: bool = True
    allow_parallel_fk: bool = False

    require_entity: bool = True
    require_event: bool = True
    max_outcome_tables: int = 1

    require_bridge_parents: bool = True
    min_bridge_parents: int = 2

    add_random_edges: bool = True
    max_edge_attempts: int = 300

    enable_motif_edges: bool = True
    enable_transition_check: bool = True
    strict_transition_check: bool = False

    seed: int = 42

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "SchemaSamplingConfig":
        schema_cfg = config.get("schema", config)

        return cls(
            min_tables=int(schema_cfg.get("min_tables", 3)),
            max_tables=int(schema_cfg.get("max_tables", 8)),
            table_count_values=tuple(
                int(v) for v in schema_cfg.get("table_count_values", ())
            ),
            table_count_weights=tuple(
                float(v) for v in schema_cfg.get("table_count_weights", ())
            ),
            feature_columns_by_table_count=tuple(
                _normalize_feature_column_range(item)
                for item in schema_cfg.get("feature_columns_by_table_count", ())
            ),
            max_depth=int(schema_cfg.get("max_depth", 3)),
            schema_depth_values=tuple(
                int(v) for v in schema_cfg.get("schema_depth_values", (1, 2, 3))
            ),
            schema_depth_weights=tuple(
                float(v) for v in schema_cfg.get("schema_depth_weights", (0.25, 0.50, 0.25))
            ),
            min_edge_density=float(schema_cfg.get("min_edge_density", 0.8)),
            max_edge_density=float(schema_cfg.get("max_edge_density", 1.4)),
            min_motifs=int(schema_cfg.get("min_motifs", 1)),
            max_motifs=int(schema_cfg.get("max_motifs", 4)),
            max_motif_rank_span=int(schema_cfg.get("max_motif_rank_span", 3)),
            require_connected=bool(schema_cfg.get("require_connected", True)),
            allow_rank_skip_edges=bool(schema_cfg.get("allow_rank_skip_edges", True)),
            allow_parallel_fk=bool(schema_cfg.get("allow_parallel_fk", False)),
            require_entity=bool(schema_cfg.get("require_entity", True)),
            require_event=bool(schema_cfg.get("require_event", True)),
            max_outcome_tables=int(schema_cfg.get("max_outcome_tables", 1)),
            require_bridge_parents=bool(schema_cfg.get("require_bridge_parents", True)),
            min_bridge_parents=int(schema_cfg.get("min_bridge_parents", 2)),
            add_random_edges=bool(schema_cfg.get("add_random_edges", True)),
            max_edge_attempts=int(schema_cfg.get("max_edge_attempts", 300)),
            enable_motif_edges=bool(schema_cfg.get("enable_motif_edges", True)),
            enable_transition_check=bool(schema_cfg.get("enable_transition_check", True)),
            strict_transition_check=bool(schema_cfg.get("strict_transition_check", False)),
            seed=int(schema_cfg.get("seed", 42)),
        )



def _normalize_feature_column_range(item: Mapping[str, Any]) -> Dict[str, int]:
    if not isinstance(item, Mapping):
        raise ValueError("feature_columns_by_table_count entries must be mappings.")

    table_count_range = item.get("table_count_range")
    if table_count_range is not None:
        if not isinstance(table_count_range, Sequence) or len(table_count_range) != 2:
            raise ValueError("table_count_range must be a two-item sequence.")
        table_count_min = int(table_count_range[0])
        table_count_max = int(table_count_range[1])
    else:
        table_count_min = int(item.get("table_count_min", item.get("min_table_count")))
        table_count_max = int(item.get("table_count_max", item.get("max_table_count")))

    feature_columns = item.get("feature_columns", item.get("effective_feature_columns"))
    if feature_columns is not None:
        if not isinstance(feature_columns, Sequence) or len(feature_columns) != 2:
            raise ValueError("feature_columns must be a two-item sequence.")
        min_columns = int(feature_columns[0])
        max_columns = int(feature_columns[1])
    else:
        min_columns = int(item.get("min_columns", item.get("min_feature_columns")))
        max_columns = int(item.get("max_columns", item.get("max_feature_columns")))

    return {
        "table_count_min": table_count_min,
        "table_count_max": table_count_max,
        "min_columns": min_columns,
        "max_columns": max_columns,
    }


# ---------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------

class SchemaSampler:
    """
    Sample anonymous relational database schemas.

    Typical use:
        sampler = SchemaSampler.default(seed=42)
        schema = sampler.sample_schema(schema_id="db_000001")
        schema.save_json("outputs/schemas/db_000001/schema.json")
    """

    def __init__(
        self,
        config: Optional[SchemaSamplingConfig] = None,
        motif_library: Optional[MotifLibrary] = None,
        role_sampler: Optional[RoleSampler] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or SchemaSamplingConfig()
        if seed is not None:
            self.config.seed = seed

        self.rng = random.Random(self.config.seed)
        self.motif_library = motif_library or MotifLibrary.default()
        self.role_sampler = role_sampler or RoleSampler.default(seed=self.config.seed)

        self.allowed_transitions: Dict[Role, Tuple[Role, ...]] = {
            normalize_role(src): tuple(normalize_role(dst) for dst in dsts)
            for src, dsts in DEFAULT_ALLOWED_TRANSITIONS.items()
        }

        self._active_max_depth = self.config.max_depth
        self._validate_config()

    @classmethod
    def default(cls, seed: int = 42) -> "SchemaSampler":
        return cls(config=SchemaSamplingConfig(seed=seed), seed=seed)

    @classmethod
    def from_config_dict(cls, config: Mapping[str, Any]) -> "SchemaSampler":
        sampling_config = SchemaSamplingConfig.from_dict(config)
        return cls(config=sampling_config, seed=sampling_config.seed)

    def sample_many(
        self,
        num_schemas: int,
        schema_id_prefix: str = "db",
    ) -> List[SampledSchema]:
        if num_schemas <= 0:
            raise ValueError("num_schemas must be positive.")

        schemas = []
        for i in range(num_schemas):
            schema_id = f"{schema_id_prefix}_{i:06d}"
            schemas.append(self.sample_schema(schema_id=schema_id))
        return schemas

    def sample_schema(self, schema_id: Optional[str] = None) -> SampledSchema:
        schema_id = schema_id or f"schema_{uuid.uuid4().hex[:8]}"

        num_tables = self._sample_num_tables()
        node_ids = [f"T{i}" for i in range(num_tables)]
        target_depth = self._sample_schema_depth(num_tables=num_tables)
        self._active_max_depth = target_depth

        motifs = self._sample_motifs(num_tables=num_tables)
        motif_required_roles = self._required_role_counts_for_motifs(
            motifs=motifs,
            num_tables=num_tables,
        )

        role_inventory = self.role_sampler.sample_inventory(
            num_nodes=num_tables,
            required_roles=motif_required_roles,
        )

        node_roles = self._assign_inventory_to_nodes(node_ids, role_inventory)
        self._repair_bridge_parent_roles(node_roles)
        ranks = self._sample_ranks(node_roles)

        edges: List[Edge] = []
        motif_records: List[Dict[str, Any]] = []

        if self.config.enable_motif_edges:
            for motif in motifs:
                realized_edges, mapping = self._try_realize_motif(
                    motif=motif,
                    node_roles=node_roles,
                    ranks=ranks,
                    existing_edges=edges,
                )
                edges.extend(realized_edges)

                motif_records.append(
                    {
                        "name": motif.name,
                        "family": motif.family,
                        "motif_type": motif.motif_type,
                        "rank_span": motif.rank_span,
                        "role_edges": list(motif.role_edges),
                        "mapping": mapping,
                        "realized_edges": list(realized_edges),
                    }
                )

        if self.config.require_bridge_parents:
            self._ensure_bridge_parent_edges(edges=edges, node_roles=node_roles, ranks=ranks)

        if self.config.require_connected:
            self._ensure_weak_connectivity(edges=edges, node_roles=node_roles, ranks=ranks)

        self._ensure_schema_depth(edges=edges, node_roles=node_roles, target_depth=target_depth)

        if self.config.add_random_edges:
            target_edges = self._sample_target_edge_count(num_tables)
            self._add_random_edges(
                edges=edges,
                node_roles=node_roles,
                ranks=ranks,
                target_edges=target_edges,
            )

        edge_list = sorted(edges, key=lambda x: (x[0], x[1]))
        ranks = self._repair_ranks_from_edges(node_ids=node_ids, edges=edge_list, ranks=ranks)
        violations = self._validate_edges(edge_list, node_roles, ranks)
        actual_schema_depth = max(ranks.values(), default=0)

        if self.config.strict_transition_check and violations:
            message = json.dumps(violations[:10], indent=2, ensure_ascii=False)
            raise ValueError(f"Invalid sampled schema:\n{message}")

        nodes = self._build_nodes(node_roles=node_roles, ranks=ranks, edges=edge_list)
        foreign_keys = self._build_foreign_keys(nodes=nodes, edges=edge_list)

        self._attach_fk_columns(nodes=nodes, foreign_keys=foreign_keys)

        return SampledSchema(
            schema_id=schema_id,
            nodes=nodes,
            foreign_keys=foreign_keys,
            motifs=motif_records,
            violations=violations,
            metadata={
                "generator": "role_motif_time_schema_sampler",
                "fk_graph_note": (
                    "Edges are FK-support/join edges, not direct causal edges."
                ),
                "role_note": (
                    "Roles are latent structural variables used internally by the generator."
                ),
                "temporal_note": (
                    "Summary must be computed as-of cutoff; outcome should be target/future result."
                ),
                "num_sampled_motifs": len(motifs),
                "actual_schema_depth": actual_schema_depth,
                "config": {
                    "min_tables": self.config.min_tables,
                    "max_tables": self.config.max_tables,
                    "table_count_values": list(self.config.table_count_values),
                    "table_count_weights": list(self.config.table_count_weights),
                    "feature_columns_by_table_count": [
                        dict(item) for item in self.config.feature_columns_by_table_count
                    ],
                    "max_depth": self.config.max_depth,
                    "sampled_schema_depth": target_depth,
                    "schema_depth_values": list(self.config.schema_depth_values),
                    "schema_depth_weights": list(self.config.schema_depth_weights),
                    "role_row_count_ranges": {
                        role: list(bounds)
                        for role, bounds in ROLE_ROW_COUNT_RANGES.items()
                    },
                    "min_edge_density": self.config.min_edge_density,
                    "max_edge_density": self.config.max_edge_density,
                    "require_connected": self.config.require_connected,
                    "allow_rank_skip_edges": self.config.allow_rank_skip_edges,
                    "allow_parallel_fk": self.config.allow_parallel_fk,
                    "require_bridge_parents": self.config.require_bridge_parents,
                    "min_bridge_parents": self.config.min_bridge_parents,
                },
            },
        )

    # ------------------------------------------------------------------
    # Motif sampling and realization
    # ------------------------------------------------------------------

    def _effective_max_depth(self) -> int:
        return int(getattr(self, "_active_max_depth", self.config.max_depth))

    def _sample_num_tables(self) -> int:
        explicit_values = tuple(int(value) for value in self.config.table_count_values)
        values = explicit_values or tuple(range(self.config.min_tables, self.config.max_tables + 1))
        weights = tuple(float(weight) for weight in self.config.table_count_weights)

        if weights:
            return int(self.rng.choices(list(values), weights=list(weights), k=1)[0])
        if explicit_values:
            return int(self.rng.choice(values))
        return self.rng.randint(self.config.min_tables, self.config.max_tables)

    def _feature_column_range_for_table_count(self, num_tables: int) -> Optional[Tuple[int, int]]:
        for item in self.config.feature_columns_by_table_count:
            if int(item["table_count_min"]) <= num_tables <= int(item["table_count_max"]):
                return int(item["min_columns"]), int(item["max_columns"])
        return None

    def _sample_schema_depth(self, num_tables: int) -> int:
        max_possible_depth = max(1, min(self.config.max_depth, max(1, num_tables - 1)))
        pairs = [
            (int(depth), float(weight))
            for depth, weight in zip(self.config.schema_depth_values, self.config.schema_depth_weights)
            if 1 <= int(depth) <= max_possible_depth and float(weight) > 0
        ]
        if not pairs:
            return max_possible_depth
        values, weights = zip(*pairs)
        return int(self.rng.choices(list(values), weights=list(weights), k=1)[0])

    def _sample_motifs(self, num_tables: Optional[int] = None) -> List[MotifPattern]:
        k = self.rng.randint(self.config.min_motifs, self.config.max_motifs)
        if k == 0:
            return []

        candidates = self.motif_library.filter(
            max_rank_span=min(self.config.max_motif_rank_span, self._effective_max_depth()),
            priorities=["core", "recommended", "optional"],
            temporal_safe_only=True,
        )
        if not candidates:
            raise ValueError("No motif candidates after filtering.")

        selected: List[MotifPattern] = []
        for _ in range(k):
            feasible = [
                motif
                for motif in candidates
                if self._motif_set_fits_num_tables(
                    motifs=[*selected, motif],
                    num_tables=num_tables,
                )
            ]
            if not feasible:
                break

            weights = [motif.weight for motif in feasible]
            if sum(weights) <= 0:
                raise ValueError("All feasible motif weights are zero.")

            selected.append(self.rng.choices(feasible, weights=weights, k=1)[0])

        if len(selected) < self.config.min_motifs:
            raise ValueError(
                "Unable to sample the configured minimum number of motifs "
                f"within num_tables={num_tables}."
            )

        return selected

    def _required_role_counts_for_motifs(
        self,
        motifs: Sequence[MotifPattern],
        num_tables: int,
    ) -> Counter:
        required = required_role_counts_from_motifs(motifs, aggregate="max")

        if self.config.require_entity:
            required["entity"] = max(required["entity"], 1)
        if self.config.require_event and num_tables >= 2:
            required["event"] = max(required["event"], 1)
        if self.config.max_outcome_tables is not None:
            required["outcome"] = min(
                required["outcome"],
                self.config.max_outcome_tables,
            )

        return required

    def _motif_set_fits_num_tables(
        self,
        motifs: Sequence[MotifPattern],
        num_tables: Optional[int],
    ) -> bool:
        if num_tables is None:
            return True

        raw_required = required_role_counts_from_motifs(motifs, aggregate="max")
        if (
            self.config.max_outcome_tables is not None
            and raw_required["outcome"] > self.config.max_outcome_tables
        ):
            return False

        required = self._required_role_counts_for_motifs(
            motifs=motifs,
            num_tables=num_tables,
        )
        return sum(required.values()) <= num_tables

    def _try_realize_motif(
        self,
        motif: MotifPattern,
        node_roles: Mapping[NodeId, Role],
        ranks: Dict[NodeId, int],
        existing_edges: Sequence[Edge],
    ) -> Tuple[List[Edge], Dict[str, NodeId]]:
        """
        Map local motif nodes to global schema nodes and add compatible edges.

        This is a greedy first-version implementation.
        Later you can replace it with a proper motif packing algorithm.
        """
        mapping: Dict[str, NodeId] = {}
        used: set[NodeId] = set()

        local_nodes = sorted(
            motif.nodes.items(),
            key=lambda kv: self._local_node_sort_key(kv[0], kv[1], motif),
        )

        for local_node, role in local_nodes:
            candidates = [
                node
                for node, global_role in node_roles.items()
                if global_role == role and node not in used
            ]

            if not candidates:
                return [], {}

            candidates = sorted(
                candidates,
                key=lambda n: (
                    ranks[n],
                    self.rng.random(),
                ),
            )
            chosen = candidates[0]
            mapping[local_node] = chosen
            used.add(chosen)

        realized: List[Edge] = []

        for src_local, dst_local in motif.edges:
            src = mapping[src_local]
            dst = mapping[dst_local]

            if src == dst:
                continue

            src_role = node_roles[src]
            dst_role = node_roles[dst]
            original_dst_rank = ranks[dst]

            if not self._is_transition_allowed(src_role, dst_role):
                continue

            if not self._rank_allows_edge(ranks[src], ranks[dst]):
                # Try to repair by promoting child rank when possible.
                if ranks[src] < self._effective_max_depth():
                    ranks[dst] = max(ranks[dst], ranks[src] + 1)
                else:
                    continue

            if not self._rank_allows_edge(ranks[src], ranks[dst]):
                ranks[dst] = original_dst_rank
                continue

            edge = (src, dst)
            if not self.config.allow_parallel_fk and edge in existing_edges:
                ranks[dst] = original_dst_rank
                continue

            if not self._edges_fit_max_depth(
                node_ids=list(node_roles.keys()),
                edges=[*existing_edges, *realized, edge],
            ):
                ranks[dst] = original_dst_rank
                continue

            realized.append(edge)

        return realized, mapping

    @staticmethod
    def _local_node_sort_key(local_node: str, role: Role, motif: MotifPattern) -> Tuple[int, str]:
        indeg = 0
        outdeg = 0
        for src, dst in motif.edges:
            if dst == local_node:
                indeg += 1
            if src == local_node:
                outdeg += 1

        # Sources first, sinks later.
        return (indeg, -outdeg, role)

    # ------------------------------------------------------------------
    # Role and rank sampling
    # ------------------------------------------------------------------

    def _assign_inventory_to_nodes(
        self,
        node_ids: Sequence[NodeId],
        role_inventory: Sequence[Role],
    ) -> Dict[NodeId, Role]:
        if len(node_ids) != len(role_inventory):
            raise ValueError("node_ids and role_inventory must have the same length.")

        inventory = [normalize_role(r) for r in role_inventory]
        self.rng.shuffle(inventory)

        return {node: role for node, role in zip(node_ids, inventory)}

    def _repair_bridge_parent_roles(self, node_roles: Dict[NodeId, Role]) -> None:
        """
        Ensure bridge nodes have enough potential upstream parent roles.

        Bridge is a many-to-many/relation table. A bridge without parent tables is
        structurally invalid for this schema prior, so before rank and edge
        construction we make sure at least min_bridge_parents non-bridge nodes
        can point into bridge.
        """
        if not self.config.require_bridge_parents:
            return

        bridge_nodes = [node for node, role in node_roles.items() if role == "bridge"]
        if not bridge_nodes:
            return

        min_parents = max(1, int(self.config.min_bridge_parents))
        if len(node_roles) <= min_parents:
            # Not enough nodes to keep a bridge with the requested parent count.
            for node in bridge_nodes:
                node_roles[node] = "event"
            return

        parent_roles = {"class", "context", "entity"}
        parent_nodes = [node for node, role in node_roles.items() if role in parent_roles]
        needed = min_parents - len(parent_nodes)
        if needed <= 0:
            return

        bridge_set = set(bridge_nodes)
        candidates = [
            node
            for node, role in node_roles.items()
            if node not in bridge_set and role not in {"outcome", "summary", "measurement"}
        ]
        candidates.extend([node for node in node_roles if node not in bridge_set and node not in candidates])

        for node in candidates:
            if needed <= 0:
                break
            if node_roles[node] not in parent_roles:
                node_roles[node] = "entity"
                needed -= 1

        if needed > 0:
            # Last resort for pathological custom configs: reduce bridge count so
            # the remaining bridge nodes can still be valid.
            for node in reversed(bridge_nodes):
                node_roles[node] = "entity"
                needed -= 1
                if needed <= 0:
                    break

    def _sample_ranks(self, node_roles: Mapping[NodeId, Role]) -> Dict[NodeId, int]:
        ranks: Dict[NodeId, int] = {}

        for node, role in node_roles.items():
            ranks[node] = self._sample_rank_for_role(role)

        if not any(rank == 0 for rank in ranks.values()):
            # Force an upstream role to rank 0.
            candidates = [
                node
                for node, role in node_roles.items()
                if role in {"class", "context", "entity"}
            ] or list(node_roles.keys())
            chosen = self.rng.choice(candidates)
            ranks[chosen] = 0

        if len(node_roles) >= 2 and len(set(ranks.values())) == 1:
            # Ensure at least one downstream table.
            candidates = [
                node
                for node, role in node_roles.items()
                if role not in {"class", "context"}
            ] or list(node_roles.keys())
            chosen = self.rng.choice(candidates)
            ranks[chosen] = min(self._effective_max_depth(), 1)

        return ranks

    @staticmethod
    def _repair_ranks_from_edges(
        node_ids: Sequence[NodeId],
        edges: Sequence[Edge],
        ranks: Mapping[NodeId, int],
    ) -> Dict[NodeId, int]:
        if not edges:
            return {node: int(ranks[node]) for node in node_ids}

        node_set = set(node_ids)
        indeg = {node: 0 for node in node_ids}
        adj: Dict[NodeId, List[NodeId]] = {node: [] for node in node_ids}

        for src, dst in edges:
            if src not in node_set or dst not in node_set:
                continue
            adj[src].append(dst)
            indeg[dst] += 1

        queue = deque([node for node in node_ids if indeg[node] == 0])
        repaired = {node: 0 for node in node_ids}
        visited = 0

        while queue:
            node = queue.popleft()
            visited += 1
            for child in adj[node]:
                repaired[child] = max(repaired[child], repaired[node] + 1)
                indeg[child] -= 1
                if indeg[child] == 0:
                    queue.append(child)

        if visited != len(node_ids):
            return {node: int(ranks[node]) for node in node_ids}

        return repaired

    def _sample_rank_for_role(self, role: Role) -> int:
        role = normalize_role(role)
        max_depth = max(1, self._effective_max_depth())

        if role in {"class", "context"}:
            return self._weighted_choice({0: 0.85, 1: 0.15})

        if role == "entity":
            upper = min(max_depth, 2)
            choices = {0: 0.55, 1: 0.35, 2: 0.10}
            return min(self._weighted_choice(choices), upper)

        if role in {"event", "bridge"}:
            return self.rng.randint(1, max_depth)

        if role == "measurement":
            low = 1 if max_depth == 1 else 2
            return self.rng.randint(low, max_depth)

        if role == "summary":
            low = 1 if max_depth == 1 else 2
            return self.rng.randint(low, max_depth)

        if role == "outcome":
            return max_depth

        return self.rng.randint(0, max_depth)

    def _weighted_choice(self, weights: Mapping[int, float]) -> int:
        keys = list(weights.keys())
        vals = [float(weights[k]) for k in keys]
        return self.rng.choices(keys, weights=vals, k=1)[0]

    # ------------------------------------------------------------------
    # Edge construction
    # ------------------------------------------------------------------

    def _sample_target_edge_count(self, num_tables: int) -> int:
        density = self.rng.uniform(
            self.config.min_edge_density,
            self.config.max_edge_density,
        )
        target = int(round(num_tables * density))
        max_dag_edges = num_tables * (num_tables - 1) // 2
        return max(0, min(target, max_dag_edges))

    def _add_random_edges(
        self,
        edges: List[Edge],
        node_roles: Mapping[NodeId, Role],
        ranks: Mapping[NodeId, int],
        target_edges: int,
    ) -> None:
        attempts = 0

        while len(edges) < target_edges and attempts < self.config.max_edge_attempts:
            attempts += 1

            src, dst = self.rng.sample(list(node_roles.keys()), 2)
            edge = (src, dst)

            if edge in edges and not self.config.allow_parallel_fk:
                continue

            if not self._rank_allows_edge(ranks[src], ranks[dst]):
                continue

            if not self._is_transition_allowed(node_roles[src], node_roles[dst]):
                continue

            # Avoid outcome as parent for temporal safety.
            if node_roles[src] == "outcome":
                continue

            if not self._edges_fit_max_depth(
                node_ids=list(node_roles.keys()),
                edges=[*edges, edge],
            ):
                continue

            edges.append(edge)

    def _ensure_weak_connectivity(
        self,
        edges: List[Edge],
        node_roles: Mapping[NodeId, Role],
        ranks: Mapping[NodeId, int],
    ) -> None:
        components = self._weak_components(list(node_roles.keys()), edges)

        if len(components) <= 1:
            return

        components = sorted(components, key=len, reverse=True)

        base_component = components[0]
        for component in components[1:]:
            added = self._connect_two_components(
                left=base_component,
                right=component,
                edges=edges,
                node_roles=node_roles,
                ranks=ranks,
            )
            if added:
                base_component = base_component | component

    def _connect_two_components(
        self,
        left: set[NodeId],
        right: set[NodeId],
        edges: List[Edge],
        node_roles: Mapping[NodeId, Role],
        ranks: Mapping[NodeId, int],
    ) -> bool:
        candidates: List[Edge] = []

        for a in sorted(left):
            for b in sorted(right):
                for src, dst in [(a, b), (b, a)]:
                    if (src, dst) in edges and not self.config.allow_parallel_fk:
                        continue
                    if self._is_transition_allowed(node_roles[src], node_roles[dst]):
                        if node_roles[src] != "outcome" and self._edges_fit_max_depth(
                            node_ids=list(node_roles.keys()),
                            edges=[*edges, (src, dst)],
                        ):
                            candidates.append((src, dst))

        if not candidates:
            return False

        edges.append(self.rng.choice(candidates))
        return True

    @staticmethod
    def _weak_components(node_ids: Sequence[NodeId], edges: Iterable[Edge]) -> List[set[NodeId]]:
        adj: Dict[NodeId, set[NodeId]] = {node: set() for node in node_ids}

        for src, dst in edges:
            adj[src].add(dst)
            adj[dst].add(src)

        seen: set[NodeId] = set()
        components: List[set[NodeId]] = []

        for node in node_ids:
            if node in seen:
                continue

            comp = set()
            queue = deque([node])
            seen.add(node)

            while queue:
                cur = queue.popleft()
                comp.add(cur)

                for nxt in sorted(adj[cur]):
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)

            components.append(comp)

        return components

    def _ensure_schema_depth(
        self,
        edges: List[Edge],
        node_roles: Mapping[NodeId, Role],
        target_depth: int,
    ) -> None:
        target_depth = max(1, int(target_depth))
        node_ids = list(node_roles.keys())

        for _ in range(self.config.max_edge_attempts):
            ranks = self._repair_ranks_from_edges(
                node_ids=node_ids,
                edges=edges,
                ranks={node: 0 for node in node_ids},
            )
            current_depth = max(ranks.values(), default=0)
            if current_depth >= target_depth:
                return

            candidates: List[Edge] = []
            for src in sorted(node_ids):
                if ranks[src] != current_depth:
                    continue
                if node_roles[src] == "outcome":
                    continue
                for dst in sorted(node_ids):
                    if src == dst:
                        continue
                    edge = (src, dst)
                    if edge in edges and not self.config.allow_parallel_fk:
                        continue
                    if not self._is_transition_allowed(node_roles[src], node_roles[dst]):
                        continue
                    candidate_edges = [*edges, edge]
                    if not self._edges_fit_max_depth(node_ids=node_ids, edges=candidate_edges):
                        continue
                    candidate_ranks = self._repair_ranks_from_edges(
                        node_ids=node_ids,
                        edges=candidate_edges,
                        ranks={node: 0 for node in node_ids},
                    )
                    if max(candidate_ranks.values(), default=0) > current_depth:
                        candidates.append(edge)

            if not candidates:
                return
            edges.append(self.rng.choice(candidates))

    def _ensure_bridge_parent_edges(
        self,
        edges: List[Edge],
        node_roles: Dict[NodeId, Role],
        ranks: Mapping[NodeId, int],
    ) -> None:
        if not self.config.require_bridge_parents:
            return

        min_parents = max(1, int(self.config.min_bridge_parents))
        node_ids = list(node_roles.keys())
        for bridge in sorted(node for node, role in node_roles.items() if role == "bridge"):
            current_parents = {src for src, dst in edges if dst == bridge}
            if len(current_parents) >= min_parents:
                continue

            for parent in self._bridge_parent_candidates(bridge, node_roles, ranks):
                if parent in current_parents:
                    continue
                edge = (parent, bridge)
                if edge in edges and not self.config.allow_parallel_fk:
                    continue
                if not self._edges_fit_max_depth(node_ids=node_ids, edges=[*edges, edge]):
                    continue
                edges.append(edge)
                current_parents.add(parent)
                if len(current_parents) >= min_parents:
                    break

            if len(current_parents) >= min_parents:
                continue

            for parent, repaired_role in self._bridge_parent_role_repair_candidates(
                bridge=bridge,
                node_roles=node_roles,
                edges=edges,
                current_parents=current_parents,
            ):
                original_role = node_roles[parent]
                node_roles[parent] = repaired_role
                edge = (parent, bridge)
                if edge in edges and not self.config.allow_parallel_fk:
                    node_roles[parent] = original_role
                    continue
                candidate_edges = [*edges, edge]
                if not self._edges_fit_max_depth(node_ids=node_ids, edges=candidate_edges):
                    node_roles[parent] = original_role
                    continue
                if not self._edge_role_transitions_valid(candidate_edges, node_roles):
                    node_roles[parent] = original_role
                    continue
                edges.append(edge)
                current_parents.add(parent)
                if len(current_parents) >= min_parents:
                    break

            if len(current_parents) >= min_parents:
                continue

            for parent, repaired_role in self._bridge_parent_detach_repair_candidates(
                bridge=bridge,
                node_roles=node_roles,
                edges=edges,
                current_parents=current_parents,
            ):
                original_role = node_roles[parent]
                node_roles[parent] = repaired_role
                candidate_edges = [edge for edge in edges if edge[1] != parent]
                edge = (parent, bridge)
                if edge in candidate_edges and not self.config.allow_parallel_fk:
                    node_roles[parent] = original_role
                    continue
                candidate_edges.append(edge)
                if not self._edges_fit_max_depth(node_ids=node_ids, edges=candidate_edges):
                    node_roles[parent] = original_role
                    continue
                if not self._edge_role_transitions_valid(candidate_edges, node_roles):
                    node_roles[parent] = original_role
                    continue
                edges[:] = candidate_edges
                current_parents.add(parent)
                if len(current_parents) >= min_parents:
                    break

    def _bridge_parent_role_repair_candidates(
        self,
        bridge: NodeId,
        node_roles: Mapping[NodeId, Role],
        edges: Sequence[Edge],
        current_parents: set[NodeId],
    ) -> List[Tuple[NodeId, Role]]:
        parent_roles = ("entity", "context", "class")
        incoming_count = Counter(dst for _, dst in edges)
        outgoing_count = Counter(src for src, _ in edges)
        candidates: List[Tuple[Tuple[int, int, str, str], NodeId, Role]] = []

        for node, original_role in node_roles.items():
            if node == bridge or node in current_parents:
                continue
            if original_role in {"bridge", "outcome"}:
                continue

            for repaired_role in parent_roles:
                repaired_roles = dict(node_roles)
                repaired_roles[node] = repaired_role
                if not self._is_transition_allowed(repaired_role, "bridge"):
                    continue
                if not self._edge_role_transitions_valid(edges, repaired_roles):
                    continue
                score = (
                    incoming_count.get(node, 0),
                    outgoing_count.get(node, 0),
                    repaired_role,
                    node,
                )
                candidates.append((score, node, repaired_role))

        candidates.sort(key=lambda item: item[0])
        return [(node, role) for _, node, role in candidates]

    def _bridge_parent_detach_repair_candidates(
        self,
        bridge: NodeId,
        node_roles: Mapping[NodeId, Role],
        edges: Sequence[Edge],
        current_parents: set[NodeId],
    ) -> List[Tuple[NodeId, Role]]:
        parent_roles = ("entity", "context", "class")
        incoming_count = Counter(dst for _, dst in edges)
        outgoing_count = Counter(src for src, _ in edges)
        candidates: List[Tuple[Tuple[int, int, str, str], NodeId, Role]] = []

        for node, original_role in node_roles.items():
            if node == bridge or node in current_parents:
                continue
            if original_role in {"bridge", "outcome"}:
                continue

            for repaired_role in parent_roles:
                repaired_roles = dict(node_roles)
                repaired_roles[node] = repaired_role
                candidate_edges = [edge for edge in edges if edge[1] != node]
                candidate_edges.append((node, bridge))
                if not self._edge_role_transitions_valid(candidate_edges, repaired_roles):
                    continue
                score = (
                    incoming_count.get(node, 0),
                    outgoing_count.get(node, 0),
                    repaired_role,
                    node,
                )
                candidates.append((score, node, repaired_role))

        candidates.sort(key=lambda item: item[0])
        return [(node, role) for _, node, role in candidates]

    def _edge_role_transitions_valid(
        self,
        edges: Sequence[Edge],
        node_roles: Mapping[NodeId, Role],
    ) -> bool:
        for src, dst in edges:
            if src not in node_roles or dst not in node_roles:
                return False
            if src == dst:
                return False
            if self.config.enable_transition_check and not self._is_transition_allowed(
                node_roles[src],
                node_roles[dst],
            ):
                return False
        return True

    def _bridge_parent_candidates(
        self,
        bridge: NodeId,
        node_roles: Mapping[NodeId, Role],
        ranks: Mapping[NodeId, int],
    ) -> List[NodeId]:
        parent_roles = {"class", "context", "entity"}
        candidates = [
            node
            for node, role in node_roles.items()
            if node != bridge
            and role in parent_roles
            and self._is_transition_allowed(role, "bridge")
        ]
        return sorted(candidates, key=lambda node: (ranks.get(node, 0), node))

    def _edges_fit_max_depth(
        self,
        node_ids: Sequence[NodeId],
        edges: Sequence[Edge],
    ) -> bool:
        if self._has_cycle(node_ids, edges):
            return False
        ranks = self._repair_ranks_from_edges(
            node_ids=node_ids,
            edges=edges,
            ranks={node: 0 for node in node_ids},
        )
        return max(ranks.values(), default=0) <= self._effective_max_depth()

    def _rank_allows_edge(self, src_rank: int, dst_rank: int) -> bool:
        if src_rank >= dst_rank:
            return False
        if self.config.allow_rank_skip_edges:
            return True
        return dst_rank == src_rank + 1

    def _is_transition_allowed(self, src_role: Role, dst_role: Role) -> bool:
        src = normalize_role(src_role)
        dst = normalize_role(dst_role)

        if src == "outcome":
            return False

        return dst in self.allowed_transitions.get(src, tuple())

    # ------------------------------------------------------------------
    # Node, FK, and column construction
    # ------------------------------------------------------------------

    def _build_nodes(
        self,
        node_roles: Mapping[NodeId, Role],
        ranks: Mapping[NodeId, int],
        edges: Sequence[Edge],
    ) -> Dict[NodeId, SchemaNode]:
        incoming = defaultdict(list)
        outgoing = defaultdict(list)

        for src, dst in edges:
            outgoing[src].append(dst)
            incoming[dst].append(src)

        nodes: Dict[NodeId, SchemaNode] = {}

        for node_id, role in node_roles.items():
            role = normalize_role(role)
            primary_key = f"{node_id.lower()}_id"
            time_col = self._default_time_col(node_id, role)
            num_rows = self._sample_num_rows(role)
            feature_col_count = self._sample_feature_col_count(role, num_tables=len(node_roles))

            columns = [
                ColumnSpec(
                    name=primary_key,
                    dtype="int64",
                    semantic_type="primary_key",
                    is_primary_key=True,
                )
            ]

            if time_col is not None:
                columns.append(
                    ColumnSpec(
                        name=time_col,
                        dtype="datetime64",
                        semantic_type="timestamp",
                        is_time=True,
                    )
                )

            for i in range(feature_col_count):
                columns.append(self._sample_feature_column(node_id, role, i))

            if role == "outcome":
                columns.append(
                    ColumnSpec(
                        name="label",
                        dtype="float32",
                        semantic_type="outcome_label",
                        is_label_candidate=True,
                    )
                )

            nodes[node_id] = SchemaNode(
                node_id=node_id,
                role=role,
                rank=int(ranks[node_id]),
                num_rows=num_rows,
                num_columns=len(columns),
                primary_key=primary_key,
                time_col=time_col,
                columns=columns,
                metadata={
                    "indegree": len(incoming[node_id]),
                    "outdegree": len(outgoing[node_id]),
                    "profile": dict(DEFAULT_ROLE_PROFILES.get(role, {})),
                    "visible_as_feature_by_default": role != "outcome",
                    "requires_as_of_cutoff": role == "summary",
                    "feature_col_count": feature_col_count,
                },
            )

        return nodes

    def _build_foreign_keys(
        self,
        nodes: Mapping[NodeId, SchemaNode],
        edges: Sequence[Edge],
    ) -> List[ForeignKeySpec]:
        foreign_keys: List[ForeignKeySpec] = []
        child_col_counts: Counter = Counter()

        for parent, child in edges:
            parent_role = nodes[parent].role
            child_role = nodes[child].role
            key = (child, parent)
            child_col_counts[key] += 1
            suffix = "" if child_col_counts[key] == 1 else f"_{child_col_counts[key]}"
            child_col = f"{parent.lower()}_id{suffix}"

            foreign_keys.append(
                ForeignKeySpec(
                    parent_table=parent,
                    child_table=child,
                    parent_col=nodes[parent].primary_key,
                    child_col=child_col,
                    cardinality=self._infer_cardinality(parent_role, child_role),
                    optional=self._sample_optional_fk(parent_role, child_role),
                    fanout_prior=self._fanout_prior(parent_role, child_role),
                )
            )

        return foreign_keys

    def _attach_fk_columns(
        self,
        nodes: Mapping[NodeId, SchemaNode],
        foreign_keys: Sequence[ForeignKeySpec],
    ) -> None:
        for fk in foreign_keys:
            child = nodes[fk.child_table]
            if any(c.name == fk.child_col for c in child.columns):
                continue

            child.columns.append(
                ColumnSpec(
                    name=fk.child_col,
                    dtype="int64",
                    semantic_type="foreign_key",
                    is_foreign_key=True,
                    references={
                        "table": fk.parent_table,
                        "column": fk.parent_col,
                    },
                )
            )
            child.num_columns = len(child.columns)

    def _default_time_col(self, node_id: NodeId, role: Role) -> Optional[str]:
        role = normalize_role(role)
        profile = DEFAULT_ROLE_PROFILES.get(role, {})
        if profile.get("has_time_col", False):
            return f"{node_id.lower()}_time"

        # Some entities can have a creation time, but not required in schema prior.
        if role == "entity" and self.rng.random() < 0.25:
            return f"{node_id.lower()}_created_time"

        return None

    def _sample_feature_col_count(self, role: Role, num_tables: Optional[int] = None) -> int:
        role = normalize_role(role)

        if num_tables is not None and role != "outcome":
            table_count_range = self._feature_column_range_for_table_count(num_tables)
            if table_count_range is not None:
                low, high = table_count_range
                return self.rng.randint(low, high)

        ranges = {
            "class": (1, 4),
            "context": (1, 4),
            "entity": (3, 10),
            "event": (2, 8),
            "bridge": (0, 3),
            "measurement": (2, 8),
            "summary": (3, 10),
            "outcome": (0, 3),
        }
        low, high = ranges.get(role, (1, 5))
        return self.rng.randint(low, high)

    def _sample_num_rows(self, role: Role) -> int:
        role = normalize_role(role)
        low, high = ROLE_ROW_COUNT_RANGES.get(role, (100, 10_000))
        if low == high:
            return low
        return self._log_uniform_int(low, high)

    def _sample_feature_column(self, node_id: NodeId, role: Role, index: int) -> ColumnSpec:
        role = normalize_role(role)

        weights_by_role = {
            "class": {"categorical": 0.8, "numeric": 0.2},
            "context": {"categorical": 0.7, "numeric": 0.3},
            "entity": {"categorical": 0.45, "numeric": 0.55},
            "event": {"categorical": 0.35, "numeric": 0.65},
            "bridge": {"categorical": 0.6, "numeric": 0.4},
            "measurement": {"categorical": 0.15, "numeric": 0.85},
            "summary": {"categorical": 0.15, "numeric": 0.85},
            "outcome": {"categorical": 0.35, "numeric": 0.65},
        }

        weights = weights_by_role.get(role, {"categorical": 0.5, "numeric": 0.5})
        semantic_type = self.rng.choices(
            list(weights.keys()),
            weights=list(weights.values()),
            k=1,
        )[0]

        if semantic_type == "categorical":
            return ColumnSpec(
                name=f"{node_id.lower()}_cat_{index}",
                dtype="category",
                semantic_type="categorical",
            )

        return ColumnSpec(
            name=f"{node_id.lower()}_num_{index}",
            dtype="float32",
            semantic_type="numeric",
        )

    def _log_uniform_int(self, low: int, high: int) -> int:
        if low <= 0 or high < low:
            raise ValueError("Invalid log-uniform integer range.")
        x = self.rng.random()
        value = int(round(math.exp(math.log(low) + x * (math.log(high) - math.log(low)))))
        return max(low, min(value, high))

    def _infer_cardinality(self, parent_role: Role, child_role: Role) -> str:
        parent_role = normalize_role(parent_role)
        child_role = normalize_role(child_role)

        if child_role == "bridge":
            return "many_to_many_component"
        if parent_role in {"class", "context"}:
            return "one_to_many_dimension"
        if child_role in {"event", "measurement"}:
            return "one_to_many"
        if child_role == "summary":
            return "one_to_many_or_windowed"
        if child_role == "outcome":
            return "one_to_one_or_one_to_many_task"
        return "one_to_many"

    def _sample_optional_fk(self, parent_role: Role, child_role: Role) -> bool:
        parent_role = normalize_role(parent_role)
        child_role = normalize_role(child_role)

        if child_role in {"bridge", "event"}:
            return self.rng.random() < 0.05
        if parent_role in {"class", "context"}:
            return self.rng.random() < 0.15
        return self.rng.random() < 0.10

    def _fanout_prior(self, parent_role: Role, child_role: Role) -> Dict[str, Any]:
        parent_role = normalize_role(parent_role)
        child_role = normalize_role(child_role)

        if parent_role == "entity" and child_role == "event":
            return {"distribution": "zipf_lognormal", "mean_range": [2, 100], "long_tail": True}
        if parent_role in {"class", "context"}:
            return {"distribution": "categorical_long_tail", "mean_range": [5, 500], "long_tail": True}
        if child_role == "bridge":
            return {"distribution": "pair_sampling", "density_range": [0.001, 0.2]}
        if child_role == "summary":
            return {"distribution": "windowed_groupby", "as_of_cutoff": True}
        if child_role == "outcome":
            return {"distribution": "task_samples", "future_window": True}
        return {"distribution": "lognormal", "long_tail": True}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_edges(
        self,
        edges: Sequence[Edge],
        node_roles: Mapping[NodeId, Role],
        ranks: Mapping[NodeId, int],
    ) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []

        for src, dst in edges:
            if src not in node_roles or dst not in node_roles:
                violations.append(
                    {"edge": [src, dst], "reason": "unknown_node"}
                )
                continue

            if src == dst:
                violations.append(
                    {"edge": [src, dst], "reason": "self_loop"}
                )

            if not self._rank_allows_edge(ranks[src], ranks[dst]):
                violations.append(
                    {
                        "edge": [src, dst],
                        "reason": "rank_violation",
                        "src_rank": ranks[src],
                        "dst_rank": ranks[dst],
                    }
                )

            if self.config.enable_transition_check and not self._is_transition_allowed(
                node_roles[src],
                node_roles[dst],
            ):
                violations.append(
                    {
                        "edge": [src, dst],
                        "reason": "role_transition_violation",
                        "src_role": node_roles[src],
                        "dst_role": node_roles[dst],
                    }
                )

            if node_roles[src] == "outcome":
                violations.append(
                    {
                        "edge": [src, dst],
                        "reason": "temporal_safety_violation",
                        "message": "Outcome should not be parent of visible tables.",
                    }
                )

        if self.config.require_bridge_parents:
            bridge_parent_sets: Dict[NodeId, set[NodeId]] = defaultdict(set)
            for src, dst in edges:
                bridge_parent_sets[dst].add(src)

            min_parents = max(1, int(self.config.min_bridge_parents))
            for node, role in node_roles.items():
                if role != "bridge":
                    continue
                parents = sorted(bridge_parent_sets.get(node, set()))
                if len(parents) < min_parents:
                    violations.append(
                        {
                            "node": node,
                            "reason": "bridge_without_required_parents",
                            "num_parents": len(parents),
                            "min_required_parents": min_parents,
                            "parents": parents,
                        }
                    )

        if self._has_cycle(list(node_roles.keys()), edges):
            violations.append({"reason": "cycle_detected"})

        return violations

    @staticmethod
    def _has_cycle(node_ids: Sequence[NodeId], edges: Sequence[Edge]) -> bool:
        indeg = {node: 0 for node in node_ids}
        adj = {node: [] for node in node_ids}

        for src, dst in edges:
            if src not in adj or dst not in indeg:
                continue
            adj[src].append(dst)
            indeg[dst] += 1

        queue = deque([n for n in node_ids if indeg[n] == 0])
        visited = 0

        while queue:
            node = queue.popleft()
            visited += 1

            for nxt in adj[node]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)

        return visited != len(node_ids)

    def _validate_config(self) -> None:
        if self.config.min_tables <= 0:
            raise ValueError("min_tables must be positive.")
        if self.config.max_tables < self.config.min_tables:
            raise ValueError("max_tables must be >= min_tables.")
        if self.config.table_count_values:
            if any(int(value) <= 0 for value in self.config.table_count_values):
                raise ValueError("table_count_values must contain positive integers.")
            if len(set(int(value) for value in self.config.table_count_values)) != len(self.config.table_count_values):
                raise ValueError("table_count_values must not contain duplicates.")
            if any(
                int(value) < self.config.min_tables or int(value) > self.config.max_tables
                for value in self.config.table_count_values
            ):
                raise ValueError("table_count_values must stay within min_tables/max_tables.")
        if self.config.table_count_weights:
            expected = len(self.config.table_count_values) if self.config.table_count_values else (
                self.config.max_tables - self.config.min_tables + 1
            )
            if len(self.config.table_count_weights) != expected:
                raise ValueError("table_count_weights must match table_count_values, or min_tables..max_tables when values are omitted.")
            if any(float(weight) < 0 for weight in self.config.table_count_weights):
                raise ValueError("table_count_weights must be non-negative.")
            if sum(float(weight) for weight in self.config.table_count_weights) <= 0:
                raise ValueError("table_count_weights must contain a positive total weight.")
        for item in self.config.feature_columns_by_table_count:
            if int(item["table_count_min"]) <= 0:
                raise ValueError("feature_columns_by_table_count table_count_min must be positive.")
            if int(item["table_count_max"]) < int(item["table_count_min"]):
                raise ValueError("feature_columns_by_table_count table_count_max must be >= table_count_min.")
            if int(item["min_columns"]) < 0:
                raise ValueError("feature_columns_by_table_count min_columns must be non-negative.")
            if int(item["max_columns"]) < int(item["min_columns"]):
                raise ValueError("feature_columns_by_table_count max_columns must be >= min_columns.")
        if self.config.max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if not self.config.schema_depth_values:
            raise ValueError("schema_depth_values must not be empty.")
        if len(self.config.schema_depth_values) != len(self.config.schema_depth_weights):
            raise ValueError("schema_depth_values and schema_depth_weights must have the same length.")
        if any(int(depth) < 1 for depth in self.config.schema_depth_values):
            raise ValueError("schema_depth_values must be positive depths.")
        if sum(float(weight) for weight in self.config.schema_depth_weights) <= 0:
            raise ValueError("schema_depth_weights must contain a positive total weight.")
        if self.config.min_edge_density < 0:
            raise ValueError("min_edge_density must be non-negative.")
        if self.config.max_edge_density < self.config.min_edge_density:
            raise ValueError("max_edge_density must be >= min_edge_density.")
        if self.config.min_motifs < 0:
            raise ValueError("min_motifs must be non-negative.")
        if self.config.max_motifs < self.config.min_motifs:
            raise ValueError("max_motifs must be >= min_motifs.")
        if self.config.max_outcome_tables is not None and self.config.max_outcome_tables < 0:
            raise ValueError("max_outcome_tables must be non-negative.")
        if self.config.min_bridge_parents < 0:
            raise ValueError("min_bridge_parents must be non-negative.")


__all__ = [
    "Role",
    "NodeId",
    "Edge",
    "ROLE_ROW_COUNT_RANGES",
    "ColumnSpec",
    "SchemaNode",
    "ForeignKeySpec",
    "SampledSchema",
    "SchemaSamplingConfig",
    "SchemaSampler",
]