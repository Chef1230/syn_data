# src/rdb_prior/schema/motif_library.py
# -*- coding: utf-8 -*-
"""
Motif library for role/motif/time-aware synthetic relational database generation.

Important design choice:
    - Edges here are FK-support edges, not causal edges.
    - A -> B means B may reference A through FK, and A can be used as join context.
    - This does NOT mean A is necessarily a direct causal parent of B at attribute level.

Role convention:
    entity      : stable business object
    class       : low-cardinality dimension/category table
    context     : contextual dimension, such as location/platform/condition
    event       : timestamped interaction or business event
    bridge      : many-to-many relation table
    measurement : observation/detail table derived from entity/event
    summary     : historical/as-of aggregation table
    outcome     : task label or future-window result

Temporal safety convention:
    - summary should only be generated from cutoff-before historical data.
    - outcome should be treated as target/future-window label, not as visible feature.
    - outcome -> summary is forbidden by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import random


Role = str
NodeId = str
Edge = Tuple[NodeId, NodeId]


ROLE_ALIASES: Dict[str, str] = {
    "entity": "entity",
    "actor": "entity",
    "object": "entity",

    "class": "class",
    "dimension": "class",
    "dim": "class",

    "context": "context",

    "event": "event",
    "transaction": "event",
    "interaction": "event",

    "bridge": "bridge",
    "detail": "bridge",
    "relation": "bridge",

    "measurement": "measurement",
    "measure": "measurement",
    "observation": "measurement",

    "summary": "summary",
    "aggregate": "summary",
    "aggregation": "summary",

    "outcome": "outcome",
    "label": "outcome",
    "target": "outcome",
}

VALID_ROLES = frozenset(
    {
        "entity",
        "class",
        "context",
        "event",
        "bridge",
        "measurement",
        "summary",
        "outcome",
    }
)

VALID_MOTIF_TYPES = frozenset(
    {
        "chain",
        "fork",
        "collider",
        "diamond",
        "star",
        "snowflake",
        "mixed",
    }
)

VALID_PRIORITIES = frozenset({"core", "recommended", "optional"})


def normalize_role(role: str) -> str:
    """Normalize role names to canonical lowercase names."""
    key = role.strip().lower()
    if key not in ROLE_ALIASES:
        raise ValueError(f"Unknown role: {role!r}. Valid roles: {sorted(VALID_ROLES)}")
    return ROLE_ALIASES[key]


@dataclass(frozen=True)
class MotifPattern:
    """
    A local role-level motif.

    Attributes
    ----------
    name:
        Unique motif name.
    family:
        Semantic family, e.g., interaction, bridge, summary, task.
    motif_type:
        chain / fork / collider / diamond / star / snowflake / mixed.
    rank_span:
        Intended local depth/rank span of the motif.
        For example, entity -> event has rank_span=1;
        entity -> event -> summary has rank_span=2.
    nodes:
        Mapping from local node id to canonical role.
    edges:
        FK-support edges over local node ids.
    weight:
        Sampling weight. Can be overridden by config.
    priority:
        core / recommended / optional.
    temporal_safe:
        Whether this motif is safe under the default temporal convention.
    description:
        Human-readable explanation.
    """

    name: str
    family: str
    motif_type: str
    rank_span: int
    nodes: Dict[NodeId, Role]
    edges: Tuple[Edge, ...]
    weight: float = 1.0
    priority: str = "core"
    temporal_safe: bool = True
    description: str = ""
    tags: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "nodes",
            {node_id: normalize_role(role) for node_id, role in self.nodes.items()},
        )

        if self.motif_type not in VALID_MOTIF_TYPES:
            raise ValueError(
                f"Invalid motif_type={self.motif_type!r} for motif={self.name!r}."
            )

        if self.priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority={self.priority!r} for motif={self.name!r}."
            )

        if self.rank_span < 1:
            raise ValueError(f"rank_span must be >= 1 for motif={self.name!r}.")

        if self.weight < 0:
            raise ValueError(f"weight must be non-negative for motif={self.name!r}.")

        node_set = set(self.nodes)
        for src, dst in self.edges:
            if src not in node_set:
                raise ValueError(f"Unknown src node={src!r} in motif={self.name!r}.")
            if dst not in node_set:
                raise ValueError(f"Unknown dst node={dst!r} in motif={self.name!r}.")
            if src == dst:
                raise ValueError(f"Self-loop {src}->{dst} in motif={self.name!r}.")

        if not self._is_dag():
            raise ValueError(f"Motif must be DAG-like, got cycle in {self.name!r}.")

        self._validate_temporal_safety_rules()

    @property
    def roles(self) -> Tuple[Role, ...]:
        return tuple(self.nodes.values())

    @property
    def role_edges(self) -> Tuple[Tuple[Role, Role], ...]:
        return tuple((self.nodes[src], self.nodes[dst]) for src, dst in self.edges)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def contains_role(self, role: str) -> bool:
        return normalize_role(role) in set(self.nodes.values())

    def contains_any_role(self, roles: Iterable[str]) -> bool:
        target = {normalize_role(role) for role in roles}
        return bool(set(self.nodes.values()) & target)

    def contains_all_roles(self, roles: Iterable[str]) -> bool:
        target = {normalize_role(role) for role in roles}
        return target.issubset(set(self.nodes.values()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "motif_type": self.motif_type,
            "rank_span": self.rank_span,
            "nodes": dict(self.nodes),
            "edges": list(self.edges),
            "role_edges": list(self.role_edges),
            "weight": self.weight,
            "priority": self.priority,
            "temporal_safe": self.temporal_safe,
            "description": self.description,
            "tags": list(self.tags),
        }

    def with_weight(self, weight: float) -> "MotifPattern":
        return MotifPattern(
            name=self.name,
            family=self.family,
            motif_type=self.motif_type,
            rank_span=self.rank_span,
            nodes=dict(self.nodes),
            edges=tuple(self.edges),
            weight=weight,
            priority=self.priority,
            temporal_safe=self.temporal_safe,
            description=self.description,
            tags=tuple(self.tags),
        )

    def _is_dag(self) -> bool:
        """Small dependency-free DAG check."""
        indeg = {node: 0 for node in self.nodes}
        adj = {node: [] for node in self.nodes}

        for src, dst in self.edges:
            adj[src].append(dst)
            indeg[dst] += 1

        queue = [node for node, deg in indeg.items() if deg == 0]
        visited = 0

        while queue:
            node = queue.pop()
            visited += 1
            for nxt in adj[node]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)

        return visited == len(self.nodes)

    def _validate_temporal_safety_rules(self) -> None:
        """
        Conservative safety rules.

        Outcome should not be a parent of summary/event/measurement by default.
        Summary should not be generated from future outcome.
        """
        for src, dst in self.edges:
            src_role = self.nodes[src]
            dst_role = self.nodes[dst]

            if src_role == "outcome" and dst_role in {
                "summary",
                "event",
                "measurement",
                "bridge",
                "entity",
                "class",
                "context",
            }:
                raise ValueError(
                    f"Temporal-unsafe edge outcome -> {dst_role} in motif={self.name!r}. "
                    "Outcome should be target/future result, not parent of visible tables."
                )


class MotifLibrary:
    """Registry, filtering, and sampling interface for motif patterns."""

    def __init__(self, motifs: Sequence[MotifPattern]):
        if not motifs:
            raise ValueError("MotifLibrary requires at least one motif.")

        self._motifs: Tuple[MotifPattern, ...] = tuple(motifs)
        self._by_name: Dict[str, MotifPattern] = {}

        for motif in self._motifs:
            if motif.name in self._by_name:
                raise ValueError(f"Duplicate motif name: {motif.name}")
            self._by_name[motif.name] = motif

    @classmethod
    def default(cls) -> "MotifLibrary":
        return cls(build_default_motifs())

    def names(self) -> List[str]:
        return list(self._by_name.keys())

    def get(self, name: str) -> MotifPattern:
        if name not in self._by_name:
            raise KeyError(f"Unknown motif={name!r}. Available: {self.names()}")
        return self._by_name[name]

    def all(self) -> List[MotifPattern]:
        return list(self._motifs)

    def filter(
        self,
        *,
        max_rank_span: Optional[int] = None,
        motif_types: Optional[Iterable[str]] = None,
        families: Optional[Iterable[str]] = None,
        priorities: Optional[Iterable[str]] = None,
        required_roles: Optional[Iterable[str]] = None,
        excluded_roles: Optional[Iterable[str]] = None,
        temporal_safe_only: bool = True,
        tags_any: Optional[Iterable[str]] = None,
    ) -> List[MotifPattern]:
        motifs = list(self._motifs)

        if max_rank_span is not None:
            motifs = [m for m in motifs if m.rank_span <= max_rank_span]

        if motif_types is not None:
            allowed = set(motif_types)
            motifs = [m for m in motifs if m.motif_type in allowed]

        if families is not None:
            allowed = set(families)
            motifs = [m for m in motifs if m.family in allowed]

        if priorities is not None:
            allowed = set(priorities)
            motifs = [m for m in motifs if m.priority in allowed]

        if required_roles is not None:
            motifs = [m for m in motifs if m.contains_all_roles(required_roles)]

        if excluded_roles is not None:
            excluded = {normalize_role(role) for role in excluded_roles}
            motifs = [m for m in motifs if not (set(m.roles) & excluded)]

        if temporal_safe_only:
            motifs = [m for m in motifs if m.temporal_safe]

        if tags_any is not None:
            wanted = set(tags_any)
            motifs = [m for m in motifs if set(m.tags) & wanted]

        return motifs

    def sample(
        self,
        *,
        rng: Optional[random.Random] = None,
        k: int = 1,
        replace: bool = True,
        max_rank_span: Optional[int] = None,
        motif_types: Optional[Iterable[str]] = None,
        families: Optional[Iterable[str]] = None,
        priorities: Optional[Iterable[str]] = None,
        required_roles: Optional[Iterable[str]] = None,
        excluded_roles: Optional[Iterable[str]] = None,
        temporal_safe_only: bool = True,
        tags_any: Optional[Iterable[str]] = None,
    ) -> List[MotifPattern]:
        rng = rng or random.Random()

        candidates = self.filter(
            max_rank_span=max_rank_span,
            motif_types=motif_types,
            families=families,
            priorities=priorities,
            required_roles=required_roles,
            excluded_roles=excluded_roles,
            temporal_safe_only=temporal_safe_only,
            tags_any=tags_any,
        )

        if not candidates:
            raise ValueError("No motif candidates after filtering.")

        weights = [m.weight for m in candidates]
        if sum(weights) <= 0:
            raise ValueError("All candidate motif weights are zero.")

        if replace:
            return rng.choices(candidates, weights=weights, k=k)

        if k > len(candidates):
            raise ValueError(
                f"Cannot sample k={k} motifs without replacement from "
                f"{len(candidates)} candidates."
            )

        selected: List[MotifPattern] = []
        pool = list(candidates)
        pool_weights = list(weights)

        for _ in range(k):
            idx = _weighted_index(rng, pool_weights)
            selected.append(pool.pop(idx))
            pool_weights.pop(idx)

        return selected

    def apply_config(self, config: Dict[str, Any]) -> "MotifLibrary":
        """
        Return a new MotifLibrary with config-based weight overrides or disabled motifs.

        Expected config example:
            {
                "disable": ["entity_outcome_entity_collider"],
                "weights": {
                    "entity_event": 2.0,
                    "entity_event_summary_outcome_chain": 3.0
                }
            }
        """
        disabled = set(config.get("disable", []))
        weight_overrides = config.get("weights", {})

        new_motifs: List[MotifPattern] = []

        for motif in self._motifs:
            if motif.name in disabled:
                continue
            if motif.name in weight_overrides:
                new_motifs.append(motif.with_weight(float(weight_overrides[motif.name])))
            else:
                new_motifs.append(motif)

        return MotifLibrary(new_motifs)

    def role_transition_counts(self) -> Dict[Tuple[Role, Role], int]:
        counts: Dict[Tuple[Role, Role], int] = {}
        for motif in self._motifs:
            for src_role, dst_role in motif.role_edges:
                key = (src_role, dst_role)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {"motifs": [m.to_dict() for m in self._motifs]}

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def _weighted_index(rng: random.Random, weights: Sequence[float]) -> int:
    total = float(sum(weights))
    threshold = rng.random() * total
    acc = 0.0
    for idx, weight in enumerate(weights):
        acc += float(weight)
        if acc >= threshold:
            return idx
    return len(weights) - 1


def build_default_motifs() -> List[MotifPattern]:
    """
    Build the default motif library.

    The library intentionally avoids natural-language table names.
    It only contains structural roles and FK-support directions.
    """

    motifs: List[MotifPattern] = []

    def add(
        name: str,
        family: str,
        motif_type: str,
        rank_span: int,
        nodes: Dict[str, str],
        edges: Sequence[Edge],
        weight: float,
        priority: str,
        description: str,
        tags: Sequence[str] = (),
        temporal_safe: bool = True,
    ) -> None:
        motifs.append(
            MotifPattern(
                name=name,
                family=family,
                motif_type=motif_type,
                rank_span=rank_span,
                nodes=nodes,
                edges=tuple(edges),
                weight=weight,
                priority=priority,
                temporal_safe=temporal_safe,
                description=description,
                tags=tuple(tags),
            )
        )

    # ------------------------------------------------------------------
    # Depth/rank-span 1: basic database building blocks
    # ------------------------------------------------------------------
    add(
        name="entity_event_chain",
        family="interaction",
        motif_type="chain",
        rank_span=1,
        nodes={"e": "entity", "v": "event"},
        edges=[("e", "v")],
        weight=3.0,
        priority="core",
        description="A stable entity participates in or generates a timestamped event.",
        tags=["entity", "event", "core"],
    )

    add(
        name="class_entity_chain",
        family="dimension",
        motif_type="chain",
        rank_span=1,
        nodes={"c": "class", "e": "entity"},
        edges=[("c", "e")],
        weight=2.0,
        priority="core",
        description="A low-cardinality class/dimension describes an entity.",
        tags=["class", "entity", "dimension"],
    )

    add(
        name="context_event_chain",
        family="context",
        motif_type="chain",
        rank_span=1,
        nodes={"ctx": "context", "v": "event"},
        edges=[("ctx", "v")],
        weight=1.5,
        priority="core",
        description="A context table describes where or under what condition an event occurs.",
        tags=["context", "event"],
    )

    add(
        name="event_measurement_chain",
        family="measurement",
        motif_type="chain",
        rank_span=1,
        nodes={"v": "event", "m": "measurement"},
        edges=[("v", "m")],
        weight=1.8,
        priority="core",
        description="An event produces detailed observations or measurements.",
        tags=["event", "measurement"],
    )

    add(
        name="entity_event_entity_collider",
        family="interaction",
        motif_type="collider",
        rank_span=1,
        nodes={"e1": "entity", "e2": "entity", "v": "event"},
        edges=[("e1", "v"), ("e2", "v")],
        weight=3.0,
        priority="core",
        description="A multi-entity event, such as user-item, driver-race, or student-course interaction.",
        tags=["entity", "event", "collider", "core"],
    )

    add(
        name="entity_bridge_entity_collider",
        family="bridge",
        motif_type="collider",
        rank_span=1,
        nodes={"e1": "entity", "e2": "entity", "b": "bridge"},
        edges=[("e1", "b"), ("e2", "b")],
        weight=2.6,
        priority="core",
        description="A many-to-many bridge relation between two entity-like tables.",
        tags=["entity", "bridge", "many_to_many", "core"],
    )

    add(
        name="entity_event_fork",
        family="history",
        motif_type="fork",
        rank_span=1,
        nodes={"e": "entity", "v1": "event", "v2": "event"},
        edges=[("e", "v1"), ("e", "v2")],
        weight=2.4,
        priority="core",
        description="One entity connects to multiple event tables, forming a multi-event history.",
        tags=["entity", "event", "fork", "history"],
    )

    add(
        name="entity_entity_event_fork",
        family="hierarchy",
        motif_type="fork",
        rank_span=1,
        nodes={"e0": "entity", "e1": "entity", "v": "event"},
        edges=[("e0", "e1"), ("e0", "v")],
        weight=1.0,
        priority="recommended",
        description="A higher-level entity points to another entity and an event table.",
        tags=["entity", "hierarchy", "event"],
    )

    add(
        name="entity_outcome_entity_collider",
        family="task",
        motif_type="collider",
        rank_span=1,
        nodes={"e1": "entity", "e2": "entity", "y": "outcome"},
        edges=[("e1", "y"), ("e2", "y")],
        weight=0.6,
        priority="optional",
        description=(
            "Two entities jointly define a future outcome. "
            "Outcome must be treated as target, not visible input."
        ),
        tags=["entity", "outcome", "task"],
    )

    # ------------------------------------------------------------------
    # Depth/rank-span 2: common relational chains
    # ------------------------------------------------------------------
    add(
        name="entity_entity_event_chain",
        family="hierarchy",
        motif_type="chain",
        rank_span=2,
        nodes={"e0": "entity", "e1": "entity", "v": "event"},
        edges=[("e0", "e1"), ("e1", "v")],
        weight=1.6,
        priority="core",
        description="Hierarchical entity structure followed by event generation.",
        tags=["entity", "event", "hierarchy"],
    )

    add(
        name="class_entity_event_chain",
        family="dimension",
        motif_type="chain",
        rank_span=2,
        nodes={"c": "class", "e": "entity", "v": "event"},
        edges=[("c", "e"), ("e", "v")],
        weight=2.5,
        priority="core",
        description="Class describes entity; entity participates in event.",
        tags=["class", "entity", "event", "dimension"],
    )

    add(
        name="entity_event_event_chain",
        family="event_sequence",
        motif_type="chain",
        rank_span=2,
        nodes={"e": "entity", "v1": "event", "v2": "event"},
        edges=[("e", "v1"), ("v1", "v2")],
        weight=1.4,
        priority="recommended",
        description="Sequential event dependency, such as click -> purchase or order -> review.",
        tags=["entity", "event", "sequence", "temporal"],
    )

    add(
        name="entity_event_summary_chain",
        family="summary",
        motif_type="chain",
        rank_span=2,
        nodes={"e": "entity", "v": "event", "s": "summary"},
        edges=[("e", "v"), ("v", "s")],
        weight=2.8,
        priority="core",
        description="Historical events are aggregated into an as-of summary table.",
        tags=["entity", "event", "summary", "temporal"],
    )

    add(
        name="entity_event_outcome_chain",
        family="task",
        motif_type="chain",
        rank_span=2,
        nodes={"e": "entity", "v": "event", "y": "outcome"},
        edges=[("e", "v"), ("v", "y")],
        weight=1.6,
        priority="core",
        description=(
            "Historical event pattern contributes to a future-window outcome. "
            "The event part must be cutoff-before; outcome is target."
        ),
        tags=["entity", "event", "outcome", "task", "temporal"],
    )

    add(
        name="event_measurement_summary_chain",
        family="summary",
        motif_type="chain",
        rank_span=2,
        nodes={"v": "event", "m": "measurement", "s": "summary"},
        edges=[("v", "m"), ("m", "s")],
        weight=1.8,
        priority="core",
        description="Event measurements are aggregated into historical summaries.",
        tags=["event", "measurement", "summary", "temporal"],
    )

    add(
        name="entity_summary_outcome_chain",
        family="task",
        motif_type="chain",
        rank_span=2,
        nodes={"e": "entity", "s": "summary", "y": "outcome"},
        edges=[("e", "s"), ("s", "y")],
        weight=2.2,
        priority="core",
        description="Entity-level historical summary predicts future outcome.",
        tags=["entity", "summary", "outcome", "task", "temporal"],
    )

    # ------------------------------------------------------------------
    # Bridge-related motifs
    # ------------------------------------------------------------------
    add(
        name="entity_bridge_event_mixed",
        family="bridge",
        motif_type="mixed",
        rank_span=2,
        nodes={"e1": "entity", "e2": "entity", "b": "bridge", "v": "event"},
        edges=[("e1", "b"), ("e2", "b"), ("b", "v")],
        weight=1.8,
        priority="recommended",
        description="A many-to-many entity relation produces downstream events.",
        tags=["entity", "bridge", "event", "many_to_many"],
    )

    add(
        name="entity_bridge_outcome_mixed",
        family="bridge_task",
        motif_type="mixed",
        rank_span=2,
        nodes={"e1": "entity", "e2": "entity", "b": "bridge", "y": "outcome"},
        edges=[("e1", "b"), ("e2", "b"), ("b", "y")],
        weight=0.9,
        priority="recommended",
        description=(
            "A relation pair has a future outcome. "
            "Outcome must be used as target, not visible feature."
        ),
        tags=["entity", "bridge", "outcome", "task"],
    )

    add(
        name="entity_bridge_measurement_mixed",
        family="bridge",
        motif_type="mixed",
        rank_span=2,
        nodes={"e1": "entity", "e2": "entity", "b": "bridge", "m": "measurement"},
        edges=[("e1", "b"), ("e2", "b"), ("b", "m")],
        weight=1.0,
        priority="optional",
        description="A many-to-many relation has downstream measurements or relation attributes.",
        tags=["entity", "bridge", "measurement"],
    )

    # ------------------------------------------------------------------
    # Dimension/context fork motifs
    # ------------------------------------------------------------------
    add(
        name="class_entity_fork",
        family="dimension",
        motif_type="fork",
        rank_span=1,
        nodes={"c": "class", "e1": "entity", "e2": "entity"},
        edges=[("c", "e1"), ("c", "e2")],
        weight=1.1,
        priority="recommended",
        description="One class/dimension table describes multiple entity tables.",
        tags=["class", "entity", "fork", "dimension"],
    )

    add(
        name="context_event_fork",
        family="context",
        motif_type="fork",
        rank_span=1,
        nodes={"ctx": "context", "v1": "event", "v2": "event"},
        edges=[("ctx", "v1"), ("ctx", "v2")],
        weight=1.1,
        priority="recommended",
        description="One context table affects or describes multiple event tables.",
        tags=["context", "event", "fork"],
    )

    add(
        name="entity_context_event_collider",
        family="context",
        motif_type="collider",
        rank_span=1,
        nodes={"e": "entity", "ctx": "context", "v": "event"},
        edges=[("e", "v"), ("ctx", "v")],
        weight=1.4,
        priority="recommended",
        description="An event is jointly indexed by an entity and a context.",
        tags=["entity", "context", "event", "collider"],
    )

    # ------------------------------------------------------------------
    # Diamond motifs: direct + indirect paths
    # ------------------------------------------------------------------
    add(
        name="entity_event_summary_diamond",
        family="summary",
        motif_type="diamond",
        rank_span=2,
        nodes={"e": "entity", "v": "event", "s": "summary"},
        edges=[("e", "v"), ("v", "s"), ("e", "s")],
        weight=1.6,
        priority="recommended",
        description="Summary depends on both entity identity/static state and historical events.",
        tags=["entity", "event", "summary", "diamond"],
    )

    add(
        name="entity_event_outcome_diamond",
        family="task",
        motif_type="diamond",
        rank_span=2,
        nodes={"e": "entity", "v": "event", "y": "outcome"},
        edges=[("e", "v"), ("v", "y"), ("e", "y")],
        weight=1.1,
        priority="recommended",
        description="Future outcome depends on both entity attributes and historical events.",
        tags=["entity", "event", "outcome", "diamond", "task"],
    )

    add(
        name="context_event_outcome_diamond",
        family="task",
        motif_type="diamond",
        rank_span=2,
        nodes={"ctx": "context", "v": "event", "y": "outcome"},
        edges=[("ctx", "v"), ("v", "y"), ("ctx", "y")],
        weight=0.8,
        priority="optional",
        description="Context has both direct and event-mediated association with future outcome.",
        tags=["context", "event", "outcome", "diamond", "task"],
    )

    # ------------------------------------------------------------------
    # Star/hub motifs
    # ------------------------------------------------------------------
    add(
        name="entity_event_star",
        family="hub",
        motif_type="star",
        rank_span=1,
        nodes={"e": "entity", "v1": "event", "v2": "event", "v3": "event"},
        edges=[("e", "v1"), ("e", "v2"), ("e", "v3")],
        weight=1.4,
        priority="recommended",
        description="Entity as hub connecting multiple event tables.",
        tags=["entity", "event", "star", "hub"],
    )

    add(
        name="event_measurement_star",
        family="measurement",
        motif_type="star",
        rank_span=1,
        nodes={"v": "event", "m1": "measurement", "m2": "measurement"},
        edges=[("v", "m1"), ("v", "m2")],
        weight=0.8,
        priority="optional",
        description="One event generates multiple measurement/detail tables.",
        tags=["event", "measurement", "star"],
    )

    add(
        name="class_entity_event_snowflake",
        family="dimension",
        motif_type="snowflake",
        rank_span=2,
        nodes={"c0": "class", "c1": "class", "e": "entity", "v": "event"},
        edges=[("c0", "c1"), ("c1", "e"), ("e", "v")],
        weight=0.8,
        priority="optional",
        description="Hierarchical dimension/class structure followed by entity-event relation.",
        tags=["class", "entity", "event", "snowflake"],
    )

    # ------------------------------------------------------------------
    # Cold-start motifs
    # ------------------------------------------------------------------
    add(
        name="class_entity_outcome_cold_start",
        family="cold_start",
        motif_type="chain",
        rank_span=2,
        nodes={"c": "class", "e": "entity", "y": "outcome"},
        edges=[("c", "e"), ("e", "y")],
        weight=1.0,
        priority="recommended",
        description=(
            "Cold-start entity outcome can be predicted from static entity/class information "
            "when event history is missing or sparse."
        ),
        tags=["class", "entity", "outcome", "cold_start", "task"],
    )

    add(
        name="context_entity_outcome_cold_start",
        family="cold_start",
        motif_type="chain",
        rank_span=2,
        nodes={"ctx": "context", "e": "entity", "y": "outcome"},
        edges=[("ctx", "e"), ("e", "y")],
        weight=0.8,
        priority="recommended",
        description="Cold-start entity prediction using contextual/static information.",
        tags=["context", "entity", "outcome", "cold_start", "task"],
    )

    add(
        name="entity_event_summary_outcome_chain",
        family="task",
        motif_type="chain",
        rank_span=3,
        nodes={"e": "entity", "v": "event", "s": "summary", "y": "outcome"},
        edges=[("e", "v"), ("v", "s"), ("s", "y")],
        weight=2.5,
        priority="core",
        description=(
            "Canonical temporal relational prediction motif: "
            "entity -> historical events -> as-of summary -> future outcome."
        ),
        tags=["entity", "event", "summary", "outcome", "task", "temporal", "core"],
    )

    return motifs


DEFAULT_MOTIF_LIBRARY = MotifLibrary.default()


__all__ = [
    "Role",
    "NodeId",
    "Edge",
    "VALID_ROLES",
    "VALID_MOTIF_TYPES",
    "VALID_PRIORITIES",
    "normalize_role",
    "MotifPattern",
    "MotifLibrary",
    "build_default_motifs",
    "DEFAULT_MOTIF_LIBRARY",
]