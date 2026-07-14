# src/rdb_prior/schema/fk_graph.py
# -*- coding: utf-8 -*-
"""
FK-support graph utilities for synthetic relational database generation.

Important distinction
---------------------
FK graph is NOT a causal graph.

A directed edge A -> B means:
    - B may contain a foreign key referencing A;
    - A can be joined as context for B;
    - A is structurally upstream of B in the relational schema.

It does NOT necessarily mean:
    - A is a direct causal parent of B's attributes;
    - A's primary key causally generates B;
    - every variable in A causes every variable in B.

This module only handles schema-level FK-support graph operations:
    - DAG validation
    - rank inference
    - root/leaf detection
    - weak connectivity
    - role transition validation
    - motif-style local graph statistics
    - JSON export/import

Causal mechanisms should be defined in separate modules, e.g.
    causal/mechanism.py
    scm/*.py
    task/label_generator.py
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import json
import os


NodeId = str
Role = str
Edge = Tuple[NodeId, NodeId]


DEFAULT_ALLOWED_ROLE_TRANSITIONS: Dict[Role, Tuple[Role, ...]] = {
    "class": (
        "class",
        "entity",
        "event",
        "bridge",
        "summary",
        "outcome",
    ),
    "context": (
        "entity",
        "event",
        "bridge",
        "summary",
        "outcome",
    ),
    "entity": (
        "entity",
        "event",
        "bridge",
        "measurement",
        "summary",
        "outcome",
    ),
    "event": (
        "event",
        "measurement",
        "summary",
        "outcome",
    ),
    "bridge": (
        "event",
        "measurement",
        "summary",
        "outcome",
    ),
    "measurement": (
        "summary",
        "outcome",
    ),
    "summary": (
        "outcome",
    ),
    "outcome": tuple(),
}


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


def normalize_role(role: str) -> str:
    key = str(role).strip().lower()
    if key not in ROLE_ALIASES:
        raise ValueError(f"Unknown role: {role!r}")
    return ROLE_ALIASES[key]


def normalize_node_id(node: Any) -> NodeId:
    return str(node)


def normalize_edge(edge: Sequence[Any]) -> Edge:
    if len(edge) != 2:
        raise ValueError(f"Edge must have length 2, got: {edge!r}")
    return normalize_node_id(edge[0]), normalize_node_id(edge[1])


@dataclass(frozen=True)
class FKGraphValidationIssue:
    issue_type: str
    message: str
    edge: Optional[Edge] = None
    node: Optional[NodeId] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "message": self.message,
            "edge": list(self.edge) if self.edge is not None else None,
            "node": self.node,
            "payload": dict(self.payload),
        }


@dataclass
class FKGraphValidationResult:
    is_valid: bool
    issues: List[FKGraphValidationIssue] = field(default_factory=list)

    @property
    def num_issues(self) -> int:
        return len(self.issues)

    def raise_if_invalid(self) -> None:
        if self.is_valid:
            return
        preview = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        raise ValueError(f"Invalid FK graph:\n{preview}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "num_issues": self.num_issues,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass
class FKGraphStats:
    num_nodes: int
    num_edges: int
    edge_density: float
    max_rank: int
    roots: List[NodeId]
    leaves: List[NodeId]
    root_ratio: float
    leaf_ratio: float
    indegree: Dict[NodeId, int]
    outdegree: Dict[NodeId, int]
    rank: Dict[NodeId, int]
    weak_components: List[List[NodeId]]
    is_weakly_connected: bool
    motif_counts: Dict[str, int]
    role_transition_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "edge_density": self.edge_density,
            "max_rank": self.max_rank,
            "roots": list(self.roots),
            "leaves": list(self.leaves),
            "root_ratio": self.root_ratio,
            "leaf_ratio": self.leaf_ratio,
            "indegree": dict(self.indegree),
            "outdegree": dict(self.outdegree),
            "rank": dict(self.rank),
            "weak_components": [list(c) for c in self.weak_components],
            "is_weakly_connected": self.is_weakly_connected,
            "motif_counts": dict(self.motif_counts),
            "role_transition_counts": dict(self.role_transition_counts),
        }


@dataclass
class FKGraph:
    """
    Directed FK-support graph.

    Edges are interpreted as parent_table -> child_table.
    """

    node_ids: List[NodeId]
    edges: List[Edge]
    node_roles: Dict[NodeId, Role] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.node_ids = [normalize_node_id(n) for n in self.node_ids]
        self.edges = [normalize_edge(e) for e in self.edges]

        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError(f"Duplicate node ids: {self.node_ids}")

        known = set(self.node_ids)
        for src, dst in self.edges:
            if src not in known or dst not in known:
                raise ValueError(f"Edge contains unknown node: {(src, dst)}")
            if src == dst:
                raise ValueError(f"Self-loop is not allowed: {(src, dst)}")

        self.node_roles = {
            normalize_node_id(node): normalize_role(role)
            for node, role in self.node_roles.items()
        }

        unknown_role_nodes = set(self.node_roles) - set(self.node_ids)
        if unknown_role_nodes:
            raise ValueError(f"node_roles contains unknown nodes: {sorted(unknown_role_nodes)}")

    @classmethod
    def from_schema_dict(cls, schema: Mapping[str, Any]) -> "FKGraph":
        """
        Build FKGraph from SampledSchema.to_dict() style schema.

        Expected keys:
            schema["nodes"]
            schema["foreign_keys"]
        """
        nodes_obj = schema.get("nodes", {})
        if isinstance(nodes_obj, Mapping):
            node_ids = list(nodes_obj.keys())
            node_roles = {
                node_id: node_info.get("role")
                for node_id, node_info in nodes_obj.items()
                if isinstance(node_info, Mapping) and node_info.get("role") is not None
            }
        else:
            node_ids = [n["node_id"] for n in nodes_obj]
            node_roles = {
                n["node_id"]: n.get("role")
                for n in nodes_obj
                if n.get("role") is not None
            }

        edges: List[Edge] = []
        for fk in schema.get("foreign_keys", []):
            parent = fk.get("parent_table")
            child = fk.get("child_table")
            if parent is None or child is None:
                continue
            edges.append((parent, child))

        return cls(
            node_ids=node_ids,
            edges=edges,
            node_roles=node_roles,
            metadata={
                "source": "schema_dict",
                "schema_id": schema.get("schema_id"),
                "note": "FK graph edges are schema-level support edges, not causal edges.",
            },
        )

    @classmethod
    def from_json(cls, path: str) -> "FKGraph":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if "nodes" in obj and "foreign_keys" in obj:
            return cls.from_schema_dict(obj)

        return cls(
            node_ids=obj["node_ids"],
            edges=[tuple(e) for e in obj["edges"]],
            node_roles=obj.get("node_roles", {}),
            metadata=obj.get("metadata", {}),
        )

    @property
    def node_set(self) -> Set[NodeId]:
        return set(self.node_ids)

    @property
    def edge_set(self) -> Set[Edge]:
        return set(self.edges)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def parents(self, node: NodeId) -> List[NodeId]:
        node = normalize_node_id(node)
        return [src for src, dst in self.edges if dst == node]

    def children(self, node: NodeId) -> List[NodeId]:
        node = normalize_node_id(node)
        return [dst for src, dst in self.edges if src == node]

    def indegree(self) -> Dict[NodeId, int]:
        deg = {node: 0 for node in self.node_ids}
        for _, dst in self.edges:
            deg[dst] += 1
        return deg

    def outdegree(self) -> Dict[NodeId, int]:
        deg = {node: 0 for node in self.node_ids}
        for src, _ in self.edges:
            deg[src] += 1
        return deg

    def roots(self) -> List[NodeId]:
        indeg = self.indegree()
        return [node for node in self.node_ids if indeg[node] == 0]

    def leaves(self) -> List[NodeId]:
        outdeg = self.outdegree()
        return [node for node in self.node_ids if outdeg[node] == 0]

    def adjacency(self) -> Dict[NodeId, List[NodeId]]:
        adj = {node: [] for node in self.node_ids}
        for src, dst in self.edges:
            adj[src].append(dst)
        return adj

    def reverse_adjacency(self) -> Dict[NodeId, List[NodeId]]:
        radj = {node: [] for node in self.node_ids}
        for src, dst in self.edges:
            radj[dst].append(src)
        return radj

    def weak_adjacency(self) -> Dict[NodeId, List[NodeId]]:
        adj = {node: [] for node in self.node_ids}
        for src, dst in self.edges:
            adj[src].append(dst)
            adj[dst].append(src)
        return adj

    def add_edge(self, src: NodeId, dst: NodeId, validate_dag: bool = True) -> None:
        src = normalize_node_id(src)
        dst = normalize_node_id(dst)

        if src not in self.node_set or dst not in self.node_set:
            raise ValueError(f"Unknown node in edge: {(src, dst)}")
        if src == dst:
            raise ValueError(f"Self-loop is not allowed: {(src, dst)}")
        if (src, dst) in self.edge_set:
            return

        old_edges = list(self.edges)
        self.edges.append((src, dst))

        if validate_dag and not self.is_dag():
            self.edges = old_edges
            raise ValueError(f"Adding edge {(src, dst)} introduces a cycle.")

    def remove_edge(self, src: NodeId, dst: NodeId) -> None:
        edge = (normalize_node_id(src), normalize_node_id(dst))
        self.edges = [e for e in self.edges if e != edge]

    def is_dag(self) -> bool:
        return self.topological_sort_or_none() is not None

    def topological_sort_or_none(self) -> Optional[List[NodeId]]:
        indeg = self.indegree()
        adj = self.adjacency()

        queue = deque([n for n in self.node_ids if indeg[n] == 0])
        order: List[NodeId] = []

        while queue:
            node = queue.popleft()
            order.append(node)

            for child in adj[node]:
                indeg[child] -= 1
                if indeg[child] == 0:
                    queue.append(child)

        if len(order) != len(self.node_ids):
            return None

        return order

    def topological_sort(self) -> List[NodeId]:
        order = self.topological_sort_or_none()
        if order is None:
            raise ValueError("FK graph contains a cycle.")
        return order

    def infer_ranks(self) -> Dict[NodeId, int]:
        """
        Infer longest-path rank from DAG roots.

        Rank 0 means source/root table.
        Child rank is at least parent rank + 1.
        """
        order = self.topological_sort()
        rank = {node: 0 for node in self.node_ids}

        for node in order:
            for child in self.children(node):
                rank[child] = max(rank[child], rank[node] + 1)

        return rank

    def weak_components(self) -> List[List[NodeId]]:
        adj = self.weak_adjacency()
        seen: Set[NodeId] = set()
        comps: List[List[NodeId]] = []

        for node in self.node_ids:
            if node in seen:
                continue

            comp: List[NodeId] = []
            queue = deque([node])
            seen.add(node)

            while queue:
                cur = queue.popleft()
                comp.append(cur)

                for nxt in adj[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)

            comps.append(comp)

        return comps

    def is_weakly_connected(self) -> bool:
        return len(self.weak_components()) <= 1

    def ancestors(self, node: NodeId) -> Set[NodeId]:
        node = normalize_node_id(node)
        if node not in self.node_set:
            raise ValueError(f"Unknown node: {node}")

        radj = self.reverse_adjacency()
        result: Set[NodeId] = set()
        queue = deque(radj[node])

        while queue:
            cur = queue.popleft()
            if cur in result:
                continue
            result.add(cur)
            queue.extend(radj[cur])

        return result

    def descendants(self, node: NodeId) -> Set[NodeId]:
        node = normalize_node_id(node)
        if node not in self.node_set:
            raise ValueError(f"Unknown node: {node}")

        adj = self.adjacency()
        result: Set[NodeId] = set()
        queue = deque(adj[node])

        while queue:
            cur = queue.popleft()
            if cur in result:
                continue
            result.add(cur)
            queue.extend(adj[cur])

        return result

    def has_path(self, src: NodeId, dst: NodeId) -> bool:
        src = normalize_node_id(src)
        dst = normalize_node_id(dst)

        if src not in self.node_set or dst not in self.node_set:
            raise ValueError(f"Unknown node in path query: {(src, dst)}")

        return dst in self.descendants(src)

    def shortest_path_length(self, src: NodeId, dst: NodeId) -> Optional[int]:
        src = normalize_node_id(src)
        dst = normalize_node_id(dst)

        if src not in self.node_set or dst not in self.node_set:
            raise ValueError(f"Unknown node in path query: {(src, dst)}")

        if src == dst:
            return 0

        adj = self.adjacency()
        queue = deque([(src, 0)])
        seen = {src}

        while queue:
            node, dist = queue.popleft()
            for child in adj[node]:
                if child == dst:
                    return dist + 1
                if child not in seen:
                    seen.add(child)
                    queue.append((child, dist + 1))

        return None

    def all_pairs_shortest_path_lengths(self) -> Dict[NodeId, Dict[NodeId, int]]:
        result: Dict[NodeId, Dict[NodeId, int]] = {}

        for src in self.node_ids:
            result[src] = {}
            queue = deque([(src, 0)])
            seen = {src}

            while queue:
                node, dist = queue.popleft()
                result[src][node] = dist

                for child in self.children(node):
                    if child not in seen:
                        seen.add(child)
                        queue.append((child, dist + 1))

        return result

    def validate(
        self,
        *,
        require_dag: bool = True,
        require_weakly_connected: bool = False,
        node_roles: Optional[Mapping[NodeId, Role]] = None,
        allowed_role_transitions: Optional[Mapping[Role, Sequence[Role]]] = None,
        forbid_outcome_as_parent: bool = True,
        require_summary_downstream: bool = False,
        allow_duplicate_edges: bool = False,
    ) -> FKGraphValidationResult:
        issues: List[FKGraphValidationIssue] = []

        known = self.node_set

        for src, dst in self.edges:
            if src not in known:
                issues.append(
                    FKGraphValidationIssue(
                        issue_type="unknown_node",
                        message=f"Unknown source node: {src}",
                        edge=(src, dst),
                    )
                )
            if dst not in known:
                issues.append(
                    FKGraphValidationIssue(
                        issue_type="unknown_node",
                        message=f"Unknown destination node: {dst}",
                        edge=(src, dst),
                    )
                )
            if src == dst:
                issues.append(
                    FKGraphValidationIssue(
                        issue_type="self_loop",
                        message="Self-loop is not allowed in FK graph.",
                        edge=(src, dst),
                    )
                )

        if not allow_duplicate_edges:
            duplicated = [edge for edge, count in Counter(self.edges).items() if count > 1]
            for edge in duplicated:
                issues.append(
                    FKGraphValidationIssue(
                        issue_type="duplicate_edge",
                        message="Duplicate FK-support edge.",
                        edge=edge,
                    )
                )

        if require_dag and not self.is_dag():
            issues.append(
                FKGraphValidationIssue(
                    issue_type="cycle",
                    message="FK graph must be DAG-like, but a cycle was detected.",
                )
            )

        if require_weakly_connected and not self.is_weakly_connected():
            issues.append(
                FKGraphValidationIssue(
                    issue_type="disconnected_graph",
                    message="FK graph is not weakly connected.",
                    payload={"weak_components": self.weak_components()},
                )
            )

        roles = {
            **self.node_roles,
            **{
                normalize_node_id(k): normalize_role(v)
                for k, v in (node_roles or {}).items()
            },
        }

        if roles:
            issues.extend(
                self._validate_role_transitions(
                    roles=roles,
                    allowed_role_transitions=allowed_role_transitions
                    or DEFAULT_ALLOWED_ROLE_TRANSITIONS,
                    forbid_outcome_as_parent=forbid_outcome_as_parent,
                    require_summary_downstream=require_summary_downstream,
                )
            )

        return FKGraphValidationResult(is_valid=len(issues) == 0, issues=issues)

    def _validate_role_transitions(
        self,
        *,
        roles: Mapping[NodeId, Role],
        allowed_role_transitions: Mapping[Role, Sequence[Role]],
        forbid_outcome_as_parent: bool,
        require_summary_downstream: bool,
    ) -> List[FKGraphValidationIssue]:
        issues: List[FKGraphValidationIssue] = []

        normalized_transitions = {
            normalize_role(src): tuple(normalize_role(dst) for dst in dsts)
            for src, dsts in allowed_role_transitions.items()
        }

        for src, dst in self.edges:
            if src not in roles or dst not in roles:
                continue

            src_role = normalize_role(roles[src])
            dst_role = normalize_role(roles[dst])

            allowed = normalized_transitions.get(src_role, tuple())
            if dst_role not in allowed:
                issues.append(
                    FKGraphValidationIssue(
                        issue_type="role_transition_violation",
                        message=f"Role transition {src_role} -> {dst_role} is not allowed.",
                        edge=(src, dst),
                        payload={
                            "src_role": src_role,
                            "dst_role": dst_role,
                        },
                    )
                )

            if forbid_outcome_as_parent and src_role == "outcome":
                issues.append(
                    FKGraphValidationIssue(
                        issue_type="temporal_safety_violation",
                        message=(
                            "Outcome should not be an upstream parent table. "
                            "It should be a target/future-window result."
                        ),
                        edge=(src, dst),
                        payload={
                            "src_role": src_role,
                            "dst_role": dst_role,
                        },
                    )
                )

        if require_summary_downstream:
            for node, role in roles.items():
                if normalize_role(role) != "summary":
                    continue

                parent_roles = [
                    normalize_role(roles[p])
                    for p in self.parents(node)
                    if p in roles
                ]

                if not any(r in {"event", "measurement", "entity", "bridge"} for r in parent_roles):
                    issues.append(
                        FKGraphValidationIssue(
                            issue_type="summary_without_history_source",
                            message=(
                                "Summary should usually have upstream event, "
                                "measurement, entity, or bridge source."
                            ),
                            node=node,
                            payload={"parent_roles": parent_roles},
                        )
                    )

        return issues

    def motif_counts(self) -> Dict[str, int]:
        """
        Count simple 3-node local motifs on FK-support graph.

        This is intentionally lightweight:
            - chain: A -> B -> C
            - fork: A -> B and A -> C
            - collider: A -> C and B -> C
            - diamond_3edge: A -> B, A -> C, B -> C
        """
        adj = self.adjacency()
        radj = self.reverse_adjacency()

        chain_count = 0
        fork_count = 0
        collider_count = 0
        diamond_3edge_count = 0

        for middle in self.node_ids:
            for parent in radj[middle]:
                for child in adj[middle]:
                    if parent != child:
                        chain_count += 1

        for src in self.node_ids:
            children = adj[src]
            if len(children) >= 2:
                fork_count += len(children) * (len(children) - 1) // 2

        for dst in self.node_ids:
            parents = radj[dst]
            if len(parents) >= 2:
                collider_count += len(parents) * (len(parents) - 1) // 2

        edge_set = self.edge_set
        for a in self.node_ids:
            for b in adj[a]:
                for c in adj[a]:
                    if b == c:
                        continue
                    if (b, c) in edge_set:
                        diamond_3edge_count += 1

        return {
            "chain": chain_count,
            "fork": fork_count,
            "collider": collider_count,
            "diamond_3edge": diamond_3edge_count,
        }

    def role_transition_counts(self) -> Dict[str, int]:
        if not self.node_roles:
            return {}

        counts: Counter = Counter()
        for src, dst in self.edges:
            if src not in self.node_roles or dst not in self.node_roles:
                continue
            src_role = normalize_role(self.node_roles[src])
            dst_role = normalize_role(self.node_roles[dst])
            counts[f"{src_role}->{dst_role}"] += 1

        return dict(counts)

    def stats(self) -> FKGraphStats:
        indeg = self.indegree()
        outdeg = self.outdegree()
        ranks = self.infer_ranks() if self.is_dag() else {node: -1 for node in self.node_ids}
        roots = self.roots()
        leaves = self.leaves()
        comps = self.weak_components()

        max_possible_edges = self.num_nodes * (self.num_nodes - 1) / 2
        edge_density = (
            float(self.num_edges) / max_possible_edges
            if max_possible_edges > 0
            else 0.0
        )

        return FKGraphStats(
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            edge_density=edge_density,
            max_rank=max(ranks.values()) if ranks else 0,
            roots=roots,
            leaves=leaves,
            root_ratio=len(roots) / self.num_nodes if self.num_nodes > 0 else 0.0,
            leaf_ratio=len(leaves) / self.num_nodes if self.num_nodes > 0 else 0.0,
            indegree=indeg,
            outdegree=outdeg,
            rank=ranks,
            weak_components=comps,
            is_weakly_connected=len(comps) <= 1,
            motif_counts=self.motif_counts(),
            role_transition_counts=self.role_transition_counts(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_ids": list(self.node_ids),
            "edges": [list(e) for e in self.edges],
            "node_roles": dict(self.node_roles),
            "metadata": {
                **dict(self.metadata),
                "fk_graph_note": (
                    "Edges are FK-support/join edges, not direct causal edges."
                ),
            },
            "stats": self.stats().to_dict(),
        }

    def save_json(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def build_fk_graph_from_edges(
    node_ids: Sequence[NodeId],
    edges: Sequence[Edge],
    node_roles: Optional[Mapping[NodeId, Role]] = None,
    validate: bool = True,
    require_dag: bool = True,
    require_weakly_connected: bool = False,
    allow_duplicate_edges: bool = False,
) -> FKGraph:
    graph = FKGraph(
        node_ids=list(node_ids),
        edges=list(edges),
        node_roles=dict(node_roles or {}),
    )

    if validate:
        result = graph.validate(
            require_dag=require_dag,
            require_weakly_connected=require_weakly_connected,
            allow_duplicate_edges=allow_duplicate_edges,
        )
        result.raise_if_invalid()

    return graph


def load_fk_graph(path: str) -> FKGraph:
    return FKGraph.from_json(path)


def save_fk_graph(graph: FKGraph, path: str) -> None:
    graph.save_json(path)


def edges_from_foreign_keys(foreign_keys: Sequence[Mapping[str, Any]]) -> List[Edge]:
    edges: List[Edge] = []
    for fk in foreign_keys:
        parent = fk.get("parent_table")
        child = fk.get("child_table")
        if parent is None or child is None:
            continue
        edges.append((str(parent), str(child)))
    return edges


def graph_from_sampled_schema_like(schema: Any) -> FKGraph:
    """
    Build FKGraph from either:
        - SampledSchema object from schema_sampler.py
        - dict produced by SampledSchema.to_dict()
    """
    if hasattr(schema, "nodes") and hasattr(schema, "foreign_keys"):
        node_ids = list(schema.nodes.keys())
        node_roles = {
            node_id: node.role
            for node_id, node in schema.nodes.items()
        }
        edges = [
            (fk.parent_table, fk.child_table)
            for fk in schema.foreign_keys
        ]
        return FKGraph(
            node_ids=node_ids,
            edges=edges,
            node_roles=node_roles,
            metadata={
                "source": "SampledSchema",
                "schema_id": getattr(schema, "schema_id", None),
            },
        )

    if isinstance(schema, Mapping):
        return FKGraph.from_schema_dict(schema)

    raise TypeError(
        "schema must be a SampledSchema-like object or a schema dictionary."
    )


__all__ = [
    "NodeId",
    "Role",
    "Edge",
    "DEFAULT_ALLOWED_ROLE_TRANSITIONS",
    "normalize_role",
    "normalize_edge",
    "FKGraphValidationIssue",
    "FKGraphValidationResult",
    "FKGraphStats",
    "FKGraph",
    "build_fk_graph_from_edges",
    "load_fk_graph",
    "save_fk_graph",
    "edges_from_foreign_keys",
    "graph_from_sampled_schema_like",
]