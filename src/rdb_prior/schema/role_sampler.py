# src/rdb_prior/schema/role_sampler.py
# -*- coding: utf-8 -*-
"""
Role sampler for role/motif/time-aware synthetic relational database generation.

Design principles
-----------------
1. Role is a latent structural variable, not a natural-language table name.
2. Role assignment can be conditioned on:
   - node rank / depth
   - in-degree / out-degree
   - root / leaf status
   - sampled motif inventory
   - optional forced roles
3. FK graph is treated as schema-level support graph, not causal graph.
4. Outcome is treated as task/future-window target by default and should not
   become an upstream visible table.

Typical usage
-------------
    sampler = RoleSampler.default(seed=42)

    assignment = sampler.assign_roles(
        node_ids=["T0", "T1", "T2", "T3"],
        edges=[("T0", "T1"), ("T0", "T2"), ("T1", "T3")],
        required_roles=["entity", "event", "summary"],
    )

    print(assignment.node_roles)
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import json
import math
import random


Role = str
NodeId = str
Edge = Tuple[NodeId, NodeId]


# ---------------------------------------------------------------------
# Role normalization
# ---------------------------------------------------------------------

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


def normalize_role(role: str) -> str:
    key = role.strip().lower()
    if key not in ROLE_ALIASES:
        raise ValueError(f"Unknown role: {role!r}. Valid roles: {sorted(VALID_ROLES)}")
    return ROLE_ALIASES[key]


# ---------------------------------------------------------------------
# Default role transition prior
# ---------------------------------------------------------------------
# These transitions are not causal assumptions.
# They only encode which FK-support directions are structurally plausible.

DEFAULT_ALLOWED_TRANSITIONS: Dict[Role, Tuple[Role, ...]] = {
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


DEFAULT_ROLE_PROFILES: Dict[Role, Dict[str, Any]] = {
    "class": {
        "has_time_col": False,
        "is_dimension": True,
        "is_event_source": False,
        "is_task_target_candidate": False,
        "default_scm_family": ["categorical", "discrete"],
    },
    "context": {
        "has_time_col": False,
        "is_dimension": True,
        "is_event_source": False,
        "is_task_target_candidate": False,
        "default_scm_family": ["categorical", "discrete", "mixed"],
    },
    "entity": {
        "has_time_col": False,
        "is_dimension": False,
        "is_event_source": True,
        "is_task_target_candidate": True,
        "default_scm_family": ["latent_profile", "linear", "mlp", "tree"],
    },
    "event": {
        "has_time_col": True,
        "is_dimension": False,
        "is_event_source": True,
        "is_task_target_candidate": True,
        "default_scm_family": ["temporal", "mlp", "tree", "mixture"],
    },
    "bridge": {
        "has_time_col": False,
        "is_dimension": False,
        "is_event_source": False,
        "is_task_target_candidate": False,
        "default_scm_family": ["fk_matching", "pair_sampling"],
    },
    "measurement": {
        "has_time_col": True,
        "is_dimension": False,
        "is_event_source": False,
        "is_task_target_candidate": False,
        "default_scm_family": ["observation", "temporal", "tree", "mlp"],
    },
    "summary": {
        "has_time_col": True,
        "is_dimension": False,
        "is_event_source": False,
        "is_task_target_candidate": True,
        "default_scm_family": ["aggregation", "window_aggregation"],
    },
    "outcome": {
        "has_time_col": True,
        "is_dimension": False,
        "is_event_source": False,
        "is_task_target_candidate": True,
        "default_scm_family": ["task_label", "future_window"],
    },
}


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RoleSpec:
    """
    Structural prior for assigning a role to schema nodes.

    This is not a table-name prior. It is a prior over observable structural
    positions: root/leaf, rank, in-degree, out-degree, etc.
    """

    role: Role
    base_weight: float = 1.0

    min_count: int = 0
    max_count: Optional[int] = None

    min_rank: Optional[int] = None
    max_rank: Optional[int] = None

    can_be_root: bool = True
    can_be_leaf: bool = True

    root_weight: float = 1.0
    internal_weight: float = 1.0
    leaf_weight: float = 1.0

    indegree_weight: float = 0.0
    outdegree_weight: float = 0.0

    # Soft structural preferences.
    prefer_high_indegree: bool = False
    prefer_high_outdegree: bool = False
    prefer_low_indegree: bool = False
    prefer_low_outdegree: bool = False

    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", normalize_role(self.role))
        if self.base_weight < 0:
            raise ValueError(f"base_weight must be non-negative for role={self.role!r}.")
        if self.min_count < 0:
            raise ValueError(f"min_count must be non-negative for role={self.role!r}.")
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError(
                f"max_count must be >= min_count for role={self.role!r}."
            )

    def allows_rank(self, rank: int) -> bool:
        if self.min_rank is not None and rank < self.min_rank:
            return False
        if self.max_rank is not None and rank > self.max_rank:
            return False
        return True

    def allows_count(self, current_count: int) -> bool:
        if self.max_count is None:
            return True
        return current_count < self.max_count


@dataclass
class RoleSamplingConfig:
    """
    Global role sampling behavior.

    require_entity:
        Enforce at least one entity when possible.
    require_event_if_possible:
        Enforce at least one event if num_nodes >= 2.
    strict_transition_check:
        If True, invalid role transitions raise an error after assignment.
    transition_bias:
        Multiplicative reward for assignments compatible with already assigned
        neighboring roles.
    invalid_transition_penalty:
        Multiplicative penalty for assignments incompatible with already
        assigned neighboring roles.
    """

    require_entity: bool = True
    require_event_if_possible: bool = True
    strict_transition_check: bool = False

    transition_bias: float = 1.5
    invalid_transition_penalty: float = 0.15

    prefer_outcome_leaf: bool = True
    max_outcome_count: int = 1

    use_transition_soft_bias: bool = True


@dataclass
class GraphFeatures:
    node_ids: Tuple[NodeId, ...]
    edges: Tuple[Edge, ...]
    ranks: Dict[NodeId, int]
    indegree: Dict[NodeId, int]
    outdegree: Dict[NodeId, int]
    roots: Tuple[NodeId, ...]
    leaves: Tuple[NodeId, ...]
    max_rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_ids": list(self.node_ids),
            "edges": list(self.edges),
            "ranks": dict(self.ranks),
            "indegree": dict(self.indegree),
            "outdegree": dict(self.outdegree),
            "roots": list(self.roots),
            "leaves": list(self.leaves),
            "max_rank": self.max_rank,
        }


@dataclass
class RoleAssignment:
    node_roles: Dict[NodeId, Role]
    graph_features: GraphFeatures
    role_profiles: Dict[NodeId, Dict[str, Any]] = field(default_factory=dict)
    violations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def role_counts(self) -> Counter:
        return Counter(self.node_roles.values())

    def nodes_by_role(self, role: str) -> List[NodeId]:
        canonical = normalize_role(role)
        return [node for node, r in self.node_roles.items() if r == canonical]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_roles": dict(self.node_roles),
            "role_counts": dict(self.role_counts),
            "role_profiles": dict(self.role_profiles),
            "graph_features": self.graph_features.to_dict(),
            "violations": list(self.violations),
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Default role specs
# ---------------------------------------------------------------------

def build_default_role_specs() -> Dict[Role, RoleSpec]:
    """
    Default structural priors for roles.

    These values are intentionally heuristic in the first version.
    They should later be calibrated using real database statistics.
    """

    specs = {
        "class": RoleSpec(
            role="class",
            base_weight=0.8,
            min_count=0,
            max_count=None,
            min_rank=0,
            max_rank=1,
            can_be_root=True,
            can_be_leaf=False,
            root_weight=3.0,
            internal_weight=1.2,
            leaf_weight=0.1,
            indegree_weight=-0.2,
            outdegree_weight=0.4,
            prefer_low_indegree=True,
            prefer_high_outdegree=True,
            description="Low-cardinality dimension/category table.",
        ),
        "context": RoleSpec(
            role="context",
            base_weight=0.5,
            min_count=0,
            max_count=None,
            min_rank=0,
            max_rank=1,
            can_be_root=True,
            can_be_leaf=False,
            root_weight=2.5,
            internal_weight=1.2,
            leaf_weight=0.1,
            indegree_weight=-0.1,
            outdegree_weight=0.3,
            prefer_low_indegree=True,
            prefer_high_outdegree=True,
            description="Contextual dimension such as location, platform, or condition.",
        ),
        "entity": RoleSpec(
            role="entity",
            base_weight=3.0,
            min_count=1,
            max_count=None,
            min_rank=0,
            max_rank=None,
            can_be_root=True,
            can_be_leaf=True,
            root_weight=2.0,
            internal_weight=1.5,
            leaf_weight=0.7,
            indegree_weight=0.1,
            outdegree_weight=0.35,
            prefer_high_outdegree=True,
            description="Stable business object.",
        ),
        "event": RoleSpec(
            role="event",
            base_weight=3.0,
            min_count=0,
            max_count=None,
            min_rank=1,
            max_rank=None,
            can_be_root=False,
            can_be_leaf=True,
            root_weight=0.05,
            internal_weight=1.8,
            leaf_weight=1.4,
            indegree_weight=0.3,
            outdegree_weight=0.2,
            prefer_high_indegree=True,
            description="Timestamped interaction or business process event.",
        ),
        "bridge": RoleSpec(
            role="bridge",
            base_weight=0.9,
            min_count=0,
            max_count=None,
            min_rank=1,
            max_rank=None,
            can_be_root=False,
            can_be_leaf=True,
            root_weight=0.01,
            internal_weight=1.4,
            leaf_weight=1.1,
            indegree_weight=0.8,
            outdegree_weight=0.1,
            prefer_high_indegree=True,
            description="Many-to-many relation table with high FK ratio.",
        ),
        "measurement": RoleSpec(
            role="measurement",
            base_weight=0.7,
            min_count=0,
            max_count=None,
            min_rank=1,
            max_rank=None,
            can_be_root=False,
            can_be_leaf=True,
            root_weight=0.01,
            internal_weight=1.2,
            leaf_weight=1.5,
            indegree_weight=0.3,
            outdegree_weight=0.0,
            prefer_high_indegree=True,
            prefer_low_outdegree=True,
            description="Observation/detail table attached to entity or event.",
        ),
        "summary": RoleSpec(
            role="summary",
            base_weight=0.8,
            min_count=0,
            max_count=None,
            min_rank=1,
            max_rank=None,
            can_be_root=False,
            can_be_leaf=True,
            root_weight=0.01,
            internal_weight=1.2,
            leaf_weight=1.6,
            indegree_weight=0.4,
            outdegree_weight=0.1,
            prefer_high_indegree=True,
            description="As-of historical aggregation table.",
        ),
        "outcome": RoleSpec(
            role="outcome",
            base_weight=0.5,
            min_count=0,
            max_count=1,
            min_rank=1,
            max_rank=None,
            can_be_root=False,
            can_be_leaf=True,
            root_weight=0.001,
            internal_weight=0.2,
            leaf_weight=2.0,
            indegree_weight=0.4,
            outdegree_weight=-1.0,
            prefer_high_indegree=True,
            prefer_low_outdegree=True,
            description="Task label or future-window result table.",
        ),
    }

    return specs


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _normalize_node_ids(node_ids: Union[int, Sequence[NodeId]]) -> Tuple[NodeId, ...]:
    if isinstance(node_ids, int):
        if node_ids <= 0:
            raise ValueError("num_nodes must be positive.")
        return tuple(f"T{i}" for i in range(node_ids))

    result = tuple(str(n) for n in node_ids)
    if not result:
        raise ValueError("node_ids must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate node ids: {result}")
    return result


def _normalize_edges(edges: Optional[Sequence[Edge]]) -> Tuple[Edge, ...]:
    if edges is None:
        return tuple()
    return tuple((str(src), str(dst)) for src, dst in edges)


def _as_role_counter(
    required_roles: Optional[Union[Iterable[str], Mapping[str, int]]]
) -> Counter:
    counter: Counter = Counter()
    if required_roles is None:
        return counter

    if isinstance(required_roles, Mapping):
        for role, count in required_roles.items():
            canonical = normalize_role(role)
            if int(count) < 0:
                raise ValueError(f"Required count must be non-negative: {role}={count}")
            counter[canonical] += int(count)
    else:
        for role in required_roles:
            counter[normalize_role(role)] += 1

    return counter


def _infer_ranks_from_dag(
    node_ids: Sequence[NodeId],
    edges: Sequence[Edge],
) -> Dict[NodeId, int]:
    """
    Infer longest-path rank from a DAG.
    Rank 0 means root/source tables.
    """
    node_set = set(node_ids)
    indeg = {node: 0 for node in node_ids}
    adj: Dict[NodeId, List[NodeId]] = {node: [] for node in node_ids}

    for src, dst in edges:
        if src not in node_set or dst not in node_set:
            raise ValueError(f"Edge contains unknown node: {(src, dst)}")
        if src == dst:
            raise ValueError(f"Self-loop is not allowed: {(src, dst)}")
        adj[src].append(dst)
        indeg[dst] += 1

    queue = deque([n for n in node_ids if indeg[n] == 0])
    rank = {node: 0 for node in node_ids}
    visited = 0

    while queue:
        node = queue.popleft()
        visited += 1

        for nxt in adj[node]:
            rank[nxt] = max(rank[nxt], rank[node] + 1)
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)

    if visited != len(node_ids):
        raise ValueError("Cannot infer ranks because FK graph contains a cycle.")

    return rank


def compute_graph_features(
    node_ids: Union[int, Sequence[NodeId]],
    edges: Optional[Sequence[Edge]] = None,
    ranks: Optional[Mapping[NodeId, int]] = None,
) -> GraphFeatures:
    nodes = _normalize_node_ids(node_ids)
    edge_tuple = _normalize_edges(edges)

    node_set = set(nodes)
    indegree = {node: 0 for node in nodes}
    outdegree = {node: 0 for node in nodes}

    for src, dst in edge_tuple:
        if src not in node_set or dst not in node_set:
            raise ValueError(f"Edge contains unknown node: {(src, dst)}")
        if src == dst:
            raise ValueError(f"Self-loop is not allowed: {(src, dst)}")
        outdegree[src] += 1
        indegree[dst] += 1

    if ranks is None:
        rank_dict = _infer_ranks_from_dag(nodes, edge_tuple) if edge_tuple else {n: 0 for n in nodes}
    else:
        rank_dict = {str(k): int(v) for k, v in ranks.items()}
        missing = set(nodes) - set(rank_dict)
        if missing:
            raise ValueError(f"Missing ranks for nodes: {sorted(missing)}")

    roots = tuple(n for n in nodes if indegree[n] == 0)
    leaves = tuple(n for n in nodes if outdegree[n] == 0)
    max_rank = max(rank_dict.values()) if rank_dict else 0

    return GraphFeatures(
        node_ids=nodes,
        edges=edge_tuple,
        ranks=rank_dict,
        indegree=indegree,
        outdegree=outdegree,
        roots=roots,
        leaves=leaves,
        max_rank=max_rank,
    )


def required_role_counts_from_motifs(
    motifs: Iterable[Any],
    aggregate: str = "max",
) -> Counter:
    """
    Extract minimal role inventory from sampled motifs.

    aggregate="max":
        Use maximum count per role among motifs. This is suitable when local motif
        nodes can be merged into a global schema.

    aggregate="sum":
        Sum role counts across motifs. This is stricter and produces larger schemas.
    """
    if aggregate not in {"max", "sum"}:
        raise ValueError("aggregate must be either 'max' or 'sum'.")

    result: Counter = Counter()

    for motif in motifs:
        if hasattr(motif, "roles"):
            roles = list(motif.roles)
        elif isinstance(motif, Mapping) and "nodes" in motif:
            roles = list(motif["nodes"].values())
        else:
            raise TypeError(
                "Each motif must have a .roles attribute or be a dict with a 'nodes' field."
            )

        counts = Counter(normalize_role(r) for r in roles)

        if aggregate == "sum":
            result.update(counts)
        else:
            for role, count in counts.items():
                result[role] = max(result[role], count)

    return result


# ---------------------------------------------------------------------
# RoleSampler
# ---------------------------------------------------------------------

class RoleSampler:
    """
    Assign latent structural roles to anonymous schema nodes.

    This class can be used either:
    - before FK graph construction, to sample a role inventory;
    - after FK graph construction, to assign roles based on observed structure.
    """

    def __init__(
        self,
        role_specs: Optional[Mapping[Role, RoleSpec]] = None,
        config: Optional[RoleSamplingConfig] = None,
        allowed_transitions: Optional[Mapping[Role, Sequence[Role]]] = None,
        seed: Optional[int] = None,
    ):
        self.role_specs: Dict[Role, RoleSpec] = dict(role_specs or build_default_role_specs())
        self.config = config or RoleSamplingConfig()
        self.rng = random.Random(seed)

        if allowed_transitions is None:
            allowed_transitions = DEFAULT_ALLOWED_TRANSITIONS

        self.allowed_transitions: Dict[Role, Tuple[Role, ...]] = {
            normalize_role(src): tuple(normalize_role(dst) for dst in dsts)
            for src, dsts in allowed_transitions.items()
        }

        self._validate_specs()

        if self.config.max_outcome_count is not None and "outcome" in self.role_specs:
            outcome_spec = self.role_specs["outcome"]
            self.role_specs["outcome"] = RoleSpec(
                role=outcome_spec.role,
                base_weight=outcome_spec.base_weight,
                min_count=outcome_spec.min_count,
                max_count=self.config.max_outcome_count,
                min_rank=outcome_spec.min_rank,
                max_rank=outcome_spec.max_rank,
                can_be_root=outcome_spec.can_be_root,
                can_be_leaf=outcome_spec.can_be_leaf,
                root_weight=outcome_spec.root_weight,
                internal_weight=outcome_spec.internal_weight,
                leaf_weight=outcome_spec.leaf_weight,
                indegree_weight=outcome_spec.indegree_weight,
                outdegree_weight=outcome_spec.outdegree_weight,
                prefer_high_indegree=outcome_spec.prefer_high_indegree,
                prefer_high_outdegree=outcome_spec.prefer_high_outdegree,
                prefer_low_indegree=outcome_spec.prefer_low_indegree,
                prefer_low_outdegree=outcome_spec.prefer_low_outdegree,
                description=outcome_spec.description,
            )

    @classmethod
    def default(cls, seed: Optional[int] = None) -> "RoleSampler":
        return cls(seed=seed)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        seed: Optional[int] = None,
    ) -> "RoleSampler":
        """
        Build sampler from a dictionary.

        Example config:
            {
              "sampling": {
                "require_entity": true,
                "require_event_if_possible": true,
                "strict_transition_check": false,
                "max_outcome_count": 1
              },
              "roles": {
                "entity": {"base_weight": 3.0, "min_count": 1},
                "event": {"base_weight": 3.0, "min_rank": 1},
                "summary": {"base_weight": 0.8}
              }
            }
        """
        sampling_cfg = config.get("sampling", {})
        sampler_cfg = RoleSamplingConfig(
            require_entity=bool(sampling_cfg.get("require_entity", True)),
            require_event_if_possible=bool(sampling_cfg.get("require_event_if_possible", True)),
            strict_transition_check=bool(sampling_cfg.get("strict_transition_check", False)),
            transition_bias=float(sampling_cfg.get("transition_bias", 1.5)),
            invalid_transition_penalty=float(sampling_cfg.get("invalid_transition_penalty", 0.15)),
            prefer_outcome_leaf=bool(sampling_cfg.get("prefer_outcome_leaf", True)),
            max_outcome_count=int(sampling_cfg.get("max_outcome_count", 1)),
            use_transition_soft_bias=bool(sampling_cfg.get("use_transition_soft_bias", True)),
        )

        specs = build_default_role_specs()
        role_overrides = config.get("roles", {})

        for role_name, override in role_overrides.items():
            role = normalize_role(role_name)
            base = specs.get(role, RoleSpec(role=role))

            specs[role] = RoleSpec(
                role=role,
                base_weight=float(override.get("base_weight", base.base_weight)),
                min_count=int(override.get("min_count", base.min_count)),
                max_count=(
                    None
                    if override.get("max_count", base.max_count) is None
                    else int(override.get("max_count", base.max_count))
                ),
                min_rank=(
                    None
                    if override.get("min_rank", base.min_rank) is None
                    else int(override.get("min_rank", base.min_rank))
                ),
                max_rank=(
                    None
                    if override.get("max_rank", base.max_rank) is None
                    else int(override.get("max_rank", base.max_rank))
                ),
                can_be_root=bool(override.get("can_be_root", base.can_be_root)),
                can_be_leaf=bool(override.get("can_be_leaf", base.can_be_leaf)),
                root_weight=float(override.get("root_weight", base.root_weight)),
                internal_weight=float(override.get("internal_weight", base.internal_weight)),
                leaf_weight=float(override.get("leaf_weight", base.leaf_weight)),
                indegree_weight=float(override.get("indegree_weight", base.indegree_weight)),
                outdegree_weight=float(override.get("outdegree_weight", base.outdegree_weight)),
                prefer_high_indegree=bool(
                    override.get("prefer_high_indegree", base.prefer_high_indegree)
                ),
                prefer_high_outdegree=bool(
                    override.get("prefer_high_outdegree", base.prefer_high_outdegree)
                ),
                prefer_low_indegree=bool(
                    override.get("prefer_low_indegree", base.prefer_low_indegree)
                ),
                prefer_low_outdegree=bool(
                    override.get("prefer_low_outdegree", base.prefer_low_outdegree)
                ),
                description=str(override.get("description", base.description)),
            )

        transitions = config.get("allowed_transitions", DEFAULT_ALLOWED_TRANSITIONS)

        return cls(
            role_specs=specs,
            config=sampler_cfg,
            allowed_transitions=transitions,
            seed=seed,
        )

    def sample_inventory(
        self,
        num_nodes: int,
        required_roles: Optional[Union[Iterable[str], Mapping[str, int]]] = None,
        allowed_roles: Optional[Iterable[str]] = None,
    ) -> List[Role]:
        """
        Sample a multiset of roles before assigning them to specific nodes.
        """
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")

        allowed = tuple(
            sorted(
                {normalize_role(r) for r in allowed_roles}
                if allowed_roles is not None
                else self.role_specs.keys()
            )
        )
        allowed_set = set(allowed)

        required = self._build_required_counts(num_nodes, required_roles)

        if any(role not in allowed_set for role in required):
            missing = [role for role in required if role not in allowed_set]
            raise ValueError(f"Required roles are not allowed: {missing}")

        if sum(required.values()) > num_nodes:
            raise ValueError(
                f"Required role count {sum(required.values())} exceeds num_nodes={num_nodes}."
            )

        counts: Counter = Counter()
        inventory: List[Role] = []

        for role, count in required.items():
            for _ in range(count):
                inventory.append(role)
                counts[role] += 1

        while len(inventory) < num_nodes:
            candidates = [
                role
                for role in allowed
                if self.role_specs[role].allows_count(counts[role])
            ]
            if not candidates:
                raise ValueError("No candidate roles left while sampling role inventory.")

            weights = [self.role_specs[role].base_weight for role in candidates]
            role = self.rng.choices(candidates, weights=weights, k=1)[0]
            inventory.append(role)
            counts[role] += 1

        self.rng.shuffle(inventory)
        return inventory

    def assign_roles(
        self,
        node_ids: Union[int, Sequence[NodeId]],
        edges: Optional[Sequence[Edge]] = None,
        ranks: Optional[Mapping[NodeId, int]] = None,
        required_roles: Optional[Union[Iterable[str], Mapping[str, int]]] = None,
        motif_required_roles: Optional[Union[Iterable[str], Mapping[str, int]]] = None,
        forced_roles: Optional[Mapping[NodeId, str]] = None,
        strict_transition_check: Optional[bool] = None,
    ) -> RoleAssignment:
        """
        Assign roles to schema nodes.

        Parameters
        ----------
        node_ids:
            Number of nodes or explicit node ids.
        edges:
            FK-support edges.
        ranks:
            Optional precomputed ranks. If absent, ranks are inferred from DAG.
        required_roles:
            Additional required role counts.
        motif_required_roles:
            Required roles derived from sampled motifs.
        forced_roles:
            Fixed role assignment for some nodes.
        strict_transition_check:
            Override config.strict_transition_check.
        """
        features = compute_graph_features(node_ids=node_ids, edges=edges, ranks=ranks)
        forced = {
            str(node): normalize_role(role)
            for node, role in (forced_roles or {}).items()
        }

        for node in forced:
            if node not in set(features.node_ids):
                raise ValueError(f"forced_roles contains unknown node: {node}")

        required = self._build_required_counts(len(features.node_ids), required_roles)
        required.update(_as_role_counter(motif_required_roles))

        # RoleSpec min_count is also respected.
        for role, spec in self.role_specs.items():
            if spec.min_count > 0:
                required[role] = max(required[role], spec.min_count)

        if sum(required.values()) > len(features.node_ids):
            raise ValueError(
                f"Required role count {sum(required.values())} exceeds "
                f"number of nodes {len(features.node_ids)}."
            )

        assignment: Dict[NodeId, Role] = dict(forced)
        counts: Counter = Counter(assignment.values())

        self._check_forced_roles(assignment, features)

        # Step 1: satisfy required role counts.
        for role, required_count in required.items():
            while counts[role] < required_count:
                node = self._choose_node_for_role(
                    role=role,
                    assignment=assignment,
                    counts=counts,
                    features=features,
                )
                assignment[node] = role
                counts[role] += 1

        # Step 2: fill remaining nodes.
        for node in features.node_ids:
            if node in assignment:
                continue

            role = self._choose_role_for_node(
                node=node,
                assignment=assignment,
                counts=counts,
                features=features,
            )
            assignment[node] = role
            counts[role] += 1

        # Step 3: validate transitions.
        violations = self.validate_role_transitions(assignment, features.edges)

        should_strict = (
            self.config.strict_transition_check
            if strict_transition_check is None
            else strict_transition_check
        )
        if should_strict and violations:
            formatted = json.dumps(violations[:10], indent=2, ensure_ascii=False)
            raise ValueError(f"Role transition violations found:\n{formatted}")

        profiles = {
            node: self.build_role_profile(node, assignment[node], features)
            for node in features.node_ids
        }

        return RoleAssignment(
            node_roles=assignment,
            graph_features=features,
            role_profiles=profiles,
            violations=violations,
        )

    def validate_role_transitions(
        self,
        node_roles: Mapping[NodeId, Role],
        edges: Sequence[Edge],
    ) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []

        for src, dst in edges:
            src_role = normalize_role(node_roles[src])
            dst_role = normalize_role(node_roles[dst])

            allowed = self.allowed_transitions.get(src_role, tuple())
            if dst_role not in allowed:
                violations.append(
                    {
                        "edge": [src, dst],
                        "src_role": src_role,
                        "dst_role": dst_role,
                        "reason": "role_transition_not_allowed",
                    }
                )

        return violations

    def build_role_profile(
        self,
        node: NodeId,
        role: Role,
        features: GraphFeatures,
    ) -> Dict[str, Any]:
        role = normalize_role(role)
        base_profile = dict(DEFAULT_ROLE_PROFILES.get(role, {}))

        base_profile.update(
            {
                "node_id": node,
                "role": role,
                "rank": features.ranks[node],
                "indegree": features.indegree[node],
                "outdegree": features.outdegree[node],
                "is_root": node in set(features.roots),
                "is_leaf": node in set(features.leaves),
            }
        )

        if role == "outcome":
            base_profile["visible_as_feature_by_default"] = False
            base_profile["temporal_safety_note"] = (
                "Outcome should be used as target/future-window result, "
                "not as an input feature."
            )

        if role == "summary":
            base_profile["requires_as_of_cutoff"] = True
            base_profile["temporal_safety_note"] = (
                "Summary must be computed only from cutoff-before historical data."
            )

        return base_profile

    def _build_required_counts(
        self,
        num_nodes: int,
        required_roles: Optional[Union[Iterable[str], Mapping[str, int]]],
    ) -> Counter:
        required = _as_role_counter(required_roles)

        if self.config.require_entity and num_nodes >= 1:
            required["entity"] = max(required["entity"], 1)

        if self.config.require_event_if_possible and num_nodes >= 2:
            required["event"] = max(required["event"], 1)

        return required

    def _check_forced_roles(
        self,
        assignment: Mapping[NodeId, Role],
        features: GraphFeatures,
    ) -> None:
        counts = Counter(assignment.values())

        for node, role in assignment.items():
            if role not in self.role_specs:
                raise ValueError(f"No RoleSpec defined for role={role!r}.")

            spec = self.role_specs[role]
            rank = features.ranks[node]
            is_root = node in set(features.roots)
            is_leaf = node in set(features.leaves)

            if is_root and not spec.can_be_root:
                raise ValueError(f"Forced role {role!r} cannot be root node {node!r}.")
            if is_leaf and not spec.can_be_leaf:
                raise ValueError(f"Forced role {role!r} cannot be leaf node {node!r}.")
            if not spec.allows_rank(rank):
                raise ValueError(
                    f"Forced role {role!r} does not allow rank={rank} for node={node!r}."
                )

        for role, count in counts.items():
            spec = self.role_specs[role]
            if spec.max_count is not None and count > spec.max_count:
                raise ValueError(
                    f"Forced role count exceeds max_count: role={role}, "
                    f"count={count}, max_count={spec.max_count}"
                )

    def _choose_node_for_role(
        self,
        role: Role,
        assignment: Mapping[NodeId, Role],
        counts: Counter,
        features: GraphFeatures,
    ) -> NodeId:
        role = normalize_role(role)

        candidates: List[NodeId] = []
        weights: List[float] = []

        for node in features.node_ids:
            if node in assignment:
                continue

            score = self._role_node_score(
                role=role,
                node=node,
                assignment=assignment,
                counts=counts,
                features=features,
            )
            if score > 0:
                candidates.append(node)
                weights.append(score)

        if not candidates:
            raise ValueError(f"No valid node candidate for required role={role!r}.")

        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _choose_role_for_node(
        self,
        node: NodeId,
        assignment: Mapping[NodeId, Role],
        counts: Counter,
        features: GraphFeatures,
    ) -> Role:
        candidates: List[Role] = []
        weights: List[float] = []

        for role in self.role_specs:
            score = self._role_node_score(
                role=role,
                node=node,
                assignment=assignment,
                counts=counts,
                features=features,
            )
            if score > 0:
                candidates.append(role)
                weights.append(score)

        if not candidates:
            raise ValueError(f"No valid role candidate for node={node!r}.")

        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _role_node_score(
        self,
        role: Role,
        node: NodeId,
        assignment: Mapping[NodeId, Role],
        counts: Counter,
        features: GraphFeatures,
    ) -> float:
        role = normalize_role(role)

        if role not in self.role_specs:
            return 0.0

        spec = self.role_specs[role]
        current_count = counts[role]

        if not spec.allows_count(current_count):
            return 0.0

        rank = features.ranks[node]
        indeg = features.indegree[node]
        outdeg = features.outdegree[node]
        is_root = node in set(features.roots)
        is_leaf = node in set(features.leaves)

        if is_root and not spec.can_be_root:
            return 0.0
        if is_leaf and not spec.can_be_leaf:
            return 0.0
        if not spec.allows_rank(rank):
            return 0.0

        score = max(spec.base_weight, 0.0)
        if score == 0:
            return 0.0

        if is_root:
            score *= spec.root_weight
        elif is_leaf:
            score *= spec.leaf_weight
        else:
            score *= spec.internal_weight

        score *= self._degree_multiplier(spec, indeg=indeg, outdeg=outdeg)
        score *= self._role_specific_multiplier(role, indeg=indeg, outdeg=outdeg, is_root=is_root, is_leaf=is_leaf)

        if self.config.use_transition_soft_bias:
            score *= self._transition_context_multiplier(
                role=role,
                node=node,
                assignment=assignment,
                features=features,
            )

        return max(score, 0.0)

    def _degree_multiplier(self, spec: RoleSpec, indeg: int, outdeg: int) -> float:
        multiplier = 1.0

        if spec.indegree_weight != 0:
            multiplier *= math.exp(spec.indegree_weight * indeg)

        if spec.outdegree_weight != 0:
            multiplier *= math.exp(spec.outdegree_weight * outdeg)

        if spec.prefer_high_indegree:
            multiplier *= 1.0 + 0.25 * indeg
        if spec.prefer_high_outdegree:
            multiplier *= 1.0 + 0.25 * outdeg
        if spec.prefer_low_indegree:
            multiplier *= 1.0 / (1.0 + 0.5 * indeg)
        if spec.prefer_low_outdegree:
            multiplier *= 1.0 / (1.0 + 0.5 * outdeg)

        return max(multiplier, 0.0)

    def _role_specific_multiplier(
        self,
        role: Role,
        indeg: int,
        outdeg: int,
        is_root: bool,
        is_leaf: bool,
    ) -> float:
        """
        Additional soft constraints for role-specific structural patterns.
        """
        if role == "bridge":
            # Bridge usually has multiple parent FK references.
            if indeg >= 2:
                return 2.0
            if indeg == 1:
                return 0.5
            return 0.05

        if role == "outcome":
            # Outcome should preferably be a downstream leaf.
            if outdeg > 0:
                return 0.05 if self.config.prefer_outcome_leaf else 0.4
            if is_leaf:
                return 2.0
            return 0.5

        if role == "summary":
            # Summary usually has upstream historical sources.
            if indeg == 0:
                return 0.1
            return 1.0 + 0.2 * indeg

        if role == "measurement":
            # Measurement should be attached to event/entity/bridge.
            if indeg == 0:
                return 0.1
            return 1.0

        if role in {"class", "context"}:
            # Dimension/context tables should usually be upstream.
            if is_root:
                return 1.5
            if outdeg == 0:
                return 0.2
            return 1.0

        if role == "event":
            # Event should normally have at least one upstream entity/context.
            if is_root:
                return 0.02
            if indeg == 0:
                return 0.2
            return 1.0 + 0.1 * indeg

        return 1.0

    def _transition_context_multiplier(
        self,
        role: Role,
        node: NodeId,
        assignment: Mapping[NodeId, Role],
        features: GraphFeatures,
    ) -> float:
        """
        Softly encourage role assignments compatible with already assigned neighbors.
        """
        multiplier = 1.0

        for src, dst in features.edges:
            if src == node and dst in assignment:
                src_role = role
                dst_role = assignment[dst]
                if self._is_transition_allowed(src_role, dst_role):
                    multiplier *= self.config.transition_bias
                else:
                    multiplier *= self.config.invalid_transition_penalty

            elif dst == node and src in assignment:
                src_role = assignment[src]
                dst_role = role
                if self._is_transition_allowed(src_role, dst_role):
                    multiplier *= self.config.transition_bias
                else:
                    multiplier *= self.config.invalid_transition_penalty

        return multiplier

    def _is_transition_allowed(self, src_role: Role, dst_role: Role) -> bool:
        src = normalize_role(src_role)
        dst = normalize_role(dst_role)
        return dst in self.allowed_transitions.get(src, tuple())

    def _validate_specs(self) -> None:
        missing = set(VALID_ROLES) - set(self.role_specs)
        if missing:
            raise ValueError(f"Missing RoleSpec for roles: {sorted(missing)}")

        for role in self.role_specs:
            normalize_role(role)


__all__ = [
    "Role",
    "NodeId",
    "Edge",
    "VALID_ROLES",
    "ROLE_ALIASES",
    "DEFAULT_ALLOWED_TRANSITIONS",
    "DEFAULT_ROLE_PROFILES",
    "normalize_role",
    "RoleSpec",
    "RoleSamplingConfig",
    "GraphFeatures",
    "RoleAssignment",
    "build_default_role_specs",
    "compute_graph_features",
    "required_role_counts_from_motifs",
    "RoleSampler",
]