import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from syn_data.src.rdb_prior.schema.fk_graph import graph_from_sampled_schema_like
from syn_data.src.rdb_prior.schema.schema_sampler import (
    ROLE_ROW_COUNT_RANGES,
    SchemaSampler,
    SchemaSamplingConfig,
)


class SchemaSamplerTests(unittest.TestCase):
    def test_default_sampler_respects_capacity_depth_and_connectivity(self):
        sampler = SchemaSampler.default(seed=42)
        schemas = sampler.sample_many(50)

        self.assertEqual(50, len(schemas))
        for schema in schemas:
            self.assertEqual([], schema.violations)
            actual_depth = max(node.rank for node in schema.nodes.values())
            sampled_depth = schema.metadata["config"]["sampled_schema_depth"]
            self.assertEqual(actual_depth, schema.metadata["actual_schema_depth"])
            self.assertLessEqual(actual_depth, sampled_depth)
            self.assertIn(sampled_depth, sampler.config.schema_depth_values)
            self.assertTrue(graph_from_sampled_schema_like(schema).is_weakly_connected())
            self._assert_bridge_parent_constraints(schema)

    def test_schema_depth_sampling_prefers_depth_two(self):
        sampler = SchemaSampler.default(seed=777)
        schemas = sampler.sample_many(300)

        sampled_depth_counts = {depth: 0 for depth in sampler.config.schema_depth_values}
        actual_depth_counts = {depth: 0 for depth in sampler.config.schema_depth_values}
        for schema in schemas:
            sampled_depth = schema.metadata["config"]["sampled_schema_depth"]
            actual_depth = schema.metadata["actual_schema_depth"]
            self.assertIn(sampled_depth, {1, 2, 3})
            self.assertIn(actual_depth, {1, 2, 3})
            self.assertLessEqual(actual_depth, sampled_depth)
            sampled_depth_counts[sampled_depth] += 1
            actual_depth_counts[actual_depth] += 1

        self.assertGreater(sampled_depth_counts[2], sampled_depth_counts[1])
        self.assertGreater(sampled_depth_counts[2], sampled_depth_counts[3])
        self.assertGreater(actual_depth_counts[2], actual_depth_counts[1])
        self.assertGreater(actual_depth_counts[2], actual_depth_counts[3])

    def test_role_row_count_ranges_are_enforced(self):
        sampler = SchemaSampler.default(seed=2026)
        schemas = sampler.sample_many(100)

        for schema in schemas:
            metadata_ranges = schema.metadata["config"]["role_row_count_ranges"]
            self.assertEqual(
                {role: list(bounds) for role, bounds in ROLE_ROW_COUNT_RANGES.items()},
                metadata_ranges,
            )
            for node in schema.nodes.values():
                low, high = ROLE_ROW_COUNT_RANGES[node.role]
                self.assertGreaterEqual(node.num_rows, low)
                self.assertLessEqual(node.num_rows, high)

    def test_bridge_tables_have_required_parents(self):
        sampler = SchemaSampler.default(seed=123)
        schemas = sampler.sample_many(100)

        self.assertTrue(any("bridge" in schema.role_counts for schema in schemas))
        for schema in schemas:
            self._assert_bridge_parent_constraints(schema)

    def _assert_bridge_parent_constraints(self, schema):
        parents_by_child = {}
        for fk in schema.foreign_keys:
            parents_by_child.setdefault(fk.child_table, set()).add(fk.parent_table)

        for node_id, node in schema.nodes.items():
            if node.role == "bridge":
                self.assertGreaterEqual(
                    len(parents_by_child.get(node_id, set())),
                    2,
                    f"bridge table {node_id} should have at least two parent tables",
                )

    def test_seed_is_reproducible_across_processes(self):
        script = """
import json
from syn_data.src.rdb_prior.schema.schema_sampler import SchemaSampler
schema = SchemaSampler.default(seed=7).sample_schema('x')
print(json.dumps({
    'edges': schema.edges,
    'roles': [node.role for node in schema.nodes.values()],
    'ranks': [node.rank for node in schema.nodes.values()],
    'rows': [node.num_rows for node in schema.nodes.values()],
}, sort_keys=True))
"""
        first = subprocess.check_output([sys.executable, "-c", script], cwd=ROOT, text=True)
        second = subprocess.check_output([sys.executable, "-c", script], cwd=ROOT, text=True)

        self.assertEqual(first, second)

    def test_parallel_fk_preserves_duplicate_edges_and_columns(self):
        config = SchemaSamplingConfig(
            min_tables=2,
            max_tables=2,
            min_motifs=0,
            max_motifs=0,
            add_random_edges=False,
            require_connected=False,
            allow_parallel_fk=True,
        )
        sampler = SchemaSampler(config=config, seed=1)
        nodes = sampler._build_nodes(
            node_roles={"T0": "entity", "T1": "event"},
            ranks={"T0": 0, "T1": 1},
            edges=[("T0", "T1"), ("T0", "T1")],
        )
        foreign_keys = sampler._build_foreign_keys(
            nodes=nodes,
            edges=[("T0", "T1"), ("T0", "T1")],
        )
        sampler._attach_fk_columns(nodes=nodes, foreign_keys=foreign_keys)

        self.assertEqual(["t0_id", "t0_id_2"], [fk.child_col for fk in foreign_keys])
        self.assertEqual(
            ["t0_id", "t0_id_2"],
            [col.name for col in nodes["T1"].columns if col.is_foreign_key],
        )


if __name__ == "__main__":
    unittest.main()