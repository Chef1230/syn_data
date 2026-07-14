import tempfile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from syn_data.src.rdb_prior.generators.base import GenerationContext
from syn_data.src.rdb_prior.generators.database_generator import RelationalDatabaseGenerator
from syn_data.src.rdb_prior.schema.schema_sampler import SchemaSampler, SchemaSamplingConfig


class RelationalGenerationTests(unittest.TestCase):
    def test_generate_default_schema_tables(self):
        schema = SchemaSampler.default(seed=42).sample_schema("db_000001")
        context = GenerationContext(seed=42, row_scale=0.001, max_rows_per_table=1000)
        result = RelationalDatabaseGenerator(seed=42).generate_database(schema, context=context)

        self.assertEqual(set(schema.nodes), set(result["tables"]))
        for table_id, df in result["tables"].items():
            self.assertGreaterEqual(len(df), 1)
            self.assertIn(table_id, result["metadata"]["tables"])

    def test_summary_and_outcome_metadata_contracts(self):
        cfg = SchemaSamplingConfig(min_tables=6, max_tables=8, min_motifs=2, max_motifs=4, seed=2)
        sampler = SchemaSampler(config=cfg, seed=2)
        schema = next(
            (
                candidate
                for candidate in sampler.sample_many(20, schema_id_prefix="db_summary")
                if {node.role for node in candidate.nodes.values()} >= {"summary", "outcome"}
            ),
            None,
        )
        self.assertIsNotNone(schema)
        context = GenerationContext(seed=2, row_scale=0.001, max_rows_per_table=1000)
        result = RelationalDatabaseGenerator(seed=2).generate_database(schema, context=context)

        summaries = [m for m in result["metadata"]["tables"].values() if m.get("role") == "summary"]
        self.assertTrue(summaries)
        for metadata in summaries:
            self.assertTrue(metadata["requires_as_of_cutoff"])
            self.assertTrue(metadata["used_only_history_before_cutoff"])
            self.assertEqual([], metadata["leakage_violations"])

        outcomes = [m for m in result["metadata"]["tables"].values() if m.get("role") == "outcome"]
        self.assertTrue(outcomes)
        for metadata in outcomes:
            self.assertFalse(metadata["visible_as_feature_by_default"])
            self.assertTrue(metadata["is_task_target"])

    def test_save_database(self):
        schema = SchemaSampler.default(seed=42).sample_schema("db_000001")
        context = GenerationContext(seed=42, row_scale=0.001, max_rows_per_table=1000)
        generator = RelationalDatabaseGenerator(seed=42)
        result = generator.generate_database(schema, context=context)

        with tempfile.TemporaryDirectory() as tmp:
            paths = generator.save_database(tmp, result["tables"], result["metadata"])
            self.assertTrue(Path(paths["metadata"]).exists())
            for table_id in result["tables"]:
                self.assertTrue(Path(paths[table_id]).exists())


if __name__ == "__main__":
    unittest.main()