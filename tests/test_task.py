import tempfile
import sys
import unittest
from pathlib import Path

try:
    import h5py
except ImportError:  # pragma: no cover - depends on optional RDBPFN export dependency.
    h5py = None

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from syn_data.src.rdb_prior.generators.base import GenerationContext
from syn_data.src.rdb_prior.generators.database_generator import RelationalDatabaseGenerator
from syn_data.src.rdb_prior.schema.schema_sampler import SchemaSampler, SchemaSamplingConfig
from syn_data.src.rdb_prior.task.label_generator import LabelGenerator
from syn_data.src.rdb_prior.task.rdbpfn_exporter import (
    RDBPFNExportConfig,
    build_rdbpfn_sample,
    export_task_directories_to_h5,
)
from syn_data.src.rdb_prior.task.task_exporter import save_task_bundle
from syn_data.src.rdb_prior.task.task_sampler import TaskSampler


class TaskGenerationTests(unittest.TestCase):
    def test_outcome_task_reads_existing_label_and_hides_outcome(self):
        schema, result, spec, bundle = self._build_bundle("outcome", seed=2)

        del schema
        self.assertEqual("existing_outcome_label", spec.label_source_mode)
        self.assertIn(spec.target_source_table, bundle.feature_manifest["hidden_tables"])
        self.assertNotIn(spec.target_source_table, bundle.feature_manifest["visible_tables"])
        self.assertGreater(len(bundle.labels), 0)
        self.assertTrue((pd.to_datetime(bundle.labels["outcome_time"]) > pd.Timestamp(spec.cutoff_time)).all())

        outcome_metadata = result["metadata"]["tables"][spec.target_source_table]
        self.assertFalse(outcome_metadata["visible_as_feature_by_default"])
        self.assertTrue(outcome_metadata["is_task_target"])

    def test_entity_task_derives_labels_without_mutating_table(self):
        _, result, spec, bundle = self._build_bundle("entity", seed=42)

        source_df = result["tables"][spec.target_source_table]
        self.assertEqual("derived_table_label", spec.label_source_mode)
        self.assertNotIn("label", source_df.columns)
        self.assertEqual(len(source_df), len(bundle.labels))
        self.assertEqual(spec.target_source_table, bundle.metadata["derived_from_table"])
        self._assert_derived_signal_columns_are_hidden(bundle)

    def test_event_task_uses_only_history_rows(self):
        _, result, spec, bundle = self._build_bundle("event", seed=43)

        event_df = result["tables"][spec.target_source_table]
        time_col = spec.target_time_col
        self.assertIsNotNone(time_col)
        cutoff = pd.Timestamp(spec.cutoff_time)
        history = event_df[pd.to_datetime(event_df[time_col]) < cutoff]
        self.assertEqual(len(history), len(bundle.labels))

        selected = history.set_index(spec.prediction_unit_pk).loc[bundle.labels["prediction_id"]]
        self.assertTrue((pd.to_datetime(selected[time_col]) < cutoff).all())

    def test_summary_task_requires_cutoff_safe_metadata(self):
        _, result, spec, bundle = self._build_bundle("summary", seed=2)

        summary_metadata = result["metadata"]["tables"][spec.target_source_table]
        self.assertTrue(summary_metadata["used_only_history_before_cutoff"])
        self.assertEqual([], summary_metadata["leakage_violations"])
        self.assertGreater(len(bundle.labels), 0)
        self._assert_derived_signal_columns_are_hidden(bundle)

    def test_splits_are_disjoint_and_reproducible(self):
        schema, result, spec, bundle = self._build_bundle("entity", seed=44)

        split_sets = {
            name: set(df["prediction_id"].tolist())
            for name, df in bundle.splits.items()
        }
        self.assertTrue(split_sets["train"].isdisjoint(split_sets["val"]))
        self.assertTrue(split_sets["train"].isdisjoint(split_sets["test"]))
        self.assertTrue(split_sets["val"].isdisjoint(split_sets["test"]))
        self.assertEqual(set(bundle.labels["prediction_id"]), set().union(*split_sets.values()))

        rebuilt = LabelGenerator(seed=44).build_task_bundle(
            spec=spec,
            schema=schema,
            tables=result["tables"],
            metadata=result["metadata"],
        )
        pd.testing.assert_frame_equal(bundle.labels, rebuilt.labels)
        self.assertEqual(bundle.feature_manifest, rebuilt.feature_manifest)

    def test_save_task_bundle_writes_expected_files(self):
        _, _, _, bundle = self._build_bundle("entity", seed=45)

        with tempfile.TemporaryDirectory() as tmp:
            paths = save_task_bundle(Path(tmp) / "task", bundle)
            self.assertTrue(Path(paths["task"]).exists())
            self.assertTrue(Path(paths["labels"]).exists())
            self.assertTrue(Path(paths["feature_manifest"]).exists())
            self.assertTrue(Path(paths["split_train"]).exists())
            self.assertTrue(Path(paths["split_val"]).exists())
            self.assertTrue(Path(paths["split_test"]).exists())
            labels = pd.read_parquet(paths["labels"])
            self.assertEqual(len(bundle.labels), len(labels))

    def test_build_rdbpfn_sample_from_task_bundle(self):
        _, result, _, bundle = self._build_bundle("outcome", seed=46)

        with tempfile.TemporaryDirectory() as tmp:
            db_dir = self._write_database_task_dir(Path(tmp) / "db_000001", result, bundle)
            sample = build_rdbpfn_sample(
                db_dir,
                config=RDBPFNExportConfig(total_rows=32, max_columns=16, seed=46),
                sample_seed=46,
            )
            self.assertEqual((32, sample.num_features), sample.x.shape)
            self.assertEqual((32,), sample.y.shape)
            self.assertGreater(sample.num_features, 0)
            self.assertGreater(sample.split_idx, 0)
            self.assertLess(sample.split_idx, 32)

    @unittest.skipIf(h5py is None, "h5py is not installed in this environment")
    def test_export_task_bundle_to_rdbpfn_h5(self):
        _, result, _, bundle = self._build_bundle("outcome", seed=46)

        with tempfile.TemporaryDirectory() as tmp:
            db_dir = self._write_database_task_dir(Path(tmp) / "db_000001", result, bundle)

            output = Path(tmp) / "rdbpfn_tasks.h5"
            report = export_task_directories_to_h5(
                [db_dir],
                output,
                config=RDBPFNExportConfig(total_rows=32, max_columns=16, seed=46),
            )
            self.assertEqual(1, report["num_samples"])
            self.assertTrue(output.exists())

            with h5py.File(output, "r") as h5:
                for key in [
                    "X",
                    "y",
                    "num_features",
                    "num_available_features",
                    "num_datapoints",
                    "single_eval_pos",
                    "feature_is_categorical",
                    "max_num_classes",
                ]:
                    self.assertIn(key, h5)
                self.assertEqual((1, 32, 16), h5["X"].shape)
                self.assertEqual((1, 32), h5["y"].shape)
                self.assertGreater(int(h5["num_features"][0]), 0)
                self.assertEqual(32, int(h5["num_datapoints"][0]))
                self.assertGreater(int(h5["single_eval_pos"][0]), 0)
                self.assertLess(int(h5["single_eval_pos"][0]), 32)
                self.assertEqual(2, int(h5["max_num_classes"][0]))

    def _write_database_task_dir(self, db_dir: Path, result, bundle) -> Path:
        table_dir = db_dir / "tables"
        table_dir.mkdir(parents=True)
        for table_id, df in result["tables"].items():
            df.to_parquet(table_dir / f"{table_id}.parquet", index=False)
        save_task_bundle(db_dir / "task", bundle)
        return db_dir

    def _build_bundle(self, target_role: str, seed: int):
        cfg = SchemaSamplingConfig(min_tables=6, max_tables=8, min_motifs=1, max_motifs=4, seed=seed)
        sampler = SchemaSampler(config=cfg, seed=seed)
        for schema in sampler.sample_many(80, schema_id_prefix=f"task_{target_role}"):
            if target_role not in {node.role for node in schema.nodes.values()}:
                continue
            context = GenerationContext(seed=seed, row_scale=0.001, max_rows_per_table=500)
            result = RelationalDatabaseGenerator(seed=seed).generate_database(schema, context=context)
            try:
                spec = TaskSampler(seed=seed).sample_task(
                    schema=schema,
                    tables=result["tables"],
                    metadata=result["metadata"],
                    target_source_role=target_role,
                )
                bundle = LabelGenerator(seed=seed).build_task_bundle(
                    spec=spec,
                    schema=schema,
                    tables=result["tables"],
                    metadata=result["metadata"],
                )
            except ValueError:
                continue
            return schema, result, spec, bundle
        self.fail(f"Unable to build a valid {target_role!r} task bundle.")

    def _assert_derived_signal_columns_are_hidden(self, bundle):
        target_table = bundle.spec.target_source_table
        visible_columns = set(bundle.feature_manifest["visible_columns"].get(target_table, []))
        for column in bundle.metadata["derived_signal_columns"]:
            self.assertNotIn(column, visible_columns)


if __name__ == "__main__":
    unittest.main()
