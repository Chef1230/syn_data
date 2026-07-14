import json
import sys
import tempfile
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
from syn_data.src.rdb_prior.cli import generate_task_bundles
from syn_data.src.rdb_prior.schema.schema_sampler import (
    ColumnSpec,
    ForeignKeySpec,
    SampledSchema,
    SchemaNode,
    SchemaSampler,
    SchemaSamplingConfig,
)
from syn_data.src.rdb_prior.task.label_generator import LabelGenerator
from syn_data.src.rdb_prior.task.rdbpfn_exporter import (
    RDBPFNExportConfig,
    build_rdbpfn_sample,
    export_task_directories_to_h5,
)
from syn_data.src.rdb_prior.task.task_exporter import save_task_bundle
from syn_data.src.rdb_prior.task.task_sampler import TaskSampler, TaskSpec


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

    def test_derived_entity_task_keeps_exportable_visible_feature(self):
        schema = SampledSchema(
            schema_id="derived_entity_edge",
            nodes={
                "T0": SchemaNode(
                    node_id="T0",
                    role="entity",
                    rank=0,
                    num_rows=96,
                    num_columns=4,
                    primary_key="t0_id",
                    time_col=None,
                    columns=[
                        ColumnSpec(
                            name="t0_id",
                            dtype="int64",
                            semantic_type="primary_key",
                            is_primary_key=True,
                        ),
                        ColumnSpec(name="t0_num_0", dtype="float32", semantic_type="numeric"),
                        ColumnSpec(name="t0_cat_1", dtype="category", semantic_type="categorical"),
                        ColumnSpec(name="t0_num_2", dtype="float32", semantic_type="numeric"),
                    ],
                )
            },
            foreign_keys=[],
            motifs=[],
        )
        table = pd.DataFrame(
            {
                "t0_id": range(96),
                "__latent_activity": [(idx % 7) / 7.0 for idx in range(96)],
                "__latent_quality": [(idx % 5) / 5.0 for idx in range(96)],
                "__activity_score": [1.0 for _ in range(96)],
                "t0_num_0": [float(idx % 11) for idx in range(96)],
                "t0_cat_1": [f"c{idx % 3}" for idx in range(96)],
                "t0_num_2": [float(idx % 13) for idx in range(96)],
            }
        )
        spec = TaskSpec(
            task_id="derived_entity_edge",
            task_type="binary_classification",
            target_source_table="T0",
            target_source_role="entity",
            prediction_unit_table="T0",
            prediction_unit_pk="t0_id",
            label_source_mode="derived_table_label",
            label_col="label",
            cutoff_time="2021-12-02",
            future_window_days=30,
        )
        result = {
            "tables": {"T0": table},
            "metadata": {
                "context": {"cutoff_time": "2021-12-02", "future_window_days": 30},
                "tables": {"T0": {"role": "entity"}},
            },
        }
        bundle = LabelGenerator(seed=9975).build_task_bundle(
            spec=spec,
            schema=schema,
            tables=result["tables"],
            metadata=result["metadata"],
        )

        visible_columns = set(bundle.feature_manifest["visible_columns"]["T0"])
        self.assertIn("t0_num_2", visible_columns)
        self.assertNotIn("t0_num_0", visible_columns)
        self.assertNotIn("t0_cat_1", visible_columns)
        self._assert_derived_signal_columns_are_hidden(bundle)

        with tempfile.TemporaryDirectory() as tmp:
            db_dir = self._write_database_task_dir(Path(tmp) / "db_derived_entity", result, bundle)
            sample = build_rdbpfn_sample(
                db_dir,
                config=RDBPFNExportConfig(total_rows=32, max_columns=16, seed=9975),
                sample_seed=9975,
            )
            self.assertEqual(1, sample.num_features)
            self.assertEqual((32, 1), sample.x.shape)

    def test_outcome_task_avoids_single_unit_prediction_parent(self):
        schema = SampledSchema(
            schema_id="outcome_prediction_unit_edge",
            nodes={
                "T0": SchemaNode(
                    node_id="T0",
                    role="entity",
                    rank=0,
                    num_rows=5,
                    num_columns=2,
                    primary_key="t0_id",
                    time_col=None,
                    columns=[
                        ColumnSpec(
                            name="t0_id",
                            dtype="int64",
                            semantic_type="primary_key",
                            is_primary_key=True,
                        ),
                        ColumnSpec(name="t0_num_0", dtype="float32", semantic_type="numeric"),
                    ],
                ),
                "T1": SchemaNode(
                    node_id="T1",
                    role="summary",
                    rank=1,
                    num_rows=1,
                    num_columns=2,
                    primary_key="t1_id",
                    time_col="as_of_time",
                    columns=[
                        ColumnSpec(
                            name="t1_id",
                            dtype="int64",
                            semantic_type="primary_key",
                            is_primary_key=True,
                        ),
                        ColumnSpec(name="as_of_time", dtype="datetime64", semantic_type="timestamp", is_time=True),
                    ],
                ),
                "T2": SchemaNode(
                    node_id="T2",
                    role="outcome",
                    rank=2,
                    num_rows=5,
                    num_columns=4,
                    primary_key="t2_id",
                    time_col="outcome_time",
                    columns=[
                        ColumnSpec(
                            name="t2_id",
                            dtype="int64",
                            semantic_type="primary_key",
                            is_primary_key=True,
                        ),
                        ColumnSpec(
                            name="outcome_time",
                            dtype="datetime64",
                            semantic_type="timestamp",
                            is_time=True,
                        ),
                        ColumnSpec(
                            name="label",
                            dtype="float32",
                            semantic_type="outcome_label",
                            is_label_candidate=True,
                        ),
                    ],
                ),
            },
            foreign_keys=[
                ForeignKeySpec(parent_table="T0", child_table="T2", parent_col="t0_id", child_col="t0_id"),
                ForeignKeySpec(parent_table="T1", child_table="T2", parent_col="t1_id", child_col="t1_id"),
            ],
            motifs=[],
        )
        tables = {
            "T0": pd.DataFrame({"t0_id": range(5), "t0_num_0": [0.1, 0.2, 0.3, 0.4, 0.5]}),
            "T1": pd.DataFrame({"t1_id": [0], "as_of_time": [pd.Timestamp("2021-12-02")]}),
            "T2": pd.DataFrame(
                {
                    "t2_id": range(5),
                    "t0_id": range(5),
                    "t1_id": [0, 0, 0, 0, 0],
                    "label": [0, 1, 0, 1, 0],
                    "outcome_time": [pd.Timestamp("2021-12-10")] * 5,
                }
            ),
        }
        metadata = {
            "context": {"cutoff_time": "2021-12-02", "future_window_days": 30},
            "tables": {
                "T0": {"role": "entity"},
                "T1": {"role": "summary", "used_only_history_before_cutoff": True},
                "T2": {"role": "outcome", "label_col": "label", "visible_as_feature_by_default": False},
            },
        }

        spec = TaskSampler(seed=90).sample_task(
            schema=schema,
            tables=tables,
            metadata=metadata,
            target_source_role="outcome",
        )
        self.assertEqual("T0", spec.prediction_unit_table)
        self.assertEqual("t0_id", spec.target_fk_col)

    def test_task_sampler_supports_one_to_three_distinct_tasks(self):
        schema, result = self._multi_task_fixture()

        for task_count in (1, 2, 3):
            specs = TaskSampler(seed=123).sample_tasks(
                schema=schema,
                tables=result["tables"],
                metadata=result["metadata"],
                min_tasks=task_count,
                max_tasks=task_count,
            )
            self.assertEqual(task_count, len(specs))
            self.assertEqual(task_count, len({spec.target_source_table for spec in specs}))
            self.assertEqual("summary", specs[0].target_source_role)

        first = TaskSampler(seed=123).sample_tasks(
            schema=schema,
            tables=result["tables"],
            metadata=result["metadata"],
            min_tasks=1,
            max_tasks=3,
        )
        second = TaskSampler(seed=123).sample_tasks(
            schema=schema,
            tables=result["tables"],
            metadata=result["metadata"],
            min_tasks=1,
            max_tasks=3,
        )
        self.assertEqual([spec.task_id for spec in first], [spec.task_id for spec in second])

        with self.assertRaisesRegex(ValueError, "must not exceed 3"):
            TaskSampler(seed=123).sample_tasks(
                schema=schema,
                tables=result["tables"],
                metadata=result["metadata"],
                min_tasks=1,
                max_tasks=4,
            )

    def test_generate_multiple_task_bundles_writes_index(self):
        schema, result = self._multi_task_fixture()

        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp) / "db_multi"
            report = generate_task_bundles(
                schema=schema,
                tables=result["tables"],
                metadata=result["metadata"],
                output_dir=db_dir,
                task_cfg={
                    "min_tasks_per_database": 3,
                    "max_tasks_per_database": 3,
                    "positive_rate": 0.35,
                },
                seed=123,
            )

            self.assertEqual(3, report["num_tasks"])
            self.assertEqual(db_dir / "task", Path(report["task_dirs"][0]))
            self.assertTrue((db_dir / "task" / "task.json").exists())
            self.assertTrue((db_dir / "tasks" / "task_001" / "task.json").exists())
            self.assertTrue((db_dir / "tasks" / "task_002" / "task.json").exists())

            index = json.loads(Path(report["index_path"]).read_text(encoding="utf-8"))
            self.assertEqual(3, index["requested_count"])
            self.assertEqual(3, index["generated_count"])
            self.assertEqual("../task", index["tasks"][0]["path"])
            self.assertEqual(3, len({item["target_source_table"] for item in index["tasks"]}))

    @unittest.skipIf(h5py is None, "h5py is not installed in this environment")
    def test_export_multiple_tasks_from_one_database(self):
        schema, result = self._multi_task_fixture()

        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp) / "db_multi"
            table_dir = db_dir / "tables"
            table_dir.mkdir(parents=True)
            for table_id, df in result["tables"].items():
                df.to_parquet(table_dir / f"{table_id}.parquet", index=False)
            generate_task_bundles(
                schema=schema,
                tables=result["tables"],
                metadata=result["metadata"],
                output_dir=db_dir,
                task_cfg={"min_tasks_per_database": 3, "max_tasks_per_database": 3},
                seed=123,
            )

            output = Path(tmp) / "multi_tasks.h5"
            report = export_task_directories_to_h5(
                [db_dir],
                output,
                config=RDBPFNExportConfig(total_rows=32, max_columns=16, seed=123),
            )
            self.assertEqual(1, report["num_databases"])
            self.assertEqual(3, report["num_task_bundles"])
            self.assertEqual(3, report["num_samples"])
            with h5py.File(output, "r") as h5:
                self.assertEqual((3, 32, 16), h5["X"].shape)
                self.assertEqual((3, 32), h5["y"].shape)

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

    def _multi_task_fixture(self):
        nodes = {}
        tables = {}
        table_metadata = {}
        roles = (("T0", "entity", None), ("T1", "event", "t1_time"), ("T2", "summary", "t2_time"))
        for table_id, role, time_col in roles:
            prefix = table_id.lower()
            columns = [
                ColumnSpec(
                    name=f"{prefix}_id",
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
            columns.extend(
                [
                    ColumnSpec(name=f"{prefix}_num_0", dtype="float32", semantic_type="numeric"),
                    ColumnSpec(name=f"{prefix}_num_1", dtype="float32", semantic_type="numeric"),
                ]
            )
            nodes[table_id] = SchemaNode(
                node_id=table_id,
                role=role,
                rank=0 if role == "entity" else 1,
                num_rows=32,
                num_columns=len(columns),
                primary_key=f"{prefix}_id",
                time_col=time_col,
                columns=columns,
            )
            data = {
                f"{prefix}_id": range(32),
                f"{prefix}_num_0": [float(idx) for idx in range(32)],
                f"{prefix}_num_1": [float(idx % 7) for idx in range(32)],
            }
            if time_col is not None:
                if role == "summary":
                    data[time_col] = [pd.Timestamp("2021-12-02")] * 32
                else:
                    data[time_col] = pd.date_range("2021-01-01", periods=32, freq="D")
            tables[table_id] = pd.DataFrame(data)
            table_metadata[table_id] = {
                "role": role,
                "time_col": time_col,
                "used_only_history_before_cutoff": role == "summary",
            }

        schema = SampledSchema(
            schema_id="multi_task_fixture",
            nodes=nodes,
            foreign_keys=[],
            motifs=[],
        )
        result = {
            "tables": tables,
            "metadata": {
                "context": {"cutoff_time": "2021-12-02", "future_window_days": 30},
                "tables": table_metadata,
            },
        }
        return schema, result

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
