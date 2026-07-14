import json
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from syn_data.src.rdb_prior.analysis.schema_stats import (
    EdgeInfo,
    TableInfo,
    collect_dbinfer_stats,
    collect_relbench_stats,
    collect_synthetic_stats,
    graph_metrics_row,
    main as schema_stats_main,
)


class SchemaStatsTests(unittest.TestCase):
    def test_synthetic_schema_metrics_and_fanout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "synthetic"
            self._write_synthetic_db(root / "db_000001")

            stats = collect_synthetic_stats(root)

            self.assertEqual(1, len(stats.per_db_rows))
            row = stats.per_db_rows[0]
            self.assertEqual("synthetic", row["source_group"])
            self.assertEqual(3, row["num_tables"])
            self.assertEqual(2, row["num_edges"])
            self.assertTrue(row["is_dag"])
            self.assertEqual(2, row["dag_depth"])
            self.assertEqual(1, row["root_count"])
            self.assertEqual(1, row["leaf_count"])

            edge = next(item for item in stats.edge_rows if item["parent_table"] == "A")
            self.assertEqual("ok", edge["fanout_status"])
            self.assertEqual(3, edge["parent_rows"])
            self.assertEqual(5, edge["child_rows"])
            self.assertEqual(2, edge["referenced_parent_keys"])
            self.assertAlmostEqual(2.5, edge["mean_fanout"])
            self.assertAlmostEqual(1 / 3, edge["zero_child_parent_ratio"])
            self.assertEqual(0.0, edge["orphan_fk_ratio"])

    def test_dbinfer_metadata_metrics_and_fanout(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp) / "existing_demo"
            self._write_dbinfer_db(db_dir)

            stats = collect_dbinfer_stats(Path(tmp))

            self.assertEqual(1, len(stats.per_db_rows))
            row = stats.per_db_rows[0]
            self.assertEqual("existing", row["source_group"])
            self.assertEqual("existing_demo", row["dataset_id"])
            self.assertEqual(2, row["num_tables"])
            self.assertEqual(1, row["num_edges"])
            self.assertEqual(1, row["dag_depth"])

            self.assertEqual(1, len(stats.edge_rows))
            edge = stats.edge_rows[0]
            self.assertEqual("ok", edge["fanout_status"])
            self.assertEqual("parent", edge["parent_table"])
            self.assertEqual("child", edge["child_table"])
            self.assertEqual(2, edge["referenced_parent_keys"])
            self.assertAlmostEqual(0.25, edge["orphan_fk_ratio"])

    def test_relbench_stats_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            relbench_path = Path(tmp) / "schema_graph.json"
            self._write_relbench_stats(relbench_path)

            stats = collect_relbench_stats(relbench_path)

            self.assertEqual(1, len(stats.per_db_rows))
            row = stats.per_db_rows[0]
            self.assertEqual("relbench", row["source_group"])
            self.assertEqual("rel-demo", row["dataset_id"])
            self.assertEqual(2, row["num_tables"])
            self.assertEqual(1, row["num_edges"])

            edge = stats.edge_rows[0]
            self.assertEqual("from_relbench_stats", edge["fanout_status"])
            self.assertEqual(2.5, edge["mean_fanout"])
            self.assertEqual(2.0, edge["median_fanout"])
            self.assertEqual(4.0, edge["p95_fanout"])
            self.assertEqual(5.0, edge["max_fanout"])
            self.assertAlmostEqual(0.1, edge["orphan_fk_ratio"])

    def test_cycle_graph_reports_non_dag(self):
        tables = {"A": TableInfo("A"), "B": TableInfo("B")}
        edges = [EdgeInfo("A", "B"), EdgeInfo("B", "A")]

        row = graph_metrics_row("synthetic", "cycle_db", "memory", tables, edges)

        self.assertFalse(row["is_dag"])
        self.assertIsNone(row["dag_depth"])

    def test_missing_table_records_fanout_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp) / "missing_demo"
            db_dir.mkdir()
            (db_dir / "metadata.yaml").write_text(
                """
dataset_name: missing_demo
tables:
  - name: parent
    source: parent.parquet
    format: parquet
    time_column: null
    columns:
      - name: id
        dtype: primary_key
        capacity: 3
  - name: child
    source: child.parquet
    format: parquet
    time_column: null
    columns:
      - name: id
        dtype: primary_key
      - name: parent_id
        dtype: foreign_key
        link_to: parent.id
tasks: []
""".strip(),
                encoding="utf-8",
            )

            stats = collect_dbinfer_stats(Path(tmp))

            self.assertEqual(1, len(stats.edge_rows))
            self.assertEqual("missing_table", stats.edge_rows[0]["fanout_status"])

    def test_cli_writes_report_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            synthetic_root = base / "synthetic"
            relbench_path = base / "schema_graph.json"
            output_dir = base / "analysis"
            self._write_synthetic_db(synthetic_root / "db_000001")
            self._write_relbench_stats(relbench_path)

            with contextlib.redirect_stdout(io.StringIO()):
                schema_stats_main(
                    [
                        "--relbench-stats-json",
                        str(relbench_path),
                        "--synthetic-root",
                        str(synthetic_root),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertTrue((output_dir / "summary.md").exists())
            self.assertTrue((output_dir / "per_db_metrics.csv").exists())
            self.assertTrue((output_dir / "edge_fanout.csv").exists())
            self.assertTrue((output_dir / "aggregate_comparison.csv").exists())
            self.assertTrue((output_dir / "report.json").exists())

    def _write_synthetic_db(self, db_dir: Path) -> None:
        (db_dir / "schema").mkdir(parents=True)
        (db_dir / "tables").mkdir(parents=True)
        schema = {
            "schema_id": db_dir.name,
            "nodes": {
                "A": {
                    "role": "entity",
                    "num_rows": 3,
                    "primary_key": "a_id",
                    "columns": [{"name": "a_id"}],
                },
                "B": {
                    "role": "event",
                    "num_rows": 5,
                    "primary_key": "b_id",
                    "columns": [{"name": "b_id"}, {"name": "a_id"}],
                },
                "C": {
                    "role": "summary",
                    "num_rows": 2,
                    "primary_key": "c_id",
                    "columns": [{"name": "c_id"}, {"name": "b_id"}],
                },
            },
            "foreign_keys": [
                {
                    "parent_table": "A",
                    "child_table": "B",
                    "parent_col": "a_id",
                    "child_col": "a_id",
                },
                {
                    "parent_table": "B",
                    "child_table": "C",
                    "parent_col": "b_id",
                    "child_col": "b_id",
                },
            ],
        }
        (db_dir / "schema" / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
        pd.DataFrame({"a_id": [1, 2, 3]}).to_parquet(db_dir / "tables" / "A.parquet", index=False)
        pd.DataFrame({"b_id": [10, 11, 12, 13, 14], "a_id": [1, 1, 2, 2, 2]}).to_parquet(
            db_dir / "tables" / "B.parquet",
            index=False,
        )
        pd.DataFrame({"c_id": [20, 21], "b_id": [10, 12]}).to_parquet(
            db_dir / "tables" / "C.parquet",
            index=False,
        )

    def _write_dbinfer_db(self, db_dir: Path) -> None:
        db_dir.mkdir(parents=True)
        (db_dir / "metadata.yaml").write_text(
            """
dataset_name: existing_demo
tables:
  - name: parent
    source: parent.parquet
    format: parquet
    time_column: null
    columns:
      - name: id
        dtype: primary_key
        capacity: 3
  - name: child
    source: child.parquet
    format: parquet
    time_column: null
    columns:
      - name: id
        dtype: primary_key
      - name: parent_id
        dtype: foreign_key
        link_to: parent.id
tasks: []
""".strip(),
            encoding="utf-8",
        )
        pd.DataFrame({"id": [1, 2, 3]}).to_parquet(db_dir / "parent.parquet", index=False)
        pd.DataFrame({"id": [10, 11, 12, 13], "parent_id": [1, 2, 2, 4]}).to_parquet(
            db_dir / "child.parquet",
            index=False,
        )

    def _write_relbench_stats(self, path: Path) -> None:
        payload = [
            {
                "dataset": "rel-demo",
                "nodes": [
                    {"table": "parent", "rows": 3, "pkey_col": "id"},
                    {"table": "child", "rows": 10, "pkey_col": "id"},
                ],
                "edges": [
                    {
                        "parent_table": "parent",
                        "child_table": "child",
                        "parent_pkey_col": "id",
                        "fkey_col": "parent_id",
                        "parent_rows": 3,
                        "child_rows": 10,
                        "child_nonnull_fkeys": 10,
                        "child_distinct_fkeys": 2,
                        "parent_distinct_keys": 3,
                        "missing_parent_keys": 1,
                        "fanout_parent_mean": 2.5,
                        "fanout_parent_p50": 2.0,
                        "fanout_parent_p95": 4.0,
                        "fanout_parent_max": 5.0,
                    }
                ],
            }
        ]
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
