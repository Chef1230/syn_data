import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from syn_data.src.rdb_prior.io.full_pipeline import (
    bounded_parallel_map,
    dbinfer_dataset_is_complete,
    load_resume_dbinfer_datasets,
    remove_incomplete_stage_output,
    run_dfs_stage_sequence,
)


def _write_complete_dataset(path: Path, dataset_name: str = "db_000000") -> None:
    (path / "data").mkdir(parents=True, exist_ok=True)
    (path / "tasks" / "task_0").mkdir(parents=True, exist_ok=True)
    (path / "data" / "table.bin").write_bytes(b"table")
    for split in ("train", "validation", "test"):
        (path / "tasks" / "task_0" / f"{split}.bin").write_bytes(split.encode("utf-8"))
    metadata = {
        "dataset_name": dataset_name,
        "tables": [{"name": "table", "source": "data/table.bin"}],
        "tasks": [
            {
                "name": "task_0",
                "source": "tasks/task_0/{split}.bin",
            }
        ],
    }
    (path / "metadata.yaml").write_text(json.dumps(metadata), encoding="utf-8")


class FullPipelineResumeTests(unittest.TestCase):
    def test_bounded_parallel_map_limits_concurrency(self):
        lock = threading.Lock()
        active = 0
        peak_active = 0

        def work(value):
            nonlocal active, peak_active
            with lock:
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return value * 2

        results = list(bounded_parallel_map(work, range(12), max_workers=3))

        self.assertEqual([value * 2 for value in range(12)], sorted(results))
        self.assertGreater(peak_active, 1)
        self.assertLessEqual(peak_active, 3)

    def test_dataset_completeness_checks_referenced_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            _write_complete_dataset(dataset_dir)

            self.assertEqual((True, "complete"), dbinfer_dataset_is_complete(dataset_dir))
            (dataset_dir / "tasks" / "task_0" / "test.bin").unlink()
            complete, reason = dbinfer_dataset_is_complete(dataset_dir)
            self.assertFalse(complete)
            self.assertIn("test source", reason)

    def test_resume_uses_supplied_dbinfer_root_and_export_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            dbinfer_root = Path(tmp) / "kept_dbinfer_root"
            dataset_dir = dbinfer_root / "db_000000"
            _write_complete_dataset(dataset_dir)
            report = {
                "num_exported": 1,
                "num_skipped": 0,
                "items": [
                    {
                        "database_dir": "/old/database/db_000000",
                        "output_dir": "/old/dbinfer/root/db_000000",
                        "status": "exported",
                        "reason": None,
                    }
                ],
            }
            (dbinfer_root / "export_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )

            with mock.patch(
                "syn_data.src.rdb_prior.io.full_pipeline._progress",
                side_effect=lambda iterable, _desc, _unit: iterable,
            ) as progress:
                datasets = load_resume_dbinfer_datasets(dbinfer_root)

            self.assertEqual([dataset_dir], datasets)
            progress.assert_called_once()
            _, description, unit = progress.call_args.args
            self.assertEqual("Validating DBInfer datasets", description)
            self.assertEqual("db", unit)

    def test_resume_can_skip_per_dataset_dbinfer_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            dbinfer_root = Path(tmp) / "kept_dbinfer_root"
            dbinfer_root.mkdir()
            report = {
                "num_exported": 1,
                "items": [
                    {
                        "output_dir": "/old/dbinfer/root/db_000000",
                        "status": "exported",
                    }
                ],
            }
            (dbinfer_root / "export_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )

            with mock.patch(
                "syn_data.src.rdb_prior.io.full_pipeline.dbinfer_dataset_is_complete"
            ) as completeness, mock.patch(
                "syn_data.src.rdb_prior.io.full_pipeline._progress"
            ) as progress:
                datasets = load_resume_dbinfer_datasets(
                    dbinfer_root,
                    validate_datasets=False,
                )

            self.assertEqual([dbinfer_root / "db_000000"], datasets)
            completeness.assert_not_called()
            progress.assert_not_called()

    def test_resume_skips_complete_stage_and_rebuilds_incomplete_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dbinfer_dir = root / "dbinfer" / "db_000000"
            workspace = root / "dfs_workspace"
            pre_dir = workspace / "dfs_1" / "pre" / "db_000000-pre-dfs"
            post_dir = workspace / "dfs_1" / "post" / "db_000000-post-dfs"
            processed_dir = workspace / "dfs_1" / "processed" / "db_000000-dfs-1"
            _write_complete_dataset(dbinfer_dir)
            _write_complete_dataset(pre_dir)
            post_dir.mkdir(parents=True)
            (post_dir / "stale.tmp").write_text("partial", encoding="utf-8")

            stages = [
                ("pre-dfs transform", dbinfer_dir, "transform", pre_dir, "pre.yaml"),
                ("dfs", pre_dir, "dfs", post_dir, "dfs.yaml"),
                ("post-dfs transform", post_dir, "transform", processed_dir, "post.yaml"),
            ]

            def fake_preprocess(_tool_root, _input, _name, output, _config, **_kwargs):
                _write_complete_dataset(Path(output))

            with mock.patch(
                "syn_data.src.rdb_prior.io.full_pipeline.run_tab2graph_preprocess",
                side_effect=fake_preprocess,
            ) as preprocess:
                report = run_dfs_stage_sequence(
                    stages=stages,
                    data_preprocessing_dir=root / "RDBPFN" / "data_preprocessing",
                    dfs_workspace_root=workspace,
                    depth=1,
                    dataset_name="db_000000",
                    resume=True,
                )

            self.assertEqual({"stages_run": 2, "stages_skipped": 1}, report)
            self.assertEqual(2, preprocess.call_count)
            self.assertFalse((post_dir / "stale.tmp").exists())
            self.assertTrue(dbinfer_dataset_is_complete(processed_dir)[0])

    def test_partial_output_cleanup_refuses_outside_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            outside = root / "outside"
            workspace.mkdir()
            outside.mkdir()
            with self.assertRaisesRegex(ValueError, "outside workspace"):
                remove_incomplete_stage_output(outside, allowed_root=workspace)
            self.assertTrue(outside.exists())

    def test_later_checkpoint_skips_missing_upstream_intermediate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dbinfer_dir = root / "dbinfer" / "db_000000"
            workspace = root / "dfs_workspace"
            pre_dir = workspace / "dfs_1" / "pre" / "db_000000-pre-dfs"
            post_dir = workspace / "dfs_1" / "post" / "db_000000-post-dfs"
            processed_dir = workspace / "dfs_1" / "processed" / "db_000000-dfs-1"
            _write_complete_dataset(dbinfer_dir)
            _write_complete_dataset(post_dir)
            stages = [
                ("pre-dfs transform", dbinfer_dir, "transform", pre_dir, "pre.yaml"),
                ("dfs", pre_dir, "dfs", post_dir, "dfs.yaml"),
                ("post-dfs transform", post_dir, "transform", processed_dir, "post.yaml"),
            ]

            def fake_preprocess(_tool_root, _input, _name, output, _config, **_kwargs):
                _write_complete_dataset(Path(output))

            with mock.patch(
                "syn_data.src.rdb_prior.io.full_pipeline.run_tab2graph_preprocess",
                side_effect=fake_preprocess,
            ) as preprocess:
                report = run_dfs_stage_sequence(
                    stages=stages,
                    data_preprocessing_dir=root / "RDBPFN" / "data_preprocessing",
                    dfs_workspace_root=workspace,
                    depth=1,
                    dataset_name="db_000000",
                    resume=True,
                )

            self.assertEqual({"stages_run": 1, "stages_skipped": 2}, report)
            self.assertEqual(1, preprocess.call_count)
            self.assertFalse(pre_dir.exists())
            self.assertTrue(dbinfer_dataset_is_complete(processed_dir)[0])


if __name__ == "__main__":
    unittest.main()
