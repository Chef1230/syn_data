import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from syn_data.src.rdb_prior.io.full_pipeline import parse_args, run_command
from syn_data.src.rdb_prior.io.pipeline_logging import (
    close_pipeline_logger,
    configure_pipeline_logger,
    resolve_pipeline_log_settings,
)


class PipelineLoggingTests(unittest.TestCase):
    def tearDown(self):
        close_pipeline_logger()

    def test_resolve_settings_precedence_and_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            settings = resolve_pipeline_log_settings(
                config={
                    "logging": {
                        "enabled": True,
                        "level": "WARNING",
                        "file": "logs/from-config.log",
                        "max_bytes": 1234,
                        "backup_count": 2,
                    }
                },
                project_root=project_root,
                log_file_override=Path("logs/from-cli.log"),
                log_level_override="DEBUG",
                environ={
                    "RDB_PRIOR_LOG_FILE": "logs/from-env.log",
                    "RDB_PRIOR_LOG_LEVEL": "ERROR",
                },
            )

            self.assertTrue(settings.enabled)
            self.assertEqual("DEBUG", settings.level_name)
            self.assertEqual((project_root / "logs/from-cli.log").resolve(), settings.path)
            self.assertEqual(1234, settings.max_bytes)
            self.assertEqual(2, settings.backup_count)

    def test_file_logger_writes_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            settings = resolve_pipeline_log_settings(
                config={"logging": {"file": "logs/pipeline.log"}},
                project_root=project_root,
                environ={},
            )
            logger = configure_pipeline_logger(settings)
            logger.info("中文日志")
            close_pipeline_logger()

            content = settings.path.read_text(encoding="utf-8")
            self.assertIn("INFO", content)
            self.assertIn("中文日志", content)

    def test_run_command_relays_stdout_and_stderr_to_one_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            settings = resolve_pipeline_log_settings(
                config={"logging": {"file": "pipeline.log"}},
                project_root=project_root,
                environ={},
            )
            configure_pipeline_logger(settings)

            terminal = io.StringIO()
            command = [
                sys.executable,
                "-u",
                "-c",
                "import sys; print('child stdout'); print('child stderr', file=sys.stderr)",
            ]
            with contextlib.redirect_stdout(terminal):
                run_command(command, cwd=project_root)
            close_pipeline_logger()

            terminal_output = terminal.getvalue()
            log_output = settings.path.read_text(encoding="utf-8")
            self.assertIn("child stdout", terminal_output)
            self.assertIn("child stderr", terminal_output)
            self.assertIn("child stdout", log_output)
            self.assertIn("child stderr", log_output)
            self.assertIn("[done]", log_output)

    def test_full_pipeline_accepts_log_cli_overrides(self):
        args = parse_args(
            [
                "--log-file",
                "outputs/logs/custom.log",
                "--log-level",
                "DEBUG",
                "--no-file-log",
                "--resume-dfs",
                "--skip-dbinfer-validation",
                "--dbinfer-root",
                "outputs/dbinfer_for_dfs/syn_v1",
            ]
        )
        self.assertEqual(Path("outputs/logs/custom.log"), args.log_file)
        self.assertEqual("DEBUG", args.log_level)
        self.assertTrue(args.no_file_log)
        self.assertTrue(args.resume_dfs)
        self.assertTrue(args.skip_dbinfer_validation)
        self.assertEqual(Path("outputs/dbinfer_for_dfs/syn_v1"), args.dbinfer_root)


if __name__ == "__main__":
    unittest.main()
