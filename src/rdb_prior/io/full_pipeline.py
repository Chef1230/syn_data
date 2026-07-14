"""Full syn_data pipeline orchestration, including optional DFS H5 exports."""

from __future__ import annotations

import argparse
import codecs
import json
import logging
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence, TypeVar

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None

from syn_data.src.rdb_prior.cli import load_config
from syn_data.src.rdb_prior.generators.base import json_ready
from syn_data.src.rdb_prior.io.pipeline_logging import (
    close_pipeline_logger,
    configure_pipeline_logger,
    get_pipeline_logger,
    resolve_pipeline_log_settings,
)

T = TypeVar("T")
_LOGGER = get_pipeline_logger()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run syn_data generation and exports.")
    parser.add_argument("--config", type=Path, default=Path("syn_data/configs/default.yaml"))
    parser.add_argument("--count", type=int, default=None, help="Override generation.num_databases.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override paths.database_output_root.")
    parser.add_argument("--jobs", type=int, default=None, help="Override generation.num_workers.")
    parser.add_argument("--skip-task", action="store_true", help="Do not create task bundles.")
    parser.add_argument("--skip-generation", action="store_true", help="Use an existing database output root instead of generating DBs.")
    parser.add_argument("--skip-rdbpfn-export", action="store_true", help="Skip raw RDBPFN H5 export.")
    parser.add_argument("--skip-dfs-export", action="store_true", help="Skip DFS-derived H5 exports.")
    parser.add_argument("--rdb-pfn-root", type=Path, default=None, help="Override dfs_export.rdb_pfn_root.")
    parser.add_argument(
        "--dbinfer-root",
        type=Path,
        default=None,
        help="Override dfs_export.dbinfer_output_root; --resume-dfs reuses this path without re-exporting.",
    )
    parser.add_argument(
        "--resume-dfs",
        action="store_true",
        help="Resume DFS from complete DBInfer/stage outputs and rerun only incomplete work.",
    )
    parser.add_argument("--dfs-h5-output-dir", type=Path, default=None, help="Override dfs_export.h5_output_dir.")
    parser.add_argument("--log-file", type=Path, default=None, help="Override logging.file.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=None,
        help="Override logging.level.",
    )
    parser.add_argument("--no-file-log", action="store_true", help="Disable pipeline file logging for this run.")
    parser.add_argument(
        "--dfs-depth",
        type=int,
        action="append",
        default=None,
        help="Override dfs_export.depths. Can be repeated, e.g. --dfs-depth 1 --dfs-depth 2.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    workspace_root = project_root.parent

    log_settings = resolve_pipeline_log_settings(
        config=config,
        project_root=project_root,
        log_file_override=args.log_file,
        log_level_override=args.log_level,
        disabled_override=args.no_file_log,
    )
    configure_pipeline_logger(log_settings)
    started_at = time.perf_counter()
    _status("=" * 72)
    _status("Starting syn_data full pipeline")
    _status(f"Config: {config_path}")
    if log_settings.path is not None:
        _status(f"Log file: {log_settings.path}")
    else:
        _status("File logging: disabled")

    try:
        report = execute_pipeline(
            args=args,
            config=config,
            config_path=config_path,
            project_root=project_root,
            workspace_root=workspace_root,
        )
        elapsed = time.perf_counter() - started_at
        report["logging"] = log_settings.to_dict()
        report["elapsed_seconds"] = round(elapsed, 3)
        report_json = json.dumps(json_ready(report), indent=2, ensure_ascii=False)
        print(report_json, flush=True)
        _LOGGER.info("Pipeline report:\n%s", report_json)
        _status(f"Pipeline completed successfully in {elapsed:.2f}s")
    except BaseException:
        elapsed = time.perf_counter() - started_at
        _LOGGER.exception("Pipeline failed after %.2fs", elapsed)
        if log_settings.path is not None:
            _write_console(
                f"Pipeline failed after {elapsed:.2f}s; see {log_settings.path} for details."
            )
        else:
            _write_console(f"Pipeline failed after {elapsed:.2f}s.")
        raise
    finally:
        close_pipeline_logger()


def execute_pipeline(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_path: Path,
    project_root: Path,
    workspace_root: Path,
) -> dict[str, Any]:
    """Run all configured stages and return their combined report."""

    database_root = resolve_project_path(
        args.output_root
        if args.output_root is not None
        else config.get("paths", {}).get("database_output_root", "outputs/databases/demo_run"),
        project_root,
    )
    _status(f"Database root: {database_root}")

    if args.skip_generation:
        if not database_root.exists():
            raise FileNotFoundError(f"Cannot skip generation because database root does not exist: {database_root}")
        _status("Generation stage skipped")
        generation_report = {
            "command": None,
            "skipped": True,
            "output_root": str(database_root),
        }
    else:
        _status("Stage 1: generating relational databases and tasks")
        generation_report = run_generation(args=args, config_path=config_path, cwd=workspace_root)

    raw_export_report = None
    if bool(config.get("rdbpfn_export", {}).get("enabled", False)) and not args.skip_rdbpfn_export:
        _status("Stage 2: exporting raw RDBPFN H5")
        raw_export_report = run_raw_rdbpfn_export(
            config_path=config_path,
            database_root=database_root,
            cwd=workspace_root,
        )
    else:
        _status("Raw RDBPFN export stage skipped")

    dfs_export_report = None
    dfs_cfg = dict(config.get("dfs_export", {}))
    if (bool(dfs_cfg.get("enabled", False)) or args.resume_dfs) and not args.skip_dfs_export:
        depths = [
            int(depth)
            for depth in (
                args.dfs_depth if args.dfs_depth is not None else dfs_cfg.get("depths", [1, 2])
            )
        ]
        _status(f"Stage 3: exporting DFS H5 at depths {depths}")
        dfs_export_report = run_dfs_exports(
            config=config,
            project_root=project_root,
            workspace_root=workspace_root,
            database_root=database_root,
            depths=depths,
            rdb_pfn_root_override=args.rdb_pfn_root,
            dbinfer_root_override=args.dbinfer_root,
            h5_output_dir_override=args.dfs_h5_output_dir,
            resume=args.resume_dfs,
        )
    else:
        _status("DFS export stage skipped")

    return {
        "command": "full-pipeline",
        "generation": generation_report,
        "rdbpfn_export": raw_export_report,
        "dfs_export": dfs_export_report,
    }


def run_generation(args: argparse.Namespace, config_path: Path, cwd: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "syn_data.src.rdb_prior.io.generate_parallel",
        "--config",
        str(config_path),
    ]
    if args.count is not None:
        cmd += ["--count", str(args.count)]
    if args.output_root is not None:
        cmd += ["--output-root", str(args.output_root)]
    if args.jobs is not None:
        cmd += ["--jobs", str(args.jobs)]
    if args.skip_task:
        cmd.append("--skip-task")
    run_command(cmd, cwd=cwd)
    return {"command": cmd}


def run_raw_rdbpfn_export(config_path: Path, database_root: Path, cwd: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "syn_data.src.rdb_prior.cli",
        "export-rdbpfn",
        "--config",
        str(config_path),
        "--database-root",
        str(database_root),
    ]
    run_command(cmd, cwd=cwd)
    return {"command": cmd}


def run_dfs_exports(
    config: Mapping[str, Any],
    project_root: Path,
    workspace_root: Path,
    database_root: Path,
    depths: list[int],
    rdb_pfn_root_override: Path | None = None,
    dbinfer_root_override: Path | None = None,
    h5_output_dir_override: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    dfs_cfg = dict(config.get("dfs_export", {}))
    dbinfer_root = resolve_project_path(
        dbinfer_root_override
        if dbinfer_root_override is not None
        else dfs_cfg.get("dbinfer_output_root", "outputs/dbinfer_for_dfs"),
        project_root,
    )
    dfs_workspace_root = resolve_project_path(dfs_cfg.get("workspace_root", "outputs/dfs_workspace"), project_root)
    h5_output_dir = resolve_project_path(
        h5_output_dir_override
        if h5_output_dir_override is not None
        else dfs_cfg.get("h5_output_dir", "../RDB_PFN/model_pretrain/pretrain_datasets"),
        project_root,
    )
    h5_output_dir.mkdir(parents=True, exist_ok=True)

    rdb_pfn_root = resolve_project_path(
        rdb_pfn_root_override if rdb_pfn_root_override is not None else dfs_cfg.get("rdb_pfn_root", "../RDB_PFN"),
        project_root,
    )
    data_preprocessing_dir = rdb_pfn_root / "data_preprocessing"
    if not data_preprocessing_dir.exists():
        raise FileNotFoundError(f"RDB_PFN data_preprocessing directory not found: {data_preprocessing_dir}")

    if resume:
        _status(f"[resume] Reusing DBInfer root without export: {dbinfer_root}")
        dataset_dirs = load_resume_dbinfer_datasets(dbinfer_root)
        dbinfer_mode = "reused"
    else:
        export_cmd = [
            sys.executable,
            "-u",
            "-m",
            "syn_data.src.rdb_prior.io.export_dbinfer_for_dfs",
            "--database-root",
            str(database_root),
            "--output-root",
            str(dbinfer_root),
            "--overwrite",
        ]
        run_command(export_cmd, cwd=workspace_root)
        dataset_dirs = discover_dbinfer_datasets(dbinfer_root)
        dbinfer_mode = "exported"
    if not dataset_dirs:
        raise RuntimeError(f"No DBInfer datasets were exported under {dbinfer_root}")
    _status(f"DBInfer datasets ready: {len(dataset_dirs)} from {dbinfer_root}")

    depth_reports = []
    for depth in _progress(depths, "DFS export depths", "depth"):
        processed_root = dfs_workspace_root / f"dfs_{depth}" / "processed"
        processed_root.mkdir(parents=True, exist_ok=True)
        pre_root = dfs_workspace_root / f"dfs_{depth}" / "pre"
        post_root = dfs_workspace_root / f"dfs_{depth}" / "post"
        stage_run_count = 0
        stage_skip_count = 0
        completed_dataset_skip_count = 0
        for dataset_dir in _progress(dataset_dirs, f"DFS-{depth} datasets", "db"):
            dataset_name = dataset_dir.name
            pre_dir = pre_root / f"{dataset_name}-pre-dfs"
            post_dir = post_root / f"{dataset_name}-post-dfs"
            processed_dir = processed_root / f"{dataset_name}-dfs-{depth}"
            if resume and dbinfer_dataset_is_complete(processed_dir)[0]:
                completed_dataset_skip_count += 1
                stage_skip_count += 3
                _status(f"[resume] DFS-{depth} {dataset_name}: processed output complete; skipping dataset")
                continue
            stages = [
                ("pre-dfs transform", dataset_dir, "transform", pre_dir, "configs/transform/pre-dfs.yaml"),
                ("dfs", pre_dir, "dfs", post_dir, f"configs/dfs/dfs-{depth}-sql.yaml"),
                ("post-dfs transform", post_dir, "transform", processed_dir, "configs/transform/post-dfs.yaml"),
            ]
            stage_report = run_dfs_stage_sequence(
                stages=stages,
                data_preprocessing_dir=data_preprocessing_dir,
                dfs_workspace_root=dfs_workspace_root,
                depth=depth,
                dataset_name=dataset_name,
                resume=resume,
            )
            stage_run_count += stage_report["stages_run"]
            stage_skip_count += stage_report["stages_skipped"]

        unsampled_h5 = h5_output_dir / str(dfs_cfg.get("unsampled_h5_name_template", "syn_dfs_{depth}_unsampled.h5")).format(depth=depth)
        final_h5 = h5_output_dir / str(dfs_cfg.get("h5_name_template", "syn_dfs_{depth}.h5")).format(depth=depth)
        merge_cmd = [
            sys.executable,
            "-u",
            "merge_dbinfer_to_h5.py",
            "--dataset-root",
            str(processed_root),
            "--output",
            str(unsampled_h5),
            "--total-rows",
            str(int(dfs_cfg.get("total_rows", config.get("rdbpfn_export", {}).get("total_rows", 256)))),
            "--max-columns",
            str(int(dfs_cfg.get("max_columns", config.get("rdbpfn_export", {}).get("max_columns", 128)))),
            "--min-train-ratio",
            str(float(dfs_cfg.get("min_train_ratio", 0.5))),
            "--max-train-ratio",
            str(float(dfs_cfg.get("max_train_ratio", 0.9))),
        ]
        run_command(merge_cmd, cwd=data_preprocessing_dir)

        filter_cfg = dict(dfs_cfg.get("filter", {}))
        if bool(filter_cfg.get("enabled", True)):
            filter_cmd = [
                sys.executable,
                "-u",
                "filter_h5_sampling_columns.py",
                str(unsampled_h5),
                str(final_h5),
                "--sampled-columns",
                str(int(filter_cfg.get("sampled_columns", 30))),
                "--max-expected-columns",
                str(int(filter_cfg.get("max_expected_columns", 10))),
                "--ratio",
                str(float(filter_cfg.get("ratio", 0.9))),
                "--safety-factor",
                str(float(filter_cfg.get("safety_factor", 1.0))),
            ]
            run_command(filter_cmd, cwd=data_preprocessing_dir)
        else:
            final_h5 = unsampled_h5

        depth_reports.append(
            {
                "depth": depth,
                "processed_root": str(processed_root),
                "unsampled_h5": str(unsampled_h5),
                "output_h5": str(final_h5),
                "stages_run": stage_run_count,
                "stages_skipped": stage_skip_count,
                "datasets_skipped_as_complete": completed_dataset_skip_count,
            }
        )

    return {
        "dbinfer_root": str(dbinfer_root),
        "dbinfer_mode": dbinfer_mode,
        "num_dbinfer_datasets": len(dataset_dirs),
        "workspace_root": str(dfs_workspace_root),
        "resume": resume,
        "depths": depth_reports,
    }


def run_dfs_stage_sequence(
    stages: Sequence[tuple[str, Path, str, Path, str]],
    data_preprocessing_dir: Path,
    dfs_workspace_root: Path,
    depth: int,
    dataset_name: str,
    resume: bool,
) -> dict[str, int]:
    """Run or resume one database's pre-DFS, DFS, and post-DFS stages."""

    stages_run = 0
    stages_skipped = 0
    resume_through_index = -1
    if resume:
        for index in range(len(stages) - 1, -1, -1):
            if dbinfer_dataset_is_complete(stages[index][3])[0]:
                resume_through_index = index
                break

    indexed_stages = list(enumerate(stages))
    for stage_index, stage in _progress(
        indexed_stages,
        f"DFS-{depth} {dataset_name} stages",
        "stage",
    ):
        stage_name, stage_input, preprocess_name, stage_output, config_file = stage
        if resume and stage_index <= resume_through_index:
            stages_skipped += 1
            _status(f"[resume] DFS-{depth} {dataset_name}: {stage_name} checkpoint available; skipping")
            continue
        if resume:
            complete, reason = dbinfer_dataset_is_complete(stage_output)
            if complete:
                stages_skipped += 1
                _status(f"[resume] DFS-{depth} {dataset_name}: {stage_name} complete; skipping")
                continue
            if stage_output.exists() or stage_output.is_symlink():
                _status(
                    f"[resume] DFS-{depth} {dataset_name}: {stage_name} incomplete ({reason}); rebuilding"
                )
                remove_incomplete_stage_output(stage_output, allowed_root=dfs_workspace_root)

        input_complete, input_reason = dbinfer_dataset_is_complete(stage_input)
        if not input_complete:
            raise RuntimeError(
                f"Cannot run DFS-{depth} {dataset_name} {stage_name}: "
                f"input {stage_input} is incomplete ({input_reason})."
            )

        _status(f"DFS-{depth} {dataset_name}: {stage_name}")
        run_tab2graph_preprocess(
            data_preprocessing_dir,
            stage_input,
            preprocess_name,
            stage_output,
            config_file,
        )
        output_complete, output_reason = dbinfer_dataset_is_complete(stage_output)
        if not output_complete:
            raise RuntimeError(
                f"DFS-{depth} {dataset_name} {stage_name} returned successfully but output "
                f"{stage_output} is incomplete ({output_reason})."
            )
        stages_run += 1
    return {"stages_run": stages_run, "stages_skipped": stages_skipped}


def load_resume_dbinfer_datasets(root: Path) -> list[Path]:
    """Load and validate the exact DBInfer export represented by ``root``."""

    root = root.resolve()
    if (root / "metadata.yaml").exists():
        complete, reason = dbinfer_dataset_is_complete(root)
        if not complete:
            raise RuntimeError(f"Cannot resume from incomplete DBInfer dataset {root}: {reason}")
        return [root]

    report_path = root / "export_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Cannot resume DFS: DBInfer export report not found: {report_path}. "
            "Pass the original --dbinfer-root, or run once without --resume-dfs."
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read DBInfer export report {report_path}: {exc}") from exc

    exported_items = [
        item
        for item in report.get("items", [])
        if isinstance(item, Mapping) and item.get("status") == "exported" and item.get("output_dir")
    ]
    expected_count = int(report.get("num_exported", len(exported_items)))
    if expected_count != len(exported_items):
        raise RuntimeError(
            f"Cannot resume DFS: {report_path} declares {expected_count} exported datasets "
            f"but contains {len(exported_items)} exported records."
        )

    dataset_dirs: list[Path] = []
    incomplete: list[str] = []
    seen_names: set[str] = set()
    for item in _progress(exported_items, "Validating DBInfer datasets", "db"):
        dataset_name = Path(str(item["output_dir"])).name
        if dataset_name in seen_names:
            incomplete.append(f"duplicate dataset name {dataset_name!r}")
            continue
        seen_names.add(dataset_name)
        dataset_dir = root / dataset_name
        complete, reason = dbinfer_dataset_is_complete(dataset_dir)
        if complete:
            dataset_dirs.append(dataset_dir)
        else:
            incomplete.append(f"{dataset_name}: {reason}")

    if incomplete:
        preview = "; ".join(incomplete[:5])
        suffix = f"; and {len(incomplete) - 5} more" if len(incomplete) > 5 else ""
        raise RuntimeError(
            f"Cannot resume DFS from {root}: {len(incomplete)} DBInfer datasets are incomplete: "
            f"{preview}{suffix}."
        )
    if not dataset_dirs:
        raise RuntimeError(f"Cannot resume DFS: no exported DBInfer datasets recorded in {report_path}.")
    return sorted(dataset_dirs)


def dbinfer_dataset_is_complete(path: Path) -> tuple[bool, str]:
    """Validate metadata and every table/task file referenced by a DBInfer dataset."""

    path = Path(path)
    metadata_path = path / "metadata.yaml"
    if not path.is_dir():
        return False, "directory does not exist"
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        return False, "metadata.yaml is missing or empty"
    try:
        metadata = load_config(metadata_path)
    except Exception as exc:
        return False, f"metadata.yaml cannot be parsed: {type(exc).__name__}: {exc}"

    tables = metadata.get("tables")
    tasks = metadata.get("tasks")
    if not isinstance(tables, list) or not tables:
        return False, "metadata contains no tables"
    if not isinstance(tasks, list) or not tasks:
        return False, "metadata contains no tasks"

    for table in tables:
        if not isinstance(table, Mapping) or not table.get("source"):
            return False, "a table is missing its source"
        source = _dataset_source_path(path, str(table["source"]))
        if not _nonempty_artifact_exists(source):
            return False, f"table source is missing or empty: {source}"

    for task in tasks:
        if not isinstance(task, Mapping) or not task.get("source"):
            return False, "a task is missing its source"
        source_template = str(task["source"])
        for split in ("train", "validation", "test"):
            try:
                source_value = source_template.format(split=split)
            except (KeyError, ValueError) as exc:
                return False, f"invalid task source template {source_template!r}: {exc}"
            source = _dataset_source_path(path, source_value)
            if not _nonempty_artifact_exists(source):
                return False, f"task {split} source is missing or empty: {source}"
    return True, "complete"


def remove_incomplete_stage_output(path: Path, allowed_root: Path) -> None:
    """Remove only a partial DFS output proven to be inside its workspace."""

    resolved_path = path.resolve()
    resolved_root = allowed_root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to remove DFS output outside workspace: {resolved_path}") from exc
    if not relative.parts:
        raise ValueError(f"Refusing to remove the DFS workspace root itself: {resolved_root}")
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _dataset_source_path(dataset_dir: Path, source: str) -> Path:
    source_path = Path(source)
    return source_path if source_path.is_absolute() else dataset_dir / source_path


def _nonempty_artifact_exists(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(path.iterdir())
    return False


def run_tab2graph_preprocess(
    data_preprocessing_dir: Path,
    input_path: Path,
    preprocess_name: str,
    output_path: Path,
    config_path: str,
) -> None:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "tab2graph.main",
        "preprocess",
        str(input_path),
        preprocess_name,
        str(output_path),
        "-c",
        config_path,
    ]
    run_command(cmd, cwd=data_preprocessing_dir)


def discover_dbinfer_datasets(root: Path) -> list[Path]:
    if (root / "metadata.yaml").exists():
        return [root]
    return [path for path in sorted(root.iterdir()) if path.is_dir() and (path / "metadata.yaml").exists()]


def run_command(cmd: Sequence[Any], cwd: Path) -> None:
    """Run a child command and tee its combined output to terminal and log."""

    command = [str(part) for part in cmd]
    printable = shlex.join(command)
    started_at = time.perf_counter()
    _status(f"[run] cwd={cwd} command={printable}")

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    log_buffer = ""
    try:
        if process.stdout is not None:
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if text:
                    _write_raw_console(text)
                    log_buffer = _log_child_lines(log_buffer, text)
        final_text = decoder.decode(b"", final=True)
        if final_text:
            _write_raw_console(final_text)
            log_buffer = _log_child_lines(log_buffer, final_text)
        if log_buffer:
            _LOGGER.info("[child] %s", _last_progress_value(log_buffer))
            if not log_buffer.endswith(("\n", "\r")):
                _write_raw_console("\n")
        return_code = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait()
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()

    elapsed = time.perf_counter() - started_at
    if return_code != 0:
        _status(
            f"[failed] exit_code={return_code} elapsed={elapsed:.2f}s command={printable}",
            level=logging.ERROR,
        )
        raise subprocess.CalledProcessError(return_code, command)
    _status(f"[done] elapsed={elapsed:.2f}s command={printable}")


def _log_child_lines(buffer: str, text: str) -> str:
    """Log completed child lines while collapsing carriage-return progress updates."""

    parts = (buffer + text).split("\n")
    for part in parts[:-1]:
        line = _last_progress_value(part).rstrip()
        if line:
            _LOGGER.info("[child] %s", line)
    return _last_progress_value(parts[-1])


def _last_progress_value(value: str) -> str:
    progress_values = value.split("\r")
    for item in reversed(progress_values):
        if item:
            return item
    return ""


def _progress(iterable: Iterable[T], desc: str, unit: str) -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, ascii=True, file=sys.stdout, dynamic_ncols=True)


def _status(message: str, level: int = logging.INFO) -> None:
    _LOGGER.log(level, message)
    _write_console(message)


def _write_console(message: str) -> None:
    if tqdm is None:
        print(message, flush=True)
    else:
        tqdm.write(message)
        sys.stdout.flush()


def _write_raw_console(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def project_root_from_config(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return Path.cwd()


def resolve_project_path(path: Any, project_root: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (project_root / value).resolve()


if __name__ == "__main__":
    main()
