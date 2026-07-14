"""Full syn_data pipeline orchestration, including optional DFS H5 exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence, TypeVar

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None

from syn_data.src.rdb_prior.cli import load_config
from syn_data.src.rdb_prior.generators.base import json_ready

T = TypeVar("T")


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
    parser.add_argument("--dfs-h5-output-dir", type=Path, default=None, help="Override dfs_export.h5_output_dir.")
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

    database_root = resolve_project_path(
        args.output_root if args.output_root is not None else config.get("paths", {}).get("database_output_root", "outputs/databases/demo_run"),
        project_root,
    )

    if args.skip_generation:
        if not database_root.exists():
            raise FileNotFoundError(f"Cannot skip generation because database root does not exist: {database_root}")
        generation_report = {
            "command": None,
            "skipped": True,
            "output_root": str(database_root),
        }
    else:
        generation_report = run_generation(args=args, config_path=config_path, cwd=workspace_root)
    raw_export_report = None
    if bool(config.get("rdbpfn_export", {}).get("enabled", False)) and not args.skip_rdbpfn_export:
        raw_export_report = run_raw_rdbpfn_export(config_path=config_path, database_root=database_root, cwd=workspace_root)

    dfs_export_report = None
    dfs_cfg = dict(config.get("dfs_export", {}))
    if bool(dfs_cfg.get("enabled", False)) and not args.skip_dfs_export:
        depths = args.dfs_depth if args.dfs_depth is not None else dfs_cfg.get("depths", [1, 2])
        dfs_export_report = run_dfs_exports(
            config=config,
            project_root=project_root,
            workspace_root=workspace_root,
            database_root=database_root,
            depths=[int(depth) for depth in depths],
            rdb_pfn_root_override=args.rdb_pfn_root,
            h5_output_dir_override=args.dfs_h5_output_dir,
        )

    report = {
        "command": "full-pipeline",
        "generation": generation_report,
        "rdbpfn_export": raw_export_report,
        "dfs_export": dfs_export_report,
    }
    print(json.dumps(json_ready(report), indent=2, ensure_ascii=False))


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
    h5_output_dir_override: Path | None = None,
) -> dict[str, Any]:
    dfs_cfg = dict(config.get("dfs_export", {}))
    dbinfer_root = resolve_project_path(dfs_cfg.get("dbinfer_output_root", "outputs/dbinfer_for_dfs"), project_root)
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
    if not dataset_dirs:
        raise RuntimeError(f"No DBInfer datasets were exported under {dbinfer_root}")

    depth_reports = []
    for depth in _progress(depths, "DFS export depths", "depth"):
        processed_root = dfs_workspace_root / f"dfs_{depth}" / "processed"
        processed_root.mkdir(parents=True, exist_ok=True)
        pre_root = dfs_workspace_root / f"dfs_{depth}" / "pre"
        post_root = dfs_workspace_root / f"dfs_{depth}" / "post"
        for dataset_dir in _progress(dataset_dirs, f"DFS-{depth} datasets", "db"):
            dataset_name = dataset_dir.name
            pre_dir = pre_root / f"{dataset_name}-pre-dfs"
            post_dir = post_root / f"{dataset_name}-post-dfs"
            processed_dir = processed_root / f"{dataset_name}-dfs-{depth}"
            stages = [
                ("pre-dfs transform", dataset_dir, "transform", pre_dir, "configs/transform/pre-dfs.yaml"),
                ("dfs", pre_dir, "dfs", post_dir, f"configs/dfs/dfs-{depth}-sql.yaml"),
                ("post-dfs transform", post_dir, "transform", processed_dir, "configs/transform/post-dfs.yaml"),
            ]
            for stage_name, stage_input, preprocess_name, stage_output, config_file in _progress(
                stages,
                f"DFS-{depth} {dataset_name} stages",
                "stage",
            ):
                _status(f"DFS-{depth} {dataset_name}: {stage_name}")
                run_tab2graph_preprocess(
                    data_preprocessing_dir,
                    stage_input,
                    preprocess_name,
                    stage_output,
                    config_file,
                )

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
            }
        )

    return {
        "dbinfer_root": str(dbinfer_root),
        "workspace_root": str(dfs_workspace_root),
        "depths": depth_reports,
    }


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
    printable = " ".join(str(part) for part in cmd)
    _status(f"[run] {printable}")
    subprocess.run([str(part) for part in cmd], cwd=str(cwd), check=True)


def _progress(iterable: Iterable[T], desc: str, unit: str) -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, ascii=True, file=sys.stdout, dynamic_ncols=True)


def _status(message: str) -> None:
    if tqdm is None:
        print(message, flush=True)
    else:
        tqdm.write(message)
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
