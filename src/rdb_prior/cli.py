"""Command line entry points for the synthetic RDB prior pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, TypeVar

try:
    import yaml
except ImportError:  # pragma: no cover - only used when PyYAML is absent.
    yaml = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for CLI progress.
    tqdm = None

from .generators.base import GenerationContext, json_ready
from .generators.database_generator import RelationalDatabaseGenerator
from .schema.fk_graph import graph_from_sampled_schema_like
from .schema.schema_sampler import SchemaSampler, SchemaSamplingConfig
from .task.label_generator import LabelGenerator
from .task.rdbpfn_exporter import RDBPFNExportConfig, export_database_root_to_h5
from .task.task_exporter import save_task_bundle
from .task.task_sampler import TaskSampler


T = TypeVar("T")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Synthetic relational DB prior pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate DBs and task bundles.")
    _add_common_args(generate_parser)
    generate_parser.add_argument("--count", type=int, default=None, help="Override generation.num_databases.")
    generate_parser.add_argument("--output-root", type=Path, default=None, help="Override paths.database_output_root.")
    generate_parser.add_argument("--skip-task", action="store_true", help="Do not create task bundles.")

    export_parser = subparsers.add_parser("export-rdbpfn", help="Export task bundles to an RDBPFN HDF5 dump.")
    _add_common_args(export_parser)
    export_parser.add_argument("--database-root", type=Path, default=None, help="Override paths.database_output_root.")
    export_parser.add_argument("--output", type=Path, default=None, help="Override rdbpfn_export.output_path.")

    full_parser = subparsers.add_parser("full-pipeline", help="Generate DBs/tasks and optionally export RDBPFN HDF5.")
    _add_common_args(full_parser)
    full_parser.add_argument("--count", type=int, default=None, help="Override generation.num_databases.")
    full_parser.add_argument("--output-root", type=Path, default=None, help="Override paths.database_output_root.")
    full_parser.add_argument("--skip-rdbpfn-export", action="store_true", help="Skip RDBPFN HDF5 export.")

    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    project_root = _project_root_from_config(config_path)

    if args.command == "generate":
        report = generate_databases_from_config(
            config=config,
            project_root=project_root,
            count_override=args.count,
            output_root_override=args.output_root,
            task_enabled_override=False if args.skip_task else None,
        )
    elif args.command == "export-rdbpfn":
        report = export_rdbpfn_from_config(
            config=config,
            project_root=project_root,
            database_root_override=args.database_root,
            output_override=args.output,
        )
    elif args.command == "full-pipeline":
        report = run_full_pipeline_from_config(
            config=config,
            project_root=project_root,
            count_override=args.count,
            output_root_override=args.output_root,
            skip_rdbpfn_export=args.skip_rdbpfn_export,
        )
    else:  # pragma: no cover - argparse enforces valid commands.
        raise ValueError(f"Unknown command: {args.command!r}")

    print(json.dumps(json_ready(report), indent=2, ensure_ascii=False))


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        if yaml is None:
            raise ImportError("PyYAML is required to read YAML config files. Install pyyaml or use JSON.")
        data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config must be a mapping, got {type(data).__name__}.")
    return dict(data)


def generate_databases_from_config(
    config: Mapping[str, Any],
    project_root: Path,
    count_override: Optional[int] = None,
    output_root_override: Optional[Path] = None,
    task_enabled_override: Optional[bool] = None,
) -> Dict[str, Any]:
    seed = int(config.get("seed", 42))
    generation_cfg = dict(config.get("generation", {}))
    task_cfg = dict(config.get("task", {}))
    paths_cfg = dict(config.get("paths", {}))

    num_databases = int(count_override if count_override is not None else generation_cfg.get("num_databases", 1))
    if num_databases <= 0:
        raise ValueError("generation.num_databases must be positive.")
    output_root = _resolve_project_path(
        output_root_override if output_root_override is not None else paths_cfg.get("database_output_root", "outputs/databases/demo_run"),
        project_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    schema_config = SchemaSamplingConfig.from_dict({"schema": dict(config.get("schema", {}))})
    schema_config.seed = seed
    sampler = SchemaSampler(config=schema_config, seed=seed)

    generated = []
    failed_tasks = []
    for index in _progress(range(num_databases), desc="Generating synthetic DBs", unit="db"):
        schema_id = f"{generation_cfg.get('schema_id_prefix', 'db')}_{index:06d}"
        schema = sampler.sample_schema(schema_id=schema_id)
        context = GenerationContext(
            seed=seed + index,
            start_time=generation_cfg.get("start_time", "2020-01-01"),
            end_time=generation_cfg.get("end_time", "2022-01-01"),
            cutoff_time=generation_cfg.get("cutoff_time"),
            future_window_days=int(generation_cfg.get("future_window_days", 30)),
            history_window_days=int(generation_cfg.get("history_window_days", 365)),
            enable_leakage_guard=bool(generation_cfg.get("enable_leakage_guard", True)),
            row_scale=float(generation_cfg.get("row_scale", 1.0)),
            max_rows_per_table=generation_cfg.get("max_rows_per_table"),
        )
        generator = RelationalDatabaseGenerator(seed=seed + index)
        result = generator.generate_database(schema=schema, context=context)

        db_dir = output_root / schema_id
        saved_paths = generator.save_database(db_dir, result["tables"], result["metadata"])
        save_schema_artifacts(schema, db_dir)

        task_enabled = bool(task_cfg.get("enabled", True))
        if task_enabled_override is not None:
            task_enabled = bool(task_enabled_override)
        task_path = None
        if task_enabled:
            try:
                task_path = generate_task_bundle(
                    schema=schema,
                    tables=result["tables"],
                    metadata=result["metadata"],
                    output_dir=db_dir,
                    task_cfg=task_cfg,
                    seed=seed + index,
                )
            except Exception as exc:
                if bool(task_cfg.get("fail_on_error", True)):
                    raise
                failed_tasks.append({"schema_id": schema_id, "reason": str(exc)})

        generated.append(
            {
                "schema_id": schema_id,
                "database_dir": str(db_dir),
                "metadata_path": saved_paths.get("metadata"),
                "task_dir": str(task_path) if task_path is not None else None,
            }
        )

    report = {
        "command": "generate",
        "seed": seed,
        "num_databases": num_databases,
        "output_root": str(output_root),
        "generated": generated,
        "failed_tasks": failed_tasks,
    }
    (output_root / "generation_index.json").write_text(
        json.dumps(json_ready(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def generate_task_bundle(
    schema: Any,
    tables: Mapping[str, Any],
    metadata: Mapping[str, Any],
    output_dir: Path,
    task_cfg: Mapping[str, Any],
    seed: int,
) -> Path:
    target_source_role = task_cfg.get("target_source_role")
    target_source_table = task_cfg.get("target_source_table")
    spec = TaskSampler(seed=seed).sample_task(
        schema=schema,
        tables=tables,
        metadata=metadata,
        target_source_role=target_source_role,
        target_source_table=target_source_table,
    )
    positive_rate = float(task_cfg.get("positive_rate", 0.35))
    bundle = LabelGenerator(seed=seed, positive_rate=positive_rate).build_task_bundle(
        spec=spec,
        schema=schema,
        tables=tables,
        metadata=metadata,
    )
    task_dir = output_dir / "task"
    save_task_bundle(task_dir, bundle)
    return task_dir


def export_rdbpfn_from_config(
    config: Mapping[str, Any],
    project_root: Path,
    database_root_override: Optional[Path] = None,
    output_override: Optional[Path] = None,
) -> Dict[str, Any]:
    paths_cfg = dict(config.get("paths", {}))
    export_cfg = dict(config.get("rdbpfn_export", {}))
    database_root = _resolve_project_path(
        database_root_override if database_root_override is not None else paths_cfg.get("database_output_root", "outputs/databases/demo_run"),
        project_root,
    )
    output_path = _resolve_project_path(
        output_override if output_override is not None else export_cfg.get("output_path", "outputs/tasks/rdbpfn_tasks.h5"),
        project_root,
    )
    h5_config = RDBPFNExportConfig(
        total_rows=int(export_cfg.get("total_rows", 256)),
        max_columns=int(export_cfg.get("max_columns", 128)),
        seed=int(config.get("seed", 42)),
        include_val_as_query=bool(export_cfg.get("include_val_as_query", True)),
        compression=export_cfg.get("compression", "lzf"),
    )
    report = export_database_root_to_h5(database_root=database_root, output_path=output_path, config=h5_config)
    report["command"] = "export-rdbpfn"
    return report


def run_full_pipeline_from_config(
    config: Mapping[str, Any],
    project_root: Path,
    count_override: Optional[int] = None,
    output_root_override: Optional[Path] = None,
    skip_rdbpfn_export: bool = False,
) -> Dict[str, Any]:
    generation_report = generate_databases_from_config(
        config=config,
        project_root=project_root,
        count_override=count_override,
        output_root_override=output_root_override,
    )
    export_cfg = dict(config.get("rdbpfn_export", {}))
    export_report = None
    if bool(export_cfg.get("enabled", False)) and not skip_rdbpfn_export:
        export_report = export_rdbpfn_from_config(
            config=config,
            project_root=project_root,
            database_root_override=Path(generation_report["output_root"]),
        )
    return {
        "command": "full-pipeline",
        "generation": generation_report,
        "rdbpfn_export": export_report,
    }


def save_schema_artifacts(schema: Any, output_dir: Path) -> None:
    schema_dir = output_dir / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)

    schema_obj = schema.to_dict() if hasattr(schema, "to_dict") else dict(schema)
    (schema_dir / "schema.json").write_text(
        json.dumps(json_ready(schema_obj), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    graph = graph_from_sampled_schema_like(schema)
    (schema_dir / "fk_graph.json").write_text(
        json.dumps(json_ready(graph.to_dict()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (schema_dir / "schema_graph.dot").write_text(_schema_to_dot(schema), encoding="utf-8")
    (schema_dir / "schema_graph.mmd").write_text(_schema_to_mermaid(schema), encoding="utf-8")


def _schema_to_dot(schema: Any) -> str:
    nodes = schema.nodes if hasattr(schema, "nodes") else schema["nodes"]
    fks = schema.foreign_keys if hasattr(schema, "foreign_keys") else schema.get("foreign_keys", [])
    lines = [
        "digraph schema {",
        "  rankdir=LR;",
        "  node [shape=box, style=rounded];",
        "  edge [fontsize=10];",
    ]
    for table_id, node in nodes.items():
        role = _read_attr(node, "role", "unknown")
        rank = _read_attr(node, "rank", "?")
        label = _dot_escape(f"{table_id}\\nrole={role}\\nrank={rank}")
        lines.append(f'  "{_dot_escape(table_id)}" [label="{label}"];')
    for fk in fks:
        parent = _read_attr(fk, "parent_table")
        child = _read_attr(fk, "child_table")
        child_col = _read_attr(fk, "child_col", "fk")
        lines.append(
            f'  "{_dot_escape(parent)}" -> "{_dot_escape(child)}" '
            f'[label="{_dot_escape(child_col)}"];'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _schema_to_mermaid(schema: Any) -> str:
    nodes = schema.nodes if hasattr(schema, "nodes") else schema["nodes"]
    fks = schema.foreign_keys if hasattr(schema, "foreign_keys") else schema.get("foreign_keys", [])
    lines = ["flowchart LR"]
    for table_id, node in nodes.items():
        role = _read_attr(node, "role", "unknown")
        rank = _read_attr(node, "rank", "?")
        lines.append(f'  {table_id}["{table_id}<br/>role={role}<br/>rank={rank}"]')
    for fk in fks:
        parent = _read_attr(fk, "parent_table")
        child = _read_attr(fk, "child_table")
        child_col = _read_attr(fk, "child_col", "fk")
        lines.append(f"  {parent} -->|{child_col}| {child}")
    return "\n".join(lines) + "\n"


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("syn_data/configs/default.yaml"),
        help="Pipeline YAML config path.",
    )


def _project_root_from_config(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return Path.cwd()


def _resolve_project_path(path: Any, project_root: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (project_root / value).resolve()


def _read_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _dot_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _progress(iterable: Iterable[T], desc: str, unit: str) -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, ascii=True, file=sys.stdout, dynamic_ncols=True)


if __name__ == "__main__":
    main()