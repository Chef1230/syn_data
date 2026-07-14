"""Generate syn_data databases in parallel, one worker task per DB.

Run from the syn_data directory, for example:

    python -m syn_data.src.rdb_prior.io.generate_parallel --config syn_data/configs/v1.yaml
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, TypeVar

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None

from syn_data.src.rdb_prior.cli import (
    generate_task_bundle,
    load_config,
    save_schema_artifacts,
)
from syn_data.src.rdb_prior.generators.base import GenerationContext, json_ready
from syn_data.src.rdb_prior.generators.database_generator import RelationalDatabaseGenerator
from syn_data.src.rdb_prior.schema.schema_sampler import SchemaSampler, SchemaSamplingConfig


T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate syn_data DBs in parallel.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Pipeline YAML config path.",
    )
    parser.add_argument("--count", type=int, default=None, help="Override generation.num_databases.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override paths.database_output_root. Relative paths resolve from syn_data/.",
    )
    parser.add_argument("--jobs", type=int, default=None, help="Override generation.num_workers.")
    parser.add_argument("--skip-task", action="store_true", help="Do not create task bundles.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    generation_cfg = dict(config.get("generation", {}))
    task_cfg = dict(config.get("task", {}))
    paths_cfg = dict(config.get("paths", {}))
    seed = int(config.get("seed", 42))
    num_databases = int(args.count if args.count is not None else generation_cfg.get("num_databases", 1))
    jobs = int(args.jobs if args.jobs is not None else generation_cfg.get("num_workers", 1))
    if jobs <= 0:
        raise ValueError("--jobs/generation.num_workers must be positive.")
    if num_databases <= 0:
        raise ValueError("--count/generation.num_databases must be positive.")

    output_root = resolve_project_path(
        args.output_root if args.output_root is not None else paths_cfg.get("database_output_root", "outputs/databases/demo_run"),
        project_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    schema_config = SchemaSamplingConfig.from_dict({"schema": dict(config.get("schema", {}))})
    schema_config.seed = seed
    sampler = SchemaSampler(config=schema_config, seed=seed)

    task_enabled = bool(task_cfg.get("enabled", True)) and not args.skip_task
    print(f"Preparing {num_databases} generation jobs...", flush=True)
    work_items = []
    for index in _progress(range(num_databases), "Preparing generation jobs", "db", total=num_databases):
        schema_id = f"{generation_cfg.get('schema_id_prefix', 'db')}_{index:06d}"
        schema = sampler.sample_schema(schema_id=schema_id)
        work_items.append(
            {
                "index": index,
                "schema_id": schema_id,
                "schema": schema,
                "seed": seed,
                "generation_cfg": generation_cfg,
                "task_cfg": task_cfg,
                "task_enabled": task_enabled,
                "output_root": output_root,
            }
        )

    if jobs == 1:
        results = [_generate_one_database(item) for item in _progress(work_items, "Generating synthetic DBs", "db")]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_generate_one_database, item) for item in work_items]
            for future in _progress(as_completed(futures), "Generating synthetic DBs", "db", total=len(futures)):
                results.append(future.result())

    results.sort(key=lambda item: item["index"])
    generated = [item["generated"] for item in results]
    failed_tasks = [failed for item in results for failed in item["failed_tasks"]]
    report = {
        "command": "generate-parallel",
        "seed": seed,
        "num_databases": num_databases,
        "jobs": jobs,
        "output_root": str(output_root),
        "generated": generated,
        "failed_tasks": failed_tasks,
    }
    (output_root / "generation_index.json").write_text(
        json.dumps(json_ready(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(report), indent=2, ensure_ascii=False))


def _generate_one_database(item: Mapping[str, Any]) -> Dict[str, Any]:
    index = int(item["index"])
    seed = int(item["seed"])
    schema_id = str(item["schema_id"])
    schema = item["schema"]
    generation_cfg = dict(item["generation_cfg"])
    task_cfg = dict(item["task_cfg"])
    output_root = Path(item["output_root"])

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

    failed_tasks = []
    task_path = None
    if bool(item["task_enabled"]):
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

    return {
        "index": index,
        "generated": {
            "schema_id": schema_id,
            "database_dir": str(db_dir),
            "metadata_path": saved_paths.get("metadata"),
            "task_dir": str(task_path) if task_path is not None else None,
        },
        "failed_tasks": failed_tasks,
    }


def project_root_from_config(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return Path.cwd()


def resolve_project_path(path: Any, project_root: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (project_root / value).resolve()


def _progress(iterable: Iterable[T], desc: str, unit: str, total: int | None = None) -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, total=total, ascii=True, file=sys.stdout, dynamic_ncols=True)


if __name__ == "__main__":
    main()
