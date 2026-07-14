# syn_data

Role/motif/time-aware synthetic relational database generation.

## One-click scripts

All scripts read `configs/default.yaml` by default. You can override the config with
`RDB_PRIOR_CONFIG=/path/to/config.yaml` or by passing a YAML file as the first
script argument.

From the repository root on Linux:

```bash
# Install runtime dependencies for generation and RDBPFN export.
bash syn_data/scripts/setup_env.sh

# Generate synthetic DBs, schema artifacts, and task bundles.
bash syn_data/scripts/run_generate.sh

# Run generation, raw RDBPFN HDF5 export, and configured DFS HDF5 exports in one command.
bash syn_data/scripts/run_full_pipeline.sh

# Run compile and tests.
bash syn_data/scripts/run_check.sh
```

Useful overrides:

```bash
bash syn_data/scripts/run_generate.sh --count 5 --output-root outputs/databases/debug_5
bash syn_data/scripts/run_full_pipeline.sh configs/default.yaml --skip-rdbpfn-export --skip-dfs-export
bash syn_data/scripts/run_full_pipeline.sh configs/default.yaml --log-level DEBUG --log-file outputs/logs/debug_pipeline.log

# The same overrides can be supplied through the shell environment.
RDB_PRIOR_LOG_LEVEL=DEBUG RDB_PRIOR_LOG_FILE=outputs/logs/debug_pipeline.log \
  bash syn_data/scripts/run_full_pipeline.sh configs/default.yaml

# Resume only the DFS stage, reusing the exact existing DBInfer root.
bash syn_data/scripts/run_full_pipeline.sh configs/v1.yaml \
  --skip-generation --skip-rdbpfn-export \
  --resume-dfs --skip-dbinfer-validation --dfs-jobs 32 \
  --dbinfer-root outputs/dbinfer_for_dfs/syn_v1
```

## Main configuration

Edit `configs/default.yaml` for the normal pipeline:

- `generation.num_databases`: number of generated DBs.
- `generation.num_workers`: number of worker processes for per-DB parallel generation.
- `logging.file`: rotating UTF-8 log containing stage and child-process output.
- `logging.level`: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`.
- `logging.max_bytes` / `logging.backup_count`: log rotation size and retained backups.
- `paths.database_output_root`: generated DB output directory.
- `schema.*`: schema sampling constraints.
- `task.target_source_role`: `null`, `outcome`, `summary`, `entity`, or `event`.
- `rdbpfn_export.output_path`: target raw `.h5` file consumed by RDBPFN training.
- `dfs_export.enabled`: whether `run_full_pipeline.sh` also exports DFS-derived H5 files.
- `dfs_export.depths`: DFS depths to export; current training config keeps `[1, 2]`.
- `--dbinfer-root`: overrides `dfs_export.dbinfer_output_root` for both normal and resumed DFS runs.
- `--resume-dfs`: enables DFS for the run, reuses that DBInfer root, skips complete pre/DFS/post outputs, and rebuilds only incomplete stages.
- `--skip-dbinfer-validation`: with `--resume-dfs`, trusts `export_report.json` and skips the slow per-dataset metadata/table/task file validation. Corrupt datasets will instead fail when DFS reaches them.
- `--dfs-jobs`: maximum databases processed concurrently inside each DFS depth. The default is `1`; use a bounded value such as `32` on a 256-core host.

Per-database DFS failures are logged and skipped so the remaining databases can
continue. At the end of the run, the full failure list (dataset, depth, stage,
error, and failed command when available) is written to
`dfs_export.h5_output_dir/dfs_failed_datasets.json` and included in the final
pipeline report. If every database fails at one depth, H5 merging for that depth
is skipped.

`--resume-dfs` requires the original `export_report.json` under `--dbinfer-root`.
The same config must retain the original `dfs_export.workspace_root`, because
that directory contains the per-depth pre/DFS/post checkpoints.

Generated DB layout:

```text
outputs/databases/sample_50/db_000000/
  tables/*.parquet
  schema/schema.json
  schema/schema_graph.dot
  schema/schema_graph.mmd
  generation_metadata.json
  task/task.json
  task/labels.parquet
  task/feature_manifest.json
```
