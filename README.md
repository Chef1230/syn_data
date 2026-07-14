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
```

## Main configuration

Edit `configs/default.yaml` for the normal pipeline:

- `generation.num_databases`: number of generated DBs.
- `generation.num_workers`: number of worker processes for per-DB parallel generation.
- `paths.database_output_root`: generated DB output directory.
- `schema.*`: schema sampling constraints.
- `task.target_source_role`: `null`, `outcome`, `summary`, `entity`, or `event`.
- `rdbpfn_export.output_path`: target raw `.h5` file consumed by RDBPFN training.
- `dfs_export.enabled`: whether `run_full_pipeline.sh` also exports DFS-derived H5 files.
- `dfs_export.depths`: DFS depths to export; current training config keeps `[1, 2]`.

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
