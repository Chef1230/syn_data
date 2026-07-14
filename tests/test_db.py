import json
import os
import tempfile
import sys
import unittest
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from syn_data.src.rdb_prior.generators.base import GenerationContext, json_ready
from syn_data.src.rdb_prior.generators.database_generator import RelationalDatabaseGenerator
from syn_data.src.rdb_prior.schema.fk_graph import graph_from_sampled_schema_like
from syn_data.src.rdb_prior.schema.schema_sampler import SchemaSampler


DEFAULT_NUM_DATABASES = int(os.environ.get("RDB_PRIOR_SAMPLE_DB_COUNT", "50"))
DEFAULT_OUTPUT_ROOT = os.environ.get("RDB_PRIOR_SAMPLE_DB_OUTPUT")


def _node_attr(node: Any, name: str, default: Any = None) -> Any:
    if isinstance(node, Mapping):
        return node.get(name, default)
    return getattr(node, name, default)


def _schema_dict(schema: Any) -> dict:
    if hasattr(schema, "to_dict"):
        return schema.to_dict()
    return dict(schema)


def _dot_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _save_schema_artifacts(schema: Any, output_dir: Path) -> None:
    """Save schema JSON plus DOT/Mermaid graph files.

    DOT can be rendered with Graphviz, for example:
        dot -Tpng schema_graph.dot -o schema_graph.png
    """
    schema_dir = output_dir / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)

    schema_obj = _schema_dict(schema)
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
        role = _node_attr(node, "role", "unknown")
        rank = _node_attr(node, "rank", "?")
        label = _dot_escape(f"{table_id}\nrole={role}\nrank={rank}")
        lines.append(f'  "{_dot_escape(table_id)}" [label="{label}"];')

    for fk in fks:
        parent = _node_attr(fk, "parent_table")
        child = _node_attr(fk, "child_table")
        child_col = _node_attr(fk, "child_col", "fk")
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
        role = _node_attr(node, "role", "unknown")
        rank = _node_attr(node, "rank", "?")
        lines.append(f'  {table_id}["{table_id}<br/>role={role}<br/>rank={rank}"]')

    for fk in fks:
        parent = _node_attr(fk, "parent_table")
        child = _node_attr(fk, "child_table")
        child_col = _node_attr(fk, "child_col", "fk")
        lines.append(f"  {parent} -->|{child_col}| {child}")
    return "\n".join(lines) + "\n"


class SampleDatabaseBatchTests(unittest.TestCase):
    def test_generate_and_save_sample_databases(self):
        """Generate sample DBs and verify parquet/metadata/schema graph output."""
        sampler = SchemaSampler.default(seed=42)
        num_databases = DEFAULT_NUM_DATABASES

        if DEFAULT_OUTPUT_ROOT:
            output_root = Path(DEFAULT_OUTPUT_ROOT)
            output_root.mkdir(parents=True, exist_ok=True)
            cleanup = None
        else:
            cleanup = tempfile.TemporaryDirectory()
            output_root = Path(cleanup.name) / "sample_dbs"

        try:
            generated_ids = []

            for i in range(num_databases):
                schema_id = f"db_{i:06d}"
                schema = sampler.sample_schema(schema_id=schema_id)
                context = GenerationContext(
                    seed=42 + i,
                    row_scale=0.001,
                    max_rows_per_table=500,
                )
                generator = RelationalDatabaseGenerator(seed=42 + i)
                result = generator.generate_database(schema, context=context)

                db_dir = output_root / schema_id
                paths = generator.save_database(
                    output_dir=db_dir,
                    tables=result["tables"],
                    metadata=result["metadata"],
                )
                _save_schema_artifacts(schema, db_dir)

                generated_ids.append(schema_id)
                self.assertEqual(set(schema.nodes), set(result["tables"]))
                self.assertTrue(Path(paths["metadata"]).exists())
                self.assertTrue((db_dir / "generation_report.json").exists())
                self.assertTrue((db_dir / "tables").is_dir())
                self.assertTrue((db_dir / "schema" / "schema.json").exists())
                self.assertTrue((db_dir / "schema" / "fk_graph.json").exists())
                self.assertTrue((db_dir / "schema" / "schema_graph.dot").exists())
                self.assertTrue((db_dir / "schema" / "schema_graph.mmd").exists())

                parquet_files = sorted((db_dir / "tables").glob("*.parquet"))
                self.assertEqual(len(schema.nodes), len(parquet_files))
                for table_id in result["tables"]:
                    self.assertTrue(Path(paths[table_id]).exists())
                    self.assertIn(table_id, result["metadata"]["tables"])

            self.assertEqual(num_databases, len(generated_ids))
            self.assertEqual("db_000000", generated_ids[0])
            self.assertEqual(f"db_{num_databases - 1:06d}", generated_ids[-1])
        finally:
            if cleanup is not None:
                cleanup.cleanup()


if __name__ == "__main__":
    unittest.main()