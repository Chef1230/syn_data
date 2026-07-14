import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from syn_data.src.rdb_prior.schema.fk_graph import FKGraph


class FKGraphTests(unittest.TestCase):
    def test_diamond_3edge_count(self):
        graph = FKGraph(
            node_ids=["A", "B", "C"],
            edges=[("A", "B"), ("A", "C"), ("B", "C")],
        )

        self.assertEqual(1, graph.motif_counts()["diamond_3edge"])

    def test_duplicate_edges_can_be_allowed_explicitly(self):
        graph = FKGraph(node_ids=["A", "B"], edges=[("A", "B"), ("A", "B")])

        self.assertFalse(graph.validate().is_valid)
        self.assertTrue(graph.validate(allow_duplicate_edges=True).is_valid)


if __name__ == "__main__":
    unittest.main()