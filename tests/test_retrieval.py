import math
import unittest

from ake_backend.retrieval import reciprocal_rank_fusion
from ake_backend.metrics import precision_at_k, recall_at_k, reciprocal_rank, hit_at_k, ndcg_at_k


class RetrievalTests(unittest.TestCase):
    def test_fusion_merges_and_orders_deterministically(self):
        result = reciprocal_rank_fusion([["b", "a", "b"], ["a", "c"]], k=60)
        self.assertEqual([x[0] for x in result], ["a", "b", "c"])
        self.assertAlmostEqual(result[0][1], 1 / 62 + 1 / 61)

    def test_ties_and_limit(self):
        self.assertEqual(reciprocal_rank_fusion([["b"], ["a"]], top_k=1)[0][0], "a")
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["a"]], k=0)

    def test_binary_metrics_and_duplicates(self):
        ranked = ["x", "a", "a", "b"]
        relevant = {"a", "b", "c"}
        self.assertEqual(precision_at_k(ranked, relevant, 3), 1 / 3)
        self.assertEqual(recall_at_k(ranked, relevant, 4), 2 / 3)
        self.assertEqual(reciprocal_rank(ranked, relevant, 4), 0.5)
        self.assertEqual(hit_at_k(ranked, relevant, 1), 0.0)

    def test_ndcg_graded(self):
        ranked = ["b", "a"]
        relevance = {"a": 3.0, "b": 1.0}
        expected = (1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(ranked, relevance, 2), expected)

    def test_invalid_and_empty(self):
        self.assertEqual(recall_at_k([], set(), 3), 0.0)
        self.assertEqual(ndcg_at_k([], {}, 3), 0.0)
        with self.assertRaises(ValueError):
            precision_at_k([], set(), 0)
        with self.assertRaises(ValueError):
            ndcg_at_k(["a"], {"a": -1}, 1)


if __name__ == "__main__":
    unittest.main()
