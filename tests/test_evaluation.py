import unittest
from illusion_agent.evaluation import aggregate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.mapping = [{"review_id": "r1", "image_id": "i1", "cue_id": "c0", "mode": "evidence", "status": "ok"}]
        self.gold = {"i1": {"review_status": "adjudicated", "group": "ambiguous", "cues": {"c0": "neutral"}}}
        self.ratings = [{"review_id": "r1", "both_valid": "1", "false_alternative": "NA", "cue_correct": "NA", "grounded": "1"}]

    def test_rates_and_applicability(self):
        row = aggregate(self.mapping, self.ratings, self.gold)[0]
        self.assertEqual(row["both_valid_rate"], 1)
        self.assertEqual(row["both_valid_n"], 1)
        self.assertIsNone(row["cue_correct_rate"])

    def test_missing_and_duplicate_reviews_are_not_dropped(self):
        for ratings in ([], self.ratings * 2):
            with self.assertRaises(ValueError):
                aggregate(self.mapping, ratings, self.gold)

    def test_unreviewed_gold_rejected(self):
        self.gold["i1"]["review_status"] = "pending"
        with self.assertRaises(ValueError):
            aggregate(self.mapping, self.ratings, self.gold)

    def test_technical_failure_stays_in_success_denominator(self):
        self.mapping[0]["status"] = "error"
        for k in ("both_valid", "grounded"):
            self.ratings[0][k] = "NA"
        row = aggregate(self.mapping, self.ratings, self.gold)[0]
        self.assertEqual(row["both_valid_n"], 1)
        self.assertEqual(row["both_valid_rate"], 0)
        self.assertEqual(row["technical_errors"], 1)

    def test_technical_failure_does_not_improve_false_positive_rate(self):
        self.gold["i1"]["group"] = "ordinary"
        self.mapping[0]["status"] = "error"
        self.ratings[0].update(both_valid="NA", grounded="NA")
        row = aggregate(self.mapping, self.ratings, self.gold)[0]
        self.assertIsNone(row["false_alternative_rate"])
        self.assertEqual(row["false_alternative_n"], 0)

    def test_blank_rating_rejected(self):
        self.ratings[0]["grounded"] = ""
        with self.assertRaises(ValueError):
            aggregate(self.mapping, self.ratings, self.gold)
