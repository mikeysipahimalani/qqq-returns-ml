"""Fast checks that work offline with the committed model-ready dataset."""

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.qqq_utils import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    chronological_split,
    load_model_data,
    regression_metrics,
    selected_random_forest,
)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_model_data(PROJECT_DIR)

    def test_dataset_schema_and_target_alignment(self):
        self.assertEqual(len(self.data), 4170)
        self.assertEqual(
            list(self.data.columns), FEATURE_COLUMNS + [TARGET_COLUMN]
        )
        # Row t's label must equal row t+1's daily return (apart from the
        # final row, whose target requires the following date's price).
        np.testing.assert_allclose(
            self.data[TARGET_COLUMN].iloc[:-1].to_numpy(),
            self.data["Daily_Return_Pct"].iloc[1:].to_numpy(),
            atol=1e-12,
        )

    def test_original_time_boundaries(self):
        train, val, test = chronological_split(self.data)
        self.assertEqual((len(train), len(val), len(test)), (2918, 624, 626))
        self.assertEqual(str(train.index.max().date()), "2021-09-02")
        self.assertEqual(str(val.index.max().date()), "2024-02-29")
        self.assertEqual(str(test.index.min().date()), "2024-03-04")
        self.assertTrue(train.index.max() < val.index.min() < test.index.min())

    def test_frozen_forest_reproduces_test_rmse(self):
        train, _, test = chronological_split(self.data)
        model = selected_random_forest().fit(
            train[FEATURE_COLUMNS], train[TARGET_COLUMN]
        )
        prediction = model.predict(test[FEATURE_COLUMNS])
        rmse = regression_metrics(test[TARGET_COLUMN], prediction)["RMSE"]
        self.assertAlmostEqual(rmse, 1.314285, places=5)


if __name__ == "__main__":
    unittest.main()
