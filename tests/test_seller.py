from nbresult import ChallengeResultTestCase


class TestSeller(ChallengeResultTestCase):
    def test_shape(self):
        self.assertEqual(self.result.shape, (2967, 15),
                         msg="Expected exactly 2967 rows and 15 columns")

    def test_median_review_score(self):
        self.assertAlmostEqual(self.result.median, 4.2, 1)

    def test_column_names(self):
        self.assertIn('share_of_one_stars', self.result.columns,
                      msg="Column 'share_of_one_stars' is missing")
        self.assertIn('share_of_five_stars', self.result.columns,
                      msg="Column 'share_of_five_stars' is missing")
