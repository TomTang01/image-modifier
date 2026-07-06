import unittest

import numpy as np

from image_combiner import combine_images, ncc_match, ssd_match


class MatchingTests(unittest.TestCase):
    def test_ssd_examples_from_notebook(self):
        img1 = np.array([[1, 2, 3, 4], [4, 5, 6, 8], [7, 8, 9, 4]])
        img2 = np.array([[1, 2, 1, 3], [6, 5, 4, 4], [9, 8, 7, 3]])

        self.assertEqual(ssd_match(img1, img2, np.array([1, 1]), np.array([1, 1]), 1), 20)
        self.assertEqual(ssd_match(img1, img2, np.array([2, 1]), np.array([2, 1]), 1), 30)
        self.assertEqual(ssd_match(img1, img2, np.array([1, 1]), np.array([2, 1]), 1), 46)

    def test_ncc_examples_from_notebook(self):
        img1 = np.array([[1, 2, 3, 4], [4, 5, 6, 8], [7, 8, 9, 4]])
        img2 = np.array([[1, 2, 1, 3], [6, 5, 4, 4], [9, 8, 7, 3]])

        self.assertAlmostEqual(ncc_match(img1, img2, np.array([1, 1]), np.array([1, 1]), 1), 0.8546, places=3)
        self.assertAlmostEqual(ncc_match(img1, img2, np.array([2, 1]), np.array([2, 1]), 1), 0.8457, places=3)
        self.assertAlmostEqual(ncc_match(img1, img2, np.array([1, 1]), np.array([2, 1]), 1), 0.6258, places=3)

    def test_synthetic_overlap_is_accepted(self):
        rng = np.random.default_rng(42)
        base = rng.random((130, 210, 3))
        image1 = base[:, :160]
        image2 = base[:, 35:195]

        result = combine_images(
            image1,
            image2,
            n_corners=220,
            ncc_radius=4,
            ncc_threshold=0.8,
            ransac_threshold=2.0,
            min_inliers=10,
            min_inlier_ratio=0.2,
            random_seed=3,
        )

        self.assertTrue(result.accepted, result.diagnostics)
        self.assertIsNotNone(result.stitched)
        self.assertGreaterEqual(result.diagnostics["inliers"], 10)

    def test_unrelated_images_are_rejected(self):
        rng = np.random.default_rng(100)
        image1 = rng.random((120, 160, 3))
        image2 = rng.random((120, 160, 3))

        result = combine_images(
            image1,
            image2,
            n_corners=180,
            ncc_radius=4,
            ncc_threshold=0.75,
            ransac_threshold=2.0,
            min_inliers=12,
            min_inlier_ratio=0.25,
            random_seed=4,
        )

        self.assertFalse(result.accepted, result.diagnostics)


if __name__ == "__main__":
    unittest.main()
