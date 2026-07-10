import unittest

import numpy as np

from extract_person import choose_sam_mask, prompt_box_from_margin, resize_for_sam, select_torch_device, to_uint8_rgb
from image_combiner import combine_images, ncc_match, ssd_match
from image_resizer import (
    RESIZE_MODE_CROP,
    RESIZE_MODE_FREE,
    RESIZE_MODE_KEEP_ASPECT,
    resize_image,
    scaled_dimensions,
)


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


class ResizeTests(unittest.TestCase):
    def test_free_resize_outputs_exact_requested_dimensions(self):
        image = np.zeros((20, 40, 3), dtype=np.uint8)

        resized = resize_image(image, 30, 10, RESIZE_MODE_FREE)

        self.assertEqual(resized.shape, (10, 30, 3))
        self.assertEqual(resized.dtype, np.uint8)

    def test_keep_aspect_resize_fits_within_bounds(self):
        image = np.zeros((50, 100, 3), dtype=np.uint8)

        resized = resize_image(image, 80, 80, RESIZE_MODE_KEEP_ASPECT)

        self.assertEqual(resized.shape, (40, 80, 3))
        self.assertEqual(resized.dtype, np.uint8)

    def test_crop_resize_outputs_exact_dimensions_from_center(self):
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(8, dtype=np.uint8)

        resized = resize_image(image, 4, 4, RESIZE_MODE_CROP)

        self.assertEqual(resized.shape, (4, 4, 3))
        self.assertTrue(np.all(resized[:, 0, 0] == 2))
        self.assertTrue(np.all(resized[:, -1, 0] == 5))

    def test_resize_converts_float_grayscale_to_uint8_rgb(self):
        image = np.ones((6, 8), dtype=float) * 0.5

        resized = resize_image(image, 4, 3, RESIZE_MODE_FREE)

        self.assertEqual(resized.shape, (3, 4, 3))
        self.assertEqual(resized.dtype, np.uint8)

    def test_scaled_dimensions_round_and_never_drop_below_one_pixel(self):
        self.assertEqual(scaled_dimensions(100, 80, 0.25), (25, 20))
        self.assertEqual(scaled_dimensions(3, 3, 0.25), (1, 1))
        self.assertEqual(scaled_dimensions(100, 80, 1.5), (150, 120))


class ExtractTests(unittest.TestCase):
    def test_prompt_box_uses_margin(self):
        box = prompt_box_from_margin(100, 80, 0.1)

        self.assertTrue(np.allclose(box, np.array([10, 8, 89, 71], dtype=float)))

    def test_resize_for_sam_scales_only_large_images(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        resized, scale = resize_for_sam(image, 50)

        self.assertEqual(resized.shape, (25, 50, 3))
        self.assertEqual(scale, 0.25)

    def test_choose_sam_mask_uses_best_valid_score(self):
        masks = np.zeros((3, 10, 10), dtype=bool)
        masks[0, :1, :1] = True
        masks[1, :5, :5] = True
        masks[2, :, :] = True
        scores = np.array([0.99, 0.8, 0.95])

        chosen = choose_sam_mask(masks, scores)

        self.assertTrue(np.array_equal(chosen, masks[1]))

    def test_to_uint8_rgb_converts_float_grayscale(self):
        image = np.ones((4, 5), dtype=float) * 0.5

        rgb = to_uint8_rgb(image)

        self.assertEqual(rgb.shape, (4, 5, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(int(rgb[0, 0, 0]), 127)

    def test_select_torch_device_prefers_cuda(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

            @staticmethod
            def current_device():
                return 0

            @staticmethod
            def get_device_name(index):
                return f"GPU {index}"

        class FakeVersion:
            cuda = "12.8"

        class FakeTorch:
            __version__ = "test"
            cuda = FakeCuda()
            version = FakeVersion()

        device_info = select_torch_device(FakeTorch)

        self.assertEqual(device_info["device"], "cuda")
        self.assertTrue(device_info["cuda_available"])
        self.assertEqual(device_info["cuda_device_name"], "GPU 0")


if __name__ == "__main__":
    unittest.main()
