import os
import numpy as np
import unittest
from automate_image_alignment import align_and_register_images, save_registered_images

class TestAutomateImageAlignment(unittest.TestCase):

    def setUp(self):
        self.image_dir = "path/to/your/image_directory"
        self.output_dir = "path/to/save/registered_images"
        self.fformat = 'tif'
        self.pad = False
        self.deg = 2
        self.avg_over = 1
        self.subpx = 1
        self.color = (0, 0, 0)

    def test_align_and_register_images(self):
        align_and_register_images(self.image_dir, self.output_dir, self.fformat, self.pad, self.deg, self.avg_over, self.subpx, self.color)
        self.assertTrue(os.path.exists(self.output_dir))

    def test_save_registered_images(self):
        image_stack = np.zeros((10, 100, 100))
        save_registered_images(image_stack, self.output_dir, self.fformat)
        self.assertTrue(os.path.exists(self.output_dir))
        for i in range(10):
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, f'registered_image_{i}.{self.fformat}')))

if __name__ == '__main__':
    unittest.main()
