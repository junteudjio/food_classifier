import unittest
import keras

from food_classifier import models

__author__ = 'Junior Teudjio'

class TestInference(unittest.TestCase):
    def test_base_model(self):
        # test if base_model can be built correctly
        model = models.base_model()
        self.assertTrue(isinstance(model, keras.Sequential))

    def test_mobilenet_model(self):
        # test if the mobilenet_model can be built correctly
        model = models.mobilenet_model()
        self.assertTrue(isinstance(model, keras.Sequential))

if __name__ == '__main__':
    unittest.main()
