import os
import unittest
import numpy as np

from food_classifier import inference

__author__ = 'Junior Teudjio'

class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = inference.Model()

    def test_predict(self):
        sandwich_pred =self.model.predict('../tests/images/sandwich/sandwich.jpg')
        sushi_pred =self.model.predict('../tests/images/sushi/sushi.jpg')

        self.assertEqual(sandwich_pred['label'], 'sandwich')
        self.assertEqual(sushi_pred['label'], 'sushi')

    def test_get_best_model_path(self):
        ckpts_path = '../tests/fake_model_ckpts'
        best_base_model_path = inference._get_best_model_path('base_model',ckpts_path)
        best_mobnet_model_path = inference._get_best_model_path('mobilenet_model', ckpts_path)

        self.assertEqual(best_base_model_path,
                         os.path.join(ckpts_path, 'base_model.ckpt-03-10.64-0.70.hdf5') )
        self.assertEqual(best_mobnet_model_path,
                         os.path.join(ckpts_path, 'mobilenet_model.ckpt-01-10.64-0.90.hdf5') )

    def test_predict_on_batch(self):
        true_labels, pred_labels = self.model.predict_on_batch(folder_path='../tests/images/')
        self.assertTrue(np.allclose(true_labels, pred_labels))
        self.assertTrue(np.allclose(np.array([0., 1.]), true_labels))

if __name__ == '__main__':
    unittest.main()
