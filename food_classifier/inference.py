from __future__ import division
import os
import glob
import logging

from PIL import Image
import numpy as np
import keras
from keras.preprocessing.image import img_to_array, load_img


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('./logs/inference.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

__author__ = 'Junior Teudjio'

ckpts_path = 'model_ckpts/_backups'

def _get_best_model_path(model_type):
    model_type_template = '{}.ckpt-*'.format(model_type)
    all_models_paths = glob.glob(os.path.join(ckpts_path, model_type_template))
    all_models_paths.sort(key=lambda path: path.split('-')[-1]) #sort by val_score

    best_model_path = all_models_paths[-1]
    best_model = keras.models.load_model(best_model_path)

    return best_model

class Model(object):
    def __init__(self):
        self.models = dict()
        try:
            self.models['base_model'] = _get_best_model_path('base_model')
        except Exception as e:
            logger.error('Unable to load model, error_message = {}'.format(repr(e)))
            raise e
        #TODO: uncomment when model is implemented and trained
        #self.models['mobilenet_model'] = _get_best_model_path('mobilenet_model')
        self.labels = {1:'sandwich', 0:'sushi'}

    def predict(self, image_path, model_type='base_model'):
        logger.info('''
        Getting prediction for
        image: {image_path}
        model_type: {model_type}'''.format(image_path=image_path, model_type=model_type))
        img = load_img(image_path)  # this is a PIL image
        if model_type == 'base_model':
            resize_shapes = (150, 150)
        else:
            resize_shapes = (160, 160) # excepted size for mobilenet_model
        img = img.resize(resize_shapes,  Image.ANTIALIAS)
        array = img_to_array(img) * 1./255 # this is a Numpy array with shape (3, 150, 150)
        array = array[np.newaxis, ...] # add another dimension for batch

        proba = self.models[model_type].predict_proba(array)[0][0]
        label_idx = 0 if proba >= 0.5 else 1
        confidence = proba if label_idx == 0 else 1. - proba

        prediction = {
            'label_idx': str(label_idx),
            'label':self.labels[label_idx],
            'confidence': str(confidence)
        }

        logger.info('''
        Prediction: {}'''.format(str(prediction)))

        return prediction
