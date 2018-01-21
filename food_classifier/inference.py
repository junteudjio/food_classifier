from __future__ import division
import os
import glob
import logging

from PIL import Image
import numpy as np
import keras
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('./logs/inference.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

__author__ = 'Junior Teudjio'
__all__ = ['Model']

ckpts_path = 'model_ckpts'

def _get_best_model_path(model_type):
    '''
    Get the best model checkpoint for a given model type.

    Parameters
    ----------
    model_type : str
        Model type to consider, one of: base_model or mobilenet_model

    Returns
    -------
    best_model : Keras model object
    '''
    model_type_template = '{}.ckpt-*'.format(model_type)
    all_models_paths = glob.glob(os.path.join(ckpts_path, model_type_template))
    all_models_paths.sort(key=lambda path: path.split('-')[-1]) #sort by val_score

    best_model_path = all_models_paths[-1]
    custom_objects = None
    if model_type == 'mobilenet_model':
        custom_objects = {'relu6': keras.applications.mobilenet.relu6,
                          'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}
    best_model = keras.models.load_model(best_model_path, custom_objects=custom_objects)

    return best_model

class Model(object):
    def __init__(self):
        self.models = dict()
        try:
            self.models['base_model'] = _get_best_model_path('base_model')
            self.models['mobilenet_model'] = _get_best_model_path('mobilenet_model')
        except Exception as e:
            logger.error('Unable to load model, error_message = {}'.format(repr(e)))
            raise e
        self.labels = {0:'sandwich', 1:'sushi'}

    def predict(self, image_path, model_type='mobilenet_model'):
        '''
        Get the prediction for an image given by a model type.

        Parameters
        ----------
        image_path : str
            Path to  the image to  get prediction.
        model_type : str
            Model type to consider, one of: base_model or mobilenet_model

        Returns
        -------
        prediction : dict {label_idx, label, confidence}
            label_idx : an str(int) representing the index of the predicted label.
            label : an str reprenting the label name.
            confidence : a 'probability' score given by the model.
        '''
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
        label_idx = 0 if proba < 0.5 else 1
        confidence = proba if label_idx == 1 else 1. - proba

        prediction = {
            'label_idx': str(label_idx),
            'label':self.labels[label_idx],
            'confidence': str(confidence)
        }

        logger.info('''
        Prediction: {}'''.format(str(prediction)))

        return prediction


    def predict_on_batch(self, folder_path, model_type='mobilenet_model'):
        '''
        Get the predictions for all the images in a given directory.

        Parameters
        ----------
        folder_path : str
            The Path to the directory containing the images partitioned in one directory for each class.
            Example: -path/to/train/ (folder_path argument)
                        -sushi/
                            -img1.png
                            -img2.png
                        -sandwich/
                            -img1.png
                            -img2.png
        model_type : str
                    Model type to consider, one of: base_model or mobilenet_model

        Returns
        -------
        true_labels, pred_labels : a tuple of numpy arrays of shape (num_of_image_counts)
        '''

        # count the total number of images in the directory
        images_count = 0
        for class_dir in os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file != '.DS_Store':
                        images_count += 1

        # set the resizing shape depending on model in use.
        if model_type == 'base_model':
            resize_shapes = (150, 150)
        else:
            resize_shapes = (160, 160)  # excepted size for mobilenet_model

        # create a keras data generator
        batch_size = 20
        datagen = ImageDataGenerator(rescale=1./255)
        generator = datagen.flow_from_directory(
            folder_path,
            target_size=resize_shapes,
            batch_size=batch_size,
            shuffle=False,
            class_mode='binary')

        pred_labels = np.zeros(shape=(images_count))
        true_labels = np.zeros(shape=(images_count))

        i = 0
        for inputs_batch, true_labels_batch in generator:
            pred_proba_batch = self.models[model_type].predict(inputs_batch)[:,0]
            pred_labels_batch = np.where(pred_proba_batch < 0.5, 0, 1)
            pred_labels[i * batch_size: (i + 1) * batch_size] = pred_labels_batch
            true_labels[i * batch_size: (i + 1) * batch_size] = true_labels_batch
            i += 1
            if i * batch_size >= images_count:
                # Note that since generators yield data indefinitely in a loop,
                # we must `break` after every image has been seen once.
                break

        return true_labels, pred_labels


if __name__ == '__main__':
    pass
    model = Model()
    #model.predict_on_batch(folder_path='data/train', model_type='base_model')
    print model.predict(image_path='data/train/sandwich/train_67.jpg', model_type='base_model')
    print model.predict(image_path='data/train/sushi/train_26.jpg', model_type='base_model')
    print model.predict(image_path='data/train/sandwich/train_67.jpg')
    print model.predict(image_path='data/train/sushi/train_26.jpg')