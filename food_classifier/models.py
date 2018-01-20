import logging

from keras.models import Sequential
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras.applications import MobileNet, VGG16

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('./logs/models.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

__author__ = 'Junior Teudjio'
__all__ = ['base_model', 'vgg16_model', 'mobilenet_model']


def base_model(init='glorot_uniform', activation='relu', batch_norm=True, dropout=0.5):
    '''
    Create an instance of the baseline model.

    Parameters
    ----------
    init: str
        Weights initialization strategy.
    activation : str
        Activation function to use.
    batch_norm : boolean
        Whether to use batch normalization or not.
    dropout : float
        Ratio of weights to turn off before the final activation function

    Returns:
        model: Sequential()
            The baseline model object
    '''
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(150, 150, 3), kernel_initializer=init))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), kernel_initializer=init))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), kernel_initializer=init))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), kernel_initializer=init))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.MaxPooling2D((2, 2)))


    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(512, activation=activation))
    model.add(layers.Dense(1, activation='sigmoid'))

    logger.info('''
    
    Created baseline model with params:
        init = {init}
        activation = {activation}
        batch_norm = {batch_norm}
        dropout = {dropout} 
        architecture = {architecture}'''.format(
        init=init,
        activation=activation,
        batch_norm=batch_norm,
        dropout=dropout,
        architecture = model.to_json()
    ))

    return model


def vgg16_model():
    pass

def mobilenet_model():
    pass
