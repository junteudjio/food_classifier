from __future__ import division
import logging
import argparse
import os
import time

import keras
import keras.optimizers as keras_optimizers
import keras.callbacks as keras_callbacks
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import utils
import models

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('./logs/train.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

__author__ = 'Junior Teudjio'


def _setup_args():
    parser = argparse.ArgumentParser()

    # data paths
    data_prefix = './data'
    train_dir = os.path.join(data_prefix, 'train')
    val_dir = os.path.join(data_prefix, 'validation')
    parser.add_argument("--data_prefix", default=data_prefix, help='Folder where to dump preprocess dataset')
    parser.add_argument("--train_dir", default=train_dir, help='Folder containing the training set images')
    parser.add_argument("--val_dir", default=val_dir, help='Folder containing the validation set images')

    # data sizes
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--val_batch_size", default=20, type=int)

    # data augmentation
    parser.add_argument("--rescale", default=1./255, type=float)
    parser.add_argument("--rotation_range", default=40, type=int)
    parser.add_argument("--width_shift_range", default=0.2, type=float)
    parser.add_argument("--height_shift_range", default=0.2, type=float)
    parser.add_argument("--shear_range", default=0.2, type=float)
    parser.add_argument("--zoom_range", default=0.2, type=float)
    parser.add_argument("--horizontal_flip", default=True, action='store_true')

    # model
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--init", default='glorot_uniform', type=str)
    parser.add_argument("--regularizer", default='l2-0.01', type=str,
                        help='Type and value of regularization to use, reg_type = l2 or l1')
    parser.add_argument("--activation", default='relu', type=str)
    parser.add_argument("--batch_norm", default=True, action='store_true')
    parser.add_argument("--model_type", default='base_model', type=str,
                        help='Type of model to train, either: base_model or mobilenet_model')

    # model callbacks
    parser.add_argument("--terminate_on_nan", default=True, action='store_true',
                        help='Callback that terminates training when a NaN loss is encountered.')

    parser.add_argument("--epochs_csv_logger", default='./logs/epochs_logs.csv', type=str,
                        help='CSV file where to save results per epoch')

    parser.add_argument("--early_stop", default=True, action='store_true',
                        help='Callback that terminates training when monitored valued not (in/de)-creasing')
    parser.add_argument("--early_stop_monitor", default='val_acc', type=str,
                        help='The metric(val_acc or val_loss) to monitor and let value trigger early stopping')
    parser.add_argument("--early_stop_mode", default='auto', type=str,
                        help='Stop when monitor value stop increasing(max) or decreasing(min)')
    parser.add_argument("--early_stop_min_delta", default=0.001, type=float,
                        help='Minimum change in the monitored quantity to qualify as an improvement')
    parser.add_argument("--early_stop_patience", default=5, type=int,
                        help='Number of epochs after which to stop if no improvement')
    parser.add_argument("--early_stop_verbose", default=1, type=int)


    # model checkpoints
    ckpts_path = 'model_ckpts'
    utils.mkdir_p(ckpts_path)
    parser.add_argument('--ckpts_prefix', type=str, default=ckpts_path, help='Parent folder of model checkpoints files')
    parser.add_argument("--ckpts_template", default='ckpt-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5',
                        type=str, help='Template name of model checkpoints files')


    # optimizer
    parser.add_argument("--optimizer_str", default='adam', type=str)
    parser.add_argument("--lr", default=0.001, type=float, help='learning rate')
    parser.add_argument("--momentum", default=0.99, type=float, help='momentum of the gradient')
    parser.add_argument("--epochs", default=40, type=int,
                        help='Number of time to go over the entire dataset during training')
    parser.add_argument("--steps_per_epoch", default=100, type=int)
    parser.add_argument("--validation_steps", default=20, type=int)


    # others
    history_plots_prefix = 'history_plots'
    utils.mkdir_p(history_plots_prefix)
    parser.add_argument('--history_plots_prefix', default=history_plots_prefix, type=str,
                        help='Parent path where to save plots created from history objects')
    parser.add_argument("--resume", default=False, action='store_true', help='Resume training or not')
    parser.add_argument('--resume_ckpt', default='base_model.ckpt-07-0.88-0.65.hdf5', type=str,
                        help='Filename of the model checkpoint to resume from')


    return parser.parse_args()

def _get_optimizer(optimizer_str, learning_rate=0.001, momentum=0.99):
    if optimizer_str in ('sgd', 'SGD'):
        return getattr(keras_optimizers, optimizer_str)(lr=learning_rate, momentum=momentum)
    return getattr(keras_optimizers, optimizer_str)(lr=learning_rate)

def _get_callbacks(args):
    logger.info('Adding callbacks to the model')
    callbacks = []

    ##  terminate_on_nan
    if args.terminate_on_nan:
        callbacks.append(keras_callbacks.TerminateOnNaN())

    ##  early stopping
    if args.early_stop:
        callbacks.append(keras_callbacks.EarlyStopping(
            monitor=args.early_stop_monitor,
            min_delta=args.early_stop_min_delta,
            patience=args.early_stop_patience,
            verbose=args.early_stop_verbose,
            mode=args.early_stop_mode
        ))

    ##  model checkpoints
    ckpts_template = '.'.join([args.model_type, args.ckpts_template])
    ckpts_path_template = os.path.join(args.ckpts_prefix, ckpts_template)
    callbacks.append(keras_callbacks.ModelCheckpoint(ckpts_path_template))

    ## epochs_csv_logger
    callbacks.append(keras_callbacks.CSVLogger(filename=args.epochs_csv_logger))

    return callbacks

def _get_datagenerators(args):
    logger.info('Getting the train/val data generators')
    train_datagen = ImageDataGenerator(
        rescale=args.rescale,
        rotation_range=args.rotation_range,
        width_shift_range=args.width_shift_range,
        height_shift_range=args.height_shift_range,
        shear_range=args.shear_range,
        zoom_range=args.zoom_range,
        horizontal_flip=args.horizontal_flip,
    )

    validation_datagen = ImageDataGenerator(rescale=args.rescale)

    target_size = (150, 150) if args.model_type == 'base_model' else (160, 160)  # else means mobilenet_model
    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=target_size,
        batch_size=args.train_batch_size,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        args.val_dir,
        target_size=target_size,
        batch_size=args.val_batch_size,
        class_mode='binary')

    return train_generator, validation_generator

def _get_model(args):
    logger.info('Constructing the model :{}'.format(args.model_type))
    start = time.time()
    if args.model_type == 'base_model':
        model = models.base_model(
            init=args.init,
            activation=args.activation,
            batch_norm=args.batch_norm,
            dropout=args.dropout,
            regularizer=args.regularizer
        )
    elif args.model_type == 'mobilenet_model':
        #TODO: need to create this class in models.py
        model = models.mobilenet_model()
    else:
        raise ValueError('Unknown model_type error, model_type should be one of (base_model, mobilenet_model)')

    logger.info('Model construction took: {}seconds'.format(time.time() - start))
    return model

def _create_history_plots(history, path_prefix):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    savepath = os.path.join(path_prefix, str(time.time()) + '.png')
    plt.savefig(savepath)

def main():
    args = _setup_args()

    # create the train/val data iterators
    train_generator, validation_generator = _get_datagenerators(args)

    # instantiate the model
    if args.resume:
        model_filepath = os.path.join(args.ckpts_prefix, args.resume_ckpt)
        logger.info('Resuming training from ckpt: {}'.format(model_filepath))
        model = keras.models.load_model(model_filepath)
    else:
        model = _get_model(args)

    # compilte the model
    optimizer = _get_optimizer(args.optimizer_str, args.lr, args.momentum)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

    # create model callbacks
    callbacks = _get_callbacks(args)

    # train loop
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=args.validation_steps,
        callbacks=callbacks
    )

    _create_history_plots(history, args.history_plots_prefix)


if __name__ == '__main__':
    main()