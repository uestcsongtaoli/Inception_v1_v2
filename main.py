from keras.utils import multi_gpu_model
import math

from model import GoogLeNet
from preprocessing import load_batch_data
from config import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"


def _main(model_name):
    # data_sets path
    train_dir = '/home/lst/datasets/cifar-10-images_train/'
    val_dir = '/home/lst/datasets/cifar-10-images_test/'

    # model
    model = GoogLeNet(input_shape=(224, 224, 3))
    # parallel model
    parallel_model = multi_gpu_model(model, gpus=2)

    # optimizers setting
    from keras import optimizers
    optimizer = optimizers.adamax(lr=0.002, decay=1e-06)
    parallel_model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                           loss_weights=[1, 0.3, 0.3],
                           optimizer=optimizer,
                           metrics=["accuracy"])
    # load data by batch
    train_generator, validation_generator, num_train, num_val = load_batch_data(train_dir, val_dir)

    # Callbacks settings
    from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=30,
                               mode='min',
                               verbose=1)

    checkpoint = ModelCheckpoint(f'{model_name}.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=1)

    tensorboard = TensorBoard(log_dir=f'./logs/{model_name}',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=False)
    # fit
    parallel_model.fit_generator(train_generator,
                                 validation_data=validation_generator,
                                 steps_per_epoch=math.ceil(num_train / batch_size),
                                 validation_steps=math.ceil(num_val / batch_size),
                                 epochs=epochs,
                                 callbacks=[tensorboard, early_stop, checkpoint],
                                 )


if __name__ == "__main__":
    name = "Inception_v1"
    _main(name)
