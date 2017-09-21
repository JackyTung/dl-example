'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback

from keras import optimizers
import argparse
import os

parser = argparse.ArgumentParser(description='train an image classifer')
parser.add_argument('--num-epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--decay', type=float, default=0)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epsilon', type=float, default=1e-08)
parser.add_argument('--loss', type=str, default='categorical_crossentropy')
parser.add_argument('--metrics', type=str, default='accuracy')
parser.add_argument('--outdir', type=str, default='output')
args = parser.parse_args()
print(args)

batch_size = args.batch_size
epochs = args.num_epoch
lr = args.lr
optimizer = args.optimizer
decay = args.decay
momentum = args.momentum
beta1 = args.beta1
beta2 = args.beta2
epsilon = args.epsilon
outdir = args.outdir
loss = args.loss
# Supported loss: mean_squared_error,  mean_absolute_error,
# mean_absolute_percentage_error, mean_squared_logarithmic_error, squared_hinge,
# hinge, categorical_hinge, logcosh, categorical_crossentropy,
# sparse_categorical_crossentropy, binary_crossentropy,
# kullback_leibler_divergence, poisson, cosine_proximity
metrics = args.metrics.split(',')
# available metrics: alias: accuracy / acc, binary_accuracy, categorical_accuracy,
# sparse_categorical_accuracy, top_k_categorical_accuracy
# sparse_top_k_categorical_accuracy

# load the network
from importlib import import_module
net = import_module('model')
model = net.get_model()

# Setup otpimizer
op = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
if optimizer == 'sgd':
    op = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
elif optimizer == 'adam':
    op = optimizers.Adam(
        lr=lr,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon,
        decay=decay)
else:
    raise Exception('unknown optimizer: ', optimizer)

model.compile(loss=loss,
              optimizer=op,
              metrics=metrics)

# prepare model output
if not os.path.exists(outdir):
    os.makedirs(outdir)
filepath = outdir + "/model-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(
    filepath,
    monitor='val_acc',
    verbose=1,
    save_best_only=False,
    mode='max')


# NOTE: Let the output format same as that of MXNet
batch_print_callback = LambdaCallback(
    on_epoch_end=lambda batch, logs: print(
        '\nINFO:root:Epoch[%d] Train-accuracy=%f\nINFO:root:Epoch[%d] Validation-accuracy=%f' %
        (batch, logs['acc'], batch, logs['val_acc'])))

callbacks_list = [checkpoint, batch_print_callback]

# todo: handle the case that the sample size is not divisible
# start training
d = net.DataGenerator(batch_size)
model.fit_generator(generator=d.training_data(),
                    steps_per_epoch=d.training_steps(),
                    epochs=epochs,
                    verbose=1,
                    validation_data=d.validation_data(),
                    validation_steps=d.validation_steps(),
                    callbacks=callbacks_list)
