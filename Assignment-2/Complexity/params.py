import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import itertools
import tensorflow as tf
import math
import functools
import operator
import typing
from pathlib import Path
tf.get_logger().setLevel('ERROR')

class DataParameters:
    M = 2 # Dimension
    s = 500 # Data quantity
    Q = 3 # Question #, 1 or 3
    # Note that Assignment 2 is basically just Q3 datawise.

    NOISE = 0

    L1 = 1 # Label value for first class
    L2 = 0 # Label value for second class

    assert L1 > L2, "Code assumes L1 > L2 in places, yet that is not true here."

    SCALE = 1000 # Amount we scale weights down by
    REGULARIZATION = 0.003
    
    GA_DUPLI_SIZE = 1 # How many bits are dedicated to duplication counter per organism
    GA_LAYER_SIZE = 4 # How many bits are dedicated to neurons in a layer per organism
    GA_INITIALIZER_SIZE = 2 # How many bits dedicated to neuron initialization per organism
    GA_LAYER_AMOUNT = 5

    def GA_BITS_PER_LAYER(): # How many bits per layer
        return (
            DataParameters.GA_DUPLI_SIZE
            + DataParameters.GA_LAYER_SIZE
            + DataParameters.GA_INITIALIZER_SIZE
        )

    def GA_GENO_SIZE(): # How many total bits per organism
        return DataParameters.GA_LAYER_AMOUNT * DataParameters.GA_BITS_PER_LAYER()

    def _internal_decode_initializer(value_in):
        if value_in == 0:
            return tf.keras.initializers.HeNormal(), 'HeNormal'
        elif value_in == 1:
            return tf.keras.initializers.RandomNormal(stddev=0.5), 'Normal'
        elif value_in == 2:
            return tf.keras.initializers.RandomUniform(minval=-1, maxval=1), 'Uniform'
        elif value_in == 3:
            return tf.keras.initializers.Zeros(), 'Zeroes'
        else:
            assert False, f"You've gone over the initializer bit limit: {value_in}!"

    def DECODE_INITIALIZER(value_in): # The weight initialization function to use in GA&GP.
        return DataParameters._internal_decode_initializer(value_in)[0]

    def INITIALIZER_STRING(value_in): # The string name of the weight initialization function
        return DataParameters._internal_decode_initializer(value_in)[1]

    # binary_crossentropy (use L1=1, L2=0, sigmoid)
    # mean_squared_error (use L1=1, L2=-1, linear) [not for classification]
    # hinge (use L1=1, L2=-1, tanh)
    LOSS = 'binary_crossentropy' # Loss function that we use
    FINAL_ACTIVATION = 'sigmoid' # Activation function at output

    MAKE_DISCRETE_PLOT = True # True if want to plot decision boundary, False for all contours

    @classmethod
    def PSOTrainable(cls):
        assert False, "Not set yet, set in nn.py"
        return 0

    USE_EARLY_STOPPING = True # True if we want to use early stopping
    def EARLY_STOPPING():
        if DataParameters.USE_EARLY_STOPPING:
            return [tf.keras.callbacks.EarlyStopping()]
        else:
            return []

    USE_LINEAR_ONLY_MODEL = False # True if want to use the model which was made assuming it only has access to linear features
    def MODEL_TO_USE(datarr):
        if DataParameters.USE_LINEAR_ONLY_MODEL:
            return DataParameters.PSOTrainable(
                [
                    tf.keras.layers.Dense(
                        units=6,
                        dtype=np.float64,
                        kernel_regularizer=tf.keras.regularizers.L2(l2=DataParameters.REGULARIZATION),
                        kernel_initializer=tf.keras.initializers.HeNormal()
                    ),
                    tf.keras.layers.ReLU(dtype=np.float64),
                    tf.keras.layers.Dense(
                        units=5,
                        dtype=np.float64,
                        kernel_regularizer=tf.keras.regularizers.L2(l2=DataParameters.REGULARIZATION),
                        kernel_initializer=tf.keras.initializers.HeNormal()
                    ),
                    tf.keras.layers.ReLU(dtype=np.float64),
                    tf.keras.layers.Dense(
                        units=4,
                        dtype=np.float64,
                        kernel_regularizer=tf.keras.regularizers.L2(l2=DataParameters.REGULARIZATION),
                        kernel_initializer=tf.keras.initializers.HeNormal()
                    ),
                    tf.keras.layers.ReLU(dtype=np.float64),
                    tf.keras.layers.Dense(
                        units=1,
                        dtype=np.float64,
                        kernel_regularizer=tf.keras.regularizers.L2(l2=DataParameters.REGULARIZATION),
                        activation=DataParameters.FINAL_ACTIVATION,
                        kernel_initializer=tf.keras.initializers.HeNormal()
                    ) # Output layer, don't forget this!!
                ],
                datarr
            )
        else:
            return DataParameters.PSOTrainable(
                [
                    tf.keras.layers.Dense(
                        units=8,
                        dtype=np.float64,
                        kernel_regularizer=tf.keras.regularizers.L2(l2=DataParameters.REGULARIZATION),
                        kernel_initializer=tf.keras.initializers.HeNormal()
                    ),
                    tf.keras.layers.ReLU(dtype=np.float64), # ReLU necessary or it won't learn nonlinear stuff
                    tf.keras.layers.Dense(
                        units=1,
                        dtype=np.float64,
                        kernel_regularizer=tf.keras.regularizers.L2(l2=DataParameters.REGULARIZATION),
                        activation=DataParameters.FINAL_ACTIVATION,
                        kernel_initializer=tf.keras.initializers.HeNormal()
                    ) # Output layer, don't forget this!!
                ],
                datarr
            )

COLAB = False #True
OUTFILE = 'all_outputs.txt'
total_imgs = 0
DIREC = 'OUTPUTS/'
Path(DIREC).mkdir(parents=True, exist_ok=True)

def printout(*args):
    if COLAB:
        print(*args)
    else:
        print(f"[SAVED TO {OUTFILE}]", *args)
        to_save = ''.join(map(str,args)) + '\n'
        f = open(DIREC + OUTFILE, "a")
        f.write(to_save)
        f.close()

def showout(filename):
    global total_imgs
    total_imgs += 1
    # Make it always save as pdf
    if '.' in filename:
        filename = '.'.join(filename.split('.')[:-1]) + '.pdf'
    else:
        filename += '.pdf'
    if COLAB:
        plt.show()
    else:
        printout(f'Saved Figure as "[{total_imgs}]{filename}"')
        plt.savefig(DIREC + f'[{total_imgs}]{filename}')
        plt.show(block=False)
        plt.clf()