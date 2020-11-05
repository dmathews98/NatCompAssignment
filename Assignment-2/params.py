import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import itertools
import tensorflow as tf
import math
import functools
import operator
from pathlib import Path
tf.get_logger().setLevel('ERROR')

class DataParameters:
    M = 2 # Dimension
    s = 500 # Data quantity
    Q = 3 # Question #, 1 or 3
    # Note that Assignment 2 is basically just Q3 datawise.

    NOISE = 0

    L1 = 1 # Label value for first class
    L2 = -1 # Label value for second class

    SCALE = 1000 # Amount we scale weights down by
    REGULARIZATION = 0.003
    GA_GENO_SIZE = 40 # How many total bits per organism
    GA_DUPLI_SIZE = 1 # How many bits are dedicated to duplication counter per organism
    GA_LAYER_SIZE = 4 # How many bits are dedicated to neurons in a layer per organism

    assert GA_GENO_SIZE / (GA_DUPLI_SIZE + GA_LAYER_SIZE) == GA_GENO_SIZE // (GA_DUPLI_SIZE + GA_LAYER_SIZE),\
        "Genone size not a multiple of the bits per organism!"

COLAB = False #True
OUTFILE = 'all_outputs.txt'
total_imgs = 0
DIREC = 'OUTPUTS'
Path(DIREC).mkdir(parents=True, exist_ok=True)

def printout(*args):
    if COLAB:
        print(*args)
    else:
        print(f"[SAVED TO {OUTFILE}]", *args)
        to_save = ''.join(map(str,args)) + '\n'
        f = open(DIREC + '/' + OUTFILE, "a")
        f.write(to_save)
        f.close()

def showout(filename):
    global total_imgs
    total_imgs += 1
    if COLAB:
        plt.show()
    else:
        printout(f'Saved Figure as "[{total_imgs}]{filename}"')
        plt.savefig(DIREC + '/' + f'[{total_imgs}]{filename}')
        plt.show(block=False)