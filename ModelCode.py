import tensorflow as tf
import functools
import operator
import numpy as np

# I'm not sure which network we'll use, so I'm gonna make the model code be general
class PSOTrainable():
    """
    This is a wrapper for a Keras model to make it easy for us to generalize
    the PSO-training to arbitrary models.
    """
    def __init__(self, layerlist, datarr):
        self.model = tf.keras.Sequential(layerlist)
        self.model.compile(loss='mean_absolute_error')
        self.model(datarr)

    def summary(self):
        return self.model.summary()

    def get_weight_count(self):
        to_return = 0
        for layer in self.model.layers:
            # Weights gives a list of shape=(N,M,K,...) arrays, so formula is
            # "∑∏shape"  to calcluate total weights in layer
            to_return += sum([
                functools.reduce(operator.mul, w.shape, 1) for w in layer.weights
            ])
        return to_return

    def set_weights(self, w):
        assert len(w) == self.get_weight_count(), "Wrong dimensionality input!"
        for layer in self.model.layers:
            weights_to_set = []
            for weightarr in layer.weights:
                total_to_take = functools.reduce(operator.mul, weightarr.shape, 1)
                taken, w = np.split(w, [total_to_take])
                taken = taken.reshape(weightarr.shape)
                weights_to_set.append(taken)
            layer.set_weights(weights_to_set)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)