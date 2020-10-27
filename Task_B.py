from InstructorPSOCode import *
from ModelCode import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import itertools
import tensorflow as tf
import math

class DataParameters:
    M = 2 # Dimension
    s = 1000 # Data quantity
    Q = 3 # Question #, 1 or 3

def generate_data_set(M, s, noise_scale):
    if DataParameters.Q == 1:
        noise = 2.0*noise_scale
        x1_mu = np.array([2 for i in range(M)])
        x_1_mu = np.array([-2 for i in range(M)])

        x1_set = np.random.multivariate_normal(x1_mu, noise*np.identity(M), size=s//2)
        x_1_set = np.random.multivariate_normal(x_1_mu, noise*np.identity(M), size=s//2)
    elif DataParameters.Q == 3:
        noise = noise_scale / 4
        def gen_spiral(deltaT):
            n = s//2
            to_return = []
            for i in range(n):
                r = i / n * 5
                t = 1.75 * i / n * 2 * math.pi + deltaT
                x = (r * math.sin(t))
                y = (r * math.cos(t))
                to_return.append(
                    np.random.multivariate_normal(
                        np.array([x, y]),
                        noise * np.identity(M),
                        size=1
                    )
                )
            return np.array(to_return)

        x1_set = gen_spiral(0).reshape((s//2, M))
        x_1_set = gen_spiral(math.pi).reshape((s//2, M))

    full_data = np.hstack([
        np.vstack([x1_set, x_1_set]),
        np.vstack([np.ones((s//2, 1)), -np.ones((s//2, 1))])
    ])
    np.random.shuffle(full_data)

    # Returns data, label pair
    return full_data[:, :-1], full_data[:, -1]

def plot_data(data, labels, nn=None, gridsize=30):
    """
    If nn is None, plot data
    If nn is not None, we plot the decision boundary
    of the NN with the best weights
    """
    rplot = np.array([
        x for x, y in zip(data, labels) if y == 1
    ])
    bplot = np.array([
        x for x, y in zip(data, labels) if y == -1
    ])

    if nn is not None:
        a = np.linspace(-5, 5, gridsize)
        def contour_helper(pos):
            to_return = nn.predict(pos.reshape(1, 2))
            return 1 if to_return[0, 0] > 0 else -1
        zz = np.zeros((gridsize, gridsize), dtype=np.int8)
        for x in range(gridsize):
            print("Drawing %d%% complete!" % int(100 * x / gridsize))
            for y in range(gridsize):
                zz[x, y] = (contour_helper(np.array([a[x], a[y]])))
        plt.contourf(a, a, zz)

    plt.plot(rplot[:, 0], rplot[:, 1], 'r.')
    plt.plot(bplot[:, 0], bplot[:, 1], 'b.')

    plt.show()

# We REALLY need the [-1, 1] constraint, otherwise
# the weights rapidly blow up
def cap(x):
    if x < -1: return -1
    if x > 1: return 1
    return x

def mapcap(x):
    return np.array([cap(y) for y in x])

def prepare_neural_net(q, datarr, labarr):
    if q == 1:
        nn = PSOTrainable(
            [tf.keras.layers.Dense(units=1, dtype=np.float64)],
            datarr
        )
    elif q == 3:
        # Here we make the model
        nn = PSOTrainable(
            [
                tf.keras.layers.Dense(units=6, dtype=np.float64),
                tf.keras.layers.ReLU(dtype=np.float64), # ReLU necessary or it won't learn nonlinear stuff
                tf.keras.layers.Dense(units=6, dtype=np.float64),
                tf.keras.layers.ReLU(dtype=np.float64), # ReLU necessary or it won't learn nonlinear stuff
                tf.keras.layers.Dense(units=1, dtype=np.float64) # Output layer, don't forget this!!
            ],
            datarr
        )
    nn.summary()

    # Sanity check, will throw error if weight calculation fails:
    nn.set_weights(np.array(list(range(nn.get_weight_count()))))
    def fitness(pos):
        nn.set_weights(pos)
        mae = nn.evaluate(
            x=datarr,
            y=labarr,
            verbose=0
        )
        return mae
    return fitness, nn.get_weight_count(), nn

# Sigmoid
def output_func(w, x):
    return expit(w@x)

def fitness_func(pos, dim, x_set, s):
    c = 0
    for i in range(int(s/2.0)):
        c += np.abs(1.0 - output_func(pos, x_set[i].T))
    for i in range(int(s/2.0), s):
        c += np.abs(-1.0 - output_func(pos, x_set[i].T))

    c /= float(s)
    # print(c)

    return c


def main():
    np.random.seed(22)
    noise_scale = 0.5

    datarr, labarr = generate_data_set(
        M=DataParameters.M,
        s=DataParameters.s,
        noise_scale=noise_scale
    )
    plot_data(datarr, labarr)

    fitness, dimensions, nn = prepare_neural_net(
        DataParameters.Q,
        datarr,
        labarr
    )

    swarm, best = PSO(
        dim=dimensions,
        w=0.75,
        a1=2.02,
        a2=2.02,
        a3=0,
        population_size=5,
        time_steps=101,
        search_range=1.0,
        fitness_func=fitness,
        constrainer=mapcap
    ).run()
    nn.set_weights(best)
    plot_data(datarr, labarr, nn)

main()