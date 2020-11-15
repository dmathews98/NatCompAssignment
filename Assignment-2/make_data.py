from params import *
from nn import *

def generate_data_set(M, s, noise_scale):
    if DataParameters.Q == 1:
        noise = 2.0*noise_scale
        x1_mu = np.array([2 for i in range(M)])
        x_1_mu = np.array([-2 for i in range(M)])

        x1_set = np.random.multivariate_normal(x1_mu, noise*np.identity(M), size=s//2)
        x_1_set = np.random.multivariate_normal(x_1_mu, noise*np.identity(M), size=s//2)
    elif DataParameters.Q == 3:
        def gen_spiral(deltaT):
            n = s//2
            to_return = []
            for i in range(n):
                r = i / n * 5
                t = 1.75 * i / n * 2 * math.pi + deltaT
                x = (r * math.sin(t)) + np.random.uniform(-1, 1) * noise_scale
                y = (r * math.cos(t)) + np.random.uniform(-1, 1) * noise_scale
                to_return.append([x, y])
            return np.array(to_return)

        x1_set = gen_spiral(0).reshape((s//2, M))
        x_1_set = gen_spiral(math.pi).reshape((s//2, M))

        # Insert x^2, y^2, sin(x), and sin(y) features.
        if not DataParameters.USE_LINEAR_ONLY_MODEL:
            x1_set = np.hstack([x1_set, x1_set**2, np.sin(x1_set)])
            x_1_set = np.hstack([x_1_set, x_1_set**2, np.sin(x_1_set)])

    full_data = np.hstack([
        np.vstack([x1_set, x_1_set]),
        np.vstack([
            DataParameters.L1 * np.ones((s//2, 1)),
            DataParameters.L2 * np.ones((s//2, 1))
        ])
    ])
    np.random.shuffle(full_data)

    # Returns data, label pair
    return full_data[:, :-1], full_data[:, -1]

plotnum = 0
def plot_data(data, labels, nn=None, gridsize=30, verbose=True, plotname=None):
    """
    If nn is None, plot data
    If nn is not None, we plot the decision boundary
    of the NN with the best weights
    """
    # [:2] necessary so that we don't plot quadratic features we engineered
    rplot = np.array([
        x[:2] for x, y in zip(data, labels) if y == DataParameters.L1
    ])
    bplot = np.array([
        x[:2] for x, y in zip(data, labels) if y == DataParameters.L2
    ])

    plt.xlim(-5, 5)
    if nn is not None:
        a = np.linspace(-5, 5, gridsize)
        def contour_helper(pos):
            # **2 is for quadratic features we engineered
            if DataParameters.USE_LINEAR_ONLY_MODEL:
                to_return = nn.predict(pos.reshape(1, 2))
            else:
                to_return = nn.predict(np.hstack([pos.reshape(1, 2), pos.reshape(1, 2)**2, np.sin(pos.reshape(1, 2))]))
            def discretizer(x):
                return DataParameters.L1 if x > (DataParameters.L1+DataParameters.L2)/2 else DataParameters.L2
            return discretizer(to_return[0, 0]) if DataParameters.MAKE_DISCRETE_PLOT else to_return[0, 0]
        zz = np.zeros((gridsize, gridsize), dtype=np.int8)
        for x in range(gridsize):
            if verbose: print("Drawing %d%% complete!" % int(100 * x / gridsize))
            for y in range(gridsize):
                zz[y, x] = (contour_helper(np.array([a[x], a[y]])))
        plt.contourf(a, a, zz, cmap=plt.cm.summer)

        if DataParameters.USE_LINEAR_ONLY_MODEL:
            rplot_pred = nn.predict(rplot)
        else:
            rplot_pred = nn.predict(np.hstack([rplot, rplot**2, np.sin(rplot)]))
        rplot_correct = np.array([x for x, y in zip(rplot, rplot_pred) if y > (DataParameters.L1+DataParameters.L2)/2])
        rplot_incorrect = np.array([x for x, y in zip(rplot, rplot_pred) if y <= (DataParameters.L1+DataParameters.L2)/2])

        if DataParameters.USE_LINEAR_ONLY_MODEL:
            bplot_pred = nn.predict(bplot)
        else:
            bplot_pred = nn.predict(np.hstack([bplot, bplot**2, np.sin(bplot)]))
        bplot_correct = np.array([x for x, y in zip(bplot, bplot_pred) if y <= (DataParameters.L1+DataParameters.L2)/2])
        bplot_incorrect = np.array([x for x, y in zip(bplot, bplot_pred) if y > (DataParameters.L1+DataParameters.L2)/2])
        if rplot_correct.shape[0] > 0:
            plt.plot(rplot_correct[:, 0], rplot_correct[:, 1], 'ro')
        if bplot_correct.shape[0] > 0:
            plt.plot(bplot_correct[:, 0], bplot_correct[:, 1], 'bo')
        if rplot_incorrect.shape[0] > 0:
            plt.plot(rplot_incorrect[:, 0], rplot_incorrect[:, 1], 'rx')
        if bplot_incorrect.shape[0] > 0:
            plt.plot(bplot_incorrect[:, 0], bplot_incorrect[:, 1], 'bx')
    else:
        plt.plot(rplot[:, 0], rplot[:, 1], 'r.')
        plt.plot(bplot[:, 0], bplot[:, 1], 'b.')

    if plotname is None:
        showout(f'plot_{plotnum}.png')#plt.show()
        plotnum+=1
    else:
        showout(plotname)

# We REALLY need the [-1, 1] constraint, otherwise
# the weights rapidly blow up
def cap(x, bound=1):
    if x < -bound: return -bound
    if x > bound: return bound
    return x

def mapcap(x, bound=1):
    return np.array([cap(y, bound=bound) for y in x])

def prepare_neural_net(q, traindata, trainlab, datarr, labarr):
    # q is outdated parameter whuch was used in CW1
    # just set it to 3.
    if q == 1:
        nn = PSOTrainable(
            [tf.keras.layers.Dense(units=1, dtype=np.float64)],
            datarr
        )
    elif q == 3:
        # Here we make the model
        nn = DataParameters.MODEL_TO_USE(datarr)
    nn.summary()

    # Sanity check, will throw error if weight calculation fails:
    #nn.set_weights(np.array(list(range(nn.get_weight_count()))))
    def fitness_f(pos:typing.List[float]):
        nn.set_weights(pos/DataParameters.SCALE)
        mae = nn.evaluate(
            x=traindata,
            y=trainlab,
            verbose=0
        )
        return mae[0]
    def evaluate_f(pos:typing.List[float]):
        nn.set_weights(pos/DataParameters.SCALE)
        mae = nn.evaluate(
            x=datarr,
            y=labarr,
            verbose=0
        )
        return mae[0]
    return fitness_f, evaluate_f, nn.get_weight_count(), nn


datarr, labarr, traindata, trainlab, testdata, testlab = None, None, None, None, None, None

def make_data():
    global datarr, labarr, traindata, trainlab, testdata, testlab
    np.random.seed(1721204)

    datarr, labarr = generate_data_set(
        M=DataParameters.M,
        s=DataParameters.s,
        noise_scale=DataParameters.NOISE
    )
    print('full data')
    plot_data(datarr, labarr, plotname='full_data_plot.png')

    split_pos = int(0.5 * datarr.shape[0])
    traindata = datarr[:split_pos, :]
    trainlab = labarr[:split_pos]
    testdata = datarr[split_pos:, :]
    testlab = labarr[split_pos:]

    print('train data')
    plot_data(traindata, trainlab, plotname='train_data_plot.png')

    print('test data')
    plot_data(testdata, testlab, plotname='test_data_plot.png')

make_data()