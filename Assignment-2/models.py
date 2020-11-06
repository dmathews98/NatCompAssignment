from params import *
from nn import *
from ga import *
from pso import *
from gp import *
from make_data import *

sgdcount = 0
def sgd(**kwargs):
    global sgdcount
    sgdcount += 1
    printout(
        "ARGS: ",
        kwargs
    )

    fitness, evaluate_func, dimensions, nn = prepare_neural_net(
        DataParameters.Q,
        traindata,
        trainlab,
        datarr,
        labarr
    )

    # This performance seems highly variable, usually appearing in the 0.8-0.9 range,
    # but can go much better if lucky.
    printout(f"Gradient Descent {sgdcount} Training for {kwargs['time_steps']} Epochs with {kwargs['population_size']} per Batch")

    scores = []
    best_net = None
    for i in range(int(kwargs['averaging'])):
        nn.set_weights(np.random.uniform(-1, 1, nn.get_weight_count()))
        nn.model.fit(
            x=traindata,
            y=trainlab,
            epochs=kwargs['time_steps'],
            verbose=0,
            batch_size=kwargs['population_size']
        )
        score = nn.evaluate(testdata, testlab)
        scores.append(score)
        if best_net is None:
            best_net = nn
        else:
            if score < min(scores):
                best_net = nn
        print(f"Model {i}: {score}")
    scores = np.array(scores)
    printout(f"Gradient Descent Over!  Avg Score: {scores.mean()} Deviation: {scores.std()} Best: {min(scores)} Worst: {max(scores)}")
    plt.plot(scores)
    showout(f'sgd_gradient_descent{sgdcount}_pop{kwargs["population_size"]}_time{kwargs["time_steps"]}_avg{kwargs["averaging"]}.png') #plt.show()
    plot_data(testdata, testlab, best_net, verbose=False, plotname="sgd_test_result{sgdcount}.png")

psocount = 0
def pso(**kwargs):
    global psocount
    psocount += 1
    printout(
        "ARGS: ",
        kwargs
    )

    fitness, evaluate_func, dimensions, nn = prepare_neural_net(
        DataParameters.Q,
        traindata,
        trainlab,
        datarr,
        labarr
    )
    
    printout(f"PSO {psocount} Training for {kwargs['time_steps']} Epochs with Population {kwargs['population_size']}")
    swarm, best = PSO(
        dim=dimensions,
        fitness_func=fitness,
        **kwargs,
        verbose=True
    ).run()
    nn.set_weights(best/DataParameters.SCALE)
    #print(nn.evaluate(testdata, testlab))
    printout(f"PSO Training Over!  Score: {nn.evaluate(testdata, testlab)}")
    
    plot_data(testdata, testlab, nn, verbose=False, plotname=f'pso{psocount}.png')

gacount = 0
def ga(**kwargs):
    global gacount
    gacount += 1
    printout(
        "ARGS: ",
        kwargs
    )

    printout(f"GA {gacount} Training for {kwargs['time_steps']} Generations with Population {kwargs['population_size']}")
    testga = GA(
        population_size=kwargs['population_size'],
        mutation_rate=kwargs['mutation_rate'],
        crossover_rate=kwargs['crossover_rate']
    )
    best, over_time = testga.run(
        traindata,
        trainlab,
        testdata,
        testlab,
        train_epochs=kwargs['train_epochs'],
        test_epochs=kwargs['test_epochs'],
        batch=kwargs['population_size'],
        generations=kwargs['time_steps']
    )
    #print(f"Best Model (Loss; {best[0]}): ")
    nn = testga.genotype_to_neural_net(best[1], datarr)
    nn.summary()
    printout(
        f"GA Best: ",
        testga.evaluate(
            traindata,
            trainlab,
            testdata,
            testlab,
            epochs=kwargs['test_epochs'],
            batch=kwargs['batch'],
            seed=1721204
        )
    )
    #print("GA Training Over!")
    #print("Plot of (Training) Accuracies over Time")
    plt.plot(np.array(over_time))
    #plt.show()
    showout('plot_of_ga_accuracies_over_time.png')
    
    plot_data(testdata, testlab, nn, verbose=False, plotname=f'ga{gacount}.png')

"""

NOTE: The grand majority of the GP evolution code is a carbon copy
of the GA evolution code.  In a perfect world, I would refactor this to have
them both inherit from a parent class that contains these functions.
However I've spent a lot of time on this code and need to focus attentions
on other courses soon, so I don't have time to do this (sorry)

"""

class GPTree():
    activations = ['ReLU', 'Sigmoid', 'Swish', 'Linear']

    def __init__(self, label: str, info: int=1, children: typing.List['GPTree']=[]):
        """
        Info specifies how many nodes in a layer, for example
        It's just additional specification beyond the label
        """
        self.label = label
        self.children = children
        self.info = info
        self.parent = None
        for child in children:
            child.parent = self

    def __eq__(self, other: 'GPTree') -> 'bool':
        self_childs = self.get_family_tree()
        other_childs = other.get_family_tree()
        for a, b in zip(self_childs, other_childs):
            eq = a.label == b.label and a.info == b.info
            if not eq:
                return False
        return True

    def add_child(self, child: 'GPTree'):
        self.children.append(child)
        child.parent = self

    def add_children(self, children: typing.List['GPTree']):
        self.children.extend(children)
        for child in children:
            child.parent = self

    def get_family_tree(self):
        """
        Compresses all children into one list
        INCLUDES SELF
        """
        return sum([child.get_family_tree() for child in self.children], [self])

    def get_random_child(self):
        # INCLUDES SELF
        return np.random.choice(self.get_family_tree())

    def mutate(self, flip_chance: float):
        to_mutate = self.get_random_child()
        if to_mutate.label == 'Input':
            # Don't mutate inputs
            return
        # Will mutate.
        # We need to decide now whether to flip between Activation type
        # (i.e. Sigmoid/ReLU) or whether to change the amount of nodes
        # in a layer
        if np.random.uniform(0, 1) < flip_chance:
            to_mutate.label = np.random.choice(GPTree.activations)
        else:
            to_mutate.info = np.random.randint(1, 11)


    def whither(self):
        to_whither = self.get_random_child()

        # Will whither
        # This just means we turn it into an input node
        to_whither.label = 'Input'
        to_whither.children = []

    def grow(self):
        to_grow = self.get_random_child()
        
        # Don't grow if not a leaf!
        if to_grow.label != 'Input':
            return

        # Will grow
        to_grow.label = np.random.choice(GPTree.activations)
        to_grow.info = np.random.randint(1, 11)
        to_grow.children = [GPTree('Input'), GPTree('Input')]

    def copy(self) -> 'GPTree':
        return GPTree(
            self.label,
            self.info,
            children = [x.copy() for x in self.children]
        )

    def __str__(self) -> 'str':
        if self.label == 'Input':
            return 'Input'
        return f"{self.label}[{self.info}]: ({','.join(map(str, self.children))})"
        

class GP():
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        crossover_rate: float,
        flip_chance: float,
        whither_rate: float,
        growth_rate: float
    ):
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.flip_chance = flip_chance
        self.population = [self.starting_organism() for i in range(self.pop_size)]
        self.best = self.population[0]
        self.whither_rate = whither_rate
        self.growth_rate = growth_rate

    def starting_organism(self):
        """
        Get starting organism to seed the population with
        """
        to_return = GPTree(
            label='ReLU',
            info=10,
            children=[
                GPTree('Input'),
                GPTree('Input')
            ]
        )
        to_return.grow()
        return to_return
    
    def genotype_to_neural_net(self, genotype: GPTree, datarr):
        """
        Genotype Specification
        Inspired by image on this site: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2952963/

        Basically, we have trees look like this:
             A
            / \
            A  A
            /\ /\
            AI II
           / \
           I I
        Where A is activation function (i.e. ReLU), D is a Dense layer of size 1
        W is a weight, which is represented by a number (perhaps a number tree)

        Labels will look like A:ReLU, A:Sigmoid, D:Dense, W:Weight for example
        """
        #print(genotype)
        datin = tf.keras.Input(shape=datarr.shape)
        def recurse(geno):
            label = geno.label
            if label in GPTree.activations:
                inputs = tf.keras.layers.Concatenate()(list(map(recurse, geno.children)))
                layer = tf.keras.layers.Dense(geno.info)(inputs)
                if label == 'ReLU':
                    activation = tf.keras.layers.ReLU(dtype=np.float64)(layer)
                elif label == 'Sigmoid':
                    activation = tf.keras.layers.Activation('sigmoid', dtype=np.float64)(layer)
                elif label == 'Swish':
                    activation = tf.keras.layers.Activation('swish', dtype=np.float64)(layer)
                elif label == 'Linear':
                    activation = tf.keras.layers.Activation('linear', dtype=np.float64)(layer)
                return activation
            elif label == 'Input':
                return tf.keras.layers.Dense(datarr.shape[1])(datin)
            else:
                assert False, f"'{label}' is not a valid layer in GP!"

        body = recurse(genotype)
        out = tf.keras.layers.Dense(1)(body)
        to_return = tf.keras.Model(inputs=datin, outputs=out)
        return to_return

    def evaluate(self, traindata, trainlab, testdata, testlab, epochs, batch, seed=None):
        if seed is not None:
            tf.random.set_seed(int(seed)) # So that we get consistent results
        nn = self.genotype_to_neural_net(self.best, traindata)
        nn.compile(loss='mean_squared_error')
        nn.fit(
            x=traindata,
            y=trainlab,
            epochs=epochs,
            verbose=0,
            batch_size=batch
        )
        return nn.evaluate(testdata, testlab)

    def run(self, traindata, trainlab, testdata, testlab, train_epochs, test_epochs, batch, generations, verbose=True):
        performances_over_time = []
        def fitness_func(genotype, epochs=train_epochs):
            tf.random.set_seed(1721204) # So that we get consistent results
            nn = self.genotype_to_neural_net(genotype, traindata)
            nn.compile(loss='mean_squared_error')
            nn.fit(
                x=traindata,
                y=trainlab,
                epochs=epochs,
                verbose=0,
                batch_size=batch
            )
            return nn.evaluate(testdata, testlab, verbose=0)
        for i in tf.range(generations):
            self.population, best_fitness = self.run_loop(fitness_func)
            performances_over_time.append(best_fitness)
            if verbose and (generations <= 20 or i%(generations//20) == 0 or i == generations - 1):
                print(f"{i+1}/{generations}: Best Fitness {best_fitness}")

        # Return the best performing network
        fitness_list = sorted([
            (fitness_func(org, epochs=test_epochs), org) for org in self.population
        ], key=lambda x: x[0])
        self.best = fitness_list[0][1]
        return fitness_list[0], performances_over_time

    def run_loop(self, fitness_func):
        fitness_list = sorted([
            (fitness_func(org), org) for org in self.population
        ], key=lambda x: x[0])
        fits, orgs = list(zip(*fitness_list))
        fits = np.array(fits)
        mean_fitness = sum(fits) / self.pop_size
        normalized_fitness = fits / mean_fitness
        absolute_best_organism = orgs[0]

        # Remainder Stochastic Sampling
        guaranteed_survivors = [int(x) for x in fits / mean_fitness]
        total_survivors = sum(guaranteed_survivors)
        total_dreamers = self.pop_size - total_survivors
        chances = np.array(normalized_fitness - guaranteed_survivors)
        alive_guaranteeds = sum([
            quantity * [org] for quantity, org
            in zip(guaranteed_survivors, orgs)
        ], [])

        # Annoyingly np.random.choice doesn't work for lists of tuples
        # so we create a dummy class T to hold the values.
        class T():
            def __init__(self, f, o):
                self.fit = f
                self.org = o
        choose_from = np.array([T(f, o) for f, o in zip(fits, orgs)])
        try:
            alive_dreamers = [
                t.org for t in
                np.random.choice(
                    choose_from,
                    total_dreamers,
                    p=chances / sum(chances)
                )
            ]
        except:
            # Once had a bug where this crashed;
            # not sure what was the cause so I've left this here in case
            # it ever happens again (seems to be rare)
            # To cope, we just grab the first few dreamers
            alive_dreamers = [x.org for x in choose_from[:total_dreamers]]

        all_alive = alive_dreamers + alive_guaranteeds
        assert len(all_alive) == self.pop_size, "Population Changed!"

        # Now crossover time
        # We do crossover by swapping two layer bitstrings as that is
        # actually meaningful for our problem.
        pairs = list(itertools.combinations(all_alive, 2))
        np.random.shuffle(pairs)
        crossovered = []
        best_org_found = False
        for idx, pair in enumerate(pairs[:self.pop_size//2]):
            a, b = pair
            if idx == self.pop_size//2 - 1 and not best_org_found:
                # We need to ensure best organism is present
                a_out = absolute_best_organism.copy()
                b_out = b.copy()
            elif (a == absolute_best_organism or b == absolute_best_organism) and not best_org_found:
                a_out = a.copy()
                b_out = b.copy()
                best_org_found = True
            elif np.random.uniform(0, 1) < self.crossover_rate:
                # Do crossover!
                # We choose a random node and then swap it with another!
                # Carrying all children with it.
                a_out = a.copy()
                b_out = b.copy()
                crosspoint_a = a_out.get_random_child()
                crosspoint_b = b_out.get_random_child()
                a_daddy = crosspoint_a.parent
                b_mommy = crosspoint_b.parent
                if a_daddy is not None:
                    a_daddy.children.remove(crosspoint_a)
                    a_daddy.add_child(crosspoint_b)
                else:
                    b_out = crosspoint_b.copy()
                if b_mommy is not None:
                    b_mommy.children.remove(crosspoint_b)
                    b_mommy.add_child(crosspoint_a)
                else:
                    a_out = crosspoint_a.copy()
            else:
                a_out = a.copy()
                b_out = b.copy()
            crossovered.extend([a_out, b_out])



        # Now mutation time
        # We have the option to only mutate alive_dreamers, but we've chosen
        # to mutate everyone!
        # We preserve the absolute best organism.

        mutated = []
        best_seen = False
        for x in crossovered:
            if x == absolute_best_organism and not best_seen:
                mutated.append(x)
                best_seen = True
            else:
                if np.random.uniform(0, 1) < self.mutation_rate:
                    x.mutate(self.flip_chance)
                if np.random.uniform(0, 1) < self.growth_rate:
                    x.grow()
                if np.random.uniform(0, 1) < self.whither_rate:
                    x.whither()
                mutated.append(x)

        assert len(mutated) == self.pop_size, "Population Changed!"
        return mutated, fits[0]
                


