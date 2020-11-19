from params import *
from nn import *

class GA():
    def __init__(self, *, population_size, mutation_rate, crossover_rate):
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = [self.random_organism() for i in range(self.pop_size)]
        self.best = self.population[0]

    def random_organism(self):
        """
        20 bit string, so choose number from 0 to 2^20-1
        """

        # Choose whether to do a random start, or start with a single neuron
        main_bits = bin(np.random.randint(0, 2**DataParameters.GA_GENO_SIZE()))[2:]

        # Might have cut off leading zeroes:
        leading_zeroes = ''.join((DataParameters.GA_GENO_SIZE()-len(main_bits)) * ['0'])
        to_return = leading_zeroes + main_bits
        assert len(to_return) == DataParameters.GA_GENO_SIZE(), "Something went wrong in GA random generation"
        return to_return

    def genotype_to_phenotype(self, genotype):
        """
        Genotype is 20 bit string, split into 5 bit segments specifying
        layers.  The first two bits being how many copies of the layer
        (interpreted as binary, 0 to 3).  The last three bits represent
        how many neurons in the layer (0 to 7, but we'll add 1 because
        the '0 neurons' case is already covere by first two bits being
        zero, so instead its 1 to 8)

        Genotype of form `str('***** ***** ***** *****')`
        (spaces added for readability but are not actually present)
        """
        def splitit(s):
            lay_size = DataParameters.GA_BITS_PER_LAYER()
            bound = DataParameters.GA_LAYER_AMOUNT
            for i in range(0, bound):
                to_yield = s[lay_size * i : lay_size * (i + 1)]
                yield ((
                    int(to_yield[0:DataParameters.GA_DUPLI_SIZE], 2),
                    int(to_yield[
                        DataParameters.GA_DUPLI_SIZE:-DataParameters.GA_INITIALIZER_SIZE
                    ], 2) + 1
                ), int(to_yield[-DataParameters.GA_INITIALIZER_SIZE:], 2))
        splits = list(splitit(genotype))
        return zip(
            sum( (tup[0] * [tup[1]] for tup, initializer in splits), []),
            [DataParameters.DECODE_INITIALIZER(initializer) for tup, initializer in splits]
        )

    def phenotype_to_neural_net(self, phenotype, datarr):
        nn = PSOTrainable(
            sum(
                ([
                    tf.keras.layers.Dense(
                        units=layer_size,
                        dtype=np.float64,
                        kernel_regularizer=tf.keras.regularizers.L2(
                            l2=DataParameters.REGULARIZATION
                        ),
                        kernel_initializer=initializer
                    ),
                    tf.keras.layers.ReLU(dtype=np.float64)
                ] for layer_size, initializer in phenotype),
                []
            ) + [
                tf.keras.layers.Dense(
                    units=1,
                    dtype=np.float64,
                    kernel_regularizer=tf.keras.regularizers.L2(
                        l2=DataParameters.REGULARIZATION
                    ),
                    activation=DataParameters.FINAL_ACTIVATION,
                    kernel_initializer=tf.keras.initializers.HeNormal()
                )
            ], # Output layer, don't forget this!!,
            datarr
        )
        return nn

    def genotype_to_neural_net(self, genotype, datarr):
        return self.phenotype_to_neural_net(
            self.genotype_to_phenotype(genotype),
            datarr
        )

    def evaluate(self, traindata, trainlab, testdata, testlab, epochs, batch, seed=None):
        if seed is not None:
            tf.random.set_seed(int(seed)) # So that we get consistent results
        nn = self.genotype_to_neural_net(self.best, traindata)
        nn.model.fit(
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
            nn.model.fit(
                x=traindata,
                y=trainlab,
                epochs=epochs,
                verbose=0,
                batch_size=batch,
                callbacks=DataParameters.EARLY_STOPPING()
            )
            return nn.evaluate(testdata, testlab, verbose=0)
        for i in tf.range(generations):
            self.population, best_fitness = self.run_loop(fitness_func)
            performances_over_time.append(best_fitness)
            if verbose and (generations <= 20 or i%(generations//20) == 0 or i == generations - 1):
                print(f"{i+1}/{generations}: Best Fitness {best_fitness}")

        # Return the best performing network
        fitness_list = sorted([
            (fitness_func(org, epochs=train_epochs)[0], org) for org in self.population
        ], key=lambda x: x[0])
        self.best = fitness_list[0][1]
        return (fitness_func(self.best, epochs=test_epochs)[0], self.best), performances_over_time

    def run_loop(self, fitness_func):
        fitness_list = sorted([
            (fitness_func(org)[0], org) for org in self.population
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
                a_out = absolute_best_organism
                b_out = b
            elif a == absolute_best_organism or b == absolute_best_organism:
                a_out = a
                b_out = b
                best_org_found = True
            elif np.random.uniform(0, 1) < self.crossover_rate:
                # Do crossover!

                # Random point crossover:
                #crossover_point = np.random.randint(1, 20)
                #a_out = a[:crossover_point] + b[crossover_point:]
                #b_out = b[:crossover_point] + a[crossover_point:]

                # Layer specification crossover
                total_size = DataParameters.GA_BITS_PER_LAYER()
                cp = np.random.randint(1, DataParameters.GA_GENO_SIZE() // total_size)
                a_out = a[:cp] + b[cp:cp+total_size] + a[cp+total_size:]
                b_out = b[:cp] + a[cp:cp+total_size] + b[cp+total_size:]
            else:
                a_out = a
                b_out = b
            crossovered.extend([a_out, b_out])


        # Now mutation time
        # We have the option to only mutate alive_dreamers, but we've chosen
        # to mutate everyone!
        # We preserve the absolute best organism.
        def flip(b):
            if np.random.uniform(0, 1) > self.mutation_rate:
                return b
            return '1' if b == '0' else '0'
        mutated = [
            ''.join(
                flip(b) for b in x
            ) if x != absolute_best_organism
            else x
            for x in crossovered
        ]

        assert len(mutated) == self.pop_size, "Population Changed!"
        return mutated, fits[0]