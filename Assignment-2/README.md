# NatCompAssignment

We test 4 different algorithms; SGD, PSO, GA, and GP.

SGD is a baseline, the rest correspond to the four parts of the coursework.

## SGD

### Parameters:

`population_size`: This is the 'batch size' in the gradient descent algorithm (called population size to emphasize parallels with PSO population size)

`time_steps`: How long to run

`averaging`: How many to run (because SGD is cheap, we can run it multiple times and get mean and std dev of the distribution of performances)

## PSO

### Parameters

`total_a`:  Sum of both alpha coefficients (1 and 2, not 3)

`a1_percent`: Percent of `total_a` that is allocated to a1, rest goes to a2.

`epsion`: How "deep into the complexity zone" we want to be.  (We calculate what omega should be by placing it on the curve between real/complex eigenvalues of PSO, and then venture into this region by `epsilon`; `epsilon` should be **negative** to go inside complex region, and positive to go inside noncomplex region)

`population_size`: How many particles to use

`time_steps`: How long to run for`

`a3`: Coefficient of the repulsion term.  We're not _really_ investigating this, but it might be nice to look at if we have spare time (for example, if we've found what we think is the 'best' PSO setup, can messing with `a3` improve it further?)

## GA

### Parameters

`population_size`: How many organisms to use

`time_steps`: How long to run

`mutation_rate`: Percent chance to flip a bit of an organism; can trigger multiple times for the same organism

`crossover_rate`: Percent chance for a pair of organisms to crossover

`train_epochs`: During the running of the algorithm, how many epochs do we train an organism for (using SGD) to get the fitness?

`test_epochs`: To get our final result, after we've identified the best organism, how many epochs do we train the best organism for.

`batch`: The batch size for the SGD trainer that trains each individual organism.

### Operators

Mutate: Flip a bit of an organism.  Each bit has an identical independant chance to flip.

Crossover: Rather than arbitrary crossover, this specifically swaps one "layer bitstring" between two organisms.  A "layer bitstring" is a string of bits that corresponds precisely to one layer; the exact shape of this can be controlled by messing with the parameters of the `DataParameters` class (either in `params.py` or through the scripting language described at the bottom of this ReadMe).

## GP

### Parameters

`population_size`: How many organisms

`time_steps`: How many generations

`mutation_rate`: Chance of a mutation.  Unlike in GA, only one mutation can happen per organism per generation.

`crossover_rate`: Chance of a crossover.

`flip_chance`: There are three types of mutations; it can either change the activation function of a layer, the amount of neurons in a layer, or the initialization function of a layer.  This is the chance of the first type (and the other two happen with equal probability)

`whither_rate`: Chance that a random node in the tree will whither away to an empty node (and loose all its children).  Note that this will never happen to the root node, and it does nothing if it selects a leaf node (no children to whither away)

`growth_rate`: Chance that a random node in the tree will grow a new child.  Note that if the random node selected already has two children, nothing will happen.

`train_epochs`: Same as in GA

`test_epochs`: Same as in GA

`batch`: Same as in GA

### Operators

Mutations: (mutually exclusive, at most one per organism)
  * Change activation function
  * Change amount of neurons in a layer (between 1 and 10)
  * Change weight initialization function ("He Normal", Uniform, Normal, All Zeroes)

Crossover: Cut off a branch of each parent and swap them

Grow: Choose a random node, and, if it is a leaf node, grow a new child (corresponds to adding a new layer downstream from chosen node).

Whither: Choose a random node, and, if it is not a leaf node or the root, whither away all descendents of chosen node (i.e. remove all downstream layers from chosen node)

## Additional Scripts

### `tree_utility.py`

Simulates a GP, but with no fitness selection, and plots the depth of the tree over time.  This allows us to understand the natural long-term behavior of the whither/growth parameters we selected, so that we can identify whether our GP results follow this trend, or if the fitness function induces some trend towards a different long-term tree depth.

## Using the Rudimentary Batch Process Language

Check `sanity_check.txt` for an example of what I use to make sure I haven't introduced a bug before pushing.  A typical line in the file will look like:

```
ALG; param1:X####, param2: X#####, param3: X#####, ...
```

i.e.:

```
SGD; population_size:I2, time_steps:I2, averaging:I2
```

The `ALG` is a specifier for what algorithm to use (SGD, PSO, GA, GP), the params correspond to the parameters listed out at the top of this ReadMe, and the 'X' correspond to an identifier specifying the datatype ('I' for int, 'F' for float, 'B' for bool)

Additionally, one can change the parameters listed in `params.py` away from their defaults (this will cause the dataset to regenerate, as some of the parameters control aspects of the data - note that this will affect all lines below the change!  Changing these parameters are permanent (although you can always manually change them back by adding another parameter change line)).  These lines are of the form:

```
DATA_PARAMETER; param:X####
```

i.e.

```
DATA_PARAMETER; USE_LINEAR_ONLY_MODEL:BTrue
```

Some interesting parameters like this that we might want to mess around with:

* `NOISE`: controls how noisy the dataset is
* `USE_LINEAR_ONLY_MODEL`: controls whether or not we allow the quadratic and sinusoidal features for learning; if False, we use a different model for SGD and PSO.
* `USE_EARLY_STOPPING`: controls whether our SGD algorithm tries to avoid overfitting by using an early stopping technique.
* `s`: How many datapoints to use.
* `SCALE`: What is the scale of the weights that we use in PSO training? (affects how influential one timestep of PSO is, as if the scale is small then one discrete motion is a relatively larger change in weights)
* `REGULARIZATION`: What is the regularization constant that we use for training?

There are also some more advanced parameters that control the exact format of the GA genotype, as well as the loss function that we use (by default we use binary crossentropy).
