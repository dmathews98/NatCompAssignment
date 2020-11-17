import numpy as np
import matplotlib.pyplot as plt

class Forest():
    def __init__(this, *, grow_rate, whither_rate, size):
        this.has_run = False
        this.grow_rate = grow_rate
        this.whither_rate = whither_rate
        class Tup():
            # To get around the 'must be one-dimensional' np.random.choice nuisance
            def __init__(self, contents):
                self.contents = contents
            def get(self, n):
                return self.contents[n]
        class Tree():
            GROWTH = grow_rate
            WHITHER = whither_rate
            def __init__(self, parent=None):
                self.parent = parent
                self.child_1 = None
                self.child_2 = None
            
            def get_family_tree(self):
                return sum(
                    [
                        child.get_family_tree() if child is not None else [Tup([None, self])]
                        for child in (self.child_1, self.child_2)
                    ],
                    [Tup([self, self])]
                )
            def get_depth(self):
                return max(
                    [
                        child.get_depth() if child is not None else 0
                        for child in (self.child_1, self.child_2)
                    ]
                ) + 1
            def get_random_descendant(self):
                return np.random.choice(self.get_family_tree())
            def grow(self):
                # Here we select a random descendant of our node,
                # and grow it iff that random descendant is `None`
                rand_child = self.get_random_descendant()
                if rand_child.get(0) is not None:
                    return
                to_grow = rand_child.get(1)
                if to_grow.child_1 is None:
                    to_grow.child_1 = Tree(parent=to_grow)
                else:
                    to_grow.child_2 = Tree(parent=to_grow)
            def whither(self):
                # Here we select a random descendant of our node,
                # and whither it iff that random descendant is not `None`
                rand_child = self.get_random_descendant()
                if rand_child.get(0) is None:
                    return
                if np.random.uniform(0, 1) < Tree.GROWTH:
                    to_grow = rand_child.get(1)
                    if to_grow.parent is not None:
                        # Never whither if it is the root
                        to_grow.child_1 = None
                        to_grow.child_2 = None
            def run(self):
                # The algorithm works by first checking if we should grow
                # and then checking if we should whither (both are allowed
                # to happen at the same time)
                if np.random.uniform(0, 1) < Tree.GROWTH:
                    self.grow()
                if np.random.uniform(0, 1) < Tree.WHITHER:
                    self.whither()
        this.contents = [Tree() for x in range(size)]

    def run_for(this, *, timesteps):
        assert not this.has_run, "Can only grow a forest once!"
        this.history = []
        for timestep in range(timesteps):
            for tree in this.contents:
                tree.run()
            this.history.append(this.get_statistics())
        this.history = np.array(this.history)

    def get_statistics(this):
        depths = np.array([x.get_depth() for x in this.contents])
        return np.array([depths.mean(), depths.std(), depths.max(), depths.min()])
        
    def plot_statistics(this, which):
        # `which` is the index into the statistics array
        to_plot = this.history[:, which]
        xs = [x+1 for x in range(len(this.history))]
        plt.plot(xs, to_plot, color='blue')
        A, B = np.polyfit(np.log(xs), to_plot, 1)
        ys = A*np.log(xs) + B
        plt.plot(xs, ys, color='orange')
        #asymptote = ys[ys.shape[0]//3:].mean()
        #plt.plot(xs, [asymptote for x in xs], color='green', linestyle='dashed')
        plt.title(f"Growth: {this.grow_rate}, Whither: {this.whither_rate}")
        plt.gca().set_ylim(bottom=1) # set lower y limit to be 1
        plt.show()

forest = Forest(grow_rate=0.1, whither_rate=0.1, size=100)
forest.run_for(timesteps=1000)
mean, std, maximum, minimum = forest.get_statistics()
print(f"Mean: {mean}, Std. Dev.: {std}, Max: {maximum}, Min: {minimum}")
forest.plot_statistics(0)