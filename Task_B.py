import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import itertools

def generate_data_set(M, s, noise_scale):
    x1_mu = np.array([2 for i in range(M)])
    x_1_mu = np.array([-2 for i in range(M)])

    noise = 2.0*noise_scale

    x1_set = np.random.multivariate_normal(x1_mu, noise*np.identity(M), size=(int(s/2.0)))
    x_1_set = np.random.multivariate_normal(x_1_mu, noise*np.identity(M), size=(int(s/2.0)))

    return np.vstack([x1_set, x_1_set])

def plot_data(x_set, s):
    plt.plot(x_set[0:int(s/2.0), 0], x_set[0:int(s/2.0), 1], 'r.')
    plt.plot(x_set[int(s/2.0):s, 0], x_set[int(s/2.0):s, 1], 'b.')
    plt.show()

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

class Particle: # all the material that is relavant at the level of the individual particles
    
    def __init__(self, dim, minx, maxx, x_set, s):
        self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        self.velocity = np.random.uniform(low=-0.01, high=0.01, size=dim)
        self.best_particle_pos = self.position
        self.dim = dim
        self.x_set = x_set
        self.s = s

        self.fitness = fitness_func(self.position,dim, x_set, s)
        self.best_particle_fitness = self.fitness   # we couldd start with very large number here, 
                                                    #but the actual value is better in case we are lucky 
                
    def setPos(self, pos):
        self.position = pos
        self.fitness = fitness_func(self.position,self.dim, self.x_set, self.s)
        if self.fitness<self.best_particle_fitness:     # to update the personal best both 
                                                        # position (for velocity update) and
                                                        # fitness (the new standard) are needed
                                                        # global best is update on swarm leven
            self.best_particle_fitness = self.fitness
            self.best_particle_pos = pos

    def updateVel(self, inertia, a1, a2, best_self_pos, best_swarm_pos):
                # Here we use the canonical version
                # V <- inertia*V + a1r1 (peronal_best - current_pos) + a2r2 (global_best - current_pos)
        cur_vel = self.velocity
        r1 = np.random.uniform(low=0, high=1, size = self.dim)
        r2 = np.random.uniform(low=0, high=1, size = self.dim)
        a1r1 = np.multiply(a1, r1)
        a2r2 = np.multiply(a2, r2)
        best_self_dif = np.subtract(best_self_pos, self.position)
        best_swarm_dif = np.subtract(best_swarm_pos, self.position)
                    # the next line is the main equation, namely the velocity update, 
                    # the velocities are added to the positions at swarm level 
        return inertia*cur_vel + np.multiply(a1r1, best_self_dif) + np.multiply(a2r2, best_swarm_dif)

class PSO: # all the material that is relavant at swarm leveel

    def __init__(self, w, a1, a2, dim, population_size, time_steps, search_range, x_set, s):

        # Here we use values that are (somewhat) known to be good
        # There are no "best" parameters (No Free Lunch), so try using different ones
        # There are several papers online which discuss various different tunings of a1 and a2
        # for different types of problems
        self.w = w # Inertia
        self.a1 = a2 # Attraction to personal best
        self.a2 = a2 # Attraction to global best
        self.dim = dim
        self.s = s
        self.x_set = x_set

        self.swarm = [Particle(dim,-search_range,search_range, x_set, s) for i in range(population_size)]
        self.time_steps = time_steps
        print('init')

        # Initialising global best, you can wait until the end of the first time step
        # but creating a random initial best and fitness which is very high will mean you
        # do not have to write an if statement for the one off case
        self.best_swarm_pos = np.random.uniform(low=-500, high=500, size=dim)
        self.best_swarm_fitness = 1e100

    def run(self):
        for t in range(self.time_steps):
            for p in range(len(self.swarm)):
                particle = self.swarm[p]

                new_position = particle.position + particle.updateVel(self.w, self.a1, self.a2, particle.best_particle_pos, self.best_swarm_pos)
                                
                if new_position@new_position > 1.0e+18: # The search will be terminated if the distance 
                                                        # of any particle from center is too large
                    print('Time:', t,'Best Pos:',self.best_swarm_pos,'Best Fit:',self.best_swarm_fitness)
                    raise SystemExit('Most likely divergent: Decrease parameter values')
 
                self.swarm[p].setPos(new_position)

                new_fitness = fitness_func(new_position,self.dim, self.x_set, self.s)

                if new_fitness < self.best_swarm_fitness:   # to update the global best both 
                                                            # position (for velocity update) and
                                                            # fitness (the new group norm) are needed
                    self.best_swarm_fitness = new_fitness
                    self.best_swarm_pos = new_position

            if t % 100 == 0: #we print only two components even it search space is high-dimensional
                print("Time: %6d,  Best Fitness: %14.6f,  Best Pos: %9.4f,%9.4f" % (t,self.best_swarm_fitness,self.best_swarm_pos[0],self.best_swarm_pos[1]), end =" ")
                if self.dim>2: 
                    print('...')
                else:
                    print('')

def main():
    np.random.seed(22)
    M = 2
    s = 100
    noise_scale = 0.5

    x_set = generate_data_set(M=M, s=s, noise_scale=noise_scale)
    plot_data(x_set, s)

    # PSO(dim=M, w=0.75, a1=2.02, a2=2.02, population_size=5, time_steps=1001, search_range=1.0, x_set=x_set, s=s).run()

main()