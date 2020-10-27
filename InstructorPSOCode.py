import numpy as np
import math

# From the tutorial!  Most of the code here is **not written by us**
# However I have modified this to take into account repulsive forces
# and fixed a bug (in main code, a1 is ignored and both params use a2)

class Particle: # all the material that is relavant at the level of the individual particles
    
    def __init__(self, dim, minx, maxx, fitness_func, constrainer):
        self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        self.velocity = np.random.uniform(low=-0.1, high=0.1, size=dim)
        self.best_particle_pos = self.position
        self.dim = dim

        self.constrainer = constrainer
        self.fitness_func = fitness_func
        self.fitness = self.fitness_func(self.position)
        self.best_particle_fitness = self.fitness   # we couldd start with very large number here, 
                                                    #but the actual value is better in case we are lucky 
                
    def setPos(self, pos):
        self.position = self.constrainer(pos)
        self.fitness = self.fitness_func(self.position)
        if self.fitness<self.best_particle_fitness:     # to update the personal best both 
                                                        # position (for velocity update) and
                                                        # fitness (the new standard) are needed
                                                        # global best is update on swarm leven
            self.best_particle_fitness = self.fitness
            self.best_particle_pos = pos

    def updateVel(self, inertia, a1, a2, a3, best_self_pos, best_swarm_pos, other_poses):
        # Here we use the canonical version + repulsion
        # V <- inertia*V + a1r1 (peronal_best - current_pos) + a2r2 (global_best - current_pos)
        # + a3r3
        cur_vel = self.velocity
        r1 = np.random.uniform(low=0, high=1, size = self.dim)
        r2 = np.random.uniform(low=0, high=1, size = self.dim)
        r3 = np.random.uniform(low=0, high=1, size = self.dim)
        a1r1 = np.multiply(a1, r1)
        a2r2 = np.multiply(a2, r2)
        def weightfunc(x, y):
            d = x - y
            return d / (np.dot(d, d))
        if a3 == 0:
            a3r3 = np.zeros(self.dim)
        else:
            a3r3 = np.multiply(
                -sum([
                    np.multiply(r3, weightfunc(x, self.position)) for x in other_poses
                ]),
                a3
            )
        best_self_dif = np.subtract(best_self_pos, self.position)
        best_swarm_dif = np.subtract(best_swarm_pos, self.position)
        # the next line is the main equation, namely the velocity update, 
        # the velocities are added to the positions at swarm level 
        return (
            inertia*cur_vel
            + np.multiply(a1r1, best_self_dif)
            + np.multiply(a2r2, best_swarm_dif)
            + a3r3
        )

class PSO: # all the material that is relavant at swarm leveel

    def __init__(self, w, a1, a2, a3, dim, population_size, time_steps, search_range, fitness_func, constrainer):

        # Here we use values that are (somewhat) known to be good
        # There are no "best" parameters (No Free Lunch), so try using different ones
        # There are several papers online which discuss various different tunings of a1 and a2
        # for different types of problems
        self.w = w # Inertia
        self.a1 = a1 # Attraction to personal best
        self.a2 = a2 # Attraction to global best
        self.a3 = a3 # Repulsive force!
        self.dim = dim
        self.fitness_func = fitness_func
        self.constrainer = constrainer

        self.swarm = [
            Particle(dim,-search_range,search_range,fitness_func=self.fitness_func,constrainer=self.constrainer)
            for i in range(population_size)
        ]
        self.time_steps = time_steps
        print('init')

        # Initialising global best, you can wait until the end of the first time step
        # but creating a random initial best and fitness which is very high will mean you
        # do not have to write an if statement for the one off case
        self.best_swarm_pos = np.random.uniform(low=-1, high=1, size=dim)
        self.best_swarm_fitness = 1e100

    def run(self):
        for t in range(self.time_steps):
            for p in range(len(self.swarm)):
                particle = self.swarm[p]

                new_position = self.constrainer(
                    particle.position + particle.updateVel(
                        self.w,
                        self.a1,
                        self.a2,
                        self.a3,
                        particle.best_particle_pos,
                        self.best_swarm_pos,
                        [val.position for idx, val in enumerate(self.swarm) if idx != p]
                    )
                )
                                
                if new_position@new_position > 1.0e+18: # The search will be terminated if the distance 
                                                        # of any particle from center is too large
                    print('Time:', t,'Best Pos:',self.best_swarm_pos,'Best Fit:',self.best_swarm_fitness)
                    raise SystemExit('Most likely divergent: Decrease parameter values')
 
                self.swarm[p].setPos(new_position)

                new_fitness = self.fitness_func(new_position)

                if new_fitness < self.best_swarm_fitness:   # to update the global best both 
                                                            # position (for velocity update) and
                                                            # fitness (the new group norm) are needed
                    self.best_swarm_fitness = new_fitness
                    self.best_swarm_pos = new_position

            if (t  + 1) % int(math.ceil(self.time_steps / 10)) == 0 or t == self.time_steps - 1: #we print only two components even it search space is high-dimensional
                print(("Time: %6d,  Best Fitness: %14.6f") % (t,self.best_swarm_fitness), end ="\n")
                print("Best Pos: " + str(self.best_swarm_pos) + "\n")
        return (list(zip(*[x.position for x in self.swarm])), self.best_swarm_pos)