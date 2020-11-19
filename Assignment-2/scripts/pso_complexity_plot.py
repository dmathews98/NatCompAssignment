import numpy as np
from models import pso
import itertools
import matplotlib.pyplot as plt
from params import showout

a1_percent = 0.5
pop = 20
time = 200

resolution = 10

mat = np.empty(shape=(resolution, resolution))
a_ = list(np.linspace(0, 4, resolution))
w_ = list(np.linspace(-1, 1, resolution))

for i, total_a in zip(itertools.count(), a_):
    for j, w in zip(itertools.count(), w_):
        print(f'total_a: {total_a}, w: {w}')
        mat[i, j] = pso(
            w=w,
            a1=a1_percent * total_a,
            a2=(1-a1_percent) * total_a,
            a3=0,
            population_size=pop,
            time_steps=time,
            search_range=1000,
            constrainer=lambda x: x,
            SILENT=True,
            GRAB_NN_SCORE=True
        )[0] # Grab loss rather than accuracy

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('a = a1 + a2')
ax.set_ylabel('w')
ax.set_title('Performance, and the Region of Complexity')
cax = ax.imshow(mat, extent=[0, 4, -1, 1], cmap='summer')
fig.colorbar(cax)
a2_ = list(np.linspace(0, 4, 100))
y_ = [-np.sqrt(4*a)+a+1 for a in a2_] # Boundary of complexity
ax.plot(a2_, y_, color='blue')
ax.set_aspect(1/ax.get_data_ratio())
ax.set_xlim(left=0, right=4)
ax.set_ylim(bottom=-1, top=1)
showout('complexity_matrix_map.png')