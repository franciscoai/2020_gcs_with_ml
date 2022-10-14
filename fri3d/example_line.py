import os
import sys

import numpy as np
from ai.fri3d.model import StaticFRi3D
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
path = os.getcwd()

sfr = StaticFRi3D(
    toroidal_height=1, poloidal_height=0.2, half_width=np.pi / 4, skew=np.pi / 9, pancaking=0.4, flattening=0.5, twist=2
)

phi = np.linspace(-sfr.half_width, sfr.half_width, 100)

fig = plt.figure()

ax = fig.add_subplot(111, projection="3d", adjustable="box")
for r in np.linspace(0, 1, 40):
    x, y, z, _ = sfr.line(r, phi, np.random.rand() * np.pi * 2)
    ax.plot(x, y, z, alpha=0.4)
ax.set_xlim3d(0.0, 1.2)
ax.set_ylim3d(-0.6, 0.6)
ax.set_zlim3d(-0.6, 0.6)

print('hecho')
plt.savefig(path + '/output/fri3d_example_line.png')
plt.show()
