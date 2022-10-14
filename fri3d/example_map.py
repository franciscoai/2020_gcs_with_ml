import os
import sys

import numpy as np
from ai.fri3d.model import StaticFRi3D
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d

path = os.getcwd()

sfr = StaticFRi3D(toroidal_height=1, poloidal_height=0.2, pancaking=0.6, skew=np.pi / 6, flattening=0.5)

xgrid = np.linspace(-0.2, 0.2, 50)
ygrid = np.linspace(-0.2, 0.2, 50)

bmap = sfr.map(1, 0, 0, [1, 0, 0], [0, 0, 1], xgrid=xgrid, ygrid=ygrid)

plt.pcolormesh(xgrid, ygrid, bmap)
plt.colorbar()

print('hecho')
plt.savefig(path + '/output/fri3d_example_map.png')
plt.show()
