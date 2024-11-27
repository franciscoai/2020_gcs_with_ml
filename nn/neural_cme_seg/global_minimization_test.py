# basin hopping global, dual annealing and differential evolution optimization for the ackley multimodal objective function
#a test case.
from scipy.optimize import basinhopping
from scipy.optimize import dual_annealing
from scipy.optimize import differential_evolution
from numpy.random import rand
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi

from numpy import arange
from numpy import meshgrid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# objective function
def objective(p):#(x, y):
    x, y = p
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
xaxis = arange(r_min, r_max, 0.1)
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective([x, y])
# create a surface plot with the jet color scheme
#figure = plt.figure()
#axis = figure.gca(projection='3d')
fig, axis = plt.subplots(subplot_kw={"projection": "3d"})
axis.plot_surface(x, y, results, cmap='jet')
# show the plot
plt.show()

run_basin_hopping = True
run_dual_annealing = True
run_evolution = True

if run_basin_hopping:
    # define range for input
    r_min, r_max = -5.0, 5.0
    # define the starting point as a random sample from the domain
    pt = r_min + rand(2) * (r_max - r_min)
    # perform the basin hopping search
    result = basinhopping(objective, pt, stepsize=0.5, niter=400)
    # summarize the result
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])
    # evaluate solution
    solution = result['x']
    evaluation = objective(solution)
    print('Solution: f(%s) = %.5f' % (solution, evaluation))
breakpoint()
if run_dual_annealing:
    # define range for input
    r_min, r_max = -5.0, 5.0
    # define the bounds on the search
    bounds = [[r_min, r_max], [r_min, r_max]]
    # perform the dual annealing search
    result = dual_annealing(objective, bounds)
    # summarize the result
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])
    # evaluate solution
    solution = result['x']
    evaluation = objective(solution)
    print('Solution: f(%s) = %.5f' % (solution, evaluation))

breakpoint()
if run_evolution:
    # define range for input
    r_min, r_max = -5.0, 5.0
    # define the bounds on the search
    bounds = [[r_min, r_max], [r_min, r_max]]
    # perform the differential evolution search
    result = differential_evolution(objective, bounds, disp=True)
    # summarize the result
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])
    # evaluate solution
    solution = result['x']
    evaluation = objective(solution)
    print('Solution: f(%s) = %.5f' % (solution, evaluation))



