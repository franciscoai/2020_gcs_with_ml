import os
import numpy as np
import cv2
import torch
import matplotlib as mpl
import math
from datetime import datetime
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
mpl.use('Agg')

__author__ = "Francisco Iglesias"
__copyright__ = "Grupo de Estudios en Heliofisica de Mendoza - https://sites.google.com/um.edu.ar/gehme"
__version__ = "0.1"
__maintainer__ = "Francisco Iglesias"
__email__ = "franciscoaiglesias@gmail.com"


def 