# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np

__author__ = "Francisco Iglesias"
__copyright__ = "Copyright 2020, Max Planck Institute for Solar System Research"
__version__ = "0.1"
__maintainer__ = "Francisco Iglesias"
__email__ = "franciscoaiglesias@gmail.com"

def low_freq_map(dim=[512,512,4],off=[0,0,0,0],var=[1,1,1,1],func=[0,1,2,3]):
	"""

	Returns an image with low-frequency, spatial variations of various types
	The aplitude, offset and shape of the variations are inputs.

	INPUTS
	See the paths in susim_def.py
	dim: Output dimensions [x,y,z]. Each [x,y] image is filled with a map
	off: off value for each image (z).
	var: Peak absolute variations around off for each image (z). For some values of func, it may be relative variation.
	func: select a function that defines the shape of the spatial variations from the list below

	Avilable functions:
	0-Patchy variations in both x and y directions.
	1-Variation only in x (vertical fringes)
	2-Variation only in y (horizontal fringes)
	3-Variation in x and y (oblique fringes)
	4-Flat (no variation, only offset, i.e. var is ignored)
	6-(assumes z=1) Flat with off value, except for a single small square area that has off+var (see SINMAP_F6_ROI)
	7-(assumes z=1) Same as 6 but with a larger square area (see SINMAP_F7_ROI)
	8-(assumes z=1) Flat with off value, except for a single small centered circle that has off+var
	11-(assumes z=1) Simulates a bias map, i.e. strong group-column-wise (var) plus milder single-row-wise variations (var/SINMAP_F11_RATIO)
	12-(assumes z=1) Tiangular target with illuminated part at off and dark areas equal to var*off
    13-Same as 0 but with random patches location

	OUTPUTS
	arr: numpy array of dimensions dim with the map
	
	"""
	#CONSTANTS
	#6
	SINMAP_F6_ROI=[800,850,800,850] # ROI that defines the square that has off+var values
	#7
	SINMAP_F7_ROI=[800,1150,800,1150] # ROI that defines the square that has off+var values
	#8
	SINMAP_F8_RAD = 100.	# Radius of the circle that has off+var values
	#11
	SINMAP_F11_GROUP=64 # Number of columns that have the same strong column-wise component of the bias
	SINMAP_F11_RATIO=2. # rms amplitud ratio of column-wise variations/row-wise variations

	if np.max(func) > 13 or np.min(func) < 0:
		logging.error("The input func is not recognized")
		exit(1)
	out=np.zeros((dim))
	x = np.arange(0, dim[0])
	y = np.arange(0, dim[1])
	xx, yy = np.meshgrid(x, y)#, sparse=True)
	if len(dim)< 3:
		nl=1
	else:
		nl=dim[2]
	for i in range(nl):
		if func[i]==0:
			tmp=var[i]*(np.sin(xx/700.)+ np.sin(yy/300.)+np.sin(xx/200+yy/400.))/3.
		if func[i]==13:
			rnd = np.random.uniform(low=dim[0]/9, high=dim[0]/16, size=(4))
			tmp=var[i]*(np.sin(xx/rnd[0])+ np.sin(yy/rnd[1])+np.sin(xx/rnd[2]+yy/rnd[3]))/3.			
		if func[i]==1:
			tmp=var[i]*np.sin(xx/150.)
		if func[i]==2:
			tmp=var[i]*np.sin(yy/150.)
		if func[i]==3:
			tmp=var[i]*np.sin(xx/200+yy/200.)
		if func[i]==4:
			tmp=np.zeros((dim[0],dim[1]))
		if func[i]==6:
			out=np.zeros((dim[0],dim[1]))
			out[SINMAP_F6_ROI[0]:SINMAP_F6_ROI[1],SINMAP_F6_ROI[2]:SINMAP_F6_ROI[3]]=var[i]
			out+=off[i]
			return out
		if func[i]==7:
			out=np.zeros((dim[0],dim[1]))
			out[SINMAP_F7_ROI[0]:SINMAP_F7_ROI[1],SINMAP_F7_ROI[2]:SINMAP_F7_ROI[3]]=var[i]	
			out+=off[i]
			return out		
		if func[i]==8:	
			x = np.arange(0, dim[0])
			y = np.arange(0, dim[1])
			cx = np.fix(x.size/2).astype(int)
			cy = np.fix(y.size/2).astype(int)
			mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < SINMAP_F8_RAD**2			
			out = np.zeros((y.size, x.size))
			out[mask]=var[i]
			out+=off[i]
			return out				
		if func[i]==11:		
			out=np.zeros((dim[0],dim[1]))+off[0]
			bla=0.5*np.sin(np.arange(dim[1])/0.3)+0.2*np.sin(np.arange(dim[1])/73.)+0.3*np.sin(np.arange(dim[1])/1350.)
			out+=np.reshape(np.repeat(bla*off[0]*var[0]/SINMAP_F11_RATIO,dim[0]),(dim[1],dim[0]))
			bla=int(dim[1]//SINMAP_F11_GROUP)
			bla=0.5*np.sin(np.arange(bla)/0.3)+0.2*np.sin(np.arange(bla)/3.5)+0.3*np.sin(np.arange(bla)/15.5)
			bla*=off[0]*var[0]
			for j in range(int(dim[1]//SINMAP_F11_GROUP)):
				out[:,j*SINMAP_F11_GROUP:(j+1)*SINMAP_F11_GROUP]+=bla[j]
			return out
		if func[i]==12:
			out=np.zeros((dim[0],dim[1]))+off[0]*var[0]
			out[np.triu_indices(dim[0],m=dim[1])]=off[0]
			return out
		if nl ==1:
			out=tmp-np.mean(tmp)+off[i]
		else:
			out[:,:,i]=tmp-np.mean(tmp)+off[i]
	return out

	##################