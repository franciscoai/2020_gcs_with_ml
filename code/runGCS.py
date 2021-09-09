runGCSfrom PyQt5 import QtWidgets, QtGui, QtCore
from astropy.io import fits
from pyGCS import *
from GCSgui import *
import sunpy
from sunpy.coordinates.ephemeris import get_horizons_coord
import datetime


# flag if using LASCO data from ISSI which has STEREOlike headers already
ISSIflag = True

# Read in your data
mainpath  = '/Users/ckay/GCSfit/'
eventpath = 'event17/'
c2path    = mainpath+eventpath+'c2/' 
c3path    = mainpath+eventpath+'c3/' 
cor2apath = mainpath+eventpath+'cor2/a/' 
cor2bpath = mainpath+eventpath+'cor2/b/'

# ISSI data (hierarchy folders)
fnameA1 = getFile(cor2apath, '20020502_125')
fnameB1 = getFile(cor2bpath, '20020502_125')
fnameL1  = getFile(c3path, '20020502_125')  
fnameA2 = getFile(cor2apath, '20020502_165')
fnameB2 = getFile(cor2bpath, '20020502_165')
fnameL2  = getFile(c3path, '20020502_165')  


# Non ISSI data (all in one folder)
'''thisPath = mainpath+eventpath
fnameA1 = getFile(thisPath, '20130522_080', ext='A.fts')
fnameB1 = getFile(thisPath, '20130522_080', ext='B.fts')
fnameL1  = getFile(thisPath, '20130522_075', ext='C2.fts') 
fnameA2 = getFile(thisPath, '20130522_132', ext='A.fts')
fnameB2 = getFile(thisPath, '20130522_132', ext='B.fts')
fnameL2  = getFile(thisPath, '20130522_132', ext='C2.fts')'''


# STEREO A
myfitsA1 = fits.open(fnameA1)
ima1 = myfitsA1[0].data
hdra1 = myfitsA1[0].header
myfitsA2 = fits.open(fnameA2)
ima2 = myfitsA2[0].data
hdra2 = myfitsA2[0].header

# STEREO B
myfitsB1 = fits.open(fnameB1)
imb1 = myfitsB1[0].data
hdrb1 = myfitsB1[0].header
myfitsB2 = fits.open(fnameB2)
imb2 = myfitsB2[0].data
hdrb2 = myfitsB2[0].header

# LASCO
if ISSIflag:
    myfitsL1 = fits.open(fnameL1)
    imL1 = myfitsL1[0].data
    hdrL1 = myfitsL1[0].header
    myfitsL2 = fits.open(fnameL2)
    imL2 = myfitsL2[0].data
    hdrL2 = myfitsL2[0].header
else:
    myfitsL1 = fits.open(fnameL1)
    imL1 = myfitsL1[0].data
    myfitsL1[0].header['OBSRVTRY'] = 'SOHO'
    coordL1 = get_horizons_coord(-21, datetime.datetime.strptime(myfitsL1[0].header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"), 'id')
    coordL1carr = coordL1.transform_to(sunpy.coordinates.frames.HeliographicCarrington)
    coordL1ston = coordL1.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
    myfitsL1[0].header['CRLT_OBS'] = coordL1carr.lat.deg
    myfitsL1[0].header['CRLN_OBS'] = coordL1carr.lon.deg
    myfitsL1[0].header['HGLT_OBS'] = coordL1ston.lat.deg
    myfitsL1[0].header['HGLN_OBS'] = coordL1ston.lon.deg
    hdrL1 = myfitsL1[0].header
    myfitsL2 = fits.open(fnameL2)
    imL2 = myfitsL2[0].data
    myfitsL2[0].header['OBSRVTRY'] = 'SOHO'
    coordL2 = get_horizons_coord(-21, datetime.datetime.strptime(myfitsL2[0].header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"), 'id')
    coordL2carr = coordL2.transform_to(sunpy.coordinates.frames.HeliographicCarrington)
    coordL2ston = coordL2.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
    myfitsL2[0].header['CRLT_OBS'] = coordL2carr.lat.deg
    myfitsL2[0].header['CRLN_OBS'] = coordL2carr.lon.deg
    myfitsL2[0].header['HGLT_OBS'] = coordL2ston.lat.deg
    myfitsL2[0].header['HGLN_OBS'] = coordL2ston.lon.deg
    hdrL2 = myfitsL2[0].header


# Options showing pyGCS for one, two, or three satellites ---------------|
# It just needs to be passed the correct images and headers

# Three Sats
headers = [hdra2, hdrL2, hdrb2]
ims = [np.transpose(ima2 - ima1), np.transpose(imL2 - imL1), np.transpose(imb2 - imb1)]

#headers = [hdrb2, hdrb2]
#ims = [np.transpose(imb2 - imb1), np.transpose(imb2 - imb1)]


# Two Sats
#headers = [hdrb2, hdra2]
#ims = [np.transpose(imb2 - imb1),  np.transpose(ima2 - ima1)]

# One Sat
#headers = [hdrL2]
#ims = [np.transpose(imL2 - imL1)]



# Option to control the density of points in GCS shape -----------------|
# ns = [nleg, ncirc, ncross]

ns =[3,10,31]      
#ns =[5,20,50]      

# Get the sat and inst information from the headers --------------------|
nSats = len(headers)        
sats = [[hdr['OBSRVTRY'], hdr['DETECTOR']] for hdr in headers]

# Get the location of sats and the range of each image -----------------|
satpos, plotranges = processHeaders(headers)  

# Pass everything to the GUI -------------------------------------------|
runGCSgui(ims, satpos, plotranges, sats, ns)


