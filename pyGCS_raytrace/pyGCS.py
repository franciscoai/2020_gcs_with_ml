import numpy as np
from numpy.linalg import inv
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import sys
import sunpy
from sunpy.coordinates.ephemeris import get_horizons_coord, get_body_heliographic_stonyhurst
import datetime

pi = np.pi
dtor = pi/180.


def rotx(vec, ang):
    # Rotate a 3D vector by ang (input in degrees) about the x-axis
    ang *= dtor
    yout = np.cos(ang) * vec[1] - np.sin(ang) * vec[2]
    zout = np.sin(ang) * vec[1] + np.cos(ang) * vec[2]
    return [vec[0], yout, zout]


def roty(vec, ang):
    # Rotate a 3D vector by ang (input in degrees) about the y-axis
    ang *= dtor
    xout = np.cos(ang) * vec[0] + np.sin(ang) * vec[2]
    zout = -np.sin(ang) * vec[0] + np.cos(ang) * vec[2]
    return [xout, vec[1], zout]


def rotz(vec, ang):
    # Rotate a 3D vector by ang (input in degrees) about the y-axis
    ang *= dtor
    xout = np.cos(ang) * vec[0] - np.sin(ang) * vec[1]
    yout = np.sin(ang) * vec[0] + np.cos(ang) * vec[1]
    return [xout, yout, vec[2]]


def SPH2CART(sph_in):
    r = sph_in[0]
    colat = (90. - sph_in[1]) * dtor
    lon = sph_in[2] * dtor
    x = r * np.sin(colat) * np.cos(lon)
    y = r * np.sin(colat) * np.sin(lon)
    z = r * np.cos(colat)
    return [x, y, z]


def cmecloud(ang, hin, nleg, ncirc, k, ncross, hIsLeadingEdge=True):
    # This generates a horizontal GCS shape with nose along x axis and axis in xy plane
    h = hin
    # convert from distance of nose to Thernisien h (length of leg)
    if hIsLeadingEdge:
        h = hin*(1.-k)*np.cos(ang)/(1.+np.sin(ang))

    # Compute the shell points from axis and radius
    axisPTS, crossrads, betas = shellSkeleton(ang, h, nleg, ncirc, k)
    nbp = axisPTS.shape[0]
    theta = np.linspace(0, 360*(1-1./ncross), ncross, endpoint=True)*dtor

    # Put things into massive arrays to avoid real for loops
    thetaMEGA = np.array([theta]*nbp).reshape([1, -1])
    crMEGA = np.array([[crossrads[i]]*ncross for i in range(nbp)]).reshape([-1])
    betaMEGA = np.array([[betas[i]]*ncross for i in range(nbp)]).reshape([-1])
    axisMEGA = np.array([[axisPTS[i]]*ncross for i in range(nbp)]).reshape([-1, 3])

    # Calc the cross section vect in xyz
    radVec = crMEGA*(np.array([np.sin(thetaMEGA)*np.sin(betaMEGA), np.sin(thetaMEGA) *
                    np.cos(betaMEGA), np.cos(thetaMEGA)]))     # Add to the axis to get the full shell
    shell = np.transpose(radVec).reshape([ncross*nbp, 3])+axisMEGA

    return np.array(shell)


def shellSkeleton(alpha, h, nleg, ncirc, k):
    # Determine the xyz position of axis, cross section radius, and beta angle
    gamma = np.arcsin(k)

    # Calculate the leg axis
    hrange = np.linspace(1, h, nleg)
    leftLeg = np.zeros([nleg, 3])
    leftLeg[:, 1] = -np.sin(alpha)*hrange
    leftLeg[:, 0] = np.cos(alpha)*hrange
    rightLeg = np.zeros([nleg, 3])
    rightLeg[:, 1] = np.sin(alpha)*hrange
    rightLeg[:, 0] = np.cos(alpha)*hrange
    rLeg = np.tan(gamma) * np.sqrt(rightLeg[:, 1]**2 + rightLeg[:, 0]**2)

    legBeta = np.ones(nleg) * -alpha

    rightCirc = np.zeros([ncirc, 3])
    leftCirc = np.zeros([ncirc, 3])

    # Calculate the circle axis
    beta = np.linspace(-alpha, pi/2, ncirc, endpoint=True)
    b = h/np.cos(alpha)  # b thernisien
    rho = h*np.tan(alpha)

    X0 = (rho+b*k**2*np.sin(beta))/(1-k**2)
    rc = np.sqrt((b**2*k**2-rho**2)/(1-k**2)+X0**2)

    rightCirc[:, 1] = X0*np.cos(beta)
    rightCirc[:, 0] = b+X0*np.sin(beta)
    leftCirc[:, 1] = -rightCirc[:, 1]
    leftCirc[:, 0] = rightCirc[:, 0]

    # Group into a list
    # radius of cross section
    crossrads = np.zeros(2*(nleg+ncirc)-3)
    crossrads[:nleg] = rLeg[:nleg]
    crossrads[-nleg:] = rLeg[:nleg][::-1]
    crossrads[nleg:nleg+ncirc-1] = rc[1:]
    crossrads[nleg+ncirc-1:-nleg] = rc[1:-1][::-1]

    # beta angle
    betas = np.zeros(2*(nleg+ncirc)-3)
    betas[:nleg] = legBeta[:nleg]
    betas[-nleg:] = pi-legBeta[:nleg][::-1]
    betas[nleg:nleg+ncirc-1] = beta[1:]
    betas[nleg+ncirc-1:-nleg] = pi-beta[1:-1][::-1]

    # xyz of axis
    axisPTS = np.zeros([2*(nleg+ncirc)-3, 3])
    axisPTS[:nleg, :] = rightLeg[:nleg, :]
    axisPTS[-nleg:, :] = leftLeg[:nleg][::-1, :]
    axisPTS[nleg:nleg+ncirc-1, :] = rightCirc[1:, :]
    axisPTS[nleg+ncirc-1:-nleg, :] = leftCirc[1:-1][::-1, :]

    return axisPTS, crossrads, betas


def getGCS(CMElon, CMElat, CMEtilt, height, k, ang, satpos, do_rotate_lat=None, nleg=5, ncirc=20, ncross=30):
    cloud = cmecloud(ang*dtor, height, nleg, ncirc, k, ncross)
    # in order (of parens) rotx to tilt, roty by -lat, rotz by lon
    # cloud = np.transpose(rotz(roty(rotx(np.transpose(cloud), CMEtilt),-CMElat),CMElon))

    clouds = []
    # dSat = 213. # was assuming L1 for sat projections which no longer use
    if do_rotate_lat is None:
        do_rotate_lat = [False]*len(satpos)
    #for sat in satpos:
    #Modified 28/02/2024 by D.Lloveras to obtain a better match with IDL in Lasco C2 (satpos[1])
    #do_rotate_lat list of boolean values to rotate or not the cloud in latitude. Should be True in case of Lasco-C2, False in any other case.
    for i, sat in enumerate(satpos):
        # rot funcs like things transposed
        cXYZ = np.transpose(cloud)
        # Rot to correct tilt and Lat, matches IDL better not including satlat
        if do_rotate_lat[i] == True:
            #Lasco-C2 correction in HGLAT
            cXYZ = roty(rotx(cXYZ, CMEtilt), -(CMElat-sat[1]))
        if do_rotate_lat[i] == False:
            cXYZ = roty(rotx(cXYZ, CMEtilt), -CMElat)
        # Project in Lat (not done in IDL)
        # satXYZ = SPH2CART([dSat,sat[1],sat[0]])
        # gamma = np.arctan((cXYZ[2]-satXYZ[2])/(np.sqrt(satXYZ[0]**2+satXYZ[1]**2)-np.sqrt(cXYZ[0]**2+cXYZ[1]**2)))
        # zcp = dSat*np.tan(gamma+sat[1]*3.14159/180.)
        # cXYZ[2] = zcp

        # Rot to correct Lon
        cXYZ = rotz(cXYZ, -(sat[0]-CMElon))
        # Project in Lon (not done in IDL)
        # satXYZ = rotz(satXYZ, -sat[0])
        # gamma = np.arctan(cXYZ[1]/(satXYZ[0]-cXYZ[0]))
        # ycp = dSat * np.tan(gamma)
        # cXYZ[1] = ycp

        # correct for roll ang
        cXYZ = rotx(cXYZ, -sat[2])

        clouds.append(np.transpose(cXYZ))
    # for row in cloud:
    #    print (i, row)
    #    i+=1

    return np.array(clouds)


def processHeaders(headers):
    satpos = []
    plotranges = []
    for i in range(len(headers)):
        # Stonyhurst coords for the sats
        thisHead = headers[i]
        # GEHME corrects header for SOHO 2023/08/07
        try: # 
            if thisHead['TELESCOP'] == 'SOHO': 
                # Header fix
                thisHead['OBSRVTRY'] = 'SOHO'
                coordL2 = get_horizons_coord(-21, datetime.datetime.strptime(thisHead['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"), 'id')
                #coordL2ston = get_body_heliographic_stonyhurst('-21', time=datetime.datetime.strptime(thisHead['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"))
                #coordL2carr = coordL2ston.transform_to(sunpy.coordinates.frames.HeliographicCarrington)
                coordL2carr = coordL2.transform_to(sunpy.coordinates.frames.HeliographicCarrington(observer='earth'))
                coordL2ston = coordL2.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
                thisHead['DSUN_OBS'] = coordL2.radius.m
                thisHead['CRLT_OBS'] = coordL2carr.lat.deg
                thisHead['CRLN_OBS'] = coordL2carr.lon.deg
                thisHead['HGLT_OBS'] = coordL2ston.lat.deg
                thisHead['HGLN_OBS'] = coordL2ston.lon.deg
        except:
            continue
        # GEHME changed HGLN_OBS to CRLN_OBS : 2023/06/30
        satpos.append([float(thisHead['CRLN_OBS']), float(thisHead['HGLT_OBS']), float(thisHead['CROTA'])])
        # GEHME changed to use the correct Sun center 2023/08/07
        rSun = float(thisHead['RSUN'])  # in arcsecs
        # xaxrange = float(thisHead['CDELT1']) * int(thisHead['NAXIS1'])/rSun/2. 
        # yaxrange = float(thisHead['CDELT2']) * int(thisHead['NAXIS2'])/rSun/2.
        # plotranges.append([-xaxrange, xaxrange, -yaxrange, yaxrange])
        xaxrange = [float(thisHead['CRPIX1'])*float(thisHead['CDELT1'])/rSun, (float(thisHead['NAXIS1'])-float(thisHead['CRPIX1']))*float(thisHead['CDELT1'])/rSun]
        yaxrange = [float(thisHead['CRPIX2'])*float(thisHead['CDELT2'])/rSun, (float(thisHead['NAXIS2'])-float(thisHead['CRPIX2']))*float(thisHead['CDELT2'])/rSun]
        plotranges.append([-xaxrange[0], xaxrange[1], -yaxrange[0], yaxrange[1]])            
    return satpos, plotranges


def getFile(dirIn, name, ext='.fts'):
    res = glob.glob(dirIn+'*'+name+'*'+ext)
    if len(res) == 1:
        return res[0]
    elif len(res) > 1:
        print('More than one matching file!')
        print('Pick from:')
        for i in range(len(res)):
            print(res[i])
        sys.exit()
    elif len(res) == 0:
        print('No file found!')
        sys.exit()


# Can use this if you want to run pyGCS directly (no GUI)
if __name__ == '__main__':
    # Read in fits files
    mainpath = '/Users/ckay/GCSfit/'
    eventpath = 'event17/'
    c2path = mainpath+eventpath+'c2/'
    c3path = mainpath+eventpath+'c3/'
    cor2apath = mainpath+eventpath+'cor2/a/'
    cor2bpath = mainpath+eventpath+'cor2/b/'
    fnameA1 = 'event17_20020502_125400_04c2A.fts'
    fnameB1 = 'event17_20020502_125450_04c2B.fts'
    fnameL1 = 'event17_20020502_124625_c2.fts'
    fnameA2 = 'event17_20020502_165400_04c2A.fts'
    fnameB2 = 'event17_20020502_165450_04c2B.fts'
    fnameL2 = 'event17_20020502_164547_c2.fts'

    myfitsA1 = fits.open(cor2apath+fnameA1)
    ima1 = myfitsA1[0].data
    hdra1 = myfitsA1[0].header
    myfitsA2 = fits.open(cor2apath+fnameA2)
    ima2 = myfitsA2[0].data
    hdra2 = myfitsA2[0].header

    myfitsB1 = fits.open(cor2bpath+fnameB1)
    imb1 = myfitsB1[0].data
    hdrb1 = myfitsB1[0].header
    myfitsB2 = fits.open(cor2bpath+fnameB2)
    imb2 = myfitsB2[0].data
    hdrb2 = myfitsB2[0].header

    myfitsL1 = fits.open(c2path+fnameL1)
    imL1 = myfitsL1[0].data
    hdrL1 = myfitsL1[0].header
    myfitsL2 = fits.open(c2path+fnameL2)
    imL2 = myfitsL2[0].data
    hdrL2 = myfitsL2[0].header

    headers = [hdrb2, hdrL2, hdra2]
    ims = [np.transpose(imb2 - imb1), np.transpose((imL2 - imL1)), np.transpose(ima2 - ima1)]

    satpos, plotranges = processHeaders(headers)
    clouds = getGCS(0, 30., 50., 12., 0.3, 30, satpos, nleg=5, ncirc=20, ncross=40)

    nSat = len(satpos)

    # Plot everything
    fig, axes = plt.subplots(1, nSat, figsize=(5*nSat+2, 7))
    for i in range(nSat):
        axes[i].scatter(clouds[i][:, 1], clouds[0][:, 2], s=5, c='lime', linewidths=0)
        cent = 1  # 0.01*np.mean(np.abs(diff))
        axes[i].imshow(ims[i], vmin=-cent, vmax=cent, cmap=cm.binary, zorder=0, extent=plotranges[i])
    plt.show()
