"""PURPOSE:
Extract pointing parameters from wcs header to initialize raytrace.
INPUTS:
  instr : detector
  imsize : size of the image to be simulated
  scchead : image fits header
  pcin : PC matrix
  flagfovpix: set if fovpix is defined by user

OUTPUTS:
  the different parameters needed for the raytracing. """
def rtgetinstrwcsparam(instr,imsize,scchead,fovpix,crpix,obsang,pc,imszratio,projtypepreset=projtypepreset,pv2_1=pv2_1,rollang=rollang,crval=crval,pcin=pcin,flagfovpix=flagfovpix):
    