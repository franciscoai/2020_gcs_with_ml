; ---- wrapper that calls the C raytracing program

pro rtwlline,sbtot,sbpol,snetot,angle=angle,imsize=imsize,$
             fovpix=fovpix,nbpix=nbpix,$
             obspos=obspos,obsang=obsang,$
             nepos=nepos,neang=neang,$
             losnbp=losnbp,losrange=losrange,$
             modelid=modelid,modparam=modparam,$
             save=save,file=file,fakelasco=fakelasco,$
             c2image=c2image,c3image=c3image,quiet=quiet,$
             cor1=cor1,cor2=cor2,hi1=hi1,hi2=hi2,c1fov=c1fov,$
             unload=unload,hlonlat=hlonlat,xdr=xdr,secchiab=secchiab,$
             instr=instr,limbdark=limbdark

;+
;  $Id: rtwlline.pro,v 1.1 2006/09/08 16:03:22 nathan Exp $
; 
; PURPOSE: 
;  Raytracing program to generate a radial profile
;
; CATEGORY:
;  raytracing, simulation, 3d
;
; INPUTS:
; ! all the distances must be given in Rsun
; ! and the angles in rad
; angle : angle of the radial profile in radian
; nbpix : size of the output profile in pix
; imsize : [xs,ys] size of the image (not calculated here, just for
; the scale)
; fovpix : fov angle of one pixel in rad 
; -- observer position and attitude
; obspos : [x,y,z] position of the observer in the Sun basis
; obsang : [ax,ay,az] orientation of the observer, 
;          z is the optical axis 
; -- Ne position and attitude
; nepos : [x,y,z] position of the Ne reference in the Sun basis
; neang : [ax,ay,az] orientation of the Ne
; -- LOS params
; losnbp : number of step for the integration along the LOS
; losrange : [lstart,lend] range for the integration along the LOS
;            in Rsun. The origin of the LOS is the orthogonal
;            projection of the Sun cntr on that LOS.
; modelid : model id number
; modparam : parameters of the model
; save : put path and filename in that variable (without extention) 
;        if you want to save the results in a .fits binary table.
; file : density cube filename for density 4 and 5
; fakelasco : put fake lasco header information in the fits header
; quiet : disable display of raytracing parameters
; unload : set to force unloading the C++ program (actualy .so lib):
;          useful when recompiling the C program
; hlonlat : [Hlon,Hlat,Hrot] heliographic lon and lat of the center of
; the disk, rotation angle corresponding to the projection of the
; north pole, counterclockwise.
; xdr : save into xdr format instead of fits table. 'save' keyword
;       must be set for xdr to take effect.
; secchiab : 'A' or 'B', to select Ahead or Behind spacecraft, for
;            secchi only
; instr : txt instrument preset, to select from the list above:
; -- Instrument FOV preset
; c1, c2, c3 : lasco C1, C2, C3
; cor1, cor2 : Secchi Cor1, Cor2
; hi1, hi2 : Secchi Hi1, Hi2
; limbdark : limb darkening coeff: default 0.58
;
; OUTPUTS:
;  sbtot : structure for the total brightness profile
;  sbpol : structure for the polarized brightness profile
;  snetot : structure for the LOS integrated electron density profile
;
;-

; ---- use rtwlseg instead now

rtwlseg,sbtot,sbpol,snetot,imsize=imsize,$
             fovpix=fovpix,angle=angle,nbpix=nbpix,$
             obspos=obspos,obsang=obsang,$
             nepos=nepos,neang=neang,$
             losnbp=losnbp,losrange=losrange,$
             modelid=modelid,modparam=modparam,$
             save=save,file=file,fakelasco=fakelasco,$
             c2image=c2image,c3image=c3image,quiet=quiet,$
             cor1=cor1,cor2=cor2,hi1=hi1,hi2=hi2,c1fov=c1fov,$
             unload=unload,hlonlat=hlonlat,xdr=xdr,secchiab=secchiab,$
             instr=instr,limbdark=limbdark



return
end
