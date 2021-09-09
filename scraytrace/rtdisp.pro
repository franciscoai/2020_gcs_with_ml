pro rtdisp,sbt,min,max,addoccult=addoccult,nolog=nolog,wnd=wnd,im=im,_extra=e,pos=pos,nosuncont=nosuncont,mocc=mocc,msun=msun


;+
;  $Id: rtdisp.pro,v 1.3 2010/09/08 15:48:53 thernis Exp $
;
; PURPOSE:
;  Tool useful to display raytracing simulations 
;
; CATEGORY:
;  raytracing, 3d, visualization
;
; INPUTS:
;  sbt : output structure of raytracewl
;  min,max : min and max of displaying
;  addoccult : /addoccult : add an occulter if a predefined instrument
;                           FOV is present in the fitsheader of the image
;              addoccult=2.2 : add an occulter of 2.2 Rsun
;  /nolog : linear scale instead of log
;  wnd=2 : specify the ID of the window display
;  im : force the image to be displayed. Useful to display sums, etc...
;  pos : position for tvscl
;  nosuncont : /nosuncont : disable plot of the sun 
;              contour in the occulter. Should
;              be used in conjonction with addoccult
;
; OUTPUTS:
;  Display on screen
;  mocc : occuter mask
;  msun : sun mask
;
;-

; ---- put the image in a local variable
if n_elements(im) eq 0 then im=sbt.im
if n_elements(pos) eq 0 then pos=0

if n_elements(min) eq 0 then begin
    m=where(sbt.im gt 1e-30,cnt)
    if cnt gt 0 then min=min(sbt.im[m]) else min=1e-30
endif
if n_elements(max) eq 0 then max=max(sbt.im)
if max eq 1e-30 then max=1e-29


; ---- add an occulter if requested
if n_elements(addoccult) ne 0 then begin
    ; -- extract info from fits header
    crpix1=sxpar(sbt.fitshead,'crpix1')
    crpix2=sxpar(sbt.fitshead,'crpix2')
    imsize=fltarr(2)
    imsize[0]=sxpar(sbt.fitshead,'naxis1')
    imsize[1]=sxpar(sbt.fitshead,'naxis2')
    solar_r= (atan(1,sxpar(sbt.fitshead,'dsun_obs')/(onersun()*1e3))*!radeg)/sxpar(sbt.fitshead,'cdelt1')    ;float(sxpar(sbt.fitshead,'SOLAR_R'))
    rtdetec=strtrim(sxpar(sbt.fitshead,'RTDETEC'),2)
    ; -- overread occulter size if specified by user
    if (addoccult eq 1) and (size(addoccult,/type) ne 2) then rtdetec=''

    case rtdetec of
        'c2image' : begin
            ; -- sun center in the LASCO-C2 images
            center=rt_t_param('c2','center')
            
            ; -- occulter center shift compare to
            ;    the Sun center for that instrument
            occenter=rt_t_param('c2','occenter')

            ; -- solar_r C2 full resolution, D=210.Rsun
            solar_r_fullreso=82.5391 ; - pix
            ; - compute shift at the simulation resolution
            ratioresol=(solar_r/solar_r_fullreso)

            if addoccult ne 1 then begin
                occulterradius=float(addoccult)*solar_r
                cntrshiftlasco=[0.,0] ; -- no shift if no default val
            endif else begin
                occulterradius=ratioresol*rt_t_param('c2','occulter')
                cntrshiftlasco=ratioresol*(occenter-center)
            endelse

        end
        'c3image' : begin
            ; -- sun center in the LASCO-C2 images
            center=rt_t_param('c3','center')
            
            ; -- occulter center shift compare to
            ;    the Sun center for that instrument
            occenter=rt_t_param('c3','occenter')

            ; -- solar_r C3 full resolution, D=210.Rsun
            solar_r_fullreso=17.5055 ; - pix
            ; - compute shift at the simulation resolution
            ratioresol=(solar_r/solar_r_fullreso)

            if addoccult ne 1 then begin
                occulterradius=float(addoccult)*solar_r
                cntrshiftlasco=[0.,0] ; -- no shift if no default val
            endif else begin
                occulterradius=ratioresol*rt_t_param('c3','occulter')
                cntrshiftlasco=ratioresol*(occenter-center)
            endelse


        end
        'c1fov' : begin
            occulterradius=float(1.1)*solar_r
            cntrshiftlasco=[0.,0]

        end
        else : begin
            occulterradius=float(addoccult)*solar_r
            cntrshiftlasco=[0.,0]
        end
    endcase

    ; ---- sun circle mask
    msksun=mkcirc(imsize[0],imsize[1],crpix1,crpix2,solar_r*0.9)-mkcirc(imsize[0],imsize[1],crpix1,crpix2,solar_r)
    msun=where(msksun gt 0,cntsun)

    ; ---- occulter mask
    mskocc=mkcirc(imsize[0],imsize[1],crpix1+cntrshiftlasco[0],crpix2+cntrshiftlasco[1],occulterradius)
    mocc=where(mskocc gt 0,cntocc)

    ; ---- add mask in the images
    if cntocc gt 0 then im[mocc]=0.
    if cntsun gt 0 and not keyword_set(nosuncont) then im[msun]=1.

endif

if keyword_set(nolog) then begin if n_elements(wnd) ne 0 then wnd,wnd,bytscl(im,min,max),/tv,_extra=e else tv,bytscl(im,min,max),pos

endif else if n_elements(wnd) ne 0 then wnd,wnd,bytscl(alog10(im > min < max),alog10(min),alog10(max)),/tv,_extra=e else tv,bytscl(alog10(im > min < max),alog10(min),alog10(max)),pos


return
end
