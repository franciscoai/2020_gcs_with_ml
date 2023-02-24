function rt_t_param, detector, key 
;+
; PROJECT:
;      SOHO - LASCO
; NAME:
;     rt_t_param
; PURPOSE:
;     Gets main telescope parameters
; CATEGORY:
;  laboratory, calibration
;
; CALLING SEQUENCE:
;     value = rt_t_param( detector, key )
; EXAMPLE
;     pixsze = rt_t_param('C2','PIXEL')
; INPUTS:
;     detector   'C1','C2','C3'                       STRING
;     key        parameter name                       STRING
;                'PIXEL'   pixelsize  (in mm)
;                'SCALE'   angular scale for pixel  (in deg)
;                'FOCAL'   equivalent focal (in mm)
;                'FIELD'   field (in deg)
;                'DFIELD'  diagonal field (in deg)
;                'BIAS'    electric bias of each amplifier (in ADU)
;                            get byas from  offset_bias.pro (needs the header)
;                'OCCULTER' radius of occlusion due to internal oculter
;                           in 'CCD pixel units'
;                'DISTORTION' set of coefs of distortion (to be used in mm)
;                              NOTE: drho = a0*rho+a1*rho^3+a2*rho^5
;                'IDISTORTION' set of coefs of distortion correction (in mm)
;                'DISCNTR' distortion center (in mm)
;                'CENTER'  Sun center (in pixels)
;                'OCCENTER' occulter center (in pixels)
; KEYWORD INPUT
;     none
; OUTPUTS:
;     value      searched value
; OUTPUTS INPUTS:
;     none
; PROCEDURE:
;     Gets parameters from a set of internal data 
; CALLED PROCEDURES:
;     none
;
; HISTORY:
;     Def. and code: A.Llebaria (LAS-CNRS)    Aug 1996 
;     Corrected and modified by Dr. M.-V. Bout on March, 4th, 1998
; NOTA :
;
;   Distortion parameters:               ; rho*(a0+a1*rho^2+a2*rho^4)
;   history
;           pm.distortion = [0.0060519645, -0.00014672423, 2.0899603e-07]
; list9604_1ora_00_g_ldist.dat
;           pm.distortion = [0.0051344125, -0.00012233862, 1.0978595e-07]  
; list9603_1ora_00_a_ldist.dat
;           pm.distortion = [0.0044836143, -0.00011276272, 5.9968042e-08]  
; list9602_2ora_00_a_ldist.dat
;           pm.distortion = [0.0051344125, -0.00012233862, 1.0978595e-07]  
; list9603_1ora_00_a_ldist.dat
;           pm.distortion = [0.0056511656, -0.00013827504, 1.6940201e-07]  
; list9604_1ora_00_h_ldist.dat
;
; CVSLOG:
;
; $Id: rt_t_param.pro,v 1.1 2010/09/08 15:45:43 thernis Exp $
;-
pm = {PARAMETERS, PIXEL: 0.0, SCALE: 0.0, FOCAL: 0.0, FIELD: 0.0, $
DFIELD: 0.0, BIAS: 0.0, OCCULTER: 0.0, DISTORTION: [0.,0.,0.],$
DISCNTR: [0.,0.], CENTER: [0.,0.]}
value = -1
dtctr = strupcase(detector) 
case 1 of
  dtctr eq 'C1': begin
                  pm.pixel = 0.021
                  pm.focal = 768.
                  pm.bias  = 332.
                  pm.occulter = 170.
                  pm.distortion = [0.,0.,0.]
                  pm.center = [511.5,511.5]                          ; (Sun)
                  occenter  = [511.5,511.5]
                  discenter = [511.5,511.5]
                 end
  dtctr eq 'C2': begin
                  pm.pixel = 0.021
                  pm.focal = 364.
                  pm.bias  = 470.
                  pm.occulter = 166.
                  pm.distortion = [.0051344125, -.00012233862, 1.0978595e-07]  
;                             rho*(a0+a1*rho^2+a2*rho^4)
                  idistortion = [-.00508753, .000119011, -5.30006e-08, -1.20326e-10]
;                             rhom*(b0+b1*rho^2+...0
;                  pm.center = [513.5,505.5]                          ; (Sun)
                  pm.center = [512.5,504.5]                          ; (Sun)
                  occenter  = [512.,506.]
                  discenter = [512.5,504.5]
                 end
  dtctr eq 'C3': begin
                  pm.pixel = 0.021
;                 pm.focal = 77.6
                  pm.focal = 77.2
                  pm.bias  = 319.
;                 pm.occulter = 33.
                  pm.occulter = 67.
                  pm.distortion = [-.0151657,.000165455]   ; rho*(a0+a1*rho^2)
                  idistortion = [.0153525, -.000173515, 6.34659e-08 ]  
;                                                            rhom*(b0+b1*rho^2+...0
                  pm.center = [518.,532.]    ; 518., 532.  (31/03/98)   (Sun)
                  occenter  = [516.3,529.5]
                  discenter  = [516.3,529.5] 
                 end
  else: begin
     message, 'Unknown telescope name (C1,C2 or C3 only are allowed)',/info 
         return, value
        end
endcase
pm.scale = (pm.pixel/pm.focal)*!radeg
pm.field = pm.scale*1024
pm.dfield = sqrt(2.)*pm.field
pm.discntr = discenter*pm.pixel
keywd = strupcase(key)
case 1 of
   keywd eq 'PIXEL' : value = pm.pixel
   keywd eq 'SCALE' : value = pm.scale
   keywd eq 'FOCAL' : value = pm.focal
   keywd eq 'FIELD' : value = pm.field
   keywd eq 'DFIELD' : value = pm.dfield
   keywd eq 'BIAS' : value = pm.bias
   keywd eq 'BIAIS' : value = pm.bias
   keywd eq 'OCCULTER' : value = pm.occulter
   keywd eq 'DISTORTION' : value = pm.distortion
   keywd eq 'DISTORSION' : value = pm.distortion
   keywd eq 'IDISTORTION' : value = idistortion
   keywd eq 'IDISTORSION' : value = idistortion
   keywd eq 'DISCNTR' : value = pm.discntr
   keywd eq 'CENTER'  : value = pm.center
   keywd eq 'OCCENTER'  : value = occenter
   keywd eq '?' :begin
        print,'PIXEL    pixelsize  (in mm)'
        print,'SCALE    angular scale for pixel  (in deg)'
        print,'FOCAL    equivalent focal (in mm)'
        print,'FIELD    field (in deg)'
        print,'DFIELD   diagonal field (in deg)'
        print,'BIAS     electric bias of each amplifier (in ADU)'
        print,'OCCULTER  radius of occlusion due to internal oculter'
        print,'          in CCD pixel units'
        print,'DISTORTION  set of coefs of distortion (to be used in mm)'
        print,'            NOTE: drho = a0*rho+a1*rho^3+a2*rho^5'
        print,'DISCNTR  Distortion center (in mm)'
        print,'CENTER   Sun center (in pixels)'
        print,'OCCENTER occulter center (in pixels)'
                 end
   keywd eq 'ALL' :begin
	help,pm,/st
        print,'OCCENTER:',occenter
		   end
   else : message, 'Unknown parameter',/info
endcase
return, value
end
