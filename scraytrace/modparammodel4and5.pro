;+
; PURPOSE:
;  Format model parameter vector for models 4 and 5
; INPUT:
;  file : density cube filename
; OUTPUT:
;  modparam : the model parameter vector
;  rco : radial vector
;  phico : latitude vector
;  thetaco : longitude vector
;  dens : the density cube
;-

pro modparammodel4and5,file,modparam,rco,phico,thetaco,dens

load_dens,dens,rco,phico,thetaco,file=file
modparam=[rco,phico,thetaco,reform(dens,n_elements(dens))]

return
end