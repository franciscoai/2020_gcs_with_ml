;+
; $Id: wire_29.pro,v 1.1 2006/09/08 16:03:23 nathan Exp $
; 
; PURPOSE:
;  Generates a wire frame for model 29
; CATEGORY:
;  raytracing, visualization, 3d
;-

pro wire_29,x,y,z,modparam=modparam ;d,s,

; Create a wire frame of a bow shock and display it

if n_elements(modparam) eq 0 then begin
    d=1.
    s=1.5
    h=1.
endif else begin
    d=modparam[0]
    s=modparam[1]
    h=modparam[2]
end

s = 1/h
s = s/2
s = s+2
d = d+(h/5)

;if n_elements (d) eq 0 then d=1
;if n_elements(s) eq 0 then s=1.5
                      ; Calculate cylindrical coord
th = makex(0,360,1.)
r = maken(0,3,n_elements(th))

; Bow shock Equation

z = (d/s)*(r/d)^s

cyl  = transpose([[th],[r],[z]])
cart = cv_coord(from_cyl=cyl,/to_rect)


x=cart(0,*)
y=cart(1,*)
z=cart(2,*)

return
end
