;+
; $Id: rotmat.pro,v 1.2 2012/03/12 13:52:50 thernis Exp $
;
; PURPOSE: compute a rotation matrix
; CATEGORY: data handling, 3d, mathematics
; INPUTS: 
;  a : the angle of rotation in rad
;  ax : the axis of rotation: 1: X, 2: Y, 3: Z
; OUTPUTS:
;  return: the rotation matrix
;-
function rotmat,a,ax,inverse=inverse
if ~keyword_set(inverse) then begin
    case ax of
        1 : r=[[1.,0,0],$
               [0.,cos(a),sin(a)],$
               [0.,-sin(a),cos(a)]]
        2 : r=[[cos(a),0,-sin(a)],$
               [0,1,0],$
               [sin(a),0,cos(a)]]
        3 : r=[[cos(a),sin(a),0],$
               [-sin(a),cos(a),0],$
               [0,0,1]]
        else : message,'Bad axis ID'
    endcase 
endif else begin
    case ax of
        1 : r=[[1.,0,0],$
               [0.,cos(a),-sin(a)],$
               [0.,sin(a),cos(a)]]
        2 : r=[[cos(a),0,sin(a)],$
               [0,1,0],$
               [-sin(a),0,cos(a)]]
        3 : r=[[cos(a),-sin(a),0],$
               [sin(a),cos(a),0],$
               [0,0,1]]
        else : message,'Bad axis ID'
    endcase 
endelse
 
return,r
end
