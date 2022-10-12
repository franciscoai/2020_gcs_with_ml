;+
; PURPOSE: Sigmoid function
;
; CATEGORIES:
;  mathematics
;
; INPUTS:
;  x : array of x axis abscissa
;  height : height of the transition
;  xshift : position of the transition
;  stiffness : stiffness of the transition
;
; OUTPUTS:
;  return : a sigmoid
;
; $Id: sigmoid.pro,v 1.1 2009/02/05 20:36:55 thernis Exp $
; $Log: sigmoid.pro,v $
; Revision 1.1  2009/02/05 20:36:55  thernis
; First commit
;
;-

function sigmoid,x,height,xshift,stiffness
return,height/(1.+exp(-stiffness*(x-xshift)))
end