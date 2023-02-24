;+
; PURPOSE: Draw with the mouse within a IDL display window
;
; OUTPUTS:
;  return : a list of points that have been drawn
;
; $Id: draw.pro,v 1.2 2009/02/05 20:16:34 thernis Exp $
; $Log: draw.pro,v $
; Revision 1.2  2009/02/05 20:16:34  thernis
; Just add logs
;
;-

function draw,color=color,quiet=quiet

if n_elements(color) eq 0 then color=255


if ~keyword_set(quiet) then print,'Click right to quit'

nbcoords=0L
cursor,x0,y0,/down,/device

butstate=!mouse.button
if butstate eq 1 then begin
	plots,x0,y0,/device,psym=3,color=color
	points=[x0,y0,0.]
	nbcoords+=3L
endif


while (butstate ne 4) do begin
cursor,x,y,/change,/device
butstate=!mouse.button

if butstate eq 1 then begin
	plots,x,y,/device,psym=3,color=color
	if ((points[nbcoords-3] ne x) || (points[nbcoords-2] ne y)) then begin
		points=[points,x,y,0.]
		nbcoords+=3L
	endif
endif

endwhile

nbp=n_elements(points)
if nbp gt 0 then points=reform(points,3,nbp/3)

return,points
end