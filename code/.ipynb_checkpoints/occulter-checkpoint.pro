PRO basiclasco

files=findfile('F:\Work\Data\soho\lasco\level_05\000715\2*.fts')   	; decirle donde buscar los archivos
n_imgs=n_elements(files)
imgcube=fltarr(1024, 1024, n_imgs)
window, 0, xsize=1024, ysize=1024

for i=0, n_imgs-1 DO BEGIN
	im=readfits(files[i], hdr)
	hstr=lasco_fitshdr2struct(hdr)
	
	asolr = GET_SOLAR_RADIUS(hdr)
	arcs = GET_SEC_PIXEL(hdr,FULL=fhsize)
	r_sun=asolr/arcs
	sunc = GET_SUN_CENTER(hdr,FULL=fhsize, /NOCHECK)
	r_occ = r_sun * 2.1   <----- Yasmin fijate acá el 2.1 representa el tamaño del occulter, eso cambia de instrumento a instrumento
	r_occ_out = r_sun * 8.
	fillcol=min(im2)  ;esta y la que sigue son para elegir los colores de relleno y de línea
	fillline=max(im2)

	;;;;;;;;;inner & outer mask, label;;;;;;;;;;;;
	sizeimg=hstr.naxis1
	window, 6, xsize=sizeimg, ysize=sizeimg, /pixmap
	tmpmask=fltarr(sizeimg, sizeimg) & tmpmask(*) = 0
	wset, 6
	tv, tmpmask
	TVCIRCLE, r_occ_out,sunc.xcen,sunc.ycen, /FILL, COLOR=1		 ;outer mask
	TVCIRCLE, r_occ, sunc.xcen, sunc.ycen, /FILL, COLOR=2			;occulter
	TVCIRCLE, r_sun, sunc.xcen, sunc.ycen, COLOR=3, THICK=2		;solar circle
	label = hstr.detector+' '+strmid(hstr.date_obs+' '+hstr.time_obs,0,16)
	XYOUTS, 15., 20., label, /DEVICE, CHARSIZE=3, COLOR=4
	tmpmask=tvrd()
	occ = where(tmpmask EQ 2)
	suncir=where(tmpmask EQ 3)
	outer = WHERE(tmpmask EQ 0)
	text=where(tmpmask EQ 4)

	IF (occ(0) NE -1) THEN im2(occ)=fillcol
	IF (suncir(0) NE -1) THEN im2(suncir)=fillline
	im2(outer)=fillcol
	im2(text)=fillline

	loadct, 3
	wset, 0
	tvscl, sigrange(im2)
	imgcube[*,*,i]=im2

ENDFOR

wdelete, 0
bmin=300	;bmin & bmax pueden variar! para C3 serian 400 y 900 respectivamente.
bmax=1900

generic_movie, imgcube, bmin, bmax   ;esto es si queres hacer movie con las imagenes del cubo

END
