;+
;  $Id: rt_generic_movie.pro,v 1.1 2006/09/08 16:03:21 nathan Exp $
;
; Project     : SOHO - LASCO/EIT
;                   
; Name        : RT_GENERIC_MOVIE
;               
; Purpose     : Load various data formats into pixmaps and call WRUNMOVIE for animation. Addapted for raytrace.
;               
; Explanation : This procedure is a generic interface to converting data
;		(image files or data cubes) into the LASCO MVI format.
;		The user can then save the movie in MVI format and use
;		routines such as wrunmovie.pro, combine_mvi.pro, mvi2mpg.pro, 
;		put_mvi.pro etc. for MVI manipulation.
;               
; Use         : RT_GENERIC_MOVIE, data[, bmin, bmax], /SXT, /MK3, /TRUECOLOR
;
; Inputs      : data : can be:
;		       - 3 dimensional byte array (nx,ny,len)
;		       - name of file containing names of images
;		       - strarr of names of images
;		    	(images can be of type: .fits, .gif, .pict, .tiff)
;
; Optional Inputs: bmin, bmax : Minimum and maximum DN for BYTSCL.
;               
; Outputs     : None.
;               
; Keywords    : 
;	/SXT	: Data is YOHKOH/SXT FITS files.
;	/MK3    : Data is Mauna-Loa Mark3 Coronagraph rpb files.
;       /PICT   : Data are PICT files
;	PAN=.5  : Resize to half
;	LABEL	Set equal to array of labels to put in lower left corner of frames
;	/RDIFF	: Do running difference movie
;	TRUECOLOR	; Set if images are true color
;
; Calls       : WRUNMOVIE
;
; Side effects: Creates multiple pixmaps.
;               
; Category: visualization, raytracing, animation
;               
; See Also    : WMKMOVIE.PRO is a widget front-end to making LASCO/EIT movies.
;               
; Prev. Hist. : None.
;
; Written     : Scott Paswaters, NRL, 1996.
;               
; Modified    : SEP 16 Dec 97 - Added /SXT keyword
;                             - Added /MK3 keyword
;               SEP 06 Jan 98 - Added .pict, .tiff support
;               DW  21 Jan 99 - Y2K Fix for Yohkoh date
;		NBR  9 Apr 99 - Added .jpg support
;		NBR 14 May 99 - Add PICT keyword
;		NBR 15 Sep 99 - Add PAN keyword
;		NBR  7 Jun 01 - Add PNG support
;		NBR  3 Jan 02 - Add LABEL keyword
;		NBR  1 May 02 - Add RDIFF keyword
;		NBR  3 Sep 02 - Add capability for floating point FITS
;		RAH 16 SEP 04 - Flag for true color
;
; Version     : 
;
; SCCS variables for IDL use
; 
; %W% %H% :LASCO IDL LIBRARY
;
;-            

PRO RT_GENERIC_MOVIE, data, bmin, bmax, SXT=sxt, MK3=mk3, PICT=pict, PAN=pan, LABEL=label, $
	RDIFF=rdiff,TRUECOLOR=truecolor

   d = DATATYPE(data)
   IF (DATATYPE(bmin) EQ 'UND') AND (DATATYPE(bmax) EQ 'UND') THEN scl=1 ELSE BEGIN
      scl=0   
      IF (DATATYPE(bmin) EQ 'UND') THEN bmin = 0
      IF (DATATYPE(bmax) EQ 'UND') THEN bmax = 255
  ENDELSE

      IF (DATATYPE(bmin) EQ 'UND') THEN bmin = 0
      IF (DATATYPE(bmax) EQ 'UND') THEN bmax = 255

   CASE (d) OF
      'BYT' : BEGIN
             sz = SIZE(data)
             hsize=sz(1)
             vsize=sz(2)
             len = sz(3)
             readflag=0
         END
      'INT' : BEGIN
             sz = SIZE(data)
             hsize=sz(1)
             vsize=sz(2)
             len = sz(3)
             readflag=0
         END
      'FLO' : BEGIN
             sz = SIZE(data)
             hsize=sz(1)
             vsize=sz(2)
             len = sz(3)
             readflag=0
         END
      'STR' : BEGIN
             IF (N_ELEMENTS(data) EQ 1) THEN BEGIN	;** read image names from file
                names = READLIST(data, len)
             ENDIF ELSE BEGIN				;** image names have been passed
                names = data
                len = N_ELEMENTS(names)
             ENDELSE
             readflag=1
         END
      ELSE : BEGIN
             PRINT, '%%RT_GENERIC_MOVIE: Unknown type for data: ', d
             RETURN
         END
   ENDCASE

   win_index = INTARR(len)

   ahdr = {filename:'',detector:'',time_obs:'',date_obs:'',filter:'',polar:'',sector:''}
   all_hdr = REPLICATE(ahdr, len)
   sunxcen=0.0
   sunycen=0.0
   sec_pix=0.0

r=indgen(256)
tvlct,r,r,r

   FOR i=0, len-1 DO BEGIN

      IF (readflag EQ 0) THEN BEGIN
         img = data(*,*,i)
      ENDIF ELSE BEGIN

         BREAK_FILE, names(i), a, dir, name, ext
         ext = STRLOWCASE(ext)

         CASE (1) OF

            (ext EQ '.fts') OR (ext EQ '.fits') : BEGIN
               img = READFITS(names(i), hdr)

; Separate case for Yohkoh/SXT data

               IF KEYWORD_SET(SXT) THEN BEGIN
                                ;Fill in generic structure
                  all_hdr(i).filename = names(i)
                  all_hdr(i).detector = 'SXT'
                  all_hdr(i).time_obs = FXPAR(hdr, 'TIME-OBS')
                  tt = FXPAR(hdr, 'DATE-OBS')
                                ;From SXT --> Lasco date format
                  IF(STRMID(tt,6,7) LT '90') THEN century = '19' ELSE century = '20'
                  all_hdr(i).date_obs = century + STRMID(tt, 6, 7) + $
                   STRMID(tt, 2, 4) + STRMID(tt, 0, 2)
                  all_hdr(i).filter = FXPAR(hdr, 'FILTER-B')
                  sunxcen = FXPAR(hdr, 'CRPIX1')
                  sunycen = FXPAR(hdr, 'CRPIX2')
                  resolut = FXPAR(hdr, 'RESOLUT')
                  CASE resolut OF
                     'Full    ': sec_pix = 2.46
                     'Half    ': sec_pix = 4.92
                     'Qrtr    ': sec_pix = 9.84
                  ENDCASE 
               ENDIF
               IF (scl EQ 1)  THEN BEGIN
                  scl=0
                  bmax = MAX(img)
                  bmin = MIN(img)
               ENDIF
            END

            (ext EQ '.gif') : BEGIN
               READ_GIF, names(i), img, r, g, b
               TVLCT, r, g, b
            END

            (ext EQ '.tif') OR (ext EQ '.tiff') : BEGIN
               img = TIFF_READ(names(i), r, g, b, ORDER=order)
               IF (DATATYPE(r) NE 'UND') THEN TVLCT, r, g, b
               IF (order EQ 1) THEN img = ROTATE(img, 7)
            END

            (ext EQ '.pct') OR (ext EQ '.pict') OR (ext EQ '.pic') OR keyword_set(PICT): BEGIN
               READ_PICT, names(i), img, r, g, b
               IF (DATATYPE(r) NE 'UND') THEN TVLCT, r, g, b
            END

	    (ext EQ '.jpg') : BEGIN
		READ_JPEG, names(i), img, ctable, COLORS=!D.N_COLORS-1
		TVLCT, ctable
	    END

	    (ext EQ '.png') : BEGIN
		img = READ_PNG(names[i],r,g,b)
		TVLCT, r,g,b
	    END

            (KEYWORD_SET(MK3)):BEGIN	;** Mark3 512x512 bytearr intarr .rpB files ex. 97d296.19_32.ch0.rpB
               OPENR, IN, names(i), /GET_LUN
               a=ASSOC(IN, INTARR(512,512))
               img = a(0)
               CLOSE, IN
               FREE_LUN, IN
               namelen = STRLEN(names(i))
               aname = STRMID(names(i), namelen-20, 20)
               doy = STRMID(aname,3,3)
               yy = STRMID(aname,0,2)
               hh = STRMID(aname,7,2)
               mm = STRMID(aname,10,2)
               ahdr.filename=aname
               ahdr.detector='MK3'
               ahdr.time_obs=hh+':'+mm
               ahdr.date_obs=UTC2STR(DOY2UTC(FIX(doy), FIX(yy)), /DATE_ONLY)
               all_hdr(i) = ahdr
               sunxcen=255.5
               sunycen=255.5
               sec_pix=9.19
            END

            ELSE : BEGIN
                   PRINT, '%%RT_GENERIC_MOVIE: Unsupported file extension: ', ext
                   RETURN
            END

         ENDCASE

         sz = SIZE(img)
         IF (sz(0) EQ 3) THEN BEGIN
            truecolor=1
            hsize=sz(2)
            vsize=sz(3)
         ENDIF ELSE BEGIN
            truecolor=0
            hsize=sz(1)
            vsize=sz(2)
         ENDELSE

      ENDELSE
	IF keyword_set(RDIFF) THEN BEGIN

		IF i EQ 0 THEN BEGIN
			mdn = median(img)
			mn = MIN(img, max=mx)
			timg = BYTSCL(img,mn + (mdn-mn)/2.,mdn + (mx-mdn)/2.)
			dimg = timg-128
		ENDIF ELSE dimg = img-prev
		prev=img
		img = dimg
	ENDIF
	IF keyword_set(PAN) THEN BEGIN
		hsize=hsize*pan
		vsize=vsize*pan
		IF (pan MOD 2) EQ 0 or (2 MOD pan) EQ 0 THEN img = REBIN(TEMPORARY(img),hsize,vsize) $
			ELSE img = CONGRID(TEMPORARY(img),hsize,vsize,/interp)
	ENDIF


      WINDOW, XSIZE = hsize, YSIZE = vsize, /PIXMAP, /FREE
      win_index(i) = !D.WINDOW
      IF (KEYWORD_SET(truecolor)) THEN TV,BYTSCL(img,bmin,bmax),TRUE=1 ELSE $
      TV, BYTSCL(img, bmin, bmax)
	IF keyword_set(LABEL) THEN xyouts,15,15,label[i],/device
      print,'Done with image ',i+1, ' of ', len
   ENDFOR

   WRUNMOVIE, win_index, SUNXCEN=sunxcen, SUNYCEN=sunycen, SEC_PIX=sec_pix, HDRS=all_hdr, NAMES=names, $
       truecolor=truecolor

END
