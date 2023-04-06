PRO gcs_testcase_full        
;To load images from CORs A&B plus LASCO and call rtsccguicloud with forward model from Thernisien et al. (2009)
secchipath='/gehme/data/stereo/secchi/L0/'  
lascopath='/gehme/data/soho/lasco/level_05/c2/' 

;event 1
;SECCHI A
ima=sccreadfits(secchipath+'/a/img/cor2/20110317/level1/20110317_133900_04c2A.fts', hdreventa)
imaprev=sccreadfits(secchipath+'/a/img/cor2/20110317/level1/20110317_132400_04c2A.fts',hdreventpa)
;SECCHI B
imb=sccreadfits(secchipath+'/b/img/cor2/20110317/level1/20110317_133900_04c2B.fts', hdreventb)
imbprev=sccreadfits(secchipath+'/b/img/cor2/20110317/level1/20110317_132400_04c2B.fts', hdreventpb)

;event 2
;SECCHI A
;ima=sccreadfits(secchipath+'a/img/cor2/20110303/20110303_080800_d4c2a.fts', hdreventa)
;imaprev=sccreadfits(secchipath+'a/img/cor2/20110303/20110303_020800_d4c2a.fts',hdreventpa)  
;SECCHI B
;imb=sccreadfits(secchipath+'b/img/cor2/20110303/20110303_080915_n4c2b.fts', hdreventb)
;imbprev=sccreadfits(secchipath+'b/img/cor2/20110303/20110303_040915_n4c2b.fts', hdreventpb)


;******************** COR2 ******************** 
;******************* COR2A ********************

;opens images of interest and backgrounds

;prepares nice images
;ima=sigma_filter(ima,3,n_sigma=1.5, /iterate)
;imaprev=sigma_filter(imaprev,3,n_sigma=1.5, /iterate)
ima=sigrange(ima)
imaprev=sigrange(imaprev)
a=rebin(ima-imaprev, 512,512) ; de aqui en adelante mi imagen de interés es 'a'
afil=sigma_filter(a,3,n_sigma=1.5, /iterate)
ma=get_smask(hdreventa)  ;gets mask
maskcor=where((rebin(ma,512,512)) EQ 0)
a=sigma_filter(a,2,n_sigma=2., /iterate)
a(maskcor)=min(a)
window, 17, xsize=512, ysize=512
tvscl, a >(-28.)<40.

;gets coordinates of Sun center
wcs = fitshead2wcs(hdreventa)
suncenter = wcs_get_pixel(wcs, [0,0])

;draws limb
arcs = hdreventa.cdelt1          ;arcsec per pixel
asolr = hdreventa.rsun           ;solar radius in arcsec
r_sun = asolr/arcs
TVCIRCLE, r_sun*0.25, suncenter[0]/4., (suncenter[1]/4.), COLOR=255, THICK=1

charsz=1.5
time_color=255
xouts=10 & youts=10
XYOUTS, xouts, youts, strmid(hdreventa.date_obs,0,4)+'/'+strmid(hdreventa.date_obs,5,2)+$
'/'+strmid(hdreventa.date_obs,8,2)+' '+strmid(hdreventa.date_obs,11,8),/DEVICE, CHARSIZE=charsz, COLOR=time_color;, font=1
camt=hdreventa.detector+'-'+strmid(hdreventa.filename,strpos(hdreventa.filename,'.f')-1,1)
cyouts=youts+ charsz*10 + 5
XYOUTS, 512-10, youts, 'STEREO/'+strupcase(camt), /DEVICE,  CHARSIZE=charsz, COLOR=time_color, alignment=1.0

;saves image from screen
TVLCT, R, G, B, /GET
ima = TVRD()
wdelete, 17
;******************* COR2B ********************
;prepares nice images
;b=hist_equal((rebin((ma*(ima)-ma*(imaprev)),512,512)), per=1.)   ;better for COR1:per=5. ;COR2:per=1.
;b=sigrange(rebin(mb*sigrange(imb)-mb*sigrange(imbprev),512,512)) ;better for COR2 ;ESTE USÉ PARA LAS IMÁGENES DEL PPT
imb=sigrange(imb)
imbprev=sigrange(imbprev)
b=rebin(imb-imbprev, 512,512) ; de aqui en adelante mi imagen de interés es 'b'
afil=sigma_filter(b,3,n_sigma=1.5, /iterate)
mb=get_smask(hdreventb)  ;gets mask
maskcor=where((rebin(mb,512,512)) EQ 0)
b=sigma_filter(b,2,n_sigma=2., /iterate)
b(maskcor)=min(b)
window, 15, xsize=512, ysize=512
tvscl, b >(-28.)<40.

;gets coordinates of Sun center
wcs = fitshead2wcs(hdreventb)
suncenter = wcs_get_pixel(wcs, [0,0])

;draws limb
arcs = hdreventb.cdelt1          ;arcsec per pixel
asolr = hdreventb.rsun           ;solar radius in arcsec
r_sun = asolr/arcs
TVCIRCLE, r_sun*0.25, suncenter[0]/4., (suncenter[1]/4.), COLOR=255, THICK=1

charsz=1.5
time_color=255
xouts=10 & youts=10
XYOUTS, xouts, youts, strmid(hdreventb.date_obs,0,4)+'/'+strmid(hdreventb.date_obs,5,2)+$
'/'+strmid(hdreventb.date_obs,8,2)+' '+strmid(hdreventb.date_obs,11,8),/DEVICE, CHARSIZE=charsz, COLOR=time_color;, font=1
camt=hdreventb.detector+'-'+strmid(hdreventb.filename,strpos(hdreventb.filename,'.f')-1,1)
cyouts=youts+ charsz*10 + 5
XYOUTS, 512-10, youts, 'STEREO/'+strupcase(camt), /DEVICE,  CHARSIZE=charsz, COLOR=time_color, alignment=1.0

;saves image from screen
TVLCT, R, G, B, /GET
imb = TVRD()
wdelete, 15

;********************LASCO********************

;opens LASCO of interest and background
lasco1=readfits(lascopath+'20110303/22363800.fts', lasco1hdr)
lasco0=readfits(lascopath+'20110303/22363788.fts', lasco0hdr)
lascohdr = LASCO_FITSHDR2STRUCT(lasco1hdr)

;gets coordinates of Sun center
suncenx=lascohdr.crpix1
sunceny=lascohdr.crpix2
;cosmic ray removal
;c3_clean_v1, lasco1, suncenx, sunceny
;c3_clean_v1, lasco0, suncenx, sunceny

print, 'DATE_OBS:  ', lascohdr.time_obs
lasco1=rebin(lasco1, 512, 512)
lasco0=rebin(lasco0, 512, 512)
;lasco=rebin(((lasco1-lasco0)>(-2.e-11)<4.0e-11),512,512)   ;para C3:  >(-3.e-12)<3.e-12), 512,512)
;lasco=sigrange(lasco1-lasco0)
;lasco=hist_equal(lasco1-lasco0, per=0.2)
lasco= (lasco1-lasco0)>(-180)<500         ;>(-0.83e-9) <7.5e-10

;masks occulter
arcs = GET_SEC_PIXEL(lascohdr, FULL=sizeimg)          ;arcsec per pixel
asolr = GET_SOLAR_RADIUS(lascohdr)           ;solar radius in arcsec
r_sun = asolr/arcs
r_occ = r_sun/2. * 2.2		; 4. for C3
r_occ_out = r_sun/2. * 7.	; 31.5 for C3

sizeimg=512
WINDOW, 16, XSIZE=sizeimg, YSIZE=sizeimg, /PIXMAP    ; FFV
tmp_img = lasco
tmp_img(*) = 0
TV,tmp_img
TVCIRCLE, r_occ_out,suncenx/2., sunceny/2., /FILL, COLOR=1
TVCIRCLE, r_occ, suncenx/2., sunceny/2., /FILL, COLOR=2
tmp_img = TVRD()
outer = WHERE(tmp_img EQ 0)
occ = where(tmp_img EQ 2)
maskfill= min(lasco);ESTE USÉ PARA LAS IMÁGENES DEL PPT 100  ;for C2: 700
lasco(occ) = maskfill
lasco(outer) = maskfill

window, 18, xsize=512, ysize=512
tvscl, lasco

;draws limb
TVCIRCLE, r_sun*0.5, suncenx/2., sunceny/2., COLOR=255, THICK=1

;displays date & time
;device, set_font='Helvetica', /TT_Font
charsz=1.5
time_color=255
xouts=10 & youts=10
XYOUTS, xouts, youts, strmid(lascohdr.date_obs,0,4)+'/'+strmid(lascohdr.date_obs,5,2)+$
'/'+strmid(lascohdr.date_obs,8,2)+' '+strmid(lascohdr.time_obs,0,8),/DEVICE, CHARSIZE=charsz, COLOR=time_color;, font=1
camt=lascohdr.detector;+'-'+strmid(lascohdr.filename,strpos(lascohdr.filename,'.f')-1,1)
cyouts=youts+ charsz*10 + 5
XYOUTS, 512-10, youts, 'SOHO/LASCO '+strupcase(camt), /DEVICE,  CHARSIZE=charsz, COLOR=time_color, alignment=1.0

;saves image from screen
TVLCT, R, G, B, /GET
lasco = TVRD()
wdelete,18

;******************** EUVI ********************

;EUVIs
;imeuvib=sccreadfits(secchipath+'/b/img/euvi/20130617/preped/20130617_124530_14euB.fts', heuvib)
;imeuvia=sccreadfits(secchipath+'/a/img/euvi/20130617/preped/20130617_124530_14euA.fts', heuvia)
;This one replaces one of the EUVIs for SDO AIA
;imeuvib=sccreadfits('/media/hebe/Datos/Data/SDO/AIA/20101214/preped/AIA20101214_153407_0193.fits', heuvib)
;secchi_prep,eveuvia,heuvia,imeuvia,/PRECOMMCORRECT_ON
;secchi_prep,eveuvib,heuvib,imeuvib,/PRECOMMCORRECT_ON
;imea=alog10(rebin(imeuvia,512,512) > 1)
;imeb=alog10(rebin(imeuvib,512,512) > 1)



;******************** calls application ********************
;calls application
SAVE_FILE='/gehme/projects/2020_gcs_with_ml/data/gcs_idl/gcs_testcase_full.sav'
Result = FILE_TEST(SAVE_FILE)
IF Result EQ 1 THEN begin
print,"*************************************leyendo archivo "
restore, SAVE_FILE
endif

rtsccguicloud, ima, imb, hdreventa, hdreventb, sgui=sguiout, sparaminit=sguiout ;imlasco=lasco, hdrlasco=lascohdr ;ocout=oc, imeuvia=imea, hdreuvia=heuvia, imeuvib=imeb, hdreuvib=heuvib

Result = FILE_TEST(SAVE_FILE)
IF Result EQ 0 THEN save, filename=SAVE_FILE, sguiout

savepath='/gehme/projects/2020_gcs_with_ml/data/gcs_testcase_output'		;GERA dónde guardará imágenes de ajuste, editar para cada fecha-evento

;  sgui : returns a structure containing all the different parameters of the GUI.
;    sgui.lon : longitude Carrington.
;    sgui.lat : latitude.
;    sgui.rot : tilt angle or rotation around the model axis of symmetry. 0 is parallel to the equator.
;    sgui.han : half angle between the model's feet.
;    sgui.hgt : height, in Rsun.
;    sgui.rat : aspect ratio

; pasamos los datos a grados y le restamos la longitud Carrington para pasarlos a coordenadas Stonyhurst
carr=tim2carr(hdreventa.date_obs)	; esto nos devuelve la longitud L_0 Carrington para el día
lon=sguiout.lon*180./!dpi
lon=sguiout.lon*180./!dpi-carr
if (lon lt 0.) then lon=360.+lon

; pasamos los datos a grados
lat=sguiout.lat*180./!dpi
tilt=sguiout.rot*180./!dpi
half=sguiout.han*180./!dpi

;saves all GCS parameters in csv file
csvfile=savepath + strmid(hdreventa.filename,0,13) +'_params.csv'
openw, 1, csvfile
printf,1,hdreventa.date_obs,',',lascohdr.time_obs,',',lon,',',lat,',',tilt,',',sguiout.hgt,',',sguiout.rat,',',half
close, 1

;saves xyz point coordinates of the wireframe in ascii file
;openw, 1, '/Users/hebe/Work/Data/stereo/xyz_2020-11-26_24Rs.txt'
;printf, 1, oc
;close, 1

;saves image from screen
;window 1 = COR B
;window 0 = COR A
;window 20 = LASCO

wset,20
TVLCT, R, G, B, /GET
imagetmp = TVRD(True=1);color24)
imgfilesave = savepath + strmid(lascohdr.date_obs,0,4)+strmid(lascohdr.date_obs,5,2)+strmid(lascohdr.date_obs,8,2)+$
'_'+strmid(lascohdr.time_obs,0,2)+strmid(lascohdr.time_obs,3,2)+'_lasco'	;(LASCO)
WRITE_PNG, imgfilesave+'.png', imagetmp,r,g,b

wset, 0
TVLCT, R, G, B, /GET
imagetmp = TVRD(True=1);color24)
imgfilesave = savepath + strmid(hdreventa.filename,0,13) +'_cor2a'	;(STA)
WRITE_PNG, imgfilesave+'.png', imagetmp,r,g,b

wset, 1
TVLCT, R, G, B, /GET
imagetmp = TVRD(True=1);color24)
imgfilesave = savepath + strmid(hdreventb.filename,0,13) +'_cor2b'	;(STB)
WRITE_PNG, imgfilesave+'.png', imagetmp,r,g,b

END

