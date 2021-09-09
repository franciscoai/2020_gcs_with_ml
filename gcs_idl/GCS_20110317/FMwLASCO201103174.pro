PRO FMwLASCO201103174
@gcs_config
;To load images from CORs A&B plus LASCO and call rtsccguicloud with forward model from Thernisien et al. (2009)
;Path to the save file where the GUI parameters will be saved/loaded
SAVE_FILE=GCSD_PATH+'/GCS_20110317/4.sav'
secchipath=DATA_PATH+'/stereo/secchi/L0'
lascopath=DATA_PATH+'/soho/lasco/level_1/c2'
;opens images of interest and backgrounds
; ;RSYNC DATA 
; bla=' /media/sf_iglesias_data/data' ; local copy
; spawn,'rsync -azvh -R '+secchipath+'/a/img/cor2/20110317/level1/20110317_135400_04c2A.fts'+ bla
; spawn,'rsync -azvh -R '+secchipath+'/a/img/cor2/20110317/level1/20110317_133900_04c2A.fts'+ bla
; spawn,'rsync -azvh -R '+secchipath+'/b/img/cor2/20110317/level1/20110317_135400_04c2B.fts'+ bla
; spawn,'rsync -azvh -R '+secchipath+'/b/img/cor2/20110317/level1/20110317_133900_04c2B.fts'+ bla
; spawn,'rsync -azvh -R '+lascopath+'/20110317/25365453.fts'+ bla
; spawn,'rsync -azvh -R '+lascopath+'/20110317/25365452.fts'+ bla
; spawn,'rsync -azvh -R '+secchipath+'/b/img/euvi/20110317/preped/20110317_124530_14euB.fts'+ bla
; spawn,'rsync -azvh -R '+secchipath+'/a/img/euvi/20110317/preped/20110317_124530_14euA.fts'+ bla
; stop 
;SECCHI A
ima=sccreadfits(secchipath+'/a/img/cor2/20110317/level1/20110317_135400_04c2A.fts', hdreventa)
imaprev=sccreadfits(secchipath+'/a/img/cor2/20110317/level1/20110317_133900_04c2A.fts',hdreventpa)
;SECCHI B
imb=sccreadfits(secchipath+'/b/img/cor2/20110317/level1/20110317_135400_04c2B.fts', hdreventb)
imbprev=sccreadfits(secchipath+'/b/img/cor2/20110317/level1/20110317_133900_04c2B.fts', hdreventpb)
;gets masks and scales
ma=get_smask(hdreventa)
mb=get_smask(hdreventb)
;a=bytscl(rebin(alog10(ma*(ima-imaprev) > 1e-12 < 1e-10),512,512))
;b=bytscl(rebin(alog10(mb*(imb-imbprev) > 1e-12 < 1e-10),512,512))
;a=sigrange(rebin((ma*(ima)-ma*(imaprev)),512,512))   ;better for COR1
;b=sigrange(rebin((mb*(imb)-mb*(imbprev)),512,512)) ;better for COR1
a=sigrange(rebin(ma*sigrange(ima)-ma*sigrange(imaprev),512,512)) ;better for COR2
b=sigrange(rebin(mb*sigrange(imb)-mb*sigrange(imbprev),512,512)) ;better for COR2
;opens LASCO of interest and background
lasco1=readfits(lascopath+'/20110317/25365453.fts', lasco1hdr);+'/C3/L0/20130328/32334111.fts', lasco1hdr)
lasco0=readfits(lascopath+'/20110317/25365452.fts', lasco0hdr);+'/C3/L0/20130328/32334110.fts', lasco0hdr)
lascohdr = LASCO_FITSHDR2STRUCT(lasco1hdr)
print, 'DATE_OBS:  ', lascohdr.date_obs
lasco1=rebin(lasco1, 512, 512)
lasco0=rebin(lasco0, 512, 512)
;lasco=rebin(((lasco1-lasco0)>(-2.e-11)<4.0e-11),512,512)   ;para C3:  >(-3.e-12)<3.e-12), 512,512)
lasco=sigrange(lasco1-lasco0)
lasco= lasco>(-0.5e-10) <0.5e-10
;lasco= lasco>(-0.83e-9) <7.5e-10
;EUVIs
imeuvib=sccreadfits(secchipath+'/b/img/euvi/20110317/preped/20110317_124530_14euB.fts', heuvib)
imeuvia=sccreadfits(secchipath+'/a/img/euvi/20110317/preped/20110317_124530_14euA.fts', heuvia)
;This one replaces one of the EUVIs for SDO AIA
;imeuvib=sccreadfits(DATA_PATH+'/sdo/aia/L1/193/20110317/preped/-', heuvib)
;secchi_prep,eveuvia,heuvia,imeuvia,/PRECOMMCORRECT_ON
;secchi_prep,eveuvib,heuvib,imeuvib,/PRECOMMCORRECT_ON
imea=alog10(rebin(imeuvia,512,512) > 1)
imeb=alog10(rebin(imeuvib,512,512) > 1)
;calls application
Result = FILE_TEST(SAVE_FILE)
IF Result EQ 1 THEN restore, SAVE_FILE

rtsccguicloud, a, b, hdreventa, hdreventb, imlasco=lasco, hdrlasco=lascohdr, imeuvia=imea, $
	hdreuvia=heuvia, imeuvib=imeb, hdreuvib=heuvib, sgui=sgui, sparaminit=sgui, ssim=ssim

help, ssim

stop

;save, filename=SAVE_FILE, sgui
END

