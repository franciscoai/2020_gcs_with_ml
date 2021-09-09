;Loads input data to run rtsccguicloud with the forward model from Thernisien et al. (2009)
pro FMwLASCO20110317m1

@gcs_config
;Differential image scaling [SECCHIA,SECCHIA, SECCHIB,SECCHIB, AIA, AIA]
RNG=1.*[-1,1,-1,1,-0.1,0.1]

;Global Paths
SAVE_FILE=GCSD_PATH+'/GCS_20110317/m1.sav'
secchipath=DATA_PATH+'/stereo/secchi/L0'
lascopath=DATA_PATH+'/soho/lasco/level_1/c2'
sdopath=DATA_PATH+'/sdo'
PLAY_MOVIE=[0,0,0] ; set to 1 to play the corresponding movie [euvia, euvib, aia]

;Reads and prepeares data
;date for both stereos
SDATE0='20110317'
SDATE ='20110317'
STIME0='110530'
STIME ='111530'

; play movies
if PLAY_MOVIE[0] then begin
 euvia_mov=DATA_PATH+'/stereo/secchi/L0/a/img/euvi/'+SDATE0+'/preped/'+SDATE0+'_EUVIA195_rdiff.mvi'
 scc_playmovie, euvia_mov
endif
if PLAY_MOVIE[1] then begin
 euvia_mov=DATA_PATH+'/stereo/secchi/L0/b/img/euvi/'+SDATE0+'/preped/'+SDATE0+'_EUVIB195_rdiff.mvi'
 scc_playmovie, euvia_mov
endif
if PLAY_MOVIE[2] then begin
 euvia_mov=DATA_PATH+'/sdo/aia/L1/193/'+SDATE0+'/preped/'+SDATE0+'_AIA193_rdiff.mvi'
 scc_playmovie, euvia_mov
endif

;SECCHIA/EUVI (to be used as CORA)
peuvia0=secchipath+'/a/img/euvi/'+SDATE0+'/preped/'+SDATE0+'_'+STIME0+'_14euA.fts'
peuvia =secchipath+'/a/img/euvi/'+SDATE+'/preped/'+SDATE+'_'+STIME+'_14euA.fts'
if ~file_test(peuvia0) then begin 
  bla=secchipath+'/a/img/euvi/'+SDATE0+'/'+SDATE0+'_'+STIME0+'_n4euA.fts'
  euvia0=sccreadfits(bla,euviahdr0)
  secchi_prep,bla,euviahdr0,euvia0,/PRECOMMCORRECT_ON, /write_fts, savepath=file_dirname(peuvia0) ; preprocessing 
endif
if ~file_test(peuvia) then begin 
  bla =secchipath+'/a/img/euvi/'+SDATE+'/'+SDATE+'_'+STIME+'_n4euA.fts' 
  euvia =sccreadfits(bla,euviahdr)
  secchi_prep,bla,euviahdr,euvia,/PRECOMMCORRECT_ON, /write_fts, savepath=file_dirname(peuvia) ; preprocessing 
endif
euvia0=sccreadfits(peuvia0,euviahdr0)
euvia =sccreadfits(peuvia,euviahdr)
euvia=sigrange(rebin(euvia-euvia0,512,512))

;SECCHIB/EUVI (to be used as CORB)
peuvib0=secchipath+'/b/img/euvi/'+SDATE0+'/preped/'+SDATE0+'_'+STIME0+'_14euB.fts'
peuvib =secchipath+'/b/img/euvi/'+SDATE+'/preped/'+SDATE+'_'+STIME+'_14euB.fts'
if ~file_test(peuvib0) then begin 
  bla=secchipath+'/b/img/euvi/'+SDATE0+'/'+SDATE0+'_'+STIME0+'_n4euB.fts'
  euvib0=sccreadfits(bla,euvibhdr0)
secchi_prep,bla,euvibhdr0,euvib0,/PRECOMMCORRECT_ON, /write_fts, savepath=file_dirname(peuvib0); preprocessing 
endif
if ~file_test(peuvib) then begin 
  bla =secchipath+'/b/img/euvi/'+SDATE+'/'+SDATE+'_'+STIME+'_n4euB.fts' 
  euvib =sccreadfits(bla,euvibhdr)
  secchi_prep,bla,euvibhdr,euvib,/PRECOMMCORRECT_ON, /write_fts, savepath=file_dirname(peuvib) ; preprocessing 
endif
euvib0=sccreadfits(peuvib0,euvibhdr0)
euvib =sccreadfits(peuvib,euvibhdr)
euvib=sigrange(rebin(euvib-euvib0,512,512))

;SDO/AIA (to be used as LASCO)
DATE0=['2011','03','17']
TIME0=['11','05','07']
DATE=['2011','03','17']
TIME=['11','15','07']

; AIA preprocessing if necesary
oaia0=sdopath+'/aia/L1/193/'+strjoin(DATE0)+'/preped/AIA'+strjoin(DATE0)+'_'+strjoin(TIME0)+'_0193.fits'
oaia =sdopath+'/aia/L1/193/'+strjoin(DATE)+'/preped/AIA'+strjoin(DATE)+'_'+strjoin(TIME)+'_0193.fits'
if ~file_test(oaia0) then begin 
  paia0=sdopath+'/aia/L1/193/'+strjoin(DATE0)+'/aia.lev1.193A_'+DATE0[0]+'-'+DATE0[1]+'-'+$ 
   DATE0[2]+'T'+TIME0[0]+'_'+TIME0[1]+'_'+TIME0[2]+'.84Z.image_lev1.fits'
  aia_prep, paia0, -1, outdir=file_dirname(oaia0),/do_write_fits ; preprocessing
endif
if ~file_test(oaia) then begin
  paia =sdopath+'/aia/L1/193/'+strjoin(DATE)+'/aia.lev1.193A_'+DATE[0]+'-'+DATE[1]+'-'+$ 
   DATE[2]+'T'+TIME[0]+'_'+TIME[1]+'_'+TIME[2]+'.84Z.image_lev1.fits'
  aia_prep, paia, -1, outdir=file_dirname(oaia),/do_write_fits ; preprocessing
endif

;reads AIA preprocessed data
aia0=sccreadfits(oaia0, aiahdr)
aia =sccreadfits(oaia, aiahdr)
aia=sigrange(rebin(aia-aia0, 512, 512))

;diff images scaling
euvia=smooth(bytscl(euvia,min=mean(euvia)+RNG[0]*stddev(euvia), max=mean(euvia)+RNG[1]*stddev(euvia)),3)
euvib=smooth(bytscl(euvib,min=mean(euvib)+RNG[2]*stddev(euvib), max=mean(euvib)+RNG[3]*stddev(euvib)),3)
aia=smooth(bytscl(aia,min=mean(aia)+RNG[4]*stddev(aia), max=mean(aia)+RNG[5]*stddev(aia)),3)


;calls application, loads and saves the fit parameters in SAVE_FLIE
Result = file_test(SAVE_FILE)
if Result eq 1 then restore, SAVE_FILE
rtsccguicloud, euvia, euvib, euviahdr, euvibhdr, imlasco=aia, hdrlasco=aiahdr, sgui=sgui, sparaminit=sgui
save, filename=SAVE_FILE, sgui

end

