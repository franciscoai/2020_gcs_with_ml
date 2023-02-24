;+
; PURPOSE: Compute CME contour from hand drawn contour
;
; $Id: cloudautofit_computecontourmap.pro,v 1.2 2011/12/01 21:05:16 nathan Exp $
; $Log: cloudautofit_computecontourmap.pro,v $
; Revision 1.2  2011/12/01 21:05:16  nathan
; use select_windows for windows compat
;
; Revision 1.1  2009/02/05 20:16:43  thernis
; First commit
;
;-

pro cloudautofit_computecontourmap,contoura,contourb,hdra,hdrb,imdistA,imdistB,masksunA,masksunB,sumA,sumB,imcontA,imcontB,hglghtcontfactor=hglghtcontfactor

; ---- compute distance map for the A and B contours
set_plot,'z'
device,set_resolution=[512,512]
erase
plots,contoura[0,*],contoura[1,*],/device
imcontA=tvrd()
imcontA/=max(imcontA)
erase
plots,contourb[0,*],contourb[1,*],/device
imcontB=tvrd()
imcontB/=max(imcontB)
select_windows	

; -- generate kernel
kernel=dist(1024)
kernel*=-1
kerminmax=minmax(kernel)
kernel=(kernel-kerminmax[0])/(kerminmax[1]-kerminmax[0])
kernel=exp((kernel^2)/0.02)

; -- use fft convolution
imdistA=fftconvol512(imcontA,kernel)
imdistB=fftconvol512(imcontB,kernel)
; - reinforce edge contrast
if n_elements(hglghtcontfactor) eq 0 then hglghtcontfactor=1.25
ma=where(imcontA eq 1,cnta)
if cnta gt 0 then imdistA[ma]=hglghtcontfactor
mb=where(imcontB eq 1,cntb)
if cntb gt 0 then imdistB[mb]=hglghtcontfactor
; -- normalize
imdistA/=max(imdistA)
sumA=float(cnta) ;total(imdistA)
imdistB/=max(imdistB)
sumB=float(cntb) ;total(imdistB)


; ---- compute occulting mask
; --- for A
wcsa=fitshead2wcs(hdra)
suncntrpix=wcs_get_pixel(wcsa,[0.,0.])
suncntrpix=piximchangereso(suncntrpix,-2)
masksunA=mkcirc(512,512,suncntrpix[0],suncntrpix[1],getpixperrsun(hdra)/4.*3.)
masksunA=float(-fix(masksunA)+1)
; --- for B
wcsb=fitshead2wcs(hdrb)
suncntrpix=wcs_get_pixel(wcsb,[0.,0.])
suncntrpix=piximchangereso(suncntrpix,-2)
masksunB=mkcirc(512,512,suncntrpix[0],suncntrpix[1],getpixperrsun(hdrb)/4.*3.)
masksunB=float(-fix(masksunB)+1)


return
end
