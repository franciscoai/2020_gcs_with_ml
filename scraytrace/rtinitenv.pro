;+
;  $Id: rtinitenv.pro,v 1.10 2010/09/08 15:48:21 thernis Exp $
;
; PURPOSE:
;  Init environment variables for the raytrace soft
;
; CATEGORY:
;  raytracing, data handling, os
;
; INPUTS:
;  forceinit : force reinitialization of the variables
;  forcelibfile : force the path and filename of the shared object to
;                 test out. Useful when the user want to bypass the
;                 ssw default shared object. As an alternative, the
;                 user can also set the envvar RT_FORCELIBFILE.
;  forcelibthread : force the path and filename of the shared object 
;                   using threadings. Note that the boost package 
;                   abd boost_thread must be installed on your machine.
;                   The libboost_thread.so should be present in the
;                   shared library search path (LD_LIBRARY_PATH on unix and linux).
;                   See the scraytrace user's guide for more information.
;
; DESCRIPTION:
;  Defines different environment variables depending on the 
;  operating system.
;   RT_PATH : path where the IDL raytrace sources are
;   RT_SOSUBPATH : subdirectory where the library are: this is
;                  determined function of the OS and processor type
;   RT_SOFILENAME : the filename of the library
;   RT_RUNFROM : 'ssw' if run from ssw
;                'local' if run from a local copy
;   RT_FORCELIBFILE : Set to the path and filename of the shared
;                     object in order to bypass the ssw precompiled one.
;   RT_LIBFILE : contains the shared object path and filename.
;   RT_LIBFILETHREAD : contains the path and filename of the shared object
;                      used for multi-threading.
;
;-
pro rtinitenv,forceinit=forceinit,forcelibfile=forcelibfile,forcelibthread=forcelibthread,help=help

if keyword_set(help) then begin
	print,'pro rtinitenv,forceinit=forceinit,forcelibfile=forcelibfile,forcelibthread=forcelibthread,help=help'

	print,'RT_PATH : ',getenv('RT_PATH')
	print,'RT_SOFILENAME : ',getenv('RT_SOFILENAME')
	print,'RT_SOTHREADFILENAME : ',getenv('RT_SOTHREADFILENAME')
	print,'RT_SOEXTENSION : ',getenv('RT_SOEXTENSION')
	print,'RT_RUNFROM : ',getenv('RT_RUNFROM')
	print,'RT_DATAPATH : ',getenv('RT_DATAPATH')
	print,'SSW : ',getenv('SSW')
	print,'RT_SOSUBPATH : ',getenv('RT_SOSUBPATH')
	print,'RT_FORCELIBFILE : ',getenv('RT_FORCELIBFILE')
	print,'RT_LIBFILE : ',getenv('RT_LIBFILE')
	print,'RT_FORCELIBTHREAD : ',getenv('RT_FORCELIBTHREAD')
	print,'RT_LIBFILETHREAD : ',getenv('RT_LIBFILETHREAD')
	return
endif

; ---- memory if already done
common rtinitenv,flagdone,flagfailed

; ---- return right away if already done
if n_elements(flagdone) eq 0 then begin 
    flagdone=0B
    flagfailed=0B
endif

if not flagfailed and flagdone and not keyword_set(forceinit) and not keyword_set(forcelibfile) and not keyword_set(forcelibthread) then return

; ---- find out in which dir that prog is located
which,'rtinitenv.pro',outfile=x,/quiet
rtpath = STRMID(x,0,strlen(x)-29)
setenv,'RT_PATH='+rtpath
sofilename='libraytrace'
sothreadfilename='libraytracethread'

case strlowcase(!version.os_family) of
    'unix' : if strlowcase(!version.os) eq 'darwin' then sofileextension='dylib' else sofileextension='so'
    'win' : sofileextension='dll'
    else : message,'I don''t deal with that OS family...'
endcase

setenv,'RT_SOFILENAME='+sofilename+'.'+sofileextension
setenv,'RT_SOTHREADFILENAME='+sothreadfilename+'.'+sofileextension
setenv,'RT_SOEXTENSION='+sofileextension

psep=path_sep()

; ---- figure out if ran from solarsoft
sswpath=getenv('SSW')
if strmid(rtpath,strlen(sswpath)) eq sswpath then begin
    ; ---- it's ran from ssw
    setenv,'RT_RUNFROM=ssw'
    ; ---- set the data path
    setenv,'RT_DATAPATH='+getenv('SCC_DATA')+psep+'scraytrace'
endif else begin
    ; ---- else it should be ran from a local copy
    ;      in that case the lib should be in the lib sub dir
    setenv,'RT_RUNFROM=local'
    ; ---- set the data path
    setenv,'RT_DATAPATH='+rtpath+psep+'data'+psep+'scraytrace'
endelse

; ---- Now chose the right lib depending on OS
oslibpath=!version.os+psep+!version.arch
setenv,'RT_SOSUBPATH=lib'+psep+oslibpath
libfile=rtpath+psep+getenv('RT_SOSUBPATH')+psep+getenv('RT_SOFILENAME')
libfilethread=rtpath+psep+getenv('RT_SOSUBPATH')+psep+getenv('RT_SOTHREADFILENAME')

; ---- force shared library if requested
if keyword_set(forcelibfile) gt 0 then setenv,'RT_FORCELIBFILE='+forcelibfile
rtforcelibfile=getenv('RT_FORCELIBFILE')
if strlen(rtforcelibfile) gt 0 && ~keyword_set(forceinit) then begin
    setenv,'RT_LIBFILE='+rtforcelibfile 
    libfile=rtforcelibfile
endif else setenv,'RT_LIBFILE='+libfile

; ---- force shared library using boost thread if requested
if keyword_set(forcelibthread) gt 0 then setenv,'RT_FORCELIBTHREAD='+forcelibthread
rtforcelibthread=getenv('RT_FORCELIBTHREAD')
if strlen(rtforcelibthread) gt 0 && ~keyword_set(forceinit) then begin
    setenv,'RT_LIBFILETHREAD='+rtforcelibthread
    libfilethread=rtforcelibthread
endif else setenv,'RT_LIBFILETHREAD='+libfilethread

; -- check if the OS and arch version of the lib is available
libexist=file_test(libfile)
if not libexist then begin
    case 1 of
        (getenv('RT_RUNFROM') eq 'ssw') : begin
            message,'Cannot find '+libfile,/info
            message,'The library for your operating system and architechture does not exist.',/info
            message,'Found OS:'+!version.os+', architecture: '+!version.arch,/info
            message,'Please see the Solar Corona Raytrace manual if you would like to compile the sources for this configuration.',/info
            message,'You can use RT_FORCELIBFILE environment variable or',/info
            message,'forcelibfile keyword for rtinitenv in order to force',/info
            message,'the use of a library if one is known to be available for your achitecture.',/info
            flagfailed=1b
        end
        (getenv('RT_RUNFROM') eq 'local') : begin
            message,'The rtinitenv you are using does not seem to be from the SSW tree.',/info
            message,'Cannot find the shared object library.',/info
            message,'Please use RT_FORCELIBFILE environment variable or',/info
            message,'forcelibfile keyword for rtinitenv in order to force',/info
            message,'the use of a library if one is known to be available for your achitecture.',/info
        end
        else : 

    endcase

    
endif

; ---- set the flag to remember that the init has been done
flagdone=1B

return
end

;
; CVSLOG:
;  $Log: rtinitenv.pro,v $
;  Revision 1.10  2010/09/08 15:48:21  thernis
;  Print all environment variables when /help
;
;  Revision 1.9  2009-02-23 22:30:55  thernis
;  Fix bug with forcelibthread
;
;  Revision 1.8  2009/02/10 16:45:37  thernis
;  - Implement uv integration (experimental for now)
;  - Implement multi threading raytracing
;
;  Revision 1.7  2008/07/03 20:08:05  thernis
;  Implement help keyword
;
;  Revision 1.6  2008/05/16 20:08:49  thernis
;  Implement RT_SOEXTENSION environment variable
;
;  Revision 1.5  2007/07/23 20:12:11  thernis
;  Fix erroneous path
;
;  Revision 1.4  2007/07/20 20:21:34  ontivero
;  fix path of the shared object
;
;  Revision 1.3  2006/11/01 20:22:50  thernis
;  Modify choice of library extension depending on the OS. WARNING: implemented but not tested yet for darwin Mac OS.
;
;  Revision 1.2  2006/10/30 21:51:27  thernis
;  Add new keyword to allow forcing the path and filename of the shared object library.
;
;
;
