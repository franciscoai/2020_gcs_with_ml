pro compare
  
sObj = OBJ_NEW('IDL_Savefile', '/gehme/projects/2020_gcs_with_ml/repo/gcs_idl/arguments/input_ok.sav')
sContents = sObj->Contents()
PRINT, sContents.N_VAR

sNames = sObj->Names()
PRINT, "PRINTING INPUTS"
PRINT, sNames

RESTORE, '/gehme/projects/2020_gcs_with_ml/repo/gcs_idl/arguments/input_ok.sav'


FOR i = 0, (sContents.N_VAR -1) DO begin
    void = EXECUTE('bla = ' + sNames[i])
    print,'*************', sNames[i]
    help, bla
    print, "min: ",min(bla)
    print, "mean: ",mean(bla,/DOUBLE)
    print, "max: ",max(bla)
    ;print, bla
    A = GET_KBRD()
    print, A
    ENDFOR



sObj_out = OBJ_NEW('IDL_Savefile', '/gehme/projects/2020_gcs_with_ml/repo/gcs_idl/arguments/output_ok.sav')
sContents_out = sObj_out->Contents()
PRINT, sContents_out.N_VAR

sNames_out = sObj_out->Names()
PRINT, "PRINTING OUTPUTS"
PRINT, sNames_out

RESTORE, '/gehme/projects/2020_gcs_with_ml/repo/gcs_idl/arguments/output_ok.sav'


FOR i = 0, (sContents_out.N_VAR -1) DO begin
    void = EXECUTE('bla = ' + sNames_out[i])
    print,'*************', sNames_out[i]
    help, bla
    print, "min: ",min(bla)
    print, "mean: ",mean(bla,/DOUBLE)
    print, "max: ",max(bla)
    ;print, bla
    A = GET_KBRD()
    print, A
    ENDFOR    

end