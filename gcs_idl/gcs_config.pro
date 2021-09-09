;+
;*************************************************************************
; Francisco A. Iglesias  - UTN-FRM/GEAA - franciscoaiglesias@hotmail.com
;
; Include file with general constants 
;
; History:
;	20180704 - First version
;*************************************************************************
;-

;Path to the dir containing /sdo ,/soho and /stereo data directories as well as the /Polar_Observations dir.
DATA_PATH='/media/sf_iglesias_data/data/media/gehme/data'
;Path with our GCS data directories
GCSD_PATH='/media/sf_onedrive_utn/work/repository/2020_gcs_with_ml/gcs_idl'
; LASCO proc images Path
LASCO_PATH=DATA_PATH+'/soho/lasco/level_1/c2'
;Linux File premissions and format
CHMOD_DIR='775'o              ; Permissions of the output directories
CHMOD_FILE='664'o             ; Permissions of the output files
NEWLINE=string([13B])         ; New line charachter
