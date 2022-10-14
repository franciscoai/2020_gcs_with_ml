;+
; $Id: makewire.pro,v 1.1 2006/09/08 16:03:20 nathan Exp $
;
; PURPOSE:
;  Generate a wireframe for the dragger in raytrace
;
; CATEGORY:
;  raytracing, visualization, 3d
;-


PRO makewire,modelid,neangx=neangx,neangy=neangy,neangz=neangz,modparam=modparam

modelid = strtrim(string(modelid),2)
desc=[0]

case modelid of
    ;27: Spherical Shell
    '27': wire_27,x,y,z,modparam=modparam
    ;28: Cylindrical Shell (SOLID)
    '28': wire_28,x,y,z,desc,modparam=modparam
    ;29: BOW SHOCK:
    '29': wire_29,x,y,z,modparam=modparam ;modparam[0],[1],[2] ;x,y,z,d,s,h
    ;31: graduated cyl.:
    '31': wire_31,x,y,z,desc,modparam=modparam

    ELSE: begin
        cmd = "wire_" + modelid + ",x,y,z"
        tempy = execute(cmd)
    end
ENDCASE

;========================

which,'wire_'+modelid+'.pro',outfile=outfile

dgr = "dragger, x, y, z, POLYGONS=desc, /solid, neangx=neangx,neangy=neangy,neangz=neangz"

if outfile ne '' then $
  tempy = execute(dgr) else begin
    print,'Sorry, this wireframe has not been written.'
 
    base = WIDGET_BASE(TITLE="Error", xsize=300, ysize=100, /COLUMN, $
                      XOFFSET=300, YOFFSET=300)
    label_text = WIDGET_LABEL(base, value="Wireframe model does not exist", /ALIGN_CENTER)
    label_text = WIDGET_LABEL(base, value="Please see MANUAL for help", /ALIGN_CENTER)
    label_blank = WIDGET_LABEL(base, value='')
    exit_butt = WIDGET_BUTTON(base, value="OK", /ALIGN_CENTER, EVENT_PRO="EXIT_PRO")
    WIDGET_CONTROL, base, /REALIZE
    XMANAGER, 'Error', base
    print,'Sorry, this wireframe has not been written, please see MANUAL for help'
    return
end

END
