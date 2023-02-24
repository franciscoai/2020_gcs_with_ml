function rvb2txtcolor,rvb

color=0l
bvr=reverse(rvb)

for i=0,2 do color+=bvr[i]*(256l^i)

return,string(color,form='(z6.6)')
end
