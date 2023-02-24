function txtcolor2rvb,txtcolor

color=0l
reads,txtcolor,color,form='(z)'

r=color / (256l^2)
v=(color mod (256l^2)) / 256l
b=(color mod (256l^2)) mod 256l

return,[r,v,b]
end
