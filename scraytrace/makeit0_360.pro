pro makeit0_360,lonslider

while (lonslider lt 0.) do lonslider+=360.
while (lonslider ge 360.) do lonslider-=360.

return
end


function makeit0_360,lonslider
lonsliderout=lonslider
makeit0_360,lonsliderout
return,lonsliderout
end