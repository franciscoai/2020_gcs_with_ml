function calcmodel54distjunc,leadingedgeheight,k,hang,reverse=reverse
if keyword_set(reverse) then out=leadingedgeheight/((1.-k)*cos(hang)/(1.+sin(hang))) else out=leadingedgeheight*(1.-k)*cos(hang)/(1.+sin(hang))
return,out
end