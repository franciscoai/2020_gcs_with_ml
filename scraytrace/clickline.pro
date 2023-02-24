function clickline

print,'Click first point of line...'

cursor,x1,y1,/down,/device
cursor,xxx,yyy,/up

plots,x1,y1,psym=1,/device

print,'Click second point of line...'

cursor,x2,y2,/down,/device
cursor,xxx,yyy,/up

plots,x1,y1,/device
plots,x2,y2,/continue,/device
plots,x2,y2,psym=1,/device

return,[[x1,y1],[x2,y2]]
end