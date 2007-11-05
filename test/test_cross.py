from hedge.tools import cross, cross2
import pylinear.array as num

v1 = num.array([1,2,3])
v2 = num.array([4,5,6])

print cross(v1, v2)
print cross2(v1, v2)
