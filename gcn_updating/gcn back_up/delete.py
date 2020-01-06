import numpy as np
import math
a = [1.0, 1.0, 2.0]
b = [0.0, 180.0, 90.0]
b_rad = np.dot(b, np.pi/180.0)

print(b_rad)
c = [a*np.cos(b_rad), a*np.sin(b_rad)]

print(c)