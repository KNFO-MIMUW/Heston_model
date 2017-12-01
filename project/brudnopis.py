import numpy as np
from scipy import integrate
import math as math

theta = np.array([0.09, 0.09, -0.3, 2, 1])
theta1 = np.array([0.09, 0.09, -0.3, 20, 10])
theta2 = np.array([9, 23, -3, 20, 10])
x = np.array([2,3,4])

A = np.vstack((theta,theta1,theta2))
B = np.vstack(([0.09, 0.09, -0.3, 2, 1],[0.09, 0.09, -0.3, 20, 10]))
#print(A[1,4])
#print(len(A))
#print(A)
#print(B)
#print(max(theta))
#print(np.hstack((np.transpose(theta),theta)))
print(np.dot(A,np.transpose(A)).diagonal().max())

