
import numpy as np



x = [[1,5,2],[2,4,6],[3,6,9]]
y = [2,4,6]
print x[0]
x = np.matrix(x)
print x.shape
print type(x)

print x+y
axis0= np.sum(x,axis=0)
axis1= np.sum(x,axis=1)

print axis0, axis0.shape
print axis1, axis1.shape

z = np.array([[1,2,3],[3,4,5],[5,6,7],[7,8,9]], np.int32)

# print x[0,1]
# print
# print z.shape
# print np.sum(z,axis=0) #add from top to bottom
# print np.sum(z,axis=1) #add from leftmost to rightmost
# print type(z)
# print
# print z
# print z[2:] #from the ith row to the rest
# print z[:2] #from the beginning to the ith row
# print z[:,1:2]
# print z[:,1:3]
# #
# print z[0,:] #the exactly ith row
# print z[0]   #the same exactly ith row
# print z[:,1] #the exactly ith column in a flat line
# print z[:,1]

# print z[1:3,1:3]

a = np.matrix([[1,2,3],[3,4,5],[5,6,7]])
b = np.matrix([[1,2,3],[3,4,5],[5,6,7]])
c = np.matrix([[3,4,5],[5,6,7]])
d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print
print np.dot(a,b)
print a*b
print d.shape
print np.dot(a,d)