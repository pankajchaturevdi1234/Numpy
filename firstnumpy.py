import numpy as np
# creating numpy array

var_1=np.array([1,2,3,4])
var_2=np.array([[1,2,3,4],[5,6,7,8]])
var_3=np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]]])
print(var_1)
print(var_2)
print(var_3)

# filled with zero
ar_zero=np.zeros(4)
ar_zero1=np.zeros((3,4))
print(ar_zero)
print(ar_zero1)

# filled with one
ar_one=np.ones(3)
ar_one1=np.ones((3,4))
print(ar_one)
print(ar_one1)

# filled with empty
ar_empty=np.empty(3)
print(ar_empty)

# filled with given range
ar_range=np.arange(4)
print(ar_range)

# filled with the diagonal
ar_dia=np.eye(3)
print(ar_dia)


# creat the numpy with the random number

# By rand()
var_4=np.random.rand(4)
print(var_4)

# By randn()
var_5=np.random.randn(5)
print(var_5)

# By ranf()
var_6=np.random.ranf(4)
print(var_6)

# By randint()
var_7=np.random.randint(5,20,5)
print(var_7)

# shape and reshaping in numpy 

# shape
var_8=np.array([[1,2,3,4],[9,10,11,12]])
print(var_8)
print()
print(var_8.shape)


# reshape
var_9=np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]]])
print(var_9)
print()
x=var_9.reshape(4,3)
print(x)

# arithmetic opretion in numpy

var_10=np.array([1,2,3,4])
var_11=np.array([9,8,7,6])
print(np.add(var_10,var_11))
print(np.subtract(var_10,var_11))
print(np.multiply(var_10,var_11))
print(np.divide(var_10,var_11))
print(np.mod(var_10,var_11))
print(np.power(var_10,var_11))
print(np.reciprocal(var_10,var_11))


# Arithmetic Function
print(np.min(var_10))
print(np.max(var_10))
print(np.argmin(var_10))
print(np.sqrt(var_10))
print(np.sin(30))
print(np.cos(60))
print(np.cumsum(var_10)) #add the privious one

# Broadcasting numpy array
var_11=np.array([1,2,3,4])
var_12=np.array([9,14,7,6])
print(var_11 + var_12)
print(var_11 *var_12)


# Indexing and slicing

# indexing
var_13=np.array([5,6,7,8])
print(var_13[-1])

var_14=np.array([[12,13,16,18],[9,15,23,15]])
print(var_14[0,2])

# slicing
var_15=np.array([1,2,3,4,5,6,7,8,9])
print("start ot 5",var_15[:5])


# iterating in numpy
var_15=np.array([1,2,3,4,5,6,7,8,9])
for i in var_15:
    print(i)

var_16=np.array([[[10,11,12,13],[14,15,16,17],[18,19,20,21]]])
for j in np.nditer(var_16):
    print(j)

# Copy and view in numpy

# copy
var_17=np.array([1,2,3,4])
co=var_17.copy()
print(var_17)
print(co)

# view
vi=var_17.view()
print(vi)

# Join and split function

# join
var18=np.array([1,2,3,4])
var19=np.array([5,6,7,8])
p=np.concatenate((var18,var19)) # BY CONCATENATE
print(p)

r=np.stack((var18,var19))  # BY STACK
print(r)


# split

var20=np.array([11,12,13,14,15,1,17,19])
sp1=np.array_split(var20,3)

var21=np.array([[1,2],[3,4],[4,5]])
sp2=np.array_split(var21,3,axis=1)       # it seprate one array into multiple
print(sp1)
print(sp2)


# search array

var22=np.array([4,2,3,4,2,5,2,5,2,5,6,7,8])
t=np.where((var22%2)==0)
print(t)

y=np.where((var22 == 0))
print(y)

# search sorted array

var23=np.array([172,2,43,4,55,6,70,8,19,1])
q=np.searchsorted(var23,5)
print(q)

# sort array

var24=np.array([1,2,3,4,5,6,7,8,9,10])
print(np.sort(var24))

# filter array

var25=np.array(["a","b","c","d","e"])
s=[True,False,True,True,False]
print(var25[s])

# shuffle

var26=np.array([1,2,3,4,5,6])
np.random.shuffle(var26)
print(var26)          # it gives the random value from the given value


# unique

var27=np.array([5,7,8,9,3,42,16,8,2,3,9,1])
c=np.unique(var27,return_index=True)
d=np.unique(var27,return_counts=True)
print(c)
print(d)

# Matrix functions

# transpose

var28=np.array([[2,4,3],[7,6,9]])
print(var28)
print()
print(np.transpose(var28))

# swapaxes

print(np.swapaxes(var28,1,1))

# inverse

print(np.linalg.inv(var28))

# power

print(np.linalg.matrix_power(var28,2))

# determinate

print(np.linalg.det(var28))
