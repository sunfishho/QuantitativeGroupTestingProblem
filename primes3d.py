import csv
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import numpy as np
xOpt = []
yOpt = []
zOpt = []

with open('/Users/samflorin/Downloads/primes L=H data - 200, to .001.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if float(row[0])+float(row[1])!=1:
            xOpt.append(float(row[0]))
            yOpt.append(float(row[1]))
            zOpt.append(0)
        #z.append(float(row[2]))

def binarySearchX(p,a,b,steps):
    checking = 1/2
    stepSize = 1/4
    for i in range(steps):
        if f3(checking,p,a,b)>0:
            checking = checking+stepSize
        else:
            checking = checking-stepSize
        stepSize=stepSize/2
    return checking
def f3(x,p,a,b):
    s = 0
    for i in range(len(p)):
        s+=p[i]*(a[i]*(1-b[i])+(1-a[i])*b[i]+2*cVal(x,a[i],b[i]))
    return s-x
def cVal(x,a,b):
    p1 = 2*a*b-a-b-k(x)
    p2 = k(x)**2 + 2*(a+b-2*a*b)*k(x)+(a-b)**2
    if x<1/3:
        return (p1+p2**.5)/2
    return (p1-p2**.5)/2
def k(x):
    return -4*x**2/((3*x-1)*(x+1))
def LminusH(p,a,b,c):
    star = 0
    for i in range(len(p)):
        star+=p[i]*(a[i]*(1-b[i])+b[i]*(1-a[i])+2*c[i])
    sum = L(star)
    for i in range(len(p)):
        ai = a[i]
        bi = b[i]
        ci = c[i]
        sum -= p[i]*H([ai*bi-ci,ai*(1-bi)+ci,(1-ai)*bi+ci,(1-ai)*(1-bi)-ci])
    return sum
def H(probabilities):
    sum = 0.0
    for i in probabilities:
        if (i <= 0 and i>= -0.01):
            sum += 0
        else:
            sum -= i * math.log2(i)
    return sum
def h(x):
    return H([x, 1-x])
def L(x):
    return h(x) + 1 - x
X = []
Y = []
Z = []
AequalsB=[]
n = 100
steps = 100
for i in range(0,n):
    for j in range(0,n):
        a = i/n
        b = j/n

        x=binarySearchX([1],[a],[b],steps)
        c = cVal(x,a,b)
        X.append(a)
        Y.append(b)
        Z.append(LminusH([1],[a],[b],[c]))
        if a==b:
            AequalsB.append((a,LminusH([1],[a],[b],[c])))
print(AequalsB)
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z)
ax.scatter(xOpt,yOpt,zOpt)
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_zlabel('L minus H')

plt.show()
'''
