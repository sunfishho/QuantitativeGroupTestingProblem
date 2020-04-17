import numpy as np
import math
from itertools import product
#computes the entropy of a probability distribution given a list of probabilities "probabilities"
def H(probabilities):
    sum = 0.0
    for i in probabilities:
        if (i <= 0 and i>= -0.01):
            sum += 0
        else:
            sum -= i * math.log2(i)
    return sum

def loss(p, a, b):
      n = len(np.atleast_1d(p))
      sum = 0
      for i in range(n):
          sum += p[i] * (H([a[i], 1 - a[i]]) + H([b[i], 1 - b[i]]))
      return sum
def h(x):
    return H([x, 1-x])

def L(x):
    return h(x) + 1 - x
def LHS(p, a, b, c):
    n = len(np.atleast_1d(p))
    sum = 0
    for i in range(n):
        sum += (p[i] * (a[i] * (1 - b[i]) + b[i] * (1 - a[i]) + 2 * c[i]))
    if (sum < 0 or sum > 1):
        return False
    return L(sum)
def RHS(p, a, b, c):
    entropy_sum = 0
    n = len(np.atleast_1d(p))
#     print(a)
#     print(b)
#     print(c)
    for i in range(n):
        if ((a[i] * b[i] - c[i]) < 0 or (a[i] * (1 - b[i]) + c[i]) < 0 or (b[i] * (1 - a[i]) + c[i]) < 0 or ((1 - a[i]) * (1 - b[i]) + c[i]) < 0):
            return False

    for i in range(n):
        entropy = H([a[i] * b[i] - c[i], a[i] * (1 - b[i]) + c[i], (1 - a[i]) * b[i] + c[i], (1 - a[i]) * (1 - b[i]) - c[i]])
        entropy_sum += p[i] * entropy
    return entropy_sum

def LHS_minus_RHS(p, a, b, c):
    if (LHS(p,a,b,c) == False or RHS(p,a,b,c) == False):
        return False
    return LHS(p, a, b, c) - RHS(p, a, b, c)
def worst_c(p, a, b, alpha, iterations):
    n = len(np.atleast_1d(p))
    c = np.zeros(n, dtype = float)
    for i in range(iterations):
        for j in range(n):
            cplus = list(c)
            cplus[j] += 0.001
            cminus = list(c)
            cminus[j] -=  0.001
            dxdc = (LHS_minus_RHS(p, a, b, cplus) - LHS_minus_RHS(p, a, b, cminus))/.002
            c[j] -= dxdc * alpha
        for j in range(n):
            if (a[j] * b[j] - c[j] < 0 or a[j] * (1 - b[j]) + c[j] < 0 or b[j] * (1 - a[j]) + c[j] < 0 or (1 - a[j]) * (1 - b[j]) + c[j] < 0):
                c += dxdc * alpha
                return c

    return c
def star(p,a,b,c):
    n = len(np.atleast_1d(p))
    sum = 0
    for i in range(n):
        sum += (p[i] * (a[i] * (1 - b[i]) + b[i] * (1 - a[i]) + 2 * c[i]))
    if (sum < 0 or sum > 1):
        return False
    return sum
#LHS of the partial derivative expression
def delL(p,a,b,c):
    s = star(p,a,b,c)
    return ((1-s)/(2*s))**2

def solve_quad(a,b,c):
    s1 = (-1*b + (b**2-4*a*c)**.5)/(2*a)
    s2 = (-1*b - (b**2-4*a*c)**.5)/(2*a)
    if s1>=0 and s1<=1:
        return s1
    if s2>=0 and s2<=1:
        return s2
    return False

#finds worse c by solving for where partial derivative = 0 given old LHS
def update_c(s,pi,ai,bi):
    w = (1-ai)*bi
    x = ai * (1-bi)
    y = ai*bi
    z = (1-ai)*(1-bi)
    quad = s-1
    lin = s*w+s*x+y+z
    con = s*w*x - y*z
    return solve_quad(quad,lin,con)
def worse_c(p,a,b,c):
    s = delL(p,a,b,c)
    n = len(np.atleast_1d(p))
    c2 = np.zeros(n, dtype = float)
    for i in range(n):
        u = update_c(s,p[i],a[i],b[i])
        if u>=0 and u<=1:
            c2[i]=u
        else:
            c2[i]=c[i]
    return c2
a = np.array((0.23684, 0.23684, 0.23684))
b = np.array((0.23684, 0.23684, 0.23684))
c = np.array((0, 0, 0))
p = np.array((0.2, 0.8))
#starting at 0 vector and iterating a bunch could give the worst c vector
for i in range(100):
    c=worse_c(p,a,b,c)
print(c)
print(LHS_minus_RHS(p,a,b,c))
