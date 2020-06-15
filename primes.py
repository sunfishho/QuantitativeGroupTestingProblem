#Fix worst C calculation for n>1, x's need to be the same
import numpy as np
import math
from itertools import product
import time
import random
import matplotlib.pyplot as plt

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
def delL(p,a,b,c):
    s = star(p,a,b,c)
    if s==0:
        return False
    return ((1-s)/(2*s))**2

def solve_quad(a,b,c):
    s1 = (-1*b + (b**2-4*a*c)**.5)/(2*a)
    s2 = (-1*b - (b**2-4*a*c)**.5)/(2*a)
    return [s1,s2]

def update_c(s,pi,ai,bi):
    if s==False:
        return False
    w = (1-ai)*bi
    x = ai * (1-bi)
    y = ai*bi
    z = (1-ai)*(1-bi)
    quad = s-1
    lin = s*w+s*x+y+z
    con = s*w*x - y*z
    [s1,s2]=solve_quad(quad,lin,con)
    if s1<=ai*bi and s1<=(1-ai)*(1-bi) and s1>=-1*ai*(1-bi) and s1>=-1*bi*(1-ai):
        return s1
    return s2


def worse_c(p,a,b,c):
    s = delL(p,a,b,c)
    n = len(np.atleast_1d(p))
    c2 = np.zeros(n, dtype = float)
    for i in range(n):
        u = update_c(s,p[i],a[i],b[i])
        c2[i]=u

    return c2
def worst_c2(p,a,b,repeat):

    n = len(np.atleast_1d(p))
    c = np.zeros(n, dtype = float)
    for i in range(repeat):

        c=worse_c(p,a,b,c)
    return c

#n=1:
def f1(x,a,b):
    if x==0:
        return max(a*(1-b),b*(1-a)) - min(a*(1-b),b*(1-a))
    if x==1:
        return a*(1-b)+b*(1-a)+2*min(a*b,(1-a)*(1-b))-1
    worstC = update_c(((1-x)/(2*x))**2,1,a,b)
    return a*(1-b)+(1-a)*b + 2*worstC - x

def k(x):
    return -4*x**2/((3*x-1)*(x+1))
def kPrime(x):
    return -8*(x-1)*x/((x+1)**2*(3*x-1)**2)
def s(x,a,b):
    return (a-b)**2+k(x)**2+2*k(x)*(a+b-2*a*b)
def sPrime(x,a,b):
    return 2*k(x)*kPrime(x)+2*kPrime(x)*(a+b-2*a*b)
def graph(a,b,n):
    xs = []
    ys = []
    zs = []
    y2ndDeriv = []
    firstDeriv = []
    for x in range(1,n):
        xs.append(x/n)
        ys.append(f1(x/n,a,b))
        zs.append(0)
    firstDeriv = []
    zeroInd = 0
    for i in range(1,len(ys)):
        firstDeriv.append((ys[i]-ys[i-1])/(xs[i]-xs[i-1]))
        if ys[i-1]>=0 and ys[i]<=0:
            zeroInd = i

    secondDeriv = []
    for i in range(1,len(firstDeriv)):
        secondDeriv.append((firstDeriv[i]-firstDeriv[i-1])/(xs[i]-xs[i-1]))
    inflInd = 0

    for j in range(1,len(secondDeriv)):
        if secondDeriv[j-1]>0 and secondDeriv[j]<=0:
            inflInd = j+1
    critPts = []
    for j in range(1,len(firstDeriv)):
        if firstDeriv[j-1]*firstDeriv[j]<=0:
            critPts.append((xs[j],ys[j]))
    y1stDeriv = []
    y2ndDeriv = []
    cI = []
    for i in range(len(xs)):
        X = xs[i]
        Y = (ys[i] -  a*(1-b) - b*(1-a)+X)/2
        cI.append(Y)
        part1 = -1*(1-X)/(2*X**3)
        part2 = (Y+(1-a)*b)*(Y+(1-b)*a)
        part3 = Y**2 - a*b*(1-a)*(1-b)
        k = (2*part1*part2**2/part3 - 1)
        y1stDeriv.append(k)
        cTerm = -3*a*b*(1-a)*(1-b)
        conTerm = a*b*(1-a)*(1-b)*(2*a*b-a*b)
        y2ndDeriv.append(Y**3 + cTerm*Y + conTerm)

    plt.plot(xs,zs,'r')
    plt.plot(xs,ys,'b')
    plt.plot(xs[inflInd],ys[inflInd],'yo')
    plt.plot(xs[zeroInd],ys[zeroInd],'yo')
    plt.plot(xs,y1stDeriv,'k')
    plt.plot(xs,cI,'g')
    #plt.plot(xs,y2ndDeriv,'c')
    for i in critPts:
        plt.plot(i[0],i[1],'yo')
    #plt.show()
    '''for i in range(len(xs)):
        top = ys[i]**2 - a*b*(1-a)*(1-b)
        bottom1 = ys[i]+a*(1-b)
        bottom2 = ys[i]+b*(1-a)
        LHS = top/((bottom1*bottom2)**2) * firstDeriv[i]
        RHS =-1*(1-xs[i])/(2*xs[i]**3)
        print(LHS,RHS)'''
    x0 = xs[zeroInd]
    LHS = (-4*x0**2)/((3*x-1)*(x+1))
    RHS = (x-a+b)*(x-b+a)/(2*(a+b-2*a*b-x))
    if x0<1/3:
        print(LHS,RHS)
    return (xs[inflInd],ys[inflInd]),(xs[zeroInd],ys[zeroInd]),critPts
def simpGraph(a,b,n):
    xs = []
    ys = []
    zs = []
    for x in range(1,n):
        xs.append(x/n)
        ys.append(f1(x/n,a,b))
        zs.append(0)
    zeroInd = 0
    for i in range(1,len(ys)):
        if ys[i-1]>=0 and ys[i]<=0:
            zeroInd = i
    return xs[zeroInd]



def eq(x,y):
    return abs(x-y)<=10**-6
def f2(x,a,b):
    if x<=1/3:
        return -1*k(x)-x+((a-b)**2+k(x)**2+2*k(x)*(a+b-2*a*b))**.5
    return -1*k(x)-x-((a-b)**2+k(x)**2+2*k(x)*(a+b-2*a*b))**.5
def cVal(x,a,b):
    p1 = 2*a*b-a-b-k(x)
    p2 = k(x)**2 + 2*(a+b-2*a*b)*k(x)+(a-b)**2
    if x<1/3:
        return (p1+p2**.5)/2
    return (p1-p2**.5)/2
'''
a=.3
b=.3
x=.538
graph(a,b,100)
print(c(x,a,b))
print(update_c(((1-x)/(2*x))**2,1,a,b))
print(f1(x,a,b))
print(f2(x,a,b))'''
'''
for i in range(1,10):
    for j in range(1,10):
        a= i/10
        b=j/10
        checkAgainst = [a,b,a*b,a*(1-b),b*(1-a),(1-a)*(1-b)]
        zeroPt = graph(i/10,j/10,100)[1][0]
        string=''
        for v in checkAgainst:
            string+=str(int(zeroPt>v))
        print(string)
'''
'''
atest = np.array((random.random(), random.random(), random.random()))
btest = np.array((random.random(), random.random(), random.random()))
print(atest)
print(btest)
minStar = 1
maxStar = 0
minDelL = float('inf')
maxDelL = 0
for _ in range(10000):
    p1 = random.uniform(0, 1)
    p2 = random.uniform(0,1-p1)
    p3 = 1-p1-p2
    p = np.array((p1,p2,p3))
    w =worst_c2(p,atest,btest,25)
    print(LHS_minus_RHS(p,atest,btest,w))
'''
def f3(x,p,a,b):
    s = 0
    for i in range(len(p)):
        s+=p[i]*(a[i]*(1-b[i])+(1-a[i])*b[i]+2*cVal(x,a[i],b[i]))
    return s-x

def xVal(p,a,b,n):
    xs = []
    ys = []
    zs = []
    for x in range(1,n):
        xs.append(x/n)
        ys.append(f3(x/n,p,a,b))
        zs.append(0)
    zeroInd = 0
    for i in range(1,len(ys)):
        if ys[i-1]>=0 and ys[i]<=0:
            zeroInd = i
    return xs[zeroInd]

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

def cVal2(p,a,b,steps):
    c = [.5]*len(p)
    for i in range(steps):
        c=worse_c(p,a,b,c)
    return list(c),star(p,a,b,c)
def check3(p,a,b,n):
    xs = []
    ys = []
    already = False
    for x in range(1,n):
        xs.append(x/n)
        ys.append(f3(x/n,p,a,b))
        if x>=2:
            if ys[-1]*ys[-2]<=0:
                if already:
                    return False
                already = True
        return True

start = time.time()
n = 10
nCheck = 25
rang = []
for i in range(1,n):
    rang.append(i/n)
count = 0
'''
for a in product(rang,rang,rang):
    for b in product(rang,rang,rang):
        for p in product(rang,rang,rang):
            if count%100000==0:
                print(count)
            if check3(p,a,b,nCheck)==False:
                print(p,a,b)
            count+=1
'''
end = time.time()
#print(end-start)

def fPrime(x,a,b):
    if x<1/3:
        return -1-kPrime(x)+1/2*s(x,a,b)**(-.5)*sPrime(x,a,b)
    return -1-kPrime(x)-1/2*s(x,a,b)**(-.5)*sPrime(x,a,b)
def zeroDeriv(a,b,n):
    xs = []
    ys = []
    zeros = []
    for x in range(1,n):
        if x!=n/3:
            xs.append(x/n)
            ys.append(fPrime(x/n,a,b))
            if x>=2:
                if ys[-1]*ys[-2]<=0 and not (xs[-1]>1/3 and xs[-2]<1/3):
                    return True
    return False
def color(n1,n2):
    for x in range(1,n1):
        print(x)
        for y in range(1,n1):
            if zeroDeriv(x/n1,y/n1,n2):
                plt.plot(x/n1,y/n1,'go')
            else:
                plt.plot(x/n1,y/n1,'ro')
    plt.show()

def LPrime(x):

    return (math.log(1/2 - x/2) - math.log(x))/math.log(2)
def HPrime(x):
    return (-math.log(x)-1)/math.log(2)
def g(a,b,c):
    x = a*(1-b)+(1-a)*b+2*c
    return L(x) - H([a*b-c,a*(1-b)+c,(1-a)*b+c,(1-a)*(1-b)-c])
def gPrimeA(a,b,c,e):
    return (g(a+e,b,c)-g(a,b,c))/e
def delA(a,b,c):
    x = a*(1-b)+(1-a)*b+2*c
    p1 = LPrime(x)*(1-2*b)
    p2 = HPrime(a*b-c)*b + HPrime(a*(1-b)+c)*(1-b)+HPrime(b*(1-a)+c)*(-b) + HPrime((1-a)*(1-b)-c) *(b-1)
    return p1-p2

def gPrimeB(a,b,c,e):
    return (g(a,b+e,c)-g(a,b,c))/e
def delB(a,b,c):
    x = a*(1-b)+(1-a)*b+2*c
    p1 = LPrime(x)*(1-2*a)
    p2 = HPrime(a*b-c)*a + HPrime(a*(1-b)+c)*(-a)+HPrime(b*(1-a)+c)*(1-a) + HPrime((1-a)*(1-b)-c) *(a-1)
    return p1 - p2
def gPrimeC(a,b,c,e):
    return (g(a,b,c+e)-g(a,b,c))/e
def delC(a,b,c):
    x = a*(1-b)+(1-a)*b+2*c
    p1 = LPrime(x)*2
    p2 = HPrime(a*b-c) - HPrime(a*(1-b)+c) - HPrime(b*(1-a)+c) + HPrime((1-a)*(1-b)-c)
    return p1+p2
def Lstar(a,b,c):
    return L(a*(1-b)+(1-a)*b+2*c)
def delLA(a,b,c,e):
    return (Lstar(a+e,b,c)-Lstar(a,b,c))/e
def delLA2(a,b,c):
    return LPrime(a*(1-b)+(1-a)*b+2*c)*(1-2*b)

def aSatisfied(a,b,c,lamb):
    s = a*(1-b)+(1-a)*b+2*c
    one = a*b-c
    two = a*(1-b)+c
    three = b*(1-a)+c
    four = (1-a)*(1-b)-c
    if one<0 or two<0 or three<0 or four<0:
        return False
    l = ((1-s)/(2*s))**(1-2*b) * one**b * two**(1-b) * three ** (-b) *  (four)** (b-1)
    r = ((1-a)/a)**lamb
    return abs(l-r)<10**-4
def bSatisfied(a,b,c,lamb):
    s = a*(1-b)+(1-a)*b+2*c
    one = a*b-c
    two = a*(1-b)+c
    three = b*(1-a)+c
    four = (1-a)*(1-b)-c
    if one<0 or two<0 or three<0 or four<0:
        return False
    l = ((1-s)/(2*s))**(1-2*a) * one**a * two**(-a) * three ** (1-a) * (four)** (a-1)
    r = ((1-b)/b)**lamb

    return abs(l-r)<10**-4
def cSatisfied(a,b,c,lamb):
    s = a*(1-b)+(1-a)*b+2*c
    one = a*b-c
    two = a*(1-b)+c
    three = b*(1-a)+c
    four = (1-a)*(1-b)-c
    if one<0 or two<0 or three<0 or four<0:
        return False
    l = ((1-s)/(2*s))**2
    r= (one*four)/(two*three)
    return abs(l-r)<10**-4
def actualLambda(A,B,C):
    if (((1-A)*B)/(A*(1-B)))==0:
        return 0
    if (A*(1-B)+C)/(B*(1-A)+C) in [0,1]:
        return 0
    return math.log((((1-A)*B)/(A*(1-B))))/math.log((A*(1-B)+C)/(B*(1-A)+C))
def newLHSa(a,b,c):
    s = a*(1-b)+(1-a)*b+2*c
    one = a*b-c
    two = a*(1-b)+c
    three = b*(1-a)+c
    four = (1-a)*(1-b)-c
    if one<0 or two<0 or three<0 or four<0:
        return False
    l = ((1-s)/(2*s))**(1-2*b) * one**b * two**(1-b) * three ** (-b) * (four)** (b-1)
    predLamb = math.log(l)/math.log((1-a)/a)
    return l,predLamb
def newLHSb(a,b,c):
    s = a*(1-b)+(1-a)*b+2*c
    one = a*b-c
    two = a*(1-b)+c
    three = b*(1-a)+c
    four = (1-a)*(1-b)-c
    if one<0 or two<0 or three<0 or four<0:
        return False
    l = ((1-s)/(2*s))**(1-2*a) * one**a * two**(-a) * three ** (1-a) *  (four)** (a-1)
    predLamb = math.log(l)/math.log((1-b)/b)
    return l,predLamb
def maximizer(l):
    return h(l[0])+h(l[1])
def lagrange1(div):
    bestD1 = float('inf')
    bestD2 = float('inf')
    bestDiffVect = [0,0,0]
    bestDVal = 0
    bestABC = [0,0,0]
    grads = [[],[]]
    #bestV = 0
    #bestVDiff = 0
    for i in range(1,div):
        print(i)
        for j in range(1,div):

            a = i/div
            b = j/div
            x = simpGraph(a,b,5*div)
            c = cVal(x,a,b)

            gradF = [math.log((1-a)/a)/math.log(2),math.log((1-b)/b)/math.log(2),0]
            gradG = [delA(a,b,c),delB(a,b,c),delC(a,b,c)]
            if a!=.5 and b!=.5 and a!=b and a!=1-b:
                lamb = actualLambda(a,b,c)
                gradF = [math.log((1-a)/a)/math.log(2),math.log((1-b)/b)/math.log(2),0]
                gradG = [delA(a,b,c),delB(a,b,c),delC(a,b,c)]
                scaledF = [lamb*gradF[0],lamb*gradF[1],lamb*gradF[2]]
                diff = [abs(scaledF[0]-gradG[0]),abs(scaledF[1]-gradG[1]),abs(scaledF[2]-gradG[2])]
                length1 = (diff[0]**2 + diff[1]**2 + diff[2]**2)**.5
                length2 = abs(g(a,b,c))

                if length1<bestD1 and length2<1/(5*div):
                    bestD1 = length1
                    bestD2 = length2
                    bestDiffVect=diff
                    bestDVal = h(a)+h(b)
                    bestABC = [a,b,c]
                    grads = [gradF,gradG]

    A = bestABC[0]
    B = bestABC[1]
    C = bestABC[2]

    print('smallest difference:',bestD1)
    print('L-H', bestD2)
    print('value given:', bestDVal)
    print('g - lambda f', bestDiffVect)
    print('a,b,c that give this:', bestABC)
    print('gradients there', grads)
    print('predicted lambdas:', newLHSa(A,B,C)[1],newLHSb(A,B,C)[1])
    print('actual lambdas:', actualLambda(A,B,C))
    lamb = newLHSa(A,B,C)[1]
    print((((1-A)*B)/(A*(1-B)))**lamb,(A*(1-B)+C)/(B*(1-A)+C))
    #return bestD,bestDVal,bestABC,grads,newLHSa(A,B,C)[1],newLHSb(A,B,C)[1]
    return bestABC,bestDVal
def lagrange2(div):
    bestD1 = float('inf')
    bestD2 = float('inf')
    bestDiffVect = [0,0,0]
    bestDVal = 0
    bestPABC = [[],[],[],[]]
    grads = [[],[]]
    posses = []
    for i in range(1,div):
        posses.append(i/div)
    for a in product(posses,posses):
        print(a)
        if a[0]>.5:
            continue
        for b in product(posses,posses):
            for P in posses:

                if .5 not in a and .5 not in b:
                    start = time.time()
                    p = [P,1-P]
                    ''''if HplusH(p,a,b,[])<bestDVal:
                        continue'''
                    if HplusH(p,a,b,[])<1.5:
                        continue
                    '''W = cVal2(p,a,b,10)
                    c = W[0]
                    x = W[1]
                    '''
                    x = binarySearchX(p,a,b,10)
                    c= []
                    for i in range(len(a)):
                        c.append(cVal(x,a[i],b[i]))
                    if abs(LminusH(p,a,b,c))>1/div:
                        continue
                    end = time.time()
                    #print('Finding xs and cs',end-start)
                    start = time.time()
                    gF = gradF(p,a,b,c)
                    combinedF = gF[0]+gF[1]+gF[2]+gF[3]
                    end = time.time()
                    #print('F gradient',end-start)
                    start = time.time()
                    gG = gradG(p,a,b,c)
                    combinedG = gG[0]+gG[1]+gG[2]+gG[3]
                    end = time.time()
                    #print('G gradient',end-start)
                    start = time.time()
                    lamb = actualLambda(a[0],b[0],c[0])
                    scaledF = []
                    diff = []
                    for i in range(len(combinedG)):
                        scaledF.append(lamb*combinedF[i])
                        diff.append(abs(scaledF[i]-combinedG[i]))
                    length1 = 0
                    for i in diff:
                        length1+=i**2
                    length1=length1**.5
                    length2 = abs(LminusH(p,a,b,c))
                    if length1<bestD1 and length2<1/div:
                        bestD1 = length1
                        bestD2 = length2
                        bestDiffVect=diff
                        bestDVal = HplusH(p,a,b,c)
                        bestPABC = [p,a,b,c]
                        grads = [gF,gG]
                    end = time.time()
                    #print('Rest of the stuff',end-start)

    P = bestPABC[0]
    A = bestPABC[1]
    B = bestPABC[2]
    C = bestPABC[3]

    print('smallest difference:',bestD1)
    print('L-H', bestD2)
    print('value given:', bestDVal)
    print('g - lambda f', bestDiffVect)
    print('a,b,c that give this:', bestPABC)
    print('gradients there', grads)
    print(actualLambda(A[0],B[0],C[0]))
    print(actualLambda(A[1],B[1],C[1]))
    return bestPABC,bestDVal
def LequalsH(div):
    lessThanLine = []
    equalToLine = []
    aboveLine = []
    f=open('data'+str(div)+'.txt','w')
    for i in range(1,div):
        print(i)
        for j in range(1,div):

            a = i/div
            b = j/div
            x = simpGraph(a,b,div)
            c = cVal(x,a,b)
            if abs(g(a,b,c))<=1/(div):
                plt.plot(a,b,'go')

                f.write(str(a)+','+str(b)+'\n')

            else:
                plt.plot(a,b,'ro')

    plt.show()
def HplusH(p,a,b,c):
    s = 0
    for i in range(len(p)):
        s+=p[i]*(h(a[i])+h(b[i]))
    return s
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
def gradF(p,a,b,c):
    dA = []
    dB = []
    dC = []
    dP = []
    for i in range(len(a)):
        dA.append(p[i]*math.log2((1-a[i])/a[i]))
        dB.append(p[i]*math.log2((1-b[i])/b[i]))
        dC.append(0)
        dP.append(h(a[i])+h(b[i]))
    return [dA,dB,dC,dP]
def gradG(p,a,b,c):
    dA = []
    dB = []
    dC = []
    dP = []
    star = 0
    for i in range(len(p)):
        star+=p[i]*(a[i]*(1-b[i])+b[i]*(1-a[i])+2*c[i])

    for i in range(len(a)):
        starI = a[i]*(1-b[i])+b[i]*(1-a[i])+2*c[i]
        ai = a[i]
        bi = b[i]
        ci = c[i]
        oneI = ai*bi-ci
        twoI = ai*(1-bi)+ci
        threeI = bi*(1-ai)+ci
        fourI = (1-ai)*(1-bi)-ci
        dA.append(p[i]*(LPrime(star)*(1-2*bi)-(HPrime(oneI)*bi+HPrime(twoI)*(1-bi)+HPrime(threeI)*-bi+HPrime(fourI)*(bi-1))))
        dB.append(p[i]*(LPrime(star)*(1-2*ai)-(HPrime(oneI)*ai+HPrime(twoI)*(-ai)+HPrime(threeI)*(1-ai)+HPrime(fourI)*(ai-1))))
        dC.append(p[i]*(LPrime(star)*2+HPrime(oneI)-HPrime(twoI)-HPrime(threeI)+HPrime(fourI)))
        dP.append(LPrime(star)*starI-H([oneI,twoI,threeI,fourI]))
        #dA.append(p[i]*delA(a[i],b[i],c[i])+(LPrime(star)-LPrime(starI))*(1-2*b[i]))
        #dB.append(p[i]*delB(a[i],b[i],c[i])+(LPrime(star)-LPrime(starI))*(1-2*a[i]))
        #dC.append(p[i]*delC(a[i],b[i],c[i])+(LPrime(star)-LPrime(starI))*(2))
        #dP.append(LPrime(star)*starI-H([ai*bi-ci,ai*(1-bi)+ci,(1-ai)*bi+ci,(1-ai)*(1-bi)-ci]))
    return [dA,dB,dC,dP]
def numGradF(p,a,b,c,eps):
    dA = []
    dB = []
    dC = []
    dP = []
    for i in range(len(a)):
        aNew = a.copy()
        aNew[i]+=eps
        bNew = b.copy()
        bNew[i]+=eps
        cNew = c.copy()
        cNew[i]+=eps
        pNew = p.copy()
        pNew[i]+=eps
        dA.append((HplusH(p,aNew,b,c)-HplusH(p,a,b,c))/eps)
        dB.append((HplusH(p,a,bNew,c)-HplusH(p,a,b,c))/eps)
        dC.append((HplusH(p,a,b,cNew)-HplusH(p,a,b,c))/eps)
        dP.append((HplusH(pNew,a,b,c)-HplusH(p,a,b,c))/eps)
    return [dA,dB,dC,dP]
def numGradG(p,a,b,c,eps):
    dA = []
    dB = []
    dC = []
    dP = []
    for i in range(len(a)):
        aNew = a.copy()
        aNew[i]+=eps
        bNew = b.copy()
        bNew[i]+=eps
        cNew = c.copy()
        cNew[i]+=eps
        pNew = p.copy()
        pNew[i]+=eps
        dA.append((LminusH(p,aNew,b,c)-LminusH(p,a,b,c))/eps)
        dB.append((LminusH(p,a,bNew,c)-LminusH(p,a,b,c))/eps)
        dC.append((LminusH(p,a,b,cNew)-LminusH(p,a,b,c))/eps)
        dP.append((LminusH(pNew,a,b,c)-LminusH(p,a,b,c))/eps)
    return [dA,dB,dC,dP]
def maxAequalsB(low,high,div,searchSteps):
    bestLminusH = float('inf')
    bestA = 0
    mini = math.floor(low*div)
    maxi = math.ceil(high*div)
    for i in range(mini,maxi+1):
        if (i*100/div)%1 == 0:
            print(i/div)
        a = i/div
        b = i/div
        if low<=a<=high and low<=b<=high:
            x = binarySearchX([1],[a],[b],searchSteps)
            c = cVal(x,a,b)
            val = abs(LminusH([1],[a],[b],[c]))
            if val<bestLminusH:
                bestLminusH=val
                bestA = a
    return bestA,HplusH([1],[bestA],[bestA],[])


start = time.time()
#print(lagrange2(50))
'''
best where a = b = 0.23683775989403694, score of 1.579483836721551
'''
#print(maxAequalsB(0.2368377598,0.2368377599,10**15,500))
a=.3
b=.3
x=binarySearchX([1],[a],[b],10)
print(x,cVal(x,a,b))
print(1/6 * (3 - 6 * a + 6* a**2 - 2*3**.5* abs(-1 + 2* a)))

low = .23
high = .24
bestA = .23
bestScore = 1.57
for i in range(2,20):
    bestA,bestScore = maxAequalsB(low,high,10**i,10*i)
    low = bestA - 10**(-i)
    high = bestA + 10**(-i)
    print(i,bestA,bestScore,low,high)
end = time.time()
print(end-start)



'''a=.745
b=.78
x = simpGraph(a,b,10000)
c = cVal(x,a,b)
print(a,b,c,g(a,b,c))
print(maximizer([a,b]))

lamb = actualLambda(a,b,c)
gradF = [math.log((1-a)/a)/math.log(2),math.log((1-b)/b)/math.log(2),0]
gradG = [delA(a,b,c),delB(a,b,c),delC(a,b,c)]
scaledF = [lamb*gradF[0],lamb*gradF[1],lamb*gradF[2]]
diff = [abs(scaledF[0]-gradG[0]),abs(scaledF[1]-gradG[1]),abs(scaledF[2]-gradG[2])]
print(diff)
print(lamb)
print('predicted lambdas:', newLHSa(a,b,c)[1],newLHSb(a,b,c)[1])
'''
#LequalsH(250)
#print(lagrange(200))
#print(lagrange2(200))
#def best(div):


a=.6
b=.3
c=.1
