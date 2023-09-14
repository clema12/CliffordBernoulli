import numpy as np
import scipy.optimize as optimize
import glob
from GeneralBernoulliPlotter import GraphDataPrep
import matplotlib.pyplot as plt

keylist = [20,30,40,50,75,100,125,175,200] #Static list
BernoulliData = {}
avgL = {}
stdL = {}

for L in keylist:
    BernoulliData[L] = GraphDataPrep(f'HPCBernoulliData\SymmetricPVals\L{L}Data\L{L}P???DenseProbBernoulliDataEntropy.npy',f'HPCBernoulliData\SymmetricPVals\L{L}Data\L{L}P???DenseProbBernoulliDataMag.npy')

for L in keylist:
    avgL[L] = BernoulliData[L][2]
    stdL[L] = BernoulliData[L][3]

def collapse_data(nu, pc):
    xx = []
    yy = []
    dd = []
    for L in Llist:

        if pc in avgL[L]:
            pc = pc + 2**(-12)
       
        MASK = [n for n in range(len(ps)) if (ps[n]>minp and ps[n]<maxp)]
       
        # xx.extend([(p - pc) * L**(1/nu) for p in avgL[L][:,0]])
        # yy.extend(avgL[L][:,1])
        # dd.extend([val/np.sqrt(samples[L]) for val in stdL[L][:,1]])
       
        xx.extend([(ps[n] - pc) * L**(1/nu) for n in MASK]) 
        yy.extend([avgL[L][n] for n in MASK])  #These are dictionaries, need to store my data into dictionaries with L being a int key for the data access
        dd.extend([stdL[L][n] for n in MASK])  # also a dictionary
       
    xytriplet = list(zip(*sorted(zip(xx,yy,dd), key = lambda triplet: triplet[0])))
   
    xx = np.array(xytriplet[0])
    yy = np.array(xytriplet[1])
    dd = np.array(xytriplet[2])
   
    ybar = ((xx[2:] - xx[1:-1])*yy[:-2] - (xx[:-2] - xx[1:-1])*yy[2:])/(xx[2:] - xx[:-2])
    DeltaSqr = dd[1:-1]**2 + ((xx[2:] - xx[1:-1])/(xx[2:] - xx[:-2])*dd[:-2])**2 + ((xx[:-2] - xx[1:-1])/(xx[2:] - xx[:-2])*dd[2:])**2
   
    yy = np.array(yy[1:-1]) 

    return (yy - ybar) / np.sqrt(DeltaSqr)

def collapse_data2(vals):
    return collapse_data(*vals)

Llist = [20,30,40,50,75,100,125,175,200] #Dynamic list
BernoulliProbs = [.100,.190,.200,.210,.230,.250,.266,.270,.275,.285,.290,.295,.300,.400,.420,.440,.460,.475,.480,.485,.490,.495,.500,.505,.510,.515,.520,.525,.540,.560,.580,.600,.700,.800,.900]

results = {}

ps = np.array(BernoulliProbs)
for L in [0,20,30,40]:
    if Llist.count(L) > 0:
        Llist.remove(L)
    for i in [.475,.480,.485,.490,.495]:
        for j in [.505,.510,.515,.520,.525]:
            minp = i
            maxp = j
            vals0 = [1.01, 0.501]
            result = optimize.least_squares(collapse_data2, vals0)
            meanvals = result.x
            errorbars = np.sqrt(np.diag(np.linalg.inv(np.transpose(result.jac) @ result.jac)))
            results[(L,minp,maxp)] = np.concatenate((meanvals,errorbars),axis=0).tolist()

for vals in results.items():
    print(f'key {vals[0]} for results {vals[1]}') #first 2 in results are nu and pc followed by errorbars for nu and pc
