import numpy as np
import random
import scipy.optimize as optimize
import glob
from GeneralBernoulliPlotter import GraphDataPrep
import matplotlib.pyplot as plt

L75AncEntropypath = 'HPCAncillaBernoulliData\L75Anc\L75P???AncillaBernoulliDataEntropy.npy'
L100AncEntropypath = 'HPCAncillaBernoulliData\L100Anc\L100P???AncillaBernoulliDataEntropy.npy'
L125AncEntropypath = 'HPCAncillaBernoulliData\L125Anc\L125P???AncillaBernoulliDataEntropy.npy'
L75AncMagpath = 'HPCAncillaBernoulliData\L75Anc\L75P???AncillaBernoulliDataMag.npy'
L100AncMagpath = 'HPCAncillaBernoulliData\L100Anc\L100P???AncillaBernoulliDataMag.npy'
L125AncMagpath = 'HPCAncillaBernoulliData\L125Anc\L125P???AncillaBernoulliDataMag.npy'
L20AncEntropypath = 'HPCAncillaBernoulliData\L20Anc\L20P???*AncillaBernoulliDataEntropy.npy'
L20AncMagpath = 'HPCAncillaBernoulliData\L20Anc\L20P???*AncillaBernoulliDataMag.npy'
L30AncEntropypath = 'HPCAncillaBernoulliData\L30Anc\L30P???*AncillaBernoulliDataEntropy.npy'
L30AncMagpath = 'HPCAncillaBernoulliData\L30Anc\L30P???*AncillaBernoulliDataMag.npy'
L40AncEntropypath = 'HPCAncillaBernoulliData\L40Anc\L40P???*AncillaBernoulliDataEntropy.npy'
L40AncMagpath = 'HPCAncillaBernoulliData\L40Anc\L40P???*AncillaBernoulliDataMag.npy'
L50AncEntropypath = 'HPCAncillaBernoulliData\L50Anc\L50P???*AncillaBernoulliDataEntropy.npy'
L50AncMagpath = 'HPCAncillaBernoulliData\L50Anc\L50P???*AncillaBernoulliDataMag.npy'
L60AncEntropypath = 'HPCAncillaBernoulliData\L60Anc\L60P???*AncillaBernoulliDataEntropy.npy'
L60AncMagpath = 'HPCAncillaBernoulliData\L60Anc\L60P???*AncillaBernoulliDataMag.npy'
L150AncEntropypath = 'HPCAncillaBernoulliData\L150Anc\L125P???AncillaBernoulliDataEntropy.npy'
L150AncMagpath = 'HPCAncillaBernoulliData\L150Anc\L125P???AncillaBernoulliDataMag.npy'
L175AncEntropypath = 'HPCAncillaBernoulliData\L175Anc\L175P???AncillaBernoulliDataEntropy.npy'
L175AncMagpath = 'HPCAncillaBernoulliData\L175Anc\L175P???AncillaBernoulliDataMag.npy'
L200AncEntropypath = 'HPCAncillaBernoulliData\L200Anc\L200P???AncillaBernoulliDataEntropy.npy'
L200AncMagpath = 'HPCAncillaBernoulliData\L200Anc\L200P???AncillaBernoulliDataMag.npy'

L75AncEntropyAVG, L75AncEntropySTD, L75AncMagnetizationAVG, L75AncMagnetizationSTD = GraphDataPrep(L75AncEntropypath,L75AncMagpath)
L100AncEntropyAVG, L100AncEntropySTD, L100AncMagnetizationAVG, L100AncMagnetizationSTD = GraphDataPrep(L100AncEntropypath,L100AncMagpath)
L125AncEntropyAVG, L125AncEntropySTD, L125AncMagnetizationAVG, L125AncMagnetizationSTD = GraphDataPrep(L125AncEntropypath,L125AncMagpath)
L20AncEntropyAVG, L20AncEntropySTD, L20AncMagnetizationAVG, L20AncMagnetizationSTD = GraphDataPrep(L20AncEntropypath,L20AncMagpath)
L30AncEntropyAVG, L30AncEntropySTD, L30AncMagnetizationAVG, L30AncMagnetizationSTD = GraphDataPrep(L30AncEntropypath,L30AncMagpath)
L40AncEntropyAVG, L40AncEntropySTD, L40AncMagnetizationAVG, L40AncMagnetizationSTD = GraphDataPrep(L40AncEntropypath,L40AncMagpath)
L50AncEntropyAVG, L50AncEntropySTD, L50AncMagnetizationAVG, L50AncMagnetizationSTD = GraphDataPrep(L50AncEntropypath,L50AncMagpath)
L60AncEntropyAVG, L60AncEntropySTD, L60AncMagnetizationAVG, L60AncMagnetizationSTD = GraphDataPrep(L60AncEntropypath,L60AncMagpath)
L150AncEntropyAVG, L150AncEntropySTD, L150AncMagnetizationAVG, L150AncMagnetizationSTD = GraphDataPrep(L150AncEntropypath,L150AncMagpath)
L175AncEntropyAVG, L175AncEntropySTD, L175AncMagnetizationAVG, L175AncMagnetizationSTD = GraphDataPrep(L175AncEntropypath,L175AncMagpath)
L200AncEntropyAVG, L200AncEntropySTD, L200AncMagnetizationAVG, L200AncMagnetizationSTD = GraphDataPrep(L200AncEntropypath,L200AncMagpath)

avgL = {
    20: L20AncEntropyAVG,
    30: L30AncEntropyAVG,
    40: L40AncEntropyAVG,
    50: L50AncEntropyAVG,
    60: L60AncEntropyAVG,
    75: L75AncEntropyAVG,
    100: L100AncEntropyAVG,
    125: L125AncEntropyAVG,
    150: L150AncEntropyAVG,
    175: L175AncEntropyAVG,
    200: L200AncEntropyAVG
}

stdL = {
    20: L20AncEntropySTD,
    30: L30AncEntropySTD,
    40: L40AncEntropySTD,
    50: L50AncEntropySTD,
    60: L60AncEntropySTD,
    75: L75AncEntropySTD,
    100: L100AncEntropySTD,
    125: L125AncEntropySTD,
    150: L150AncEntropySTD,
    175: L175AncEntropySTD,
    200: L200AncEntropySTD
}

def collapse_data(nu, pc):
    xx = []
    yy = []
    dd = []
    for L in Llist:

        if pc in avgL[L]:
            pc = pc + 2**(-52)
       
        MASK = [n for n in range(len(ps)) if ps[n]>minp and ps[n]<maxp]
       
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

results = {}

Llist = [20, 30, 40, 50, 60, 75, 100, 125, 175, 200]
ps = [.1,.18,.19,.20,.21,.22,.23,.24,.25,.255,.26,.265,.270,.275,.280,.290,.300,.400]
ps = np.array(ps)
for L in [0,20,30,40]:
    if Llist.count(L) > 0:
        Llist.remove(L)
    for i in [.220,.240]:
        for j in [.280,.300,.400]:
            minp = i
            maxp = j
            vals0 = [1.0, 0.2660]
            result = optimize.least_squares(collapse_data2, vals0)
            meanvals = result.x
            errorbars = np.sqrt(np.diag(np.linalg.inv(np.transpose(result.jac) @ result.jac)))
            results[(L,minp,maxp)] = np.concatenate((meanvals,errorbars),axis=0).tolist()
            
for vals in results.items():
    print(f'key {vals[0]} for results {vals[1]}') #first 2 in results are nu and pc followed by errorbars for nu and pc