import numpy as np
from GeneralBernoulliPlotter import GraphDataPrep
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob

def func(x, a, b):
    return a*np.log(x)+b

def linfunc(x, a, b):
     return a*x+b

def DataGrab(Entropyfilepath):
    Entropy = []
    EntropyData = []

    for files in glob.glob(Entropyfilepath):
            Entropy.append(files)

    for files in Entropy:
            with open(files,'rb') as f:
                filedata = np.load(f)
                EntropyData.append(filedata)

    return EntropyData

def ListAdder(DataToAdd):
    result = np.zeros(len(DataToAdd[0][0]))
    for i in DataToAdd[0]:
        result = result + i
    result = result/1000
    result = np.delete(result,0)
    return result

BernoulliData = {}  #Make a dictionary of all the data. Indices 0 and 1 are the entropy and mag data and indices 2 and 3 are the error bar data for entropy and mag respectively

keylist = [20,30,40,50,75,100,125,150,175,200]
probkeys = [0.1, 0.19, 0.2, 0.21, 0.23, 0.25, 0.266, 0.27, 0.275, 0.285, 0.29, 0.295, 0.3]


for L in keylist:
    BernoulliData[f'{L}'] = GraphDataPrep(f'HPCBernoulliData\L{L}Data\L{L}P???DenseProbBernoulliDataEntropy.npy',f'HPCBernoulliData\L{L}Data\L{L}P???DenseProbBernoulliDataMag.npy')

#Prepare the entropy data along a system size slice so we can plot vs L instead of p:
for posp,p in enumerate(probkeys):
    templist = []
    for L in keylist:
        templist.append(BernoulliData[f'{L}'][0][posp])
    BernoulliData[f'p{p}'] = templist

opt, cov = curve_fit(func, keylist, BernoulliData[f'p{0.266}'])
print(opt[0]) # a constant for the log(L) fit
perr = np.sqrt(np.diag(cov))
print(perr[0]) # error in a respectively

# graphprobs = [0.1, 0.19, 0.2, 0.21, 0.23, 0.25, 0.266, 0.27, 0.275, 0.285, 0.29, 0.295, 0.3]

#S vs L plots for different p values
# plt.figure(1)
# for p in graphprobs:
#     plt.plot(keylist, BernoulliData[f'p{p}'], '-o',  label = f"p={p}")
# # plt.xscale('log')
# plt.ylim([0,20])
# plt.ylabel('Average Half-cut Entropy of Bernoulli Circuit')
# plt.xlabel('Number of Qubits')
# plt.title('Entropy vs System Size')
# plt.legend()

#Visualize fit for a*log(x)+b
# plt.figure(2)
# plt.plot(keylist, BernoulliData['p0.266'],'-o', label = 'data')
# plt.plot(keylist, func(keylist, *opt), 'r-', label = 'fit: a=%5.3f, b=%5.3f' % tuple(opt))
# plt.xlabel('Number of Qubits')
# plt.ylabel('Average Half-cut Entropy of Bernoulli Circuit')
# plt.title('Fitting p = pc with a*ln(x)+b')
# plt.legend()
# plt.show()


TempDataPent = DataGrab(f"HPCBernoulliData\TDData\L{150}P266TDBernoulliDataEntropy.npy")
BernoulliTDL150 = ListAdder(TempDataPent)

T = 2*150**2
temp = []
for i in range(T):
    if i%5 == 0:
        temp.append(i)
PlotNormT = np.log(np.asarray(temp)/np.sqrt((T/2)))
PlotSqrtT = np.sqrt(np.asarray(temp)/(T/2))

#Slice out a linear portion of the log plot to fit the log 
FitPlotNormT = np.array([x for x in PlotNormT if (x>2 and x<4)])
indicesPlot = [i for i,x in enumerate(PlotNormT) if (x>2 and x<4)]
FitBernoulliTDL150 = BernoulliTDL150[indicesPlot]

TDopt, TDcov = curve_fit(linfunc, FitPlotNormT, FitBernoulliTDL150)
print(TDopt[0])
print(np.sqrt(np.diag(TDcov))[0])

#Compute Zent for half-cut:

z = opt[0]/TDopt[0]
zerr = (1/TDopt[0])*np.sqrt(cov[0][0]+(TDcov[0][0]*(opt[0]/TDopt[0])**2))

print(f'The computed z value is {z} and its error is {zerr}.')



########## All of the numerical results so far:

#Numerical Z for the Half-cut ent transition: z = 1.06(2)
#Numerical Z for the Half-cut mag transition: z = 2.07(8)

#Numerical Z for the Ancilla ent transition: z = 1.002(6)
#Numerical Z for the Ancilla mag transition: z = 1.98(2)

#Numerical nu and p for Half-cut mag transition: nu = 1.00(1), pctrl = 0.5001(1)
#Numerical nu and p for Ancilla ent transition: nu = 1.22(2), pent = 0.266(3)

