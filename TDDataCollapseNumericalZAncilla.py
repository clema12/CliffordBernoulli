import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.optimize as optimize

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

#Finite difference Newton-Raphson
def NewtonRaphson(fx,x0,dx,maxiter=100):
    initialval = fx(x0)
    xs = []
    ys = []
    xs.append(x0)
    ys.append(initialval)
    for i in range(maxiter):
        newx = x0 - dx*((fx(x0+dx)-fx(x0))/(fx(x0+dx)-2*fx(x0)+fx(x0-dx)))
        newval = fx(newx)
        xs.append(newx)
        ys.append(newval)
        print(x0)
        x0 = newx
    return xs,ys

LList = [20,30,40,50,75,100,125,150]
P400LList = [20,30,40,50,75,125,150]
TList = [2*L**2 for L in LList]
P400TList = [2*L**2 for L in P400LList]


PlotNormT = {}
PlotSqrtT = {}
TDAncillaPent = {}
TDAncillaPctrl = {}
TDAncillaPentSTD = {}
TDAncillaPctrlSTD = {}

for L in LList:
    #Transpose the 3D array at the 2D data to form the correct data structure
    TempAncillaDataPent = np.array(DataGrab(f"HPCAncillaBernoulliData\TDData\L{L}P266TDAncillaBernoulliDataEntropy.npy")[0]).T
    TempAncillaDataPctrl = np.array(DataGrab(f"HPCAncillaBernoulliData\TDData\L{L}P500TDAncillaBernoulliDataEntropy.npy")[0]).T
    #Remove the first element because it double counts the initial entropy
    TempAncillaDataPent = np.delete(TempAncillaDataPent,0,0)
    TempAncillaDataPctrl = np.delete(TempAncillaDataPctrl,0,0)
    TDAncillaPent[L] = [np.average(element) for element in TempAncillaDataPent]
    TDAncillaPctrl[L] = [np.average(element) for element in TempAncillaDataPctrl]
    TDAncillaPentSTD[L] = [np.std(element)/np.sqrt(1000) for element in TempAncillaDataPent]
    TDAncillaPctrlSTD[L] = [np.std(element)/np.sqrt(1000) for element in TempAncillaDataPctrl]


#remove entanglement values for the collapse when the value isn't 0
for T in TList:
    temp = []
    for i in range(T):
        if i%5 == 0:
            temp.append(i)
    PlotNormT[T] = (np.asarray(temp)/np.sqrt((T/2)))
    PlotSqrtT[T] = np.sqrt(np.asarray(temp)/(T/2))

#change the ent dataset for the ctrl dataset to go between Zent and Zctrl
def collapse_data(z):
    xx = []
    yy = []
    dd = []
    for L in LList:
       
        # xx.extend([(p - pc) * L**(1/nu) for p in avgL[L][:,0]])
        # yy.extend(avgL[L][:,1])
        # dd.extend([val/np.sqrt(samples[L]) for val in stdL[L][:,1]])

        MASK = [n for n in range(len((TDAncillaPentSTD[L]))) if TDAncillaPentSTD[L][n] > 0]
       
        xx.extend([(PlotNormT[2*L**2][n] + 2**(-40)) / L**(z) for n in MASK]) 
        yy.extend([TDAncillaPent[L][n] for n in MASK])  #These are dictionaries, need to store my data into dictionaries with L being a int key for the data access
        dd.extend([TDAncillaPentSTD[L][n] for n in MASK])  # also a dictionary
       
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

def collapsefunc(val):
    f = 0
    n=0
    for x in collapse_data2([val]):
        f += x**2
        n += 1
    return f/n

dz = 1e-3
xvals, yvals = NewtonRaphson(collapsefunc,1.01,dz,maxiter=50)
#This need to be z = xvals[-1] + 1 since the collapse has an extra power of L implied from the variable rescaling for Zctrl, otherwise it is normal for Zent
print(f'The final z value from the optimization is {xvals[-1]}')

z0 = xvals[-1]

errorbars = (collapsefunc(z0+dz)-2*collapsefunc(z0)+collapsefunc(z0-dz))/dz**2
print(np.sqrt(1/np.abs(errorbars)))