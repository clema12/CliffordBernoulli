import numpy as np
import matplotlib.pyplot as plt
import glob

def GraphDataPrep(Entropyfilepath,Magnetizationfilepath):
    Entropy = []
    EntropyData = []
    Magnetization = []
    MagnetizationData = []

    for files in glob.glob(Entropyfilepath):
        Entropy.append(files)

    for files in glob.glob(Magnetizationfilepath):
        Magnetization.append(files)
    
    for files in Entropy:
        with open(files,'rb') as f:
            filedata = np.load(f)
            EntropyData.append(filedata)

    for files in Magnetization:
        with open(files,'rb') as f:
            filedata = np.load(f)
            MagnetizationData.append(filedata)

    EntropyAVG = [np.average(element) for element in EntropyData]
    EntropySTD = [np.std(element)/np.sqrt(1000) for element in EntropyData]

    MagnetizationAVG = [np.average(element) for element in MagnetizationData]
    MagnetizationSTD = [np.std(element)/np.sqrt(1000) for element in MagnetizationData]

    return EntropyAVG, EntropySTD, MagnetizationAVG, MagnetizationSTD