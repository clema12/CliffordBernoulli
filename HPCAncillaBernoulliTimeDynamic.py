import numpy as np
import matplotlib.pyplot as plt
import stim
import random
# from seeds import *
from seeds4000 import *
import sys

def getCutStabilizers(binaryMatrix, cuts):
    """
        - Purpose: Return only the part of the binary matrix that corresponds to the qubits we want to consider for a bipartition.
        - Inputs:
            - binaryMatrix (array of size (N, 2N)): The binary matrix for the stabilizer generators.
            - cut (integer): Location for the cut.
        - Outputs:
            - cutMatrix (array of size (N, 2cut)): The binary matrix for the cut on the left.
    """
    N = len(binaryMatrix)
    cutMatrix = np.zeros((N,2*len(cuts)))

    cutMatrix[:,:len(cuts)] = binaryMatrix[:,cuts]
    cutMatrix[:,len(cuts):] = binaryMatrix[:,[N+j for j in cuts]]

    return cutMatrix

def gf2_rank(rows):
    """
    Find rank of a matrix over GF2.

    The rows of the matrix are given as nonnegative integers, thought
    of as bit-strings.

    This function modifies the input list. Use gf2_rank(rows.copy())
    instead of gf2_rank(rows) to avoid modifying rows.
    """
    
    rank = 0
    while rows:
        pivot_row = rows.pop()
        if pivot_row:
            rank += 1
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
    return rank

def entropy(cutmatrix, numqubits):
    a = (cutmatrix > 0).tolist()
    atoints = [sum(b[j]*2**j for j in range(len(b))) for b in a]
    return gf2_rank(atoints) - numqubits

def binaryMatrix(zStabilizers):
    """
        - Purpose: Construct the binary matrix representing the stabilizer states.
        - Inputs:
            - zStabilizers (array): The result of conjugating the Z generators on the initial state.
        Outputs:
            - binaryMatrix (array of size (N, 2N)): An array that describes the location of the stabilizers in the tableau representation.
    """
    N = len(zStabilizers)
    binaryMatrix = np.zeros((N,2*N))
    r = 0 # Row number
    for row in zStabilizers:
        c = 0 # Column number
        for i in row:
            if i == 3: # Pauli Z
                binaryMatrix[r,N + c] = 1
            if i == 2: # Pauli Y
                binaryMatrix[r,N + c] = 1
                binaryMatrix[r,c] = 1
            if i == 1: # Pauli X
                binaryMatrix[r,c] = 1
            c += 1
        r += 1

    return binaryMatrix

def AncillaBernoulliCircuit(L,T,p,initseed,finalflag = True):
    MagOperators = ["I" * count + "Z" for count in range (L)]
    random.seed(initseed)
    SArray = []
    FinalSArray = []
    MeasureOutcomes = []
    s = stim.TableauSimulator()
    s.set_num_qubits(L+1)

    #maximally entangle qubits L and L+1 and apply a random L-qubit Clifford gate on the first L qubits
    inittab = s.current_inverse_tableau()
    H = stim.Tableau.from_named_gate("H")
    CNOT = stim.Tableau.from_named_gate("CNOT")
    inittab.prepend(H,[L-1])
    inittab.prepend(CNOT,[L-1,L])
    randomgate = stim.Tableau.random(L)
    inittab.prepend(randomgate,[i for i in range (L)])
    s.set_inverse_tableau(inittab)

    #Intial Entropy measure from settup up simulator, condense later
    pretableau = s.current_inverse_tableau() ** -1
    prezs = np.array([pretableau.z_output(k) for k in range(len(pretableau))])
    prebmatrix = binaryMatrix(prezs)
    precutMatrix = getCutStabilizers(prebmatrix,[L])
    initentropy = entropy(precutMatrix,1)
    SArray.append(initentropy)
    FinalSArray.append(initentropy)

    for t in range(T):
        #Bernoulli Map
        if random.random() > p: #prob 1-p
            tab = s.current_inverse_tableau()
            layer = stim.Tableau.random(2)
            swap = stim.Tableau.from_named_gate("SWAP")
            for i in range(L-1):
                tab.prepend(swap**(-1),[i, i+1])
            tab.prepend(layer**(-1),[L-2,L-1])
            s.set_inverse_tableau(tab)
            MeasureOutcomes.append(-1)

        #Control map
        else: #prob p
            if s.measure(L-1) == True: #measure last qubit with True <-> |1>
                MeasureOutcomes.append(1)
                tab2 = s.current_inverse_tableau()
                X = stim.Tableau.from_named_gate("X")
                tab2.prepend(X,[L-1]) #flips 1 -> 0, leaves 0 to control onto |000...0>
                s.set_inverse_tableau(tab2)
            else: 
                MeasureOutcomes.append(0)
            tab2 = s.current_inverse_tableau()
            swap = stim.Tableau.from_named_gate("SWAP")
            for i in range(L-1,0,-1):
                tab2.prepend(swap**(-1),[i, i-1]) #shift the bits backwards for the map to work regardless of bit flip
            s.set_inverse_tableau(tab2)
        if finalflag == False:
            if t%(T//50) == 0: #even spacing for the TD Data. Used in the 4000 realization plots but replacing (T//50) -> 5 recovers the normal TD Data set
                tableau = s.current_inverse_tableau() ** -1
                zs = np.array([tableau.z_output(k) for k in range(len(tableau))])
                bmatrix = binaryMatrix(zs)
                cutMatrix = getCutStabilizers(bmatrix,[L])
                SArray.append(entropy(cutMatrix,1))
    if finalflag == True:
        tableau = s.current_inverse_tableau() ** -1
        zs = np.array([tableau.z_output(k) for k in range(len(tableau))])
        bmatrix = binaryMatrix(zs)
        cutMatrix = getCutStabilizers(bmatrix,[L])
        FinalSArray.append(entropy(cutMatrix,1))
    rawmagnetization = 0
    for i in range(L):
        obs = stim.PauliString(MagOperators[i])
        rawmagnetization += s.peek_observable_expectation(obs)
    normmagnetization = (1/L)*rawmagnetization
    if finalflag == False: return SArray, MeasureOutcomes, normmagnetization
    else: return FinalSArray, MeasureOutcomes, normmagnetization

L = sys.argv[1]
p = sys.argv[2]
T = 2*int(L)**2
initseed = initseeds #This is the variable from seeds4000.py and in the appropriate form. To convert to 1000 reals, use the list comprehension from HPCAncillaBernoulli.py and switch the seeds import comment at the top

#start = timer()
TotalRealizationEntropy = []
TotalRealizationEntropysquared = []
TotalRealizationMagnetization = []
TotalRealizationMagnetizationsquared = []
measurementarr = []
for i in range(int(sys.argv[3])):
    RealizationEntropy,RealizationMeasure,RealizationMagnetization = AncillaBernoulliCircuit(int(L),T,float(p),initseed[i],False)
    TotalRealizationEntropy.append(RealizationEntropy)
    TotalRealizationEntropysquared.append(RealizationEntropy)
    TotalRealizationMagnetization.append(RealizationMagnetization)
    TotalRealizationMagnetizationsquared.append(RealizationMagnetization**2)
    measurementarr.append(RealizationMeasure)
TotalRealizationEntropy = np.asarray(TotalRealizationEntropy)
TotalRealizationEntropysquared = np.asarray(TotalRealizationEntropysquared)
TotalRealizationMagnetization = np.asarray(TotalRealizationMagnetization)
TotalRealizationMagnetizationsquared = np.asarray(TotalRealizationMagnetizationsquared)
measurementarr = np.asarray(measurementarr)
np.save("L" + str(L) + "P" + str(int(round(float(p)*1000,0))) + "TDAncillaBernoulliDataEntropy4000Reals",TotalRealizationEntropy)
np.save("L" + str(L) + "P" + str(int(round(float(p)*1000,0))) + "TDAncillaBernoulliDataEntropysquared4000Reals",TotalRealizationEntropysquared)
np.save("L" + str(L) + "P" + str(int(round(float(p)*1000,0))) + "TDAncillaBernoulliDataMag4000Reals",TotalRealizationMagnetization)
np.save("L" + str(L) + "P" + str(int(round(float(p)*1000,0))) + "TDAncillaBernoulliDataMagsquared4000Reals",TotalRealizationMagnetizationsquared)
np.save("L" + str(L) + "P" + str(int(round(float(p)*1000,0))) + "TDAncillaBernoulliDataMeasurements4000Reals",measurementarr)
#stop = timer()
#print(stop-start)