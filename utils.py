import numpy as np
from math import floor
def zipTuple(arr1, arr2):
    """
    Zip arrays, in little-endian (i.e. lexicographical) ordering
    """
    dim1 = arr1.shape[0]
    dim2 = arr2.shape[0]
    dimT = dim1*dim2
    
    zipped = np.dstack((
        arr1.repeat(dim2).reshape(dimT),
        arr2.repeat(dim1).reshape(dim2,dim1).T.reshape(dimT)
    )).reshape((dimT,-1))
    
    return zipped

def getComputationalBasis(n):
    """
    Returns a dictionary with computational basis
    in lexicographical ordering, for n qubits.
    """
    myDict = {}
    for i in range(2**n):
        word = ''
        for j in range(n):
            word = word + str((i//(2**j))%2)
        myDict[word] = 0
    
    return myDict

def traceGetCount(countDict, i):
    """
    Takes counts dictionary from IBM QPU.
    Returns counts for qubit(s) i, with others traced out.
    """
    dim = len(i)
    myDict = getComputationalBasis(dim)

    for j in countDict:
        shortKey = ''
        for k in i:
            shortKey = shortKey + j[k]
        
        myDict[shortKey] += countDict[j]
    
    return myDict