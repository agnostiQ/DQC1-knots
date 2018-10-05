#Numpy/scipy imports
import numpy as np

#QISkit imports
from qiskit import QuantumProgram, QuantumCircuit, ClassicalRegister, QuantumRegister, QISKitError, qasm, get_backend

#Current project imports
import utils

braids = 3
phi    = 1.618033988749895
consta = -0.309017 - 0.951057*1.j
weights= [0.38196601125010515, 0.2360679774997897]

def getNorm(writhe):
    return (-consta)**(3*writhe) * phi**(braids-1)

def comSequence(qc, qrId):
    qc.u3(2.2370357415988407, np.pi, 0, qrId)

def braid23w1(qc, qrData, qrId, inv=False):
    """
    Adds instructions to Program p,
    corresponding to weight 1 unitary
    for 2-3 braid operator.
    """
    if inv==False:
        phase = 2*np.pi/5
    else:
        phase = -2*np.pi/5
    
    qc.cu1(phase, qrData, qrId)
    comSequence(qc, qrId)
    qc.cz(qrData, qrId)
    comSequence(qc, qrId)
    qc.cu1(phase, qrData, qrId)
    
def braid12w1(qc, qrData, qrId, inv=False):
    """
    Adds instructions to Program p,
    corresponding to weight 1 unitary
    for 1-2 braid operator.
    """
    if inv==False:
        phase = 1.8849555921538759
    else:
        phase = -1.8849555921538759

    qc.cu1(phase, qrData, qrId)
    qc.u1(phase, qrData)

def braid123w2(qc, qrData, qrId, inv=False):
    """
    Adds instructions to Program p,
    corresponding to weight 2 unitary,
    for BOTH braid operators.
    """
    if inv==False:
        phase = 1.8849555921538759
    else:
        phase = -1.8849555921538759

    qc.cu1(phase, qrData, qrId)

def getProg(opSequence, qr, cr, weight=1, inState='o', mBasis='x'):
    """
    Returns program to be run on QPU or QVM.
    opSequence specifies which weight and
    braid operator to perform.
    """
    qc = QuantumCircuit(qr, cr)

    qc.h(qr[0])
    
    if inState=='x':
        qc.x(qr[1])
    
    #Main controlled-2x2-unitary sequence
    for i in opSequence:
        if weight==1:
            if i=='s12':
                braid12w1(qc, qr[0], qr[1])
            elif i=='s23':
                braid23w1(qc, qr[0], qr[1])
            elif i=='s12i':
                braid12w1(qc, qr[0], qr[1], inv=True)
            elif i=='s23i':
                braid23w1(qc, qr[0], qr[1], inv=True)
        elif weight==2:
            if i=='s12' or i=='s23':
                braid123w2(qc, qr[0], qr[1])
            elif i=='s12i' or i=='s23i':
                braid123w2(qc, qr[0], qr[1], inv=True)
    
    if mBasis=='x':
        qc.h(qr[0])
    elif mBasis=='y':
        qc.s(qr[0])
        qc.h(qr[0])
    
    qc.measure(qr, cr)

    return qc

def buildJob(opSeq, shots, backend, reps):
    """
    Shell function to call a QVM execution and to compile results
    """
    #There are always two weights to trace over regardless of braid or writhe
    #First call relevant code for weight 1.
    dimW1 = 4  #Number of quantum circuits to run for weight 1 block
    dimW2 = 2  #Number of quantum circuits to run for weight 2 block
    qp = QuantumProgram()
    qr = qp.create_quantum_register("q",2)
    cr = ClassicalRegister(2)
    qc = reps*(dimW1+dimW2)*['']
    names = reps*(dimW1+dimW2)*['']

    # res1 = np.zeros((arrZipped.shape[0],shots,1))
    idx = 0
    for rep in range(reps):
        #Now call relevant code for weight 1.
        arrState = np.array(['o','x'])
        arrBasis = np.array(['x','y'])
        arrZipped= utils.zipTuple(arrBasis,arrState)
        
        for i in arrZipped:
            qc[idx] = getProg(opSeq, qr, cr, weight=1, inState=i[1], mBasis=i[0])
            names[idx] = "Rep" + str(rep) + ",Weight 1," + i[0] + i[1]
            qp.add_circuit(names[idx], qc[idx])
            idx += 1

        #Now call relevant code for weight 2.
        arrState = np.array(['x'])
        arrBasis = np.array(['x','y'])
        arrZipped= utils.zipTuple(arrBasis,arrState)

        # res2 = np.zeros((arrZipped.shape[0],shots,1))
        for i in arrZipped:
            qc[idx] = getProg(opSeq, qr, cr, weight=2, inState=i[1], mBasis=i[0])
            names[idx] = "Rep" + str(rep) + ",Weight 2," + i[0] + i[1]
            qp.add_circuit(names[idx], qc[idx])
            idx += 1
    
    #RUN!!!
    qpCompiled = qp.compile(names, backend=backend, shots=shots)

    return qpCompiled

def computeJones(countsW1, countsW2):
    """
    Parse results from IBM QPU & compute Jone's polynom
    """
    res1Sum = np.zeros(4)
    idx = 0
    for i in countsW1:
        tracedSum = utils.traceGetCount(i, [1]) #For some stupid reason, qr[0] is ordered last
        res1Sum[idx] = tracedSum['1']/(tracedSum['0']+tracedSum['1'])
        idx += 1

    res1Sum = res1Sum.reshape(-1,2)  #Make conjugate pairs, over which to multiply weights before summing
    res1Sum = 4*(0.5*np.sum(res1Sum,axis=1)-0.5)  #Convert to tr(U) params
    res1Sum[1] = -res1Sum[1]  #Fix sign on <Y> stemming from missing Z in our definitions.

    res2Sum = np.zeros(2)
    idx = 0
    for i in countsW2:
        tracedSum = utils.traceGetCount(i, [1]) #For some stupid reason, qr[0] is ordered last
        res2Sum[idx] = tracedSum['1']/(tracedSum['0']+tracedSum['1'])
        idx += 1

    res2Sum = 2*(res2Sum-0.5)  #Convert to tr(U) params
    res2Sum[1] = -res2Sum[1]  #Fix sign on <Y> stemming from missing Z in our definitions.

    res = weights[0]*res1Sum + weights[1]*res2Sum
    res = res[0] + res[1]*1.j
    
    return res