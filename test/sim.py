import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks as fp

R = 0.047e6

NLR = 120/21.5


def AMPLITUDES(DATA,INDICES):
    AMPS = np.array([])
    if (DATA[0] > DATA[1]):
        DATA = DATA[1:]
    for i in range(INDICES.size-1):
        L = DATA[INDICES[i]]
        R = DATA[INDICES[i+1]]
        if (L < R):
            AMPS = np.append(AMPS, R-L)
    return AMPS

# Removes consecutive minimum / maximum points from the conjoined array of maximum and minimum indices
def FIX(MAXES,MINS):
    ALL = (np.sort(np.append(MAXES,MINS)))
    INDICES = np.array([])
    REMOVE = np.array([],dtype=np.int32)
    for i in range(len(ALL)):
        temp = np.argwhere(MAXES == ALL[i])
        if (temp.size != 0):
            INDICES = np.append(INDICES,1)
        temp = np.argwhere(MINS == ALL[i])
        if (temp.size != 0):
            INDICES = np.append(INDICES,0)
    for i in range(len(INDICES)-1):
        if (INDICES[i] == INDICES[i+1]):
            REMOVE = np.append(REMOVE, i)
    ALL = np.delete(ALL, np.flip(REMOVE))
    return ALL





# (R/R0) * V0
# R = 0.047M,   R0 = 0.148M,    V0 = 0.300V
constInput = (R)/(0.148e6 + 7.43e3) * 0.300

# numerical solution of ODE with a given ratio of (R/Rv)
def ivpODE(ratio):
    def thirdOrderODE(t, xt):
        x = xt[0]
        dx = xt[1]
        d2x = xt[2]
        d3x = -ratio*d2x - dx - NLR*min(x,0) - constInput
        return np.array([dx,d2x,d3x])
    return thirdOrderODE

# with a given ratio of (R/Rv) uses solve_ivp to get wave data for a chaotic system and returns voltage x
def getSol(ratio,TSTEP):
    sln = solve_ivp(ivpODE(ratio),t_span=[0,750],t_eval=np.linspace(500,750,5000),y0=np.zeros(3),max_step=TSTEP)
    return sln.t,sln.y[0],sln.y[1]

def simulate(Rv,TSTEP):
    t,x,dx = getSol(R/Rv, TSTEP)
    return t,x,dx

def simbifurcate(Rv,TSTEP):
    _,x,_ = simulate(Rv,TSTEP)
    N = fp(x)[0]
    return x[N]
    