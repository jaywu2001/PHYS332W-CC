import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

R = 0.047e6

# (R/R0) * V0
# R = 0.047M,   R0 = 0.148M,    V0 = 0.300V
constInput = (0.047e6)/(0.148e6 + 7.43e3) * 0.300

# numerical solution of ODE with a given ratio of (R/Rv)
def ivpODE(ratio):
    def thirdOrderODE(t, xt):
        x = xt[0]
        dx = xt[1]
        d2x = xt[2]
        d3x = -ratio*d2x - dx - 6.6729*min(x,0) - constInput
        return np.array([dx,d2x,d3x])
    return thirdOrderODE

# with a given ratio of (R/Rv) uses solve_ivp to get wave data for a chaotic system and returns voltage x()
def getSol(ratio,TSTEP):
    sln = solve_ivp(ivpODE(ratio),t_span=[0,1000],y0=np.zeros(3),max_step=TSTEP)
    return sln.t,sln.y[0],sln.y[1]

def sim(R,TSTEP):
    t,x,dx = getSol(83e3/R, TSTEP)
    t /= 1000
    return t,x,dx