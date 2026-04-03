import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks as fp

Rc = 0.047e6
R0 = 0.148e6 + 7.43e3
DxRatio = 120e3/22e3
V0 = Rc/R0*0.3

def func(t,xt,Rv):
    x = xt[0]
    dx = xt[1]
    d2x = xt[2]
    d3x = -Rc/Rv*d2x - dx - DxRatio*min(x,0) - V0
    return [dx,d2x,d3x]


def simulate(Rv):
    sol = solve_ivp(func,args=(Rv,),t_span=[-100,500],t_eval=np.linspace(0,312,150000),y0=np.zeros(3))
    return sol.t,sol.y[0],sol.y[1]