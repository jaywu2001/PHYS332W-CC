import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks, savgol_filter
import struct
import os

R = 47
R0 = 157
R1 = 21.85
R2 = 120

alpha = (R/R0)*0.3

lower = 0
upper = 1000
points = 5000

g1 = 0.0
g2 = 0.0
g3 = 0.0
x0 = [g1, g2, g3]

tvalues = np.linspace(lower, upper, points)

def data(filename):
    
    with open(filename, 'rb') as file:

        mag = {5:1e-9, 6:1e-6, 7:1e-3, 8:1e0, 9:1e3, 10:1e6, 11:1e9}
        
        file.seek(0x0,0)
        state1 = struct.unpack('<i',file.read(0x4))[0]
        file.seek(0x4,0)
        state2 = struct.unpack('<i',file.read(0x4))[0]
        
        file.seek(0xd4,0)
        tdiv = struct.unpack('<d',file.read(0x8))[0]
        file.seek(0xdc,0)
        tmag = mag[struct.unpack('<i',file.read(0x4))[0]]
        tdiv *= tmag
        
        file.seek(0xf4,0)
        wavelength = struct.unpack('<i',file.read(0x4))[0]
        narray = np.linspace(0,wavelength,wavelength)
            
        file.seek(0xf8,0)
        samplerate = struct.unpack('<d',file.read(0x8))[0]
        file.seek(0x100,0)
        sampleratemag = mag[struct.unpack('<i',file.read(0x4))[0]]
        samplerate *= sampleratemag

        ngrid = 14.0
        cpdiv = 25.0
        
        if (state1 == 1):
            file.seek(0x10,0)
            ch1div = struct.unpack('<d',file.read(0x8))[0]
            file.seek(0x18,0)
            ch1mag = mag[struct.unpack('<i',file.read(0x4))[0]]
            ch1div *= ch1mag
            file.seek(0x800,0)
            ch1 = np.int8(np.fromfile(file,dtype=np.uint8,count=wavelength,offset=0)-128)*(ch1div/cpdiv)
        
        if (state2 == 1):
            file.seek(0x20,0)
            ch2div = struct.unpack('<d',file.read(0x8))[0]
            file.seek(0x28,0)
            ch2mag = mag[struct.unpack('<i',file.read(0x4))[0]]
            ch2div *= ch2mag
            offset = 0x800
            file.seek(0x800,0)
            if (state1 == 1):
                file.seek(wavelength,1)
            ch2 = np.int8(np.fromfile(file,dtype=np.uint8,count=wavelength)-128)*(ch2div/cpdiv)
            
        t = -(tdiv*ngrid/2)+narray*(1/samplerate)
        
        return ch1, ch2, t

def loaddata1(filename, transient):
    
    ch1, ch2, t = data(filename)
    
    if len(ch1) < 100:
        print("Skipping {}: data too short".format(filename))
        return np.array([]), np.array([])
    
    probefactor = 10
    x = ch1 * probefactor

    aftertransients = int(len(x) * transient)
    
    if aftertransients >= len(x):
        print("Skipping {}: transients too long".format(filename))
        return np.array([]), np.array([])

    x = x[aftertransients:]
    
    if len(x) == 0:
        print("Skipping {}: empty after transients".format(filename))
        return np.array([]), np.array([])
    
    t = t[aftertransients:]
    
    return x, t

def loaddata2(filename, transient):
    
    ch1, ch2, t = data(filename)
    
    probefactor = 10
    x = ch1 * probefactor
    xdot = -ch2 * probefactor

    aftertransients = int(len(x)*transient)
    
    x = x[aftertransients:]
    
    xdot = xdot[aftertransients:]
    
    t = t[aftertransients:]

    if len(x) >= 101:
        x = savgol_filter(x, 101, 3)
        xdot = savgol_filter(xdot, 101, 3)
    else:
        print("No smoothing for {}: {} points".format(filename, len(x)))
    
    return x, xdot, t

def D(x):
    return -(R2/R1) * np.minimum(x, 0)

def dx_dt(t, xs, Rv):
    x = xs[0]
    xdot = xs[1]
    xddot = xs[2]
    Dx = D(x)
    return (xdot, xddot, -(R/Rv)*xddot - xdot + Dx - alpha)

def IVP(Rv, x0):

    return solve_ivp(dx_dt,[lower, upper], y0=x0, args=(Rv,), t_eval=tvalues, rtol=1e-10, atol=1e-10)


def PlotCombined(filename, left1, right1, left2, right2, size, transientP, transientPP, Rv, title):
    
    x1, t1 = loaddata1(filename, transientP)
    
    xpeakid1 = find_peaks(x1, distance = len(x1)//size, prominence = np.std(x1)*0.25)[0]
    peaks1 = x1[xpeakid1]

    res1 = IVP(Rv, x0)

    limit = res1.t > (upper*0.8)
    t2 = res1.t[limit]
    x2 = res1.y[0][limit]

    xpeakid2, _ = find_peaks(x2)
    peaks2 = x2[xpeakid2]

    x3, xdot3, t3 = loaddata2(filename, transientPP)

    step = max(1, len(x3)//10000)
    
    x3 = x3[::step]
    xdot3 = xdot3[::step]

    res2 = IVP(Rv, x0)
    
    limit = res2.t > (upper*0.5)
    x4 = res2.y[0][limit]
    xdot4 = res2.y[1][limit]

    fig = plt.figure(figsize=(12, 6), dpi=100)

    gs = GridSpec(3, 2, height_ratios = [1, 1, 2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[2, 1])

    ax1.plot(t1, x1, color='black', linewidth = 0.8)
    ax1.set_xlim(left1, right1)
    ax1.set_title('Experimental Data')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)')
    ax2.plot(t1, x1, color='black', linewidth = 0.8)
    ax2.plot(t1[xpeakid1], peaks1, '.', color='red', markersize = 7.5)
    ax2.set_xlim(left1, right1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Voltage (V)')
    ax3.plot(x3, xdot3, color='black', linewidth = 0.8)
    ax3.set_xlim(left2, right2)
    ax3.set_xlabel('x (V)')
    ax3.set_ylabel('-ẋ (V)')
    ax3.axis('equal')

    ax4.plot(t2, x2, color='black', linewidth = 0.8)
    ax4.set_xlim(min(t2), max(t2))
    ax4.set_title('Simulated Data')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('x')
    ax5.plot(t2, x2, color='black', linewidth = 0.8)
    ax5.plot(t2[xpeakid2], peaks2, '.', color='red', markersize = 7.5)
    ax5.set_xlim(min(t2), max(t2))
    ax5.set_xlabel('Time')
    ax5.set_ylabel('x')
    ax6.plot(x4, xdot4, color='black', linewidth = 0.8)
    ax6.set_xlim(left2, right2)
    ax6.set_xlabel('x')
    ax6.set_ylabel('-ẋ')
    ax6.axis('equal')

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()

def CombinedBifurcation(start, end, increment, size1, size2, amount, transient, titlel, titler, title):

    Rvvalues1 = np.arange(start, end, increment)
    
    Rvlist1 = []
    peaklist1 = []
    
    for i in range(1, len(Rvvalues1) + 1):
        try:
            if i < 10:
                filename = 'SDS0000'+str(i)+'.bin'
            elif i < 100:
                filename = 'SDS000'+str(i)+'.bin'
            elif i < 1000:
                filename = 'SDS00'+str(i)+'.bin'
            else:
                filename = 'SDS0'+str(i)+'.bin'
    
            if not os.path.exists(filename):
                continue
    
            x1, t1 = loaddata1(filename, transient)

            cut = int(len(x1)*0.7)
            x1 = x1[cut:]

            x1 = savgol_filter(x1, 11, 3)

            xpeakid1, _ = find_peaks(x1, distance = len(x1)//size1, prominence=np.std(x1)*0.25)
            peaks1 = x1[xpeakid1]

            if i-1 < len(Rvvalues1):
                Rv1 = Rvvalues1[i-1]
                Rvlist1.extend([Rv1] * len(peaks1))
                peaklist1.extend(peaks1)
            else:
                continue
    
        except Exception as e:
            print(filename, e)

    Rvvalues2 = np.linspace(start, end, amount)

    Rvlist2 = []
    xmaxlist2 = []

    currentx0 = x0

    for Rv2 in Rvvalues2:
        res = IVP(Rv2, currentx0)
    
        limit = res.t > (upper*0.6)
        x2 = res.y[0][limit]
        
        xpeakid2, _ = find_peaks(x2, distance = len(x2)//size2, prominence=np.std(x2)*0.05)
        peaks2 = x2[xpeakid2]
    
        currentx0 = res.y[:, -1]
    
        Rvlist2.extend([Rv2] * len(peaks2))
        xmaxlist2.extend(peaks2)

    fig = plt.figure(figsize=(12, 4), dpi=100)

    gs = GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.scatter(Rvlist1, peaklist1, s=0.1, c='black')
    ax1.set_ylim(-0.1, 1.5)
    ax1.set_title(titlel)
    ax1.set_xlabel('Rv (kΩ)')
    ax1.set_ylabel('Peak Voltage (V)')
    
    ax2.scatter(Rvlist2, xmaxlist2, s=0.1, c='black')
    ax2.set_title(titler)
    ax2.set_xlabel('Rv (kΩ)')
    ax2.set_ylabel('Peak Voltage (V)')

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()
    
def CombinedPowerSpectrumDensity(filename1, filename2, filename3, filename4, transient, title):

    x1, t1 = loaddata1(filename1, transient)
    
    n1 = len(x1)
    dt1 = t1[1] - t1[0]
    fs1 = 1/dt1

    powerspectrumdensity1, frequencies1, line1 = plt.psd(x1, NFFT=n1, Fs=fs1, return_line=True)
    plt.close()

    x2, t2 = loaddata1(filename2, transient)
    
    n2 = len(x2)
    dt2 = t2[1] - t2[0]
    fs2 = 1/dt2

    powerspectrumdensity2, frequencies2, line2 = plt.psd(x2, NFFT=n2, Fs=fs2, return_line=True)
    plt.close()

    x3, t3 = loaddata1(filename3, transient)
    
    n3 = len(x3)
    dt3 = t3[1] - t3[0]
    fs3 = 1/dt3

    powerspectrumdensity3, frequencies3, line3 = plt.psd(x3, NFFT=n3, Fs=fs3, return_line=True)
    plt.close()

    x4, t4 = loaddata1(filename4, transient)
    
    n4 = len(x4)
    dt4 = t4[1] - t4[0]
    fs4 = 1/dt4

    powerspectrumdensity4, frequencies4, line4 = plt.psd(x4, NFFT=n4, Fs=fs4, return_line=True)
    plt.close()

    fig = plt.figure(figsize=(12, 6), dpi=100)

    gs = GridSpec(4, 1, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])

    ax1.plot(frequencies1, powerspectrumdensity1, color='black', linewidth = 0.8)
    ax1.set_xlim(0.9e1, 5.50e6)
    ax1.set_title(title)
    ax1.set_xlabel('')
    ax1.tick_params(labelbottom=False)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('One Period')
    
    ax2.plot(frequencies2, powerspectrumdensity2, color='black', linewidth = 0.8)
    ax2.set_xlim(0.9e1, 5.50e6)
    ax2.set_xlabel('')
    ax2.tick_params(labelbottom=False)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylabel('Two Period')

    ax3.plot(frequencies3, powerspectrumdensity3, color='black', linewidth = 0.8)
    ax3.set_xlim(0.9e1, 5.50e6)
    ax3.set_xlabel('')
    ax3.tick_params(labelbottom=False)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylabel('Four Period')

    ax4.plot(frequencies4, powerspectrumdensity4, color='black', linewidth = 0.8)
    ax4.set_xlim(0.9e1, 5.50e6)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Chaotic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()

def FeigenbaumDeltaA(A):

    deltalist = []

    for i in range(1, len(A) - 1):
        delta = (A[i] - A[i-1])/(A[i+1] - A[i])

        deltalist.append(delta)
        
        print('Feigenbaum delta{}: {:.3f}'.format(str(i), delta))

def CombinedReturnMap(filename, size, transient, r1, r2, left1, right1, left2, right2, title):

    x, t = loaddata1(filename, transient)
    
    xpeakid, _ = find_peaks(x, distance = len(x)//size, prominence = np.std(x)*0.35)
    peaks = x[xpeakid]
    
    xn = peaks[:-r1]
    xn1 = peaks[:-r2]

    xnplus1 = peaks[r1:]
    xnplus2 = peaks[r2:]

    fig = plt.figure(figsize=(12, 6), dpi=100)

    gs = GridSpec(2, 2, height_ratios = [1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.plot(xn, xnplus1, '.', color='black', markersize = 2)
    ax1.set_title('First Return Map')
    ax1.set_xlabel('x(n)')
    ax1.set_ylabel('x(n+1)')
    ax2.plot(t, x, color='black', linewidth = 0.8)
    ax2.set_xlim(left1, right1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Voltage (V)')

    ax3.plot(xn1, xnplus2, '.', color='black', markersize = 2)
    ax3.set_title('Second Return Map')
    ax3.set_xlabel('x(n)')
    ax3.set_ylabel('x(n+2)')
    ax4.plot(t, x, color='black', linewidth = 0.8)
    ax4.set_xlim(left2, right2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Voltage (V)')

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()

filenameone = 'SDS00001.bin'
filenametwo = 'SDS00120.bin'
filenamefour = 'SDS00210.bin'
filenamechaotic1 = 'SDS00260.bin'
filenamechaotic2 = 'SDS00480.bin'

RvSimulatedOne = 50.0
RvSimulatedTwo = 61.9
RvSimulatedFour = 70.9
RvSimulatedChaotic1 = 75.9
RvSimulatedChaotic2 = 97.9

# PlotCombined(filenameone, 0.00, 0.01, -0.15, 0.35, 600, 0.2, 0.5, RvSimulatedOne, 'One Period Behaviour')
# PlotCombined(filenametwo, 0.00, 0.01, -0.15, 0.35, 600, 0.2, 0.5, RvSimulatedTwo, 'Two Period Behaviour')
# PlotCombined(filenamefour, 0.00, 0.01, -0.15, 0.35, 600, 0.2, 0.5, RvSimulatedFour, 'Four Period Behaviour')
# PlotCombined(filenamechaotic2, 0.00, 0.01, -0.15, 0.35, 600, 0.2, 0.5, RvSimulatedChaotic2, 'Chaotic Behaviour')





CombinedBifurcation(50.0, 150.0, 0.1, 500, 325, 500, 0.2, 'Experimental Bifurcation Plot', 'Simulated Bifurcation Plot', 'Bifurcation Plots')





# CombinedPowerSpectrumDensity(filenameone, filenametwo, filenamefour, filenamechaotic1, 0.3, 'Power Spectrum Densities')





A = np.array([1.174, 1.453, 1.513, 1.526])

print('---------------------------------------------------------------------')
print('Feigenbaum delta:')
print('---------------------------------------------------------------------')

# FeigenbaumDeltaA(A)





# CombinedReturnMap(filenamechaotic1, 10000, 0.5, 1, 2, 0.00, 0.015, 0.035, 0.05, 'Return Maps')
