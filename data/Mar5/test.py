import matplotlib.pyplot as plt
import numpy as np
import os
import chaoticPackage as CC

plt.ion()

tfiles = os.listdir("lower/")
files = ["lower/"+x for x in tfiles]
tfiles = os.listdir("lowmid/")
files += ["lowmid/"+x for x in tfiles]
tfiles = os.listdir("afterlower/")
files += ["afterlower/"+x for x in tfiles]
print(len(files))
rvalues = np.arange(10.0,25.0,step=0.1)
rvalues = np.append(rvalues, np.arange(25.0,50.0,step=0.1))
rvalues = np.append(rvalues, np.arange(50.0,115.05,step=0.05))

plt.figure(figsize=(16,9),dpi=100)
for i in range(len(files)):
    data = CC.getAmplitudes(files[i])
    plt.plot(np.ones(len(data))*rvalues[i],data,'k.',markersize=0.1,alpha=0.5)
plt.ylabel(r"Amplitude (V)")
plt.xlabel(r"$R_v$ (k$\Omega$)")
plt.title(r"Amplitude (V) vs. $R_v$ (k$\Omega$)")
plt.xticks(np.linspace(10.0,120.0,12))
plt.show()