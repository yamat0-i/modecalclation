import numpy as np
import matplotlib.pyplot as plt

import fibermode
import opticalfiber.he as he
import opticalfiber.te as te
import opticalfiber.tm as tm

wavelength = 785e-9
radius = np.linspace(0, 500e-9, 1000)
n_silica = 1.45
n_vacuum = 1.0
k = 2*np.pi/wavelength

betaHE11 = np.zeros(np.shape(radius))
for i in np.arange(len(radius)):
    he11 = he.HE(l=1,
                 m=1,
                 wavelength=wavelength,
                 radius=radius[i],
                 n_1=n_silica,
                 n_2=n_vacuum
                 )
    try:
        betaHE11[i] = he11.beta / he11.k
    except TypeError:
        betaHE11[i] = 0

betaHE21 = np.zeros(np.shape(radius))
for i in np.arange(len(radius)):
    he21 = he.HE(l=2,
                 m=1,
                 wavelength=wavelength,
                 radius=radius[i],
                 n_1=n_silica,
                 n_2=n_vacuum
                 )
    try:
        betaHE21[i] = he21.beta / he21.k
    except TypeError:
        betaHE21[i] = 0

betaTE01 = np.zeros(np.shape(radius))
for i in np.arange(len(radius)):
    te01 = te.TE(m=1,
                 wavelength=wavelength,
                 radius=radius[i],
                 n_1=n_silica,
                 n_2=n_vacuum,
                 )
    try:
        betaTE01[i] = te01.beta / te01.k
    except TypeError:
        betaTE01[i] = 0

betaTM01 = np.zeros(np.shape(radius))
for i in np.arange(len(radius)):
    tm01 = tm.TM(m=1,
                 wavelength=wavelength,
                 radius=radius[i],
                 n_1=n_silica,
                 n_2=n_vacuum,
                 )
    try:
        betaTM01[i] = tm01.beta / tm01.k
    except TypeError:
        betaTM01[i] = 0


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(radius, betaHE11, 'b', label='HE11')
ax.plot(radius, betaTE01, 'r', label='TE01')
ax.plot(radius, betaTM01, 'g', label='TM01')
ax.plot(radius, betaHE21, 'c', label='HE21')

ax.set_xlim(0, 500e-9)
ax.set_xticks([0, 1e-7, 2e-7, 3e-7, 4e-7, 5e-7])
ax.set_xticklabels(['0', '100', '200', '300', '400', '500'])
ax.set_xlabel('Fiber radius(nm)', fontsize=16)

ax.set_ylim([1, 1.4])
ax.set_ylabel(r'$\beta / k$', fontsize=16)

ax.legend()
plt.show()


