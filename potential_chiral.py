import numpy as np
from scipy.constants import k
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import fibermode


radius_list = np.linspace(150e-9, 350e-9, 1000)
pol = np.pi/2
phi=pol
T = 300 # [K]

L = 1000e-6
D_list = 2 * radius_list
D0 = 2 * D_list[0]
z = L * (np.log(D_list/ D0))

def linearmode(wavelength, radius, core_index, clad_index, power, pol):
    mode = fibermode.HE(wavelength=wavelength,
                        radius=radius,
                        core_index=core_index,
                        clad_index=clad_index,
                        power=power)
    rho = radius + 1e-9
    Ex = mode.Ex(rho=rho, phi=phi, pol=pol)
    Ey = mode.Ey(rho=rho, phi=phi, pol=pol)
    Ez = mode.Ez(rho=rho, phi=phi, pol=pol)
    return Ex, Ey, Ez

def circmode(wavelength, radius, core_index, clad_index, power):
    ExH, EyH, EzH = linearmode(wavelength=wavelength,
                     radius=radius,
                     core_index=core_index,
                     clad_index=clad_index,
                     power=power,
                     pol=0)
    ExV, EyV, EzV = linearmode(wavelength=wavelength,
                     radius=radius,
                     core_index=core_index,
                     clad_index=clad_index,
                     power=power,
                     pol=np.pi/2)
    
    Ex = (ExH + 1j*ExV) / np.sqrt(2)
    Ey = (EyH + 1j*EyV) / np.sqrt(2)
    Ez = (EzH + 1j*EzV) / np.sqrt(2)
    return Ex, Ey, Ez

def calc_intensity_lin(wavelength, radius, power):
    Ex, Ey, Ez = linearmode(wavelength=wavelength,
                                    radius=radius,
                                    core_index=1.45,
                                    clad_index=1.0,
                                    power=power,
                                    pol=pol)
    I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
    return I

def calc_intensity_circ(wavelength, radius, power):
    Ex, Ey, Ez = circmode(wavelength=wavelength,
                                    radius=radius,
                                    core_index=1.45,
                                    clad_index=1.0,
                                    power=power,
                                    )
    I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
    return I

def get_force(wavelength, radius, power, state):
    F = np.zeros(np.shape(radius_list))
    if state == 'Lin':
        for i in np.arange(len(radius_list)):
            F[i] = calc_intensity_lin(wavelength=wavelength,
                                    radius=radius[i],
                                    power=power)
        return F
    elif state == '+':
        for i in np.arange(len(radius_list)):
            F[i] = calc_intensity_circ(wavelength=wavelength,
                                    radius=radius[i],
                                    power=power)
        return 1.1*F
    elif state == '-':
        for i in np.arange(len(radius_list)):
            F[i] = calc_intensity_circ(wavelength=wavelength,
                                    radius=radius[i],
                                    power=power)
        return 0.9*F

def calc_potential(F):
    U = -cumulative_trapezoid(F, radius_list, initial=0)
    return U

F660_10_lin = get_force(wavelength=660e-9,
                    radius=radius_list,
                    power=10e-3,
                    state='Lin')
F660_10_p = get_force(wavelength=660e-9,
                    radius=radius_list,
                    power=10e-3,
                    state='+')
F660_10_m = get_force(wavelength=660e-9,
                    radius=radius_list,
                    power=10e-3,
                    state='-')
F785_10_Lin = get_force(wavelength=785e-9,
                    radius=radius_list,
                    power=10e-3,
                    state='Lin')

U1 = calc_potential(F660_10_lin - F785_10_Lin)
U1_set0 = U1 - np.amin(U1)
U1_norm = (U1_set0 - np.amin(U1_set0)) / (np.amax(U1_set0) + np.amin(U1_set0))
U2 = calc_potential(F660_10_p - F785_10_Lin)
U2_set0 = U2 - np.amin(U2)
U2_norm = (U2_set0 - np.amin(U1_set0)) / (np.amax(U1_set0) + np.amin(U1_set0))
U3 = calc_potential(F660_10_m - F785_10_Lin)
U3_set0 = U3 - np.amin(U3)
U3_norm = (U3_set0 - np.amin(U1_set0)) / (np.amax(U1_set0) + np.amin(U1_set0))

fig = plt.figure(layout='tight')
ax = fig.add_subplot(111)
ax.plot(radius_list, U1_norm, color='k', linestyle='solid', label=r'$P=1.0, F^{660}=F^{achiral}_{rad}$')
ax.plot(radius_list, U2_norm, color='r', linestyle='solid', label=r'$P=1.0, F^{660}=F^{achiral}_{rad}+F^{chiral}_{rad}$')
ax.plot(radius_list, U3_norm, color='b', linestyle='solid', label=r'$P=1.0, F^{660}=F^{achiral}_{rad}-F^{chiral}_{rad}$')
ax.set_xticks([1.5e-7, 1.75e-7, 2.0e-7, 2.25e-7, 2.5e-7, 2.75e-7, 3.0e-7, 3.25e-7, 3.5e-7])
ax.set_xticklabels(['300', '350', '400', '450', '500', '550', '600', '650', '700'])
ax.set_xlabel('Fiber diamter (nm)', fontsize=16)
ax.set_ylabel(r'$U$ (a.u.)', fontsize=16)
ax.legend(loc='upper right')
plt.show()
# # Parameters
# DList = 2 * np.linspace(150e-9, 350e-9, 1000)
# # Fiber taper parameters
# D0 = DList[0]
# L = 1000e-6 # Hot zone length in um

# # z dependence : z is in microns
# z = np.linspace(0, 1000e-6, 1000)
# Dz = D0 * np.exp(z / L)

# plt.figure()
# plt.plot(z, Dz)
# plt.show()