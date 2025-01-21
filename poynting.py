import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0
from scipy.special import jv, kv
import fibermode


core_index = 1.45
clad_index = 1.0
radius_list = np.linspace(150e-9, 350e-9, 1000)
P660 = 10e-3
P785 = 10e-3

def PoyntingVector_in(omega, n1, beta, s, s1, l, h, rho, phi, pol):
    coef = (omega * epsilon_0 *n1**2 * beta) / (4 * h**2)
    term1 = (1 - s) * (1 - s1) * jv(l-1, h*rho)**2
    term2 = (1 + s) * (1 + s1) * jv(l+1, h*rho)**2
    term3 = -2 * (1 - s * s1) * jv(l-1, h*rho) * jv(l+1, h*rho) * np.cos(2*(l*phi - pol))
    return coef * (term1 + term2 + term3)

def PoyntingVector_out(A, omega, n2, beta, s, s2, l, radius, h, q, rho, phi, pol):
    coef1 = np.abs(A)**2 * (omega * epsilon_0 *n2**2 * beta) / (4 * q**2)
    coef2 = jv(l, h*radius)**2 / kv(l, q*radius)**2
    term1 = (1 - s) * (1 - s2) * kv(l-1, q*rho)**2
    term2 = (1 + s) * (1 + s2) * kv(l+1, q*rho)**2
    term3 = 2 * (1 - s * s2) * kv(l-1, q*rho) * kv(l+1, q*rho) * np.cos(2*(l*phi - pol))
    return coef1 * coef2 * (term1 + term2 + term3)

def PoyntingVector_out_Circ(A, omega, n2, beta, s, s2, l, radius, h, q, rho, phi, pol):
    coef1 = np.abs(A)**2 * (omega * epsilon_0 *n2**2 * beta) / (4 * q**2)
    coef2 = jv(l, h*radius)**2 / kv(l, q*radius)**2
    term1 = (1 - s) * (1 - s2) * kv(l-1, q*rho)**2
    term2 = (1 + s) * (1 + s2) * kv(l+1, q*rho)**2
    return coef1 * coef2 * (term1 + term2) /np.sqrt(2)

Sz660_lin = np.zeros(np.shape(radius_list))
for i in np.arange(len(radius_list)):
    mode660 = fibermode.HE(wavelength=660e-9,
                           radius=radius_list[i],
                           core_index=core_index,
                           clad_index=clad_index,
                           power=P660
                           )
    A = mode660.A
    omega = mode660.omega
    n2=clad_index
    beta = mode660.beta
    s = mode660.s
    s2 = mode660.s2
    l = mode660.l
    radius = radius_list[i]
    h = mode660.h
    q = mode660.q
    rho = radius + 1e-9
    phi = np.pi/2
    pol = np.pi/2

    Sz660_lin[i] = PoyntingVector_out(A=A,
                                  omega=omega,
                                  n2=n2,
                                  beta=beta,
                                  s=s,
                                  s2=s2,
                                  l=1,
                                  radius=radius,
                                  h=h,
                                  q=q,
                                  rho=rho,
                                  phi=phi,
                                  pol=pol)
    
Sz660_lin_norm = [(Sz660_lin[i]-min(Sz660_lin))/(max(Sz660_lin)-min(Sz660_lin)) for i in range(len(Sz660_lin))]

Sz660_circ = np.zeros(np.shape(radius_list))
for i in np.arange(len(radius_list)):
    mode660_circ = fibermode.HE(wavelength=660e-9,
                                radius=radius_list[i],
                                core_index=core_index,
                                clad_index=clad_index,
                                power=P660
                                )
    A = mode660_circ.A
    omega = mode660_circ.omega
    n2=clad_index
    beta = mode660_circ.beta
    s = mode660_circ.s
    s2 = mode660_circ.s2
    l = mode660_circ.l
    radius = radius_list[i]
    h = mode660_circ.h
    q = mode660_circ.q
    rho = radius + 1e-9
    phi = np.pi/2
    pol = np.pi/2

    Sz660_circ[i] = PoyntingVector_out_Circ(A=A,
                                  omega=omega,
                                  n2=n2,
                                  beta=beta,
                                  s=s,
                                  s2=s2,
                                  l=1,
                                  radius=radius,
                                  h=h,
                                  q=q,
                                  rho=rho,
                                  phi=phi,
                                  pol=pol)
    
Sz660_circ_norm = [(Sz660_circ[i]-min(Sz660_lin))/(max(Sz660_lin)-min(Sz660_lin)) for i in range(len(Sz660_lin))]

Sz785 = np.zeros(np.shape(radius_list))
for i in np.arange(len(radius_list)):
    mode785 = fibermode.HE(wavelength=785e-9,
                           radius=radius_list[i],
                           core_index=core_index,
                           clad_index=clad_index,
                           power=P785
                           )
    A = mode785.A
    omega = mode785.omega
    n2=clad_index
    beta = mode785.beta
    s = mode785.s
    s2 = mode785.s2
    l = mode785.l
    radius = radius_list[i]
    h = mode785.h
    q = mode785.q
    rho = radius + 1e-9
    phi = np.pi/2
    pol = np.pi/2

    Sz785[i] = PoyntingVector_out(A=A,
                                  omega=omega,
                                  n2=n2,
                                  beta=beta,
                                  s=s,
                                  s2=s2,
                                  l=1,
                                  radius=radius,
                                  h=h,
                                  q=q,
                                  rho=rho,
                                  phi=phi,
                                  pol=pol)
    
Sz785_norm = [(Sz785[i]-min(Sz660_lin))/(max(Sz660_lin)-min(Sz660_lin)) for i in range(len(Sz660_lin))]

fig = plt.figure(layout='tight')
ax = fig.add_subplot(111)
ax.plot(radius_list, Sz660_lin_norm, color='red', label='660nm, Linear pol.')
ax.plot(radius_list, Sz660_circ_norm, color='red', linestyle='--', label='660nm, Circular pol.')
ax.plot(radius_list, Sz785_norm, color='green', label='785nm, Linear pol.')
ax.set_xticks([1.5e-7, 1.75e-7, 2.0e-7, 2.25e-7, 2.5e-7, 2.75e-7, 3.0e-7, 3.25e-7, 3.5e-7])
ax.set_xticklabels(['300', '350', '400', '450', '500', '550', '600', '650', '700'])
ax.set_xlabel('Fiber diamter (nm)', fontsize=16)
ax.set_ylabel(r'$\Pi_{z}$ (a.u.)', fontsize=16)
ax.legend()
plt.show()
            