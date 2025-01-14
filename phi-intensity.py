import numpy as np
import matplotlib.pyplot as plt
import fibermode


# Parameters
radius = 200e-9 # Fiber Radius [m]
a = 200e-9 # Particle size [m]

# Polar coordinates
rho = radius + a/2
phi = np.linspace(0, 2*np.pi, 1000)
phi_deg = phi * 180/np.pi  #[deg]

# Circ. pol.
mode660 = fibermode.HE(
    wavelength=660e-9,
    radius=radius,
    core_index=1.45,
    clad_index=1.0,
    power=4.0e-3
    )

Ex660H = mode660.Ex(rho, phi, pol=0)
Ey660H = mode660.Ey(rho, phi, pol=0)
Ez660H = mode660.Ez(rho, phi, pol=0)

Ex660V = mode660.Ex(rho, phi, pol=np.pi/2)
Ey660V = mode660.Ey(rho, phi, pol=np.pi/2)
Ez660V = mode660.Ez(rho, phi, pol=np.pi/2)

Ex660 = (Ex660H + 1j*Ex660V) / np.sqrt(2)
Ey660 = (Ey660H + 1j*Ey660V) / np.sqrt(2)
Ez660 = (Ez660H + 1J*Ez660V) / np.sqrt(2)

I660 = np.abs(Ex660)**2 + np.abs(Ey660)**2 + np.abs(Ez660)**2

# Lin. pol.
mode785 = fibermode.HE(
    wavelength=785e-9,
    radius=radius,
    core_index=1.45,
    clad_index=1.0,
    power=3.0e-3
)

Ex785 = mode785.Ex(rho, phi, pol=0)
Ey785 = mode785.Ey(rho, phi, pol=0)
Ez785 = mode785.Ez(rho, phi, pol=0)

I785 = np.abs(Ex785)**2 + np.abs(Ey785)**2 + np.abs(Ez785)**2

plt.figure()
plt.plot(phi_deg, I660)
plt.plot(phi_deg, I785)

plt.xlabel(r'$\phi[\text{deg}]$')
plt.ylabel(r'$|E_{x}|^{2} + |E_{y}|^{2} + |E_{z}|^{2}$')

plt.show()
