import numpy as np
import matplotlib.pyplot as plt
import fibermode


# Parameters
wavelength = 780e-9
radius = 250e-9
core_index = 1.45
clad_index = 1.0

rho_values = np.linspace(0, radius+800e-9, 1000)
phi_values = np.linspace(0, 2*np.pi, 1000)
rho, phi = np.meshgrid(rho_values, phi_values)

pol = np.pi / 2

# HE11 mode (Linear pol)
HEmode = fibermode.HE(
    wavelength=wavelength,
    radius=radius,
    core_index=core_index,
    clad_index=clad_index,
    azimuthal_order=1,
    radial_order=0
    )

Ex = HEmode.Ex(rho, phi, pol)
Ey = HEmode.Ey(rho, phi, pol)
Ez = HEmode.Ez(rho, phi, pol)

x = rho * np.cos(phi)
y = rho * np.sin(phi)

# Fiber
fib_rho_values = np.linspace(radius-1e-9, radius, 1)
fib_phi_values = np.linspace(0, 2*np.pi, 1000)
fib_rho, fib_phi = np.meshgrid(fib_rho_values, fib_phi_values)
fib_x = fib_rho * np.cos(fib_phi)
fib_y = fib_rho * np.sin(fib_phi)

# Plot
fig = plt.figure(layout='tight')

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

ax1.set_title('|E|^2')
ax2.set_title('Ex^2')
ax3.set_title('Ey^2')
ax4.set_title('Ez^2')

ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')
ax4.set_aspect('equal')

ax1.pcolormesh(x, y, np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
ax1.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax1.set_xlim(-500e-9, 500e-9)
ax1.set_ylim(-500e-9, 500e-9)

ax2.pcolormesh(x, y, np.abs(Ex)**2)
ax2.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax2.set_xlim(-500e-9, 500e-9)
ax2.set_ylim(-500e-9, 500e-9)

ax3.pcolormesh(x, y, np.abs(Ey)**2)
ax3.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax3.set_xlim(-500e-9, 500e-9)
ax3.set_ylim(-500e-9, 500e-9)

ax4.pcolormesh(x, y, np.abs(Ez)**2)
ax4.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax4.set_xlim(-500e-9, 500e-9)
ax4.set_ylim(-500e-9, 500e-9)

plt.show()
