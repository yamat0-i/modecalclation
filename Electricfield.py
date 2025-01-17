import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import fibermode


# Parameters
wavelength = 780e-9
radius = 250e-9
core_index = 1.45
clad_index = 1.0

# Polar coordinates
rho_values = np.linspace(0, radius+800e-9, 1000)
phi_values = np.linspace(-np.pi, 0, 1000)
rho, phi = np.meshgrid(rho_values, phi_values)

# Cartesian coordinates
x = rho * np.cos(phi)
y = rho * np.sin(phi)

pol = np.pi / 2

# HE11 mode (Linear pol)
HEmode = fibermode.HE(
    wavelength=wavelength,
    radius=radius,
    core_index=core_index,
    clad_index=clad_index,
    azimuthal_order=1,
    radial_order=1,
    power=10e-3
    )

Ex = HEmode.Ex(rho, phi, pol)
Ey = HEmode.Ey(rho, phi, pol)
Ez = HEmode.Ez(rho, phi, pol)
I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

# Fiber surface
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

ax1.set_title(r'$|\boldsymbol{E}|^2$')
ax2.set_title(r'$|E_{x}|^2$')
ax3.set_title(r'$|E_{y}|^2$')
ax4.set_title(r'$|E_{z}|^2$')

ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')
ax4.set_aspect('equal')

c1= ax1.pcolormesh(x, y, I)
ax1.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax1.set_xlim(-500e-9, 500e-9)
ax1.set_xticks([-500e-9, -250e-9, 0, 250e-9, 500e-9])
ax1.set_xticklabels(['-500', '-250', '0', '250', '500'])
ax1.set_ylim(-500e-9, 500e-9)
ax1.set_yticks([-500e-9, -250e-9, 0, 250e-9, 500e-9])
ax1.set_yticklabels(['-500', '-250', '0', '250', '500'])
ax1.set_ylabel(r'$y \text{[nm]}$')

ax2.pcolormesh(x, y, np.abs(Ex)**2)
ax2.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax2.set_xlim(-500e-9, 500e-9)
ax2.set_xticks([-500e-9, -250e-9, 0, 250e-9, 500e-9])
ax2.set_xticklabels(['-500', '-250', '0', '250', '500'])
ax2.set_ylim(-500e-9, 500e-9)
ax2.set_yticks([-500e-9, -250e-9, 0, 250e-9, 500e-9])
ax2.set_yticklabels(['-500', '-250', '0', '250', '500'])

ax3.pcolormesh(x, y, np.abs(Ey)**2)
ax3.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax3.set_xlim(-500e-9, 500e-9)
ax3.set_xticks([-500e-9, -250e-9, 0, 250e-9, 500e-9])
ax3.set_xticklabels(['-500', '-250', '0', '250', '500'])
ax3.set_ylim(-500e-9, 500e-9)
ax3.set_yticks([-500e-9, -250e-9, 0, 250e-9, 500e-9])
ax3.set_yticklabels(['-500', '-250', '0', '250', '500'])
ax3.set_xlabel(r'$x \text{[nm]}$')
ax3.set_ylabel(r'$y \text{[nm]}$')

ax4.pcolormesh(x, y, np.abs(Ez)**2)
ax4.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax4.set_xlim(-500e-9, 500e-9)
ax4.set_xticks([-500e-9, -250e-9, 0, 250e-9, 500e-9])
ax4.set_xticklabels(['-500', '-250', '0', '250', '500'])
ax4.set_ylim(-500e-9, 500e-9)
ax4.set_yticks([-500e-9, -250e-9, 0, 250e-9, 500e-9])
ax4.set_yticklabels(['-500', '-250', '0', '250', '500'])
ax4.set_xlabel(r'$x \text{[nm]}$')

cax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
cbar = plt.colorbar(c1, cax)
cbar.set_ticks([np.nanmin(I), 0.2*np.nanmax(I), 0.4*np.nanmax(I), 0.6*np.nanmax(I), 0.8*np.nanmax(I), np.nanmax(I)])
cbar.set_ticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.show()
