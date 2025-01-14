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

# HE11 mode (Linear pol)
modeH = fibermode.HE(
    wavelength=wavelength,
    radius=radius,
    core_index=core_index,
    clad_index=clad_index
    )

polH = 0

ExH = modeH.Ex(rho, phi, polH)
# Ey = HEmode.Ey(rho, phi, pol)
EyH = modeH.Ey(rho, phi, polH)
EzH = modeH.Ez(rho, phi, polH)


modeV = fibermode.HE(
    wavelength=wavelength,
    radius=radius,
    core_index=core_index,
    clad_index=clad_index
    )

polV = np.pi/2

ExV = modeV.Ex(rho, phi, polV)
# Ey = HEmode.Ey(rho, phi, pol)
EyV = modeV.Ey(rho, phi, polV)
EzV = modeV.Ez(rho, phi, polV)


Ex = (ExH + 1j*ExV) / np.sqrt(2)
Ey = (EyH + 1j*EyV) / np.sqrt(2)
Ez = (EzH + 1J*EzV) / np.sqrt(2)


x = rho * np.cos(phi)
y = rho * np.sin(phi)

# Fiber
fib_rho_values = np.linspace(radius-1e-9, radius, 1)
fib_phi_values = np.linspace(0, 2*np.pi, 1000)
fib_rho, fib_phi = np.meshgrid(fib_rho_values, fib_phi_values)
fib_x = fib_rho * np.cos(fib_phi)
fib_y = fib_rho * np.sin(fib_phi)

# Plot
fig = plt.figure(0, layout='tight')

ax01 = fig.add_subplot(2, 2, 1)
ax02 = fig.add_subplot(2, 2, 2)
ax03 = fig.add_subplot(2, 2, 3)
ax04 = fig.add_subplot(2, 2, 4)

ax01.set_title('|E|^2')
ax02.set_title('Ex^2')
ax03.set_title('Ey^2')
ax04.set_title('Ez^2')

ax01.set_aspect('equal')
ax02.set_aspect('equal')
ax03.set_aspect('equal')
ax04.set_aspect('equal')

ax01.pcolormesh(x, y, np.abs(ExH)**2 + np.abs(EyH)**2 + np.abs(EzH)**2)
ax01.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax01.set_xlim(-500e-9, 500e-9)
ax01.set_ylim(-500e-9, 500e-9)

ax02.pcolormesh(x, y, np.abs(ExH)**2)
ax02.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax02.set_xlim(-500e-9, 500e-9)
ax02.set_ylim(-500e-9, 500e-9)

ax03.pcolormesh(x, y, np.abs(EyH)**2)
ax03.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax03.set_xlim(-500e-9, 500e-9)
ax03.set_ylim(-500e-9, 500e-9)

ax04.pcolormesh(x, y, np.abs(EzH)**2)
ax04.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax04.set_xlim(-500e-9, 500e-9)
ax04.set_ylim(-500e-9, 500e-9)



# Plot
fig = plt.figure(1, layout='tight')

ax11 = fig.add_subplot(2, 2, 1)
ax12 = fig.add_subplot(2, 2, 2)
ax13 = fig.add_subplot(2, 2, 3)
ax14 = fig.add_subplot(2, 2, 4)

ax11.set_title('|E|^2')
ax12.set_title('Ex^2')
ax13.set_title('Ey^2')
ax14.set_title('Ez^2')

ax11.set_aspect('equal')
ax12.set_aspect('equal')
ax13.set_aspect('equal')
ax14.set_aspect('equal')

ax11.pcolormesh(x, y, np.abs(ExV)**2 + np.abs(EyV)**2 + np.abs(EzV)**2)
ax11.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax11.set_xlim(-500e-9, 500e-9)
ax11.set_ylim(-500e-9, 500e-9)

ax12.pcolormesh(x, y, np.abs(ExV)**2)
ax12.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax12.set_xlim(-500e-9, 500e-9)
ax12.set_ylim(-500e-9, 500e-9)

ax13.pcolormesh(x, y, np.abs(EyV)**2)
ax13.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax13.set_xlim(-500e-9, 500e-9)
ax13.set_ylim(-500e-9, 500e-9)

ax14.pcolormesh(x, y, np.abs(EzV)**2)
ax14.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax14.set_xlim(-500e-9, 500e-9)
ax14.set_ylim(-500e-9, 500e-9)


# Plot
fig = plt.figure(2, layout='tight')

ax21 = fig.add_subplot(2, 2, 1)
ax22 = fig.add_subplot(2, 2, 2)
ax23 = fig.add_subplot(2, 2, 3)
ax24 = fig.add_subplot(2, 2, 4)

ax21.set_title('|E|^2')
ax22.set_title('Ex^2')
ax23.set_title('Ey^2')
ax24.set_title('Ez^2')

ax21.set_aspect('equal')
ax22.set_aspect('equal')
ax23.set_aspect('equal')
ax24.set_aspect('equal')

ax21.pcolormesh(x, y, np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
ax21.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax21.set_xlim(-500e-9, 500e-9)
ax21.set_ylim(-500e-9, 500e-9)

ax22.pcolormesh(x, y, np.abs(Ex)**2)
ax22.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax22.set_xlim(-500e-9, 500e-9)
ax22.set_ylim(-500e-9, 500e-9)

ax23.pcolormesh(x, y, np.abs(Ey)**2)
ax23.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax23.set_xlim(-500e-9, 500e-9)
ax23.set_ylim(-500e-9, 500e-9)

ax24.pcolormesh(x, y, np.abs(Ez)**2)
ax24.scatter(fib_x, fib_y, marker='.', s=0.1, c='white')
ax24.set_xlim(-500e-9, 500e-9)
ax24.set_ylim(-500e-9, 500e-9)

plt.show()
