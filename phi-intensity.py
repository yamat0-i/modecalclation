import numpy as np
import matplotlib.pyplot as plt
import fibermode


# Parameters
radius = 200e-9 # Fiber Radius [m]
a = 200e-9 # Particle size [m]
core_index = 1.45
clad_index = 1.0
P660 = 4.0e-3
P785 = 3.0e-3

# Polar coordinates(global)
rho = radius + a/2
phi = np.linspace(0, 2*np.pi, 1000)
phi_deg = phi * 180/np.pi  #[deg]

def main():
    # Circ. pol.
    Ex660, Ey660, Ez660 = circmode(wavelnegth=660e-9,
                                   radius=radius,
                                   core_index=core_index,
                                   clad_index=clad_index,
                                   power=P660)
    I660 = np.abs(Ex660)**2 + np.abs(Ey660)**2 + np.abs(Ez660)**2
    #Lin. pol.
    Ex785, Ey785, Ez785 = linearmode(wavelnegth=785e-9,
                                     radius=radius,
                                     core_index=core_index,
                                     clad_index=clad_index,
                                     power=P785,
                                     pol=0)
    I785 = np.abs(Ex785)**2 + np.abs(Ey785)**2 + np.abs(Ez785)**2
    plot(I660=I660, I785=I785)
    
def linearmode(wavelnegth, radius, core_index, clad_index, power, pol):
    mode = fibermode.HE(wavelength=wavelnegth,
                        radius=radius,
                        core_index=core_index,
                        clad_index=clad_index,
                        power=power)
    Ex = mode.Ex(rho=rho, phi=phi, pol=pol)
    Ey = mode.Ey(rho=rho, phi=phi, pol=pol)
    Ez = mode.Ez(rho=rho, phi=phi, pol=pol)
    return Ex, Ey, Ez

def circmode(wavelnegth, radius, core_index, clad_index, power):
    ExH, EyH, EzH = linearmode(wavelnegth=wavelnegth,
                     radius=radius,
                     core_index=core_index,
                     clad_index=clad_index,
                     power=power,
                     pol=0)
    ExV, EyV, EzV = linearmode(wavelnegth=wavelnegth,
                     radius=radius,
                     core_index=core_index,
                     clad_index=clad_index,
                     power=power,
                     pol=np.pi/2)
    
    Ex = (ExH + 1j*ExV) / np.sqrt(2)
    Ey = (EyH + 1j*EyV) / np.sqrt(2)
    Ez = (EzH + 1J*EzV) / np.sqrt(2)
    return Ex, Ey, Ez
    
def plot(I660, I785):
    plt.figure()
    plt.plot(phi_deg, I660, color='black', label='660nm, {}mW, Circ.'.format(P660*1e3))
    plt.plot(phi_deg, I785, color='red', label='785nm, {}mW, Lin.'.format(P785*1e3))
    plt.xlabel(r'$\phi[\text{deg}]$')
    plt.ylabel(r'$|E_{x}|^{2} + |E_{y}|^{2} + |E_{z}|^{2}$')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()
