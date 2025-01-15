import numpy as np
import matplotlib.pyplot as plt
import fibermode


def main(radius, a, core_index, clad_index, P660, P785):
    # Polar coordinates(global)
    global rho
    rho = radius + a/2
    global phi
    phi = np.linspace(0, 2*np.pi, 1000)
    global phi_deg
    phi_deg = phi * 180/np.pi  #[deg]

    # Circ. pol.
    Ex660, Ey660, Ez660 = circmode(wavelength=660e-9,
                                   radius=radius,
                                   core_index=core_index,
                                   clad_index=clad_index,
                                   power=P660)
    I660 = np.abs(Ex660)**2 + np.abs(Ey660)**2 + np.abs(Ez660)**2

    #Lin. pol.
    Ex785, Ey785, Ez785 = linearmode(wavelength=785e-9,
                                     radius=radius,
                                     core_index=core_index,
                                     clad_index=clad_index,
                                     power=P785,
                                     pol=0)
    I785 = np.abs(Ex785)**2 + np.abs(Ey785)**2 + np.abs(Ez785)**2
    plot(I660=I660, Ex660=Ex660, Ey660=Ey660, Ez660=Ez660, P660=P660, 
         I785=I785, Ex785=Ex785, Ey785=Ey785, Ez785=Ez785, P785=P785,
         radius=radius, a=a)
    # intersection(I660, I785)
    

def linearmode(wavelength, radius, core_index, clad_index, power, pol):
    mode = fibermode.HE(wavelength=wavelength,
                        radius=radius,
                        core_index=core_index,
                        clad_index=clad_index,
                        power=power)
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

# def intersection(y1, y2):
#     idx = np.argwhere(np.isclose(y1, y2))
#     print(idx)

def plot(I660, Ex660, Ey660, Ez660, P660, I785, Ex785, Ey785, Ez785, P785, radius, a):
    fig = plt.figure(layout='constrained')

    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)

    ax1.plot(phi_deg, I660, color='red', label='660nm, {}mW, Circ.'.format(P660*1e3))
    ax1.plot(phi_deg, I785, color='black', label='785nm, {}mW, Lin.'.format(P785*1e3))
    # ax1.set_xlabel(r'$\phi[\text{deg}]$')
    ax1.set_ylabel(r'I')
    ax1.legend(loc='right')
    ax1.set_title('r={}nm, a={}nm'.format(radius*1e9, a*1e9))


    ax2.plot(phi_deg, np.abs(Ex660)**2, color='red', label='660nm, {}mW, Circ.'.format(P660*1e3))
    ax2.plot(phi_deg, np.abs(Ex785)**2, color='black', label='785nm, {}mW, Lin.'.format(P785*1e3))
    # ax2.set_xlabel(r'$\phi[\text{deg}]$')
    ax2.set_ylabel(r'$|E_{x}|^{2}$')
    ax2.legend(loc='right')

    ax3.plot(phi_deg, np.abs(Ey660)**2, color='red', label='660nm, {}mW, Circ.'.format(P660*1e3))
    ax3.plot(phi_deg, np.abs(Ey785)**2, color='black', label='785nm, {}mW, Lin.'.format(P785*1e3))
    # ax3.set_xlabel(r'$\phi[\text{deg}]$')
    ax3.set_ylabel(r'$|E_{y}|^{2}$')
    ax3.legend(loc='right')

    ax4.plot(phi_deg, np.abs(Ez660)**2, color='red', label='660nm, {}mW, Circ.'.format(P660*1e3))
    ax4.plot(phi_deg, np.abs(Ez785)**2, color='black', label='785nm, {}mW, Lin.'.format(P785*1e3))
    ax4.set_xlabel(r'$\phi[\text{deg}]$')
    ax4.set_ylabel(r'$|E_{z}|^{2}$')
    ax4.legend(loc='right')

    plt.show()


if __name__ == '__main__':
    # Parameters
    radius = 200e-9 # Fiber Radius [m]
    a = 200e-9 # Particle size [m]
    core_index = 1.45
    clad_index = 1.0
    P660 = 4.0e-3
    P785 = 4.0e-3
    main(radius=radius,
         a=a,
         core_index=core_index,
         clad_index=clad_index,
         P660=P660,
         P785=P785)
