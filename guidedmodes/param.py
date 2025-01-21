import numpy as np
from scipy.constants import c, epsilon_0, mu_0


a = np.linspace(0, 500, 1000) # Fiber radius
n_silica = 1.45 # Index of silica
n_vacuum = 1.0 # Index of vacuum
lam = 785e-9 # Wavelength [m]
om = 2*np.pi*c / lam # Frequency
k = om / c # Wave number