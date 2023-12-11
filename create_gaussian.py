import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def gaussian_beam(x, y, power, w_0=2.2e-3, wavelength=1550e-9):
    k = 2*np.pi/wavelength
    I_0 = 2*power/(np.pi*w_0**2)
    return I_0*np.exp(-2*(np.power(x,2)+np.power(y,2))/(w_0**2))

def wavefront_radius(z, z_0):
    return z*(1+np.power(z_0/z, 2))

def beam_radius(z, z_0, w_0):
    return w_0*np.sqrt(1+np.power(z/z_0, 2))

def phase_correct(z, z_0):
    return np.arctan(z/z_0)


fig, ax = plt.subplots()
fig.set_figwidth(6)
x = np.linspace(-2.2e-3, 2.3e-3, 200)
X, Y = np.meshgrid(x, x.copy())
p = ax.pcolor(X, Y, gaussian_beam(X, Y, 300e-3), cmap=matplotlib.cm.viridis)
plt.show()

x = np.linspace(-10e-6, 10e-6,1000)
plt.plot(np.linspace(-10e-6, 10e-6,1000), gaussian_beam(x, 0, 300e-3, w_0=2.2e-6/2))
plt.show()