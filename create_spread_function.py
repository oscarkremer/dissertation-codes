import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import trapezoid
from scipy.special import jv

na = 0.67
theta_max = np.arcsin(0.67)
focus = 3.1e-3

def apodization_function(theta, focus=3.1e-3, theta_max=np.arcsin(0.67), w_0=(4.2e-3)/2):
    f_0 = w_0/(focus*np.sin(theta_max))
    return np.exp(-np.power(np.sin(theta)/(f_0*np.sin(theta_max)),2))


def integral_i_00(rho, theta, z, focus=3.1e-3, theta_max=np.arcsin(0.67), wavenumber=2*np.pi/(1550e-9)):
    fw = apodization_function(theta, focus=focus, theta_max=theta_max)
    return fw*np.sqrt(np.cos(theta))*np.sin(theta)*(1+np.cos(theta))*jv(0, wavenumber*rho*np.sin(theta))*np.exp(1j*wavenumber*z*np.cos(theta))


def integral_i_01(rho, theta, z, focus=3.1e-3, theta_max=np.arcsin(0.67), wavenumber=2*np.pi/(1550e-9)):
    fw = apodization_function(theta, focus=focus, theta_max=theta_max)
    return fw*np.sqrt(np.cos(theta))*np.power(np.sin(theta),2)*jv(1, wavenumber*rho*np.sin(theta))*np.exp(1j*wavenumber*z*np.cos(theta))


def integral_i_02(rho, theta, z, focus=3.1e-3, theta_max=np.arcsin(0.67), wavenumber=2*np.pi/(1550e-9)):
    fw = apodization_function(theta, focus=focus, theta_max=theta_max)
    return fw*np.sqrt(np.cos(theta))*np.sin(theta)*(1-np.cos(theta))*jv(2, wavenumber*rho*np.sin(theta))*np.exp(1j*wavenumber*z*np.cos(theta))


def compute_electric_field(rho, phi, z, focus=3.1e-3, theta_max=np.arcsin(0.67), wavenumber=2*np.pi/(1550e-9)):
    thetas = np.linspace(0, theta_max, 1000)
    i_00 = trapezoid(integral_i_00(rho, thetas, z), thetas)
    i_01 = trapezoid(integral_i_01(rho, thetas, z), thetas)
    i_02 = trapezoid(integral_i_02(rho, thetas, z), thetas)
    E_amp = (1j*wavenumber*focus/2)*np.exp(-1j*wavenumber*focus)
    return E_amp*np.array([[i_00+i_02*np.cos(2*phi)],[i_02*np.sin(2*phi)],[-2j*i_01*np.cos(phi)]])

def compute_intensity_profile(xs, ys, z=0, power=250e-3, initial_waist=4.2e-3/2):
    xs, ys = np.meshgrid(xs, ys)
    rhos = np.sqrt(np.power(xs,2)+np.power(ys,2))
    phis = np.arctan2(ys, xs)

    E_0 = np.sqrt(2*power/(initial_waist*np.pi*8.85e-12*3e8))



    intensities = np.zeros(rhos.shape)
    for i in range(rhos.shape[0]):
        for j in range(rhos.shape[1]):
            module_e = E_0*np.abs(compute_electric_field(rhos[i,j], phis[i,j], z))
            intensity = np.power(module_e,2).sum()
            intensities[i, j] =intensity
    return xs, ys, intensities


def compute_intensity_gaussian_z():
    zs = np.linspace(-5e-6, 5e-6, 100)
    intensities = np.zeros(zs.shape).astype(np.complex_)
    for i in range(zs.shape[0]):
        module_e = np.abs(compute_electric_field(0, 0, zs[i]))
        intensity = np.power(module_e,2).sum()
        intensities[i] =intensity
    return zs, intensities

x_points = np.linspace(-2e-6, 2e-6, 100)
y_points = np.linspace(-2e-6, 2e-6, 100)

xs, ys, intensities = compute_intensity_profile(x_points, y_points)

from create_gaussian import gaussian_beam


def gauss_functions(x, power, w_0):
    I_0 = 2*power/(np.pi*(w_0**2))
    return I_0*np.exp(-2*np.power(x,2)/(w_0**2))

from scipy.optimize import curve_fit

params, _ = curve_fit(gauss_functions, x_points,  intensities[:, int(ys.shape[1]/2)], p0=[300e-3, 1.5e-6])
print(params)


x = np.linspace(-5e-6, 5e-6,1000)
plt.plot(np.linspace(-5e-6, 5e-6,1000), gaussian_beam(x, 0, 300e-3, w_0=2.4e-6/2))
plt.plot(x_points, intensities[:, int(ys.shape[1]/2)])
plt.plot(x_points, gauss_functions(x_points, *params))
plt.show()



x_points = np.linspace(-4e-6, 4e-6, 100)
y_points = np.linspace(-4e-6, 4e-6, 100)

xs, ys, intensities = compute_intensity_profile(x_points, y_points, z=1e-5)


params, _ = curve_fit(gauss_functions, x_points,  intensities[:, int(ys.shape[1]/2)], p0=[300e-3, 1.5e-6])
print(params)

#fig, ax = plt.subplots()
#fig.set_figwidth(6)
#p = ax.pcolor(xs, ys, intensities/intensities.max(), cmap=matplotlib.cm.viridis, vmin=0, vmax=1)
#plt.show()





'''
fig, ax = plt.subplots()
fig.set_figwidth(6)
x = np.linspace(-2.2e-3, 2.3e-3, 200)
X, Y = np.meshgrid(x, x.copy())
p = ax.pcolor(X, Y, gaussian_beam(X, Y, 300e-3), cmap=matplotlib.cm.viridis)
plt.show()
'''

