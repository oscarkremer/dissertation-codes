import numpy as np

R = (293e-9)/2
epsilon_0 = 8.85e-12
refraction_index = 1.55
wavelength = 1550e-9
wavenumber = 2*np.pi/wavelength
c = 2.99e8
epsilon = refraction_index**2
rho = 2200
power = 200e-3
volume = 4 * np.pi * R**3 / 3
m = rho * volume
epsilon_ratio = (epsilon-1)/(epsilon+2)
alpha_CM = 3 * volume * epsilon_0 * epsilon_ratio
alpha = alpha_CM/(1-epsilon_ratio*((wavenumber*R)**2+2j/3*(wavenumber*R)**3))
alpha = np.real(alpha)
print(alpha, alpha_CM)
#power = 1.77210629e-01 
waist = 1.049e-06
#waist = 1.00412525e-06
zR = 2 * np.pi * waist**2 / (2 * wavelength)
print(zR)
zR=2.81e-6
print(zR)
omega_z = np.sqrt(
    (alpha * power) / ((waist * zR) ** 2 * np.pi * m * epsilon_0 * c)
)
print(omega_z/(2*np.pi))


import numpy as np

power = 300e-3
#power = 1.77210629e-01 
waist = 1.049e-06
#waist = 1.00412525e-06
#zR = 2 * np.pi * waist**2 / (2 * wavelength)

#zR=2.81e-6
print(zR)
omega_z = np.sqrt(
    (alpha * power) / ((waist * zR) ** 2 * np.pi * m * epsilon_0 * c)
)
print(omega_z/(2*np.pi))


omega_z = np.sqrt(12*(refraction_index**2-1)*power/(rho*c*waist**4*np.pi*(refraction_index**2+2)))
print(omega_z/(2*np.pi))