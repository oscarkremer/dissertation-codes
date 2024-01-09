#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:28:54 2023

@author: thiagoguerreiro
"""

import numpy as np 
from numpy import *
import matplotlib.pyplot as plt

# All units SI

# Fundamental constants
epsilon0 = 8.854 * 1e-12 # N * m2 * C−2
c = 3 * 1e8 # m/s
hbar = 1e-34 # J * s

# Wavelength of laser
l = 1.55 * 1e-6 # m

# Wavenumber
k = 2 * np.pi / l

# Cavity length
L = 3 * 1e-2 # 1.07 * 1e-2 # m 

# Refractive index of silica
n = np.sqrt(2.07) # See table in https://arxiv.org/pdf/1902.01282.pdf

# Radius of nanoparticle
R = 70 * 1e-9 # m 

# Density of Silica 

rho = 2200 # kg * m-3

# Volume of nanoparticle
V = (4/3) * np.pi * (R**3)

# Mass of nanoparticle -- Typical order of magnitude is femtograms -- from Brandao
m =  V * rho 

# Tweezer power 
P = 0.17 # W

# Tweezer waist -- Aspelmeyer arXiv:2305.16226v1
wx =  0.67 * 1e-6 # m
wy = 0.77 * 1e-6 # m 

# Cavity frequency
omega_c = 2 * np.pi * c / l

gamma = 2*np.pi*193 * 1e3 # Cavity linewidth in Hz


# Cavity waist
w_c = np.sqrt(c*L / omega_c)

# Cavity mode volume 
Vc = ((w_c)**2) * np.pi * L / 4


# Tweezer spring frequency -- arXiv:2305.16226v1
omega_m =  (2*np.pi) * 190 * 1e3 # Hz 

# Zero point motion
q0 = np.sqrt(hbar / (2*m*omega_m))

# Polarizability
alpha = 3 * epsilon0 * V * ((n**2 - 1)/(n**2 + 2))

# Tweezer field
E0 = np.sqrt(4 * P / (np.pi * wx * wy * epsilon0 * c))

# Cavity field
Ec = np.sqrt( hbar * omega_c / (2* epsilon0 * Vc) )

# Coupling constant formula
g = (1 / hbar) * alpha * E0 * Ec * k * q0

# Relative coupling
varepsilon = g/omega_m

# Coupling proportionality constant -- ''a'' in the paper, see Eq. (86)
g0 = (omega_c**2)/g

print(varepsilon)

#print(g0)


# Cross check on g0 -- ignore this if you trust the numbers

#g_alternative = (6 * V * ((n**2 - 1)/(n**2 + 2)) / np.pi) * np.sqrt( P / (wx * wy * m * omega_m) ) * (1/(L * (c**2)))

#print(1/g_alternative)
#print(g_alternative * (omega_c)**2 / omega_m)

#g0 = (  ((alpha * E0)/(np.sqrt(np.pi * epsilon0))) * ( 1 / (np.sqrt(2 * (c**3) * (L**2) * m * omega_m  )))  )**(-1)

#g = omega_c **2 / g0


###################### Particle-particle interactions ######################


###################### Coulomb ######################

# Charge in elementary charges
N = 250

# Charge (in Coulombs)
Q = N * 1.6 * 1e-19

# Particle separation

d_coulomb = 2 * 1e-6 # in meters

# Mechanical frequency shift due to coulomb interaction -- result in Hz 

delta_omega = Q**2 / (4 * np.pi * epsilon0 * m * d_coulomb**3) 

# Modified mechanical frequencies

omega_a = np.sqrt(((2*np.pi) * 190 * 1e3)**2 - delta_omega) # Hz 

omega_b = np.sqrt(((2*np.pi) * 180 * 1e3)**2 - delta_omega) # Hz 

# Zero point fluctuations

q0_a = np.sqrt(hbar / (2*m*omega_a))

q0_b = np.sqrt(hbar / (2*m*omega_b))

# Coupling constant --Hz NOT divided by 2pi

g_coulomb = ( (Q**2 / (4 * np.pi * epsilon0 * hbar * d_coulomb**3)) * (q0_a * q0_b) ) 

###################### Optical binding ######################



###################### Cavity mediated ######################



###################### Typical force and standard deviation values ######################


kappa = omega_b / omega_a

f0 = hbar * g_coulomb / q0_b

# Typical value of the quantum-induced variance for the ground state
ell = (1/(1-kappa**2)) * (f0 / (m*omega_a*omega_b))


###################### Plots ######################

kappa = omega_b / omega_a

n_bar = 10

def h(t):
    h = 1 + (np.cos(t))**2 - 2 * np.cos(t) * np.cos(kappa * t) - 2 * kappa * np.sin(t) * np.sin(kappa * t) + (kappa**2) * (np.sin(kappa*t))**2
    return np.sqrt((2*n_bar + 1) + ((ell/q0_b)**2) * h )


sigma_0 = np.sqrt((2*n_bar + 1))

time = np.linspace(0,120,2000)

plt.rcParams.update({'font.size': 12})
plt.rcParams["axes.linewidth"] = 1.5


# plt.figure(figsize=(4.5, 2.5))

# plt.plot(time, h(time), color='C0', lw = 2)
# plt.axhline(y=sigma_0, color='C3', linestyle='--', alpha=0.6)
# plt.ylabel('$ \sigma_{X} \ [q_{0,b}] $')
# plt.xlabel(r'$ \Omega_{a} t $')
# plt.grid(alpha = 0.4)
# plt.tight_layout()


fig, ax1 = plt.subplots(figsize=(5, 3))

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.6, 0.63, 0.35, (4/5)*0.35]
ax2 = fig.add_axes([left, bottom, width, height])

ax1.plot(time, h(time), color='C0', lw = 2)
ax1.axhline(y=sigma_0, color='C3', linestyle='--', alpha=0.6, lw = 2)
#ax1.set_xlim(0,  max(time))
ax1.grid(alpha = 0.4)

ax1.text(0.3, 8.8, '(a)')
ax2.text(0.0, 4.8, '(b)')




ax2.plot(time[0:150], h(time[0:150]), color='C0', lw = 2)
ax2.axhline(y=sigma_0, color='C3', linestyle='--', alpha=0.6, lw = 2)

ax2.set_ylim(round(sigma_0,1) - 0.5,  round(sigma_0,1) + 0.5)

ax2.set_yticks([round(sigma_0,1) - 0.5, round(sigma_0,1), round(sigma_0,1) + 0.5])
ax2.set_xticks([0, round(time[150]/2,1), round(time[150],1)])
#ax2.set_yticks([])
#ax2.set_xticks([])


ax1.set_ylabel('$ \sigma_{\mathbf{q}_{b}} \ [q_{0,b}] $')
ax1.set_xlabel(r'$ \Omega_{a} t $')



plt.tight_layout()
plt.show()
