import numpy as np
import dissertation_codes.utils.constants as ct


class NParticle3D:
    def __init__(self, omegas, gamma, B, xsis, radius=147e-9, rho=2200, T=293):
        self.__omega_x__ = omegas[0]
        self.__omega_y__ = omegas[1]
        self.__omega_z__ = omegas[2]
        self.__xsi_x__ = xsis[0]
        self.__xsi_y__ = xsis[1]
        self.__xsi_z__ = xsis[2]
        self.__gamma__ = gamma
        self.T = T
        self.A = np.array([[0, 0, 0, self.__omega_x__, 0, 0],
                           [0, 0, 0, 0, self.__omega_y__, 0],
                           [0, 0, 0, 0, 0, self.__omega_z__],
                           [-self.__omega_x__, 0, 0, -self.__gamma__, 0, 0],
                           [0, -self.__omega_y__, 0, 0, -self.__gamma__, 0],
                           [0, 0, -self.__omega_z__, 0, 0, -self.__gamma__]])
        self.B = B
        self.C = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])
        self.G = np.array([[0], [0], [0], [1], [1], [1]])
        self._m_ = rho * 4 * np.pi * np.power(radius, 3) / 3
        self.zp_x = np.sqrt(ct.hbar / (2 * omegas[0] * self._m_))
        self.zp_px = np.sqrt(omegas[0] * ct.hbar * self._m_ / 2)
        self.zp_y = np.sqrt(ct.hbar / (2 * omegas[1] * self._m_))
        self.zp_py = np.sqrt(omegas[1] * ct.hbar * self._m_ / 2)
        self.zp_z = np.sqrt(ct.hbar / (2 * omegas[2] * self._m_))
        self.zp_pz = np.sqrt(omegas[2] * ct.hbar * self._m_ / 2)
        self.G_norm = np.array([[0],
                                [0],
                                [0],
                                [1/self.zp_px],
                                [1/self.zp_py],
                                [1/self.zp_pz]])

        self.thermal_force_std_x = (
            np.sqrt(2 * self.__gamma__ * self._m_ * ct.kb * T)
        )/self.zp_px
        self.thermal_force_std_y = (
            np.sqrt(2 * self.__gamma__ * self._m_ * ct.kb * T)
        )/self.zp_py
        self.thermal_force_std_z = (
            np.sqrt(2 * self.__gamma__ * self._m_ * ct.kb * T)
        )/self.zp_pz

    def step(self, states, control=0.0, delta_t=50e-2, bypass_noise=False):
        thermal_force_x = self.thermal_force_std_x * np.random.normal()
        thermal_force_y = self.thermal_force_std_y * np.random.normal()
        thermal_force_z = self.thermal_force_std_z * np.random.normal()
        if bypass_noise:
            thermal_force = np.zeros((6, 1))
        else:
            thermal_force = np.array([[0],
                                      [0],
                                      [0],
                                      [thermal_force_x],
                                      [thermal_force_y],
                                      [thermal_force_z]])
        xsi_array = np.array([[0],
                              [0],
                              [0],
                              [self.__xsi_x__*(states[0, 0]*self.zp_x)**2],
                              [self.__xsi_y__*(states[1, 0]*self.zp_y)**2],
                              [self.__xsi_z__*(states[2, 0]*self.zp_z)**2]])
        aux_omega_array = np.array([[0],
                                    [0],
                                    [0],
                                    [states[0, 0]*self.zp_x*(self.__omega_x__**2)],
                                    [states[1, 0]*self.zp_y*(self.__omega_y__**2)],
                                    [states[2, 0]*self.zp_z*(self.__omega_z__**2)]])
        state_dot = np.matmul(self.A, states) + self.B @ control + self.G_norm*(self._m_*xsi_array@aux_omega_array.T).T@self.G
        states = states + state_dot * delta_t + self.G * np.sqrt(delta_t) * thermal_force
        return states
