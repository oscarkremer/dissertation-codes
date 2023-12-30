import numpy as np
from numba import njit, jit
from numba.pycc import CC


@jit(nopython=True)
def simulation_n3d(
    A,
    B,
    thermal_std_x,
    thermal_std_y,
    thermal_std_z,
    omegas,
    x0,
    C,
    G,
    G_norm,
    G_fb,
    zps,
    xsis,
    m,
    dt,
    N_time,
):
    zp_x = zps[0]
    zp_y = zps[1]
    zp_z = zps[2]
    xsi_x = xsis[0]
    xsi_y = xsis[1]
    xsi_z = xsis[2]
    omega_x = omegas[0]
    omega_y = omegas[1]
    omega_z = omegas[2]
    state = np.zeros(shape=(N_time, 6))
    controls = np.zeros(shape=(N_time))
    current_states = np.array([[x0 * np.random.normal()],
                               [x0 * np.random.normal()],
                               [x0 * np.random.normal()],
                               [x0 * np.random.normal()],
                               [x0 * np.random.normal()],
                               [x0 * np.random.normal()]
                               ])
    for k in range(N_time):

        thermal_force_x = thermal_std_x * np.random.normal()
        thermal_force_y = thermal_std_y * np.random.normal()
        thermal_force_z = thermal_std_z * np.random.normal()
        thermal_force = np.array([[0.0],
                                  [0.0],
                                  [0.0],
                                  [thermal_force_x],
                                  [thermal_force_y],
                                  [thermal_force_z]])

#        print(xsi_x, xsi_y, xsi_z, zp_x, zp_y, zp_z, current_states)
        xsi_array = np.array([[0.0],
                              [0.0],
                              [0.0],
                              [xsi_x*(current_states[0, 0]*zp_x)**2],
                              [xsi_y*(current_states[1, 0]*zp_y)**2],
                              [xsi_z*(current_states[2, 0]*zp_z)**2]])
        aux_omega_array = np.array([[0.0],
                                    [0.0],
                                    [0.0],
                                    [current_states[0, 0]*zp_x*omega_x**2],
                                    [current_states[1, 0]*zp_y*omega_y**2],
                                    [current_states[2, 0]*zp_z*omega_z**2]])
        control = -G_fb@current_states
        non_linear_matrix = (m*xsi_array@(aux_omega_array.T)).T
        non_linear_matrix[5, 3] = 2*non_linear_matrix[5, 3]
        non_linear_matrix[5, 4] = 2*non_linear_matrix[5, 4]
        state_dot = A@current_states + B @ control - G_norm*(non_linear_matrix@G)
        current_states = current_states + state_dot * dt + G * np.sqrt(dt) * thermal_force
        state[k, :] = current_states[:, 0]
    return state, controls
