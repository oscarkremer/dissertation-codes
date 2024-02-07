import numpy as np
from numba import njit
from numba.pycc import CC


cc_param = CC("simulation_parametric")
cc_param._source_module = "dissertation_codes.simulation.simulation_paramtric"


@njit(nopython=True, cache=True)
@cc_param.export(
    "simulation_param",
    "f8[:,:]\
       (f8[:,:], f8, f8, f8, f8[:,:], f8[:], f8[:], f8, f8, i8, i8)",
)
def simulation_param(A,
                     thermal_std_x,
                     thermal_std_y,
                     thermal_std_z,
                     G,
                     omegas,
                     param_factors,
                     x0,
                     dt,
                     N_time,
                     M):
    states = np.zeros(shape=(N_time, 6))
    current_states = np.array([[7*x0/6],
                               [x0],
                               [4*x0],
                               [7*x0/6],
                               [x0],
                               [4*x0]
                               ])
    phi_x = 2 * np.arctan(current_states[0, 0]/current_states[3, 0])
    phi_x = phi_x + np.pi/2
    phi_y = 2 * np.arctan(current_states[1, 0]/current_states[4, 0])
    phi_y = phi_y + np.pi/2
    phi_z = 2 * np.arctan(current_states[2, 0]/current_states[5, 0])
    phi_z = phi_z + np.pi/2
    b_array = np.zeros(shape=(6, 1))
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
        if (M != 0) and not (k % M):
            phi_x = 2 * np.arctan(current_states[0, 0]/current_states[3, 0])
            phi_x = phi_x + (np.pi/2)
            phi_y = 2 * np.arctan(current_states[1, 0]/current_states[4, 0])
            phi_y = phi_y + (np.pi/2)
            phi_z = 2 * np.arctan(current_states[2, 0]/current_states[5, 0])
            phi_z = phi_z + (np.pi/2)
        parametric_x = param_factors[0]*np.cos(2 * omegas[0] * k * dt + phi_x)
        parametric_y = param_factors[1]*np.cos(2 * omegas[1] * k * dt + phi_y)
        parametric_z = param_factors[2]*np.cos(2 * omegas[2] * k * dt + phi_z)
        parametric_force = parametric_x + parametric_y + parametric_z
        b_array[3, 0] = current_states[0, 0]
        b_array[4, 0] = current_states[1, 0]
        b_array[5, 0] = current_states[2, 0]
        state_dot = A @ current_states + b_array * 2 * parametric_force
        current_states = (
            current_states
            + state_dot * dt
            + G * np.sqrt(dt) * thermal_force
        )
        states[k, :] = current_states[:, 0]
    return states


if __name__ == "__main__":
    cc_param.compile()
