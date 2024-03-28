import numpy as np
import matplotlib.pyplot as plt

from gym_jiminy.common.blocks.proportional_derivative_controller import integrate_zoh

ORDER = 4

state_prev = np.zeros((ORDER, 1))
state_min = np.full((ORDER, 1), np.nan)
state_max = np.full((ORDER, 1), np.nan)
dt = 0.001

t = 0.0
times, states = [t], [state_prev.copy()]
states_min, states_max = [state_min.copy()], [state_max.copy()]
while t < 2.5:
    print(t)
    state_prev[-1] = 1.0
    state_min[:, 0] = (-1.0, *((-np.inf,) * (ORDER - 1)))
    state_max[:, 0] = (+1.0, *((+np.inf,) * (ORDER - 1)))
    state_prev[:] = integrate_zoh(state_prev, state_min, state_max, dt)
    t += dt
    times.append(t)
    states.append(state_prev.copy())
    states_min.append(state_min.copy())
    states_max.append(state_max.copy())

fig, axes = plt.subplots(len(state_prev), 1, sharex=True)
for ax, data, data_min, data_max in zip(axes, *(
        np.concatenate(data, axis=-1)
        for data in (states, states_min, states_max))):
    ax.plot(times, data, '.-', color="tab:blue")
    ax.plot(times, data_min, '--', color="tab:red")
    ax.plot(times, data_max, '--', color="tab:red")
    ax.set_xlim(left=1.7)
plt.show()
