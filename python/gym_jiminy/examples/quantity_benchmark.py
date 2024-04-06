import timeit

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

import gym_jiminy.common.bases.quantity
from gym_jiminy.common.bases import QuantityManager
from gym_jiminy.common.quantities import EulerAnglesFrame

# Define number of samples for benchmarking
N_SAMPLES = 50000

# Disable caching by forcing `SharedCache.has_value` to always return `False`
setattr(gym_jiminy.common.bases.quantity.SharedCache,
        "has_value",
        property(lambda self: False))

# Instantiate a dummy environment
env = gym.make("gym_jiminy.envs:atlas")
env.reset()
env.step(env.action)

# Define quantity manager and add quantities to benchmark
nframes = len(env.pinocchio_model.frames)
quantity_manager = QuantityManager(
    env.simulator,
    {
        f"rpy_{i}": (EulerAnglesFrame, dict(frame_name=frame.name))
        for i, frame in enumerate(env.pinocchio_model.frames)
    })

# Run the benchmark for all batch size
time_per_frame_all = []
for i in range(1, nframes):
    # Reset tracking
    quantity_manager.reset(reset_tracking=True)

    # Fetch all quantities once to update dynamic computation graph
    for j, quantity in enumerate(quantity_manager.quantities.values()):
        quantity.get()
        if i == j + 1:
            break

    # Extract batched data buffer of `EulerAnglesFrame` quantities
    shared_data = quantity.requirements['data']

    # Benchmark computation of batched data buffer
    duration = timeit.timeit(
        'shared_data.get()', number=N_SAMPLES, globals={
            "shared_data": shared_data
        }) / N_SAMPLES
    time_per_frame_all.append(duration)
    print(f"Computation time (ns) for {i} frames: {duration * 1e9}")

# Plot the result if enough data is available
if len(time_per_frame_all) > 1:
    plt.figure()
    plt.plot(np.diff(time_per_frame_all) * 1e9)
    plt.xlabel("Number of frames")
    plt.ylabel("Average computation time per frame (ns)")
    plt.show()
