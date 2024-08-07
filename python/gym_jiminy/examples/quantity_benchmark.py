import timeit

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

import gym_jiminy.common.bases.quantities
from gym_jiminy.common.quantities import (
    OrientationType, QuantityManager, FrameOrientation)

# Define number of samples for benchmarking
N_SAMPLES = 50000

# Disable caching by disabling "IS_CACHED" FSM State
gym_jiminy.common.bases.quantities._IS_CACHED = (
    gym_jiminy.common.bases.quantities.QuantityStateMachine.IS_INITIALIZED)

# Instantiate a dummy environment
env = gym.make("gym_jiminy.envs:atlas")
env.reset()
env.step(env.action)

# Define quantity manager and add quantities to benchmark
quantity_manager = QuantityManager(env)
for i, frame in enumerate(env.robot.pinocchio_model.frames):
    quantity_manager[f"rpy_{i}"] = (FrameOrientation, dict(
        frame_name=frame.name, type=OrientationType.EULER))

# Run the benchmark for all batch size
time_per_frame_all = []
for i in range(1, len(env.robot.pinocchio_model.frames)):
    # Reset tracking
    quantity_manager.reset(reset_tracking=True)

    # Fetch all quantities once to update dynamic computation graph
    for j, quantity in enumerate(quantity_manager.registry.values()):
        quantity.get()
        if i == j + 1:
            break

    # Extract batched data buffer of `FrameOrientation` quantities
    shared_data = quantity.data

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
