import os
import numpy as np

from gym import core, spaces
from gym.utils import seeding

import jiminy
from jiminy_py import engine_asynchronous
from gym_jiminy.common import RenderOutMock


class RobotJiminyEnv(core.Env):
    """
    Base class for Jiminy actors in a Scene.
    These environments create single-player scenes and behave like normal Gym environments.
    """

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, robot_name, engine_py, dt):
        ####################### Configure the learning environment #############################

        self.robot_name = robot_name
        self.engine_py = engine_py
        self.dt = dt

        motors_position_idx = self.engine_py._engine.model.motors_position_idx
        joint_position_limit_upper = self.engine_py._engine.model.position_limit_upper.A1
        joint_position_limit_lower = self.engine_py._engine.model.position_limit_lower.A1
        joint_velocity_limit = self.engine_py._engine.model.velocity_limit.A1

        action_high = joint_position_limit_upper[motors_position_idx]
        action_low = joint_position_limit_lower[motors_position_idx]
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)

        obs_high = np.concatenate((joint_position_limit_upper, joint_velocity_limit))
        obs_low = np.concatenate((joint_position_limit_lower, -joint_velocity_limit))
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)

        self.state_random_high = 0.1 * np.ones(self.observation_space.shape)
        self.state_random_low = -self.state_random_high
        self.state = None
        self.viewer = None
        self.steps_beyond_done = None

        self.seed()

        ####################### Enforce some options of the engine #############################

        engine_options = self.engine_py.get_engine_options()

        engine_options["stepper"]["iterMax"] = -1 # Infinite number of iterations
        engine_options["stepper"]["sensorsUpdatePeriod"] = self.dt
        engine_options["stepper"]["controllerUpdatePeriod"] = self.dt

        self.engine_py.set_engine_options(engine_options)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.engine_py.seed(seed)
        self.state = self.engine_py.state
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=self.state_random_low,
                                            high=self.state_random_high)
        self.engine_py.reset(np.expand_dims(self.state, axis=-1))
        self.steps_beyond_done = None
        return self._get_obs()

    def render(self, mode='rgb_array', lock=None):
        # Do not suport Multi-Rendering RGB output because it is not
        # possible to create window in new tabs programmatically in
        # Gepetto viewer.

        self.engine_py.render(return_rgb_array=False, lock=lock)
        if (self.viewer is None):
            self.viewer = self.engine_py._client
        return RenderOutMock()

    def close(self):
        if (self.viewer is not None):
            self.engine_py.close()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

class RobotJiminyGoalEnv(RobotJiminyEnv, core.GoalEnv):
    """
    Base class for Jiminy actors in a Scene.
    These environments create single-player scenes and behave like normal Gym goal-environments.
    """

    def __init__(self, robot_name, engine_py, dt):
        super(RobotJiminyGoalEnv, self).__init__(robot_name, engine_py, dt)

        self.goal = self._sample_goal()

        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype=np.float64),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype=np.float64),
            observation=self.observation_space
        ))

    def reset(self):
        self.goal = self._sample_goal().copy()
        return super(RobotJiminyGoalEnv, self).reset()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()