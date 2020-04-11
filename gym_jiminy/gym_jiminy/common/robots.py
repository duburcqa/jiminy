## @file

"""
@package    gym_jiminy

@brief      Package containing python-native helper methods for Gym Jiminy Open Source.
"""

import os
import numpy as np

from gym import core, spaces
from gym.utils import seeding

from jiminy_py import core as jiminy
from jiminy_py.engine_asynchronous import EngineAsynchronous

from . import RenderOutMock


class RobotJiminyEnv(core.Env):
    """
    @brief      Base class to train a robot in Gym OpenAI using a user-specified
                Python Jiminy engine for physics computations.

                It creates an Gym environment wrapping Jiminy Engine and behaves
                like any other Gym environment.

    @details    The Python Jiminy engine must be completely initialized beforehand,
                which means that the Jiminy Robot and Controller are already setup.
                For now, the only engine available is `EngineAsynchronous`.
    """

    ## Metadata of the environment
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, robot_name : str, engine_py : EngineAsynchronous, dt : float):
        """
        @brief      Constructor

        @param[in]  robot_name  Name of the robot
        @param[in]  engine_py   Python Jiminy engine used for physics computations.
                                It must be completely initialized. For now, the
                                only engine available is `EngineAsynchronous`.
        @param[in]  dt          Desired update period of the simulation

        @return     Instance of the environment.
        """

        # ##################### Configure the learning environment ############################

        ## Name of the robot
        self.robot_name = robot_name
        ## Jiminy engine associated with the robot. It is used for physics computations.
        self.engine_py = engine_py
        ## Update period of the simulation
        self.dt = dt

        ## Extract some information from the robot
        motors_position_idx = self.engine_py._engine.robot.motors_position_idx
        joint_position_limit_upper = self.engine_py._engine.robot.position_limit_upper
        joint_position_limit_lower = self.engine_py._engine.robot.position_limit_lower
        joint_velocity_limit = self.engine_py._engine.robot.velocity_limit

        ## Action space
        action_high = joint_position_limit_upper[motors_position_idx]
        action_low = joint_position_limit_lower[motors_position_idx]
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)

        ## Observation space
        obs_high = np.concatenate((joint_position_limit_upper, joint_velocity_limit))
        obs_low = np.concatenate((joint_position_limit_lower, -joint_velocity_limit))
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)

        ## Higher bound of the hypercube associated with the initial state of the robot
        self.state_random_high = 0.1 * np.ones(self.observation_space.shape)
        ## Lower bound of the hypercube associated with the initial state of the robot
        self.state_random_low = -self.state_random_high
        ## State of the robot
        self.state = None
        self._viewer = None
        ## Number of simulation steps performed after having met the stopping criterion
        self.steps_beyond_done = None

        self.seed()

        # ##################### Enforce some options of the engine ############################

        engine_options = self.engine_py.get_engine_options()

        engine_options["stepper"]["iterMax"] = -1 # Infinite number of iterations
        engine_options["stepper"]["sensorsUpdatePeriod"] = self.dt
        engine_options["stepper"]["controllerUpdatePeriod"] = self.dt

        self.engine_py.set_engine_options(engine_options)

    def seed(self, seed=None):
        """
        @brief      Specify the seed of the simulation.

        @details    One must reset the simulation after updating the seed because
                    otherwise the behavior is undefined as it is not part of the
                    specification for Python Jiminy engines.

        @param[in]  seed    Desired seed as a Unsigned Integer 32bit
                            Optional: The seed will be randomly generated using np if omitted.

        @return     Updated seed of the simulation
        """
        self.np_random, seed = seeding.np_random(seed)
        self.engine_py.seed(seed)
        self.state = self.engine_py.state
        return [seed]

    def reset(self):
        """
        @brief      Reset the simulation.

        @details    The initial state is randomly sampled using a uniform
                    distribution between `self.state_random_low` and
                    `self.state_random_high`.

        @remark    `self.state_random_low` and `self.state_random_high` can be
                    overwritten. They are obtained by default by extracting the
                    information from the URDF file.

        @return     Initial state of the simulation
        """
        self.state = self.np_random.uniform(low=self.state_random_low,
                                            high=self.state_random_high)
        self.engine_py.reset(self.state)
        self.steps_beyond_done = None
        return self._get_obs()

    def render(self, mode=None, lock=None, **kwargs):
        """
        @brief      Render the current state of the robot in Gepetto-viewer.

        @details    Do not suport Multi-Rendering RGB output because it is not
                    possible to create window in new tabs programmatically in
                    Gepetto viewer.

        @param[in]  mode    Unused. Defined for compatibility with Gym OpenAI.
        @param[in]  lock    Unique threading.Lock for every simulation
                            Optional: Only required for parallel rendering

        @return     Fake output for compatibility with Gym OpenAI.
        """

        self.engine_py.render(return_rgb_array=False, lock=lock, **kwargs)
        return RenderOutMock()

    def close(self):
        """
        @brief      Terminate the Python Jiminy engine. Mostly defined for
                    compatibility with Gym OpenAI.
        """
        self.engine_py.close()

    def _get_obs(self):
        """
        @brief      Returns the observation.
        """
        raise NotImplementedError()


class RobotJiminyGoalEnv(RobotJiminyEnv, core.GoalEnv):
    """
    @brief      Base class to train a robot in Gym OpenAI using a user-specified
                Jiminy Engine for physics computations.

                It creates an Gym environment wrapping Jiminy Engine and behaves
                like any other Gym goal-environment.

    @details    The Jiminy Engine must be completely initialized beforehand, which
                means that the Jiminy Robot and Controller are already setup.
    """
    def __init__(self, robot_name : str, engine_py : EngineAsynchronous, dt : float):
        """
        @brief      Constructor

        @param[in]  robot_name  Name of the robot
        @param[in]  engine_py   Python Jiminy engine used for physics computations.
                                It must be completely initialized. For now, the
                                only engine available is `EngineAsynchronous`.
        @param[in]  dt          Desired update period of the simulation

        @return     Instance of the environment.
        """

        ## @var observation_space
        # @copydoc RobotJiminyEnv::observation_space

        super(RobotJiminyGoalEnv, self).__init__(robot_name, engine_py, dt)

        ## Current goal
        self.goal = self._sample_goal()

        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype=np.float64),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype=np.float64),
            observation=self.observation_space
        ))

    def reset(self):
        """
        @brief      Reset the simulation.

        @details    Sample a new goal, then call `RobotJiminyEnv.reset`.
        .
        @remark     See documentation of `RobotJiminyEnv` for details.

        @return     Initial state of the simulation
        """
        self.goal = self._sample_goal().copy()
        return super(RobotJiminyGoalEnv, self).reset()

    def _sample_goal(self):
        """
        @brief      Samples a new goal and returns it.
        """
        raise NotImplementedError()
