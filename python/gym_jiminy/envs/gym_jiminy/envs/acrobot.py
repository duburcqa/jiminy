import os
import sys
from typing import Dict, Any, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium.spaces import flatten_space

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator

from gym_jiminy.common.bases import InfoType, EngineObsType
from gym_jiminy.common.envs import BaseJiminyEnv
from gym_jiminy.common.utils import sample, copyto

if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files


# Stepper update period
STEP_DT = 0.2
# Controller update period
CONTROL_DT = 0.02
# Range of uniform sampling distribution of joint angles
THETA_RANDOM_MAX = 0.1
# Range of uniform sampling distribution of joint velocities
DTHETA_RANDOM_MAX = 0.1
# Relative height of tip to consider an episode successful for normal env
HEIGHT_REL_DEFAULT_THRESHOLD = 0.5
# Range of rel. height of tip to consider an episode successful for goal env
HEIGHT_REL_GOAL_THRESHOLD_RANGE = (-0.2, 0.98)
# Standard deviation of the noise added to the action
ACTION_NOISE = 0.0


class AcrobotJiminyEnv(BaseJiminyEnv[np.ndarray, np.ndarray]):
    """Implementation of a Gym environment for the Acrobot using Jiminy for
    both physics computations and for rendering.

    It is a specialization of BaseJiminyEnv. The acrobot is a 2-link
    pendulum with only the second joint actuated. Initially, both links point
    downwards. The goal is to swing the end-effector at a height at least the
    length of one link above the base. Both links can swing freely and can pass
    by each other, i.e. they don't collide when they have the same angle.

    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :

        [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]

    For the first link, an angle of 0 corresponds to the link pointing
    downwards. The angle of the second link is relative to the angle of the
    first link. An angle of 0 corresponds to having the same angle between the
    two links. A state of [1, 0, 1, 0, ..., ...] means that both links point
    downwards.

    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between the
    two pendulum links.

    .. note::
        - R. Sutton: Generalization in Reinforcement Learning:
             Successful Examples Using Sparse Coarse Coding (NIPS 1996)
        - R. Sutton and A. G. Barto:
             Reinforcement learning: An introduction.
             Cambridge: MIT press, 1998.
    """
    def __init__(self,
                 continuous: bool = False,
                 debug: bool = False,
                 viewer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        :param continuous: Whether the action space is continuous. If not
                           continuous, the action space has only 3 states, i.e.
                           low, zero, and high.
                           Optional: True by default.
        :param debug: Whether the debug mode must be enabled.
                      See `BaseJiminyEnv` constructor for details.
        :param viewer_kwargs: Keyword arguments used to override the original
                              default values whenever a viewer is instantiated.
                              This is the only way to pass custom arguments to
                              the viewer when calling `render` method, unlike
                              `replay` which forwards extra keyword arguments.
                              Optional: None by default.
        """
        # Backup some input arguments
        self.continuous = continuous

        # Get URDF path
        data_dir = str(files("gym_jiminy.envs") / "data/toys_models/acrobot")
        urdf_path = os.path.join(data_dir, "acrobot.urdf")

        # Instantiate robot
        robot = jiminy.Robot()
        robot.initialize(
            urdf_path, has_freeflyer=False, mesh_package_dirs=[data_dir])

        # Add motors and sensors
        motor_joint_name = "SecondArmJoint"
        encoder_joint_names = ("FirstArmJoint", "SecondArmJoint")
        motor = jiminy.SimpleMotor(motor_joint_name)
        robot.attach_motor(motor)
        motor.initialize(motor_joint_name)
        for joint_name in encoder_joint_names:
            encoder = jiminy.EncoderSensor(joint_name)
            robot.attach_sensor(encoder)
            encoder.initialize(joint_name)

        # Instantiate simulator
        simulator = Simulator(robot, viewer_kwargs=viewer_kwargs)

        # Override the default camera pose to be absolute if none is specified
        simulator.viewer_kwargs.setdefault("camera_pose", (
            (0.0, 8.0, 0.0), (np.pi/2, 0.0, np.pi), None))

        # Map between discrete actions and actual motor torque if necessary
        if not self.continuous:
            command_limit = np.asarray(motor.command_limit)
            self.AVAIL_CTRL = (-command_limit, np.array(0.0), command_limit)

        # Internal parameters used for computing termination condition
        self._tipIdx = robot.pinocchio_model.getFrameId("Tip")
        self._tipPosZMax = abs(
            robot.pinocchio_data.oMf[self._tipIdx].translation[2])

        # Configure the learning environment
        super().__init__(simulator,
                         step_dt=STEP_DT,
                         debug=debug)

        # Create some proxies for fast access
        self.__state_view = (self.observation[:self.robot.nq],
                             self.observation[-self.robot.nv:])

    def _setup(self) -> None:
        """ TODO: Write documentation.
        """
        # Call base implementation
        super()._setup()

        # Increase stepper accuracy for time-continuous control
        engine_options = self.simulator.engine.get_options()
        engine_options["stepper"]["solver"] = "runge_kutta_4"
        engine_options["stepper"]["dtMax"] = CONTROL_DT
        self.simulator.engine.set_options(engine_options)

    def _initialize_observation_space(self) -> None:
        """Configure the observation of the environment.

        Only the state is observable, while by default, the current time,
        state, and sensors data are available.
        """
        self.observation_space = flatten_space(self._get_agent_state_space())

    def refresh_observation(self, measurement: EngineObsType) -> None:
        """Update the observation based on the current simulation state.

        Only the state is observable, while by default, the current time,
        state, and sensors data are available.

        .. note::
            For goal env, in addition of the current robot state, both the
            desired and achieved goals are observable.
        """
        copyto(self.__state_view, measurement[
            'states']['agent'].values())  # type: ignore[index,union-attr]

    def _initialize_action_space(self) -> None:
        """Configure the action space of the environment.

        Replace the action space by its discrete representation depending on
        'continuous'.
        """
        if not self.continuous:
            self.action_space = gym.spaces.Discrete(len(self.AVAIL_CTRL))
        else:
            super()._initialize_action_space()

    def _sample_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a valid configuration and velocity for the robot.

        The initial state is randomly sampled using a hypercube uniform
        distribution, according to the official Gym acrobot-v1 action space.

        See documentation: https://gym.openai.com/envs/Acrobot-v1/.
        """
        theta1, theta2 = sample(
            scale=THETA_RANDOM_MAX, shape=(2,), rg=self.np_random)
        qpos = np.array([np.cos(theta1), np.sin(theta1),
                         np.cos(theta2), np.sin(theta2)])
        qvel = sample(scale=DTHETA_RANDOM_MAX, shape=(2,), rg=self.np_random)
        return qpos, qvel

    def has_terminated(self) -> Tuple[bool, bool]:
        """Determine whether the episode is over.

        It terminates (`terminated=True`) if the goal has been achieved, namely
        if the tip of the Acrobot is above 'HEIGHT_REL_DEFAULT_THRESHOLD'.
        Apart from that, there is no specific truncation condition.

        :returns: terminated and truncated flags.
        """
        # Call base implementation
        terminated, truncated = super().has_terminated()

        # Check if the agent has successfully solved the task
        tip_transform = self.robot.pinocchio_data.oMf[self._tipIdx]
        tip_position_z = tip_transform.translation[2]
        if tip_position_z > HEIGHT_REL_DEFAULT_THRESHOLD * self._tipPosZMax:
            terminated = True

        return terminated, truncated

    def compute_command(self, action: np.ndarray) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        Convert a discrete action into its actual value if necessary, then add
        noise to the action is enable.

        :param action: Desired motors efforts.
        """
        if not self.continuous:
            action = self.AVAIL_CTRL[action]
        if ACTION_NOISE > 0.0:
            action += sample(scale=ACTION_NOISE, rg=self.np_random)
        return action

    def compute_reward(self,
                       terminated: bool,
                       truncated: bool,
                       info: InfoType) -> float:
        """Compute reward at current episode state.

        Get a small negative reward till success.
        """
        return 0.0 if terminated else -1.0
