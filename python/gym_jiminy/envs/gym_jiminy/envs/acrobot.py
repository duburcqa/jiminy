import os
from importlib.resources import files
from typing import Dict, Any, Optional, Tuple, cast

import numpy as np
import gymnasium as gym

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator

from gym_jiminy.common.bases import InfoType, EngineObsType
from gym_jiminy.common.envs import BaseJiminyEnv
from gym_jiminy.common.utils import sample, get_robot_state_space


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

        # Add motor
        motor_joint_name = "SecondArmJoint"
        motor = jiminy.SimpleMotor(motor_joint_name)
        robot.attach_motor(motor)
        motor.initialize(motor_joint_name)
        for joint_name in ("FirstArmJoint", "SecondArmJoint"):
            sensor = jiminy.EncoderSensor(joint_name)
            robot.attach_sensor(sensor)
            sensor.initialize(joint_name=joint_name)

        # Instantiate simulator
        simulator = Simulator(robot, viewer_kwargs=viewer_kwargs)

        # Override the default camera pose to be absolute if none is specified
        simulator.viewer_kwargs.setdefault("camera_pose", (
            (0.0, 8.0, 0.0), (np.pi/2, 0.0, np.pi), None))

        # Map between discrete actions and actual motor torque if necessary
        if not self.continuous:
            command_limit = np.array(motor.effort_limit)
            self.AVAIL_CTRL = (-command_limit, np.array(0.0), command_limit)

        # Internal parameters used for computing termination condition
        self._tipIndex = robot.pinocchio_model.getFrameId("Tip")
        self._tipPosZMax = abs(
            robot.pinocchio_data.oMf[self._tipIndex].translation[2])

        # Configure the learning environment
        super().__init__(simulator,
                         step_dt=STEP_DT,
                         debug=debug)

    def _setup(self) -> None:
        """Configure the environment.

        In practice, it sets the stepper options to enforce a fixed-timestep
        integrator after calling the base implementation.

        .. note::
            This method must be called once, after the environment has been
            reset. This is done automatically when calling `reset` method.
        """
        # Call base implementation
        super()._setup()

        # Enforce fixed-timestep integrator.
        # It ensures calling 'step' always takes the same amount of time.
        engine_options = self.simulator.get_options()
        engine_options["stepper"]["odeSolver"] = "runge_kutta_4"
        engine_options["stepper"]["dtMax"] = CONTROL_DT
        self.simulator.set_options(engine_options)

    def _initialize_observation_space(self) -> None:
        """Configure the observation of the environment.

        Only the state is observable, while by default, the current time,
        state, and sensors data are available.
        """
        state_space = get_robot_state_space(
            self.robot, use_theoretical_model=True)
        position_space, velocity_space = state_space['q'], state_space['v']
        assert isinstance(position_space, gym.spaces.Box)
        assert isinstance(velocity_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate((position_space.low, velocity_space.low)),
            high=np.concatenate((position_space.high, velocity_space.high)),
            dtype=np.float64)
        self.observation_space.mirror_mat = (  # type: ignore[attr-defined]
            np.diag([1.0, -1.0, 1.0, -1.0, -1.0, -1.0]))

    def refresh_observation(self, measurement: EngineObsType) -> None:
        angles, velocities = measurement['measurements']['EncoderSensor']
        self.observation[[0, 2]] = np.cos(angles)
        self.observation[[1, 3]] = np.sin(angles)
        self.observation[4:] = velocities

    def _initialize_action_space(self) -> None:
        """Configure the action space of the environment.

        Replace the action space by its discrete representation depending on
        'continuous'.
        """
        if not self.continuous:
            self.action_space = cast(
                gym.Space[np.ndarray],
                gym.spaces.Discrete(len(self.AVAIL_CTRL)))
        else:
            super()._initialize_action_space()
            self.action_space.mirror_mat = (  # type: ignore[attr-defined]
                - np.eye(1))

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

    def has_terminated(self, info: InfoType) -> Tuple[bool, bool]:
        """Determine whether the episode is over.

        It terminates (`terminated=True`) if the goal has been achieved, namely
        if the tip of the Acrobot is above 'HEIGHT_REL_DEFAULT_THRESHOLD'.
        Apart from that, there is no specific truncation condition.

        :param info: Dictionary of extra information for monitoring.

        :returns: terminated and truncated flags.
        """
        # Call base implementation
        terminated, truncated = super().has_terminated(info)

        # Check if the agent has successfully solved the task
        tip_transform = self.robot.pinocchio_data.oMf[self._tipIndex]
        tip_position_z = tip_transform.translation[2]
        if tip_position_z > HEIGHT_REL_DEFAULT_THRESHOLD * self._tipPosZMax:
            terminated = True

        return terminated, truncated

    def compute_command(self, action: np.ndarray, command: np.ndarray) -> None:
        """Compute the motors efforts to apply on the robot.

        Convert a discrete action into its actual value if necessary, then add
        noise to the action is enable.

        :param action: Desired motors efforts.
        """
        command[:] = action if self.continuous else self.AVAIL_CTRL[action]
        if ACTION_NOISE > 0.0:
            command += sample(scale=ACTION_NOISE, rg=self.np_random)

    def compute_reward(self, terminated: bool, info: InfoType) -> float:
        """Compute reward at current episode state.

        Get a small negative reward till success.
        """
        return 0.0 if terminated else -1.0
