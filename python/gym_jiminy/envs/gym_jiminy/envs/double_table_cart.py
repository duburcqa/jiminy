import os
import xml
import math
import atexit
import tempfile
from functools import partial
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, Any, Optional, Dict, Union, List, Sequence
from pkg_resources import resource_filename

import numpy as np
from gym import spaces

import pinocchio as pin
from xacro import parse, process_doc
from jiminy_py.robot import BaseJiminyRobot
from jiminy_py.simulator import Simulator
from gym_jiminy.common.envs import BaseJiminyEnv
from gym_jiminy.common.utils import sample, set_value, fill


# Constants of the universe
STEP_DT = 0.02

COULOMB_FRICTION_COEFF = 0.4

INIT_POS_REL_MAX = 0.8
INIT_VEL_MAX = 0.03

COUPLING_STIFFNESS_LOG_RANGE = (4.6, 5.3)  # [5cm -> 2000N, 1cm -> 2000N]
COUPLING_STIFFNESS_DEFAULT = 1.0e5
COUPLING_DAMPING_RATIO_RANGE = (0.2, 1.0)
COUPLING_DAMPING_DEFAULT = 4.0e3

PATIENT_KP_LOG_RANGE = (3.6, 4.3)    # [5cm -> 200N, 1cm -> 200N]
PATIENT_KP_DEFAULT = 1.0e4
PATIENT_KD_RATIO_RANGE = (0.2, 1.0)
PATIENT_KD_DEFAULT = 1.0e3

FLEX_STIFFNESS_LOG_RANGE = (3.7, 4.6)  # [5000, 40000]
FLEX_STIFFNESS_DEFAULT = 1.0e4
FLEX_DAMPING_RATIO_RANGE = (0.2, 1.0)
FLEX_DAMPING_DEFAULT = 5.0e2

L_X_F_MIN_RANGE = (4.0e-2, 8.0e-2)
L_X_F_MAX_RANGE = (24.0e-2, 31.0e-2)
L_Y_F_RANGE = (3.0e-2, 6.0e-2)
I_R_XX_RANGE = (8.0, 13.0)
I_R_XZ_RANGE = (0.5, 1.5)
I_R_YY_RANGE = (6.0, 12.0)
I_R_ZZ_RANGE = (1.5, 3.0)
Z_R_RANGE = (62.0e-2, 76.0e-2)
F_X_R_MAX_RANGE = (1.5e3, 3.0e3)
F_Y_R_MAX_RANGE = (1.0e3, 2.0e3)
V_X_R_MAX_RANGE = (1.5, 2.5)
V_Y_R_MAX_RANGE = (0.5, 1.0)
M_P_L_RANGE = (20.0, 40.0)
X_P_RANGE = (-1.0e-2, 2.0e-2)
I_P_L_XX_RANGE = (2.0, 4.0)
I_P_L_YY_RANGE = (2.0, 4.0)
M_P_U_RANGE = (30.0, 50.0)
I_P_U_XX_RANGE = (2.0, 4.0)
I_P_U_YY_RANGE = (0.5, 1.0)
I_P_U_ZZ_RANGE = (1.5, 3.0)
Z_P_RANGE = (15.0e-2, 25.0e-2)
L_X_P_MIN_RANGE = (1.0e-2, 4.0e-2)
L_X_P_MAX_RANGE = (10.0e-2, 20.0e-2)
L_Y_P_RANGE = (3.0e-2, 6.0e-2)
F_X_P_MAX_RANGE = (1.0e2, 3.0e2)
F_Y_P_MAX_RANGE = (1.0e2, 3.0e2)
V_X_P_MAX_RANGE = (1.5, 4.0)
V_Y_P_MAX_RANGE = (0.5, 1.5)

TRAJ_V_MAX = 2 * np.pi  # angular velocity (rad.s-1)
TRAJ_MARGINS = (1.0e-2, 1.0e-2)



def _find_joints_state_idx(robot: BaseJiminyRobot, pattern: str):
    joints_idx = [i for i, joint_name in enumerate(robot.rigid_joints_names)
                  if pattern in joint_name]
    return (np.array([robot.rigid_joints_position_idx[i] for i in joints_idx]),
            np.array([robot.rigid_joints_velocity_idx[i] for i in joints_idx]))


def _generate_model(xacro_path: str,
                    hardware_path: Optional[str] = None,
                    has_flexiblity: bool = True,
                    model_dict: Optional[Dict[str, float]] = None,
                    simulator: Optional[Simulator] = None,
                    simulator_kwargs: Optional[Dict[str, Any]] = None,
                    debug: bool = False) -> Simulator:
    # Generate URDF by updating and processing xacro
    doc = parse(None, xacro_path)
    if model_dict is not None:
        for elem in doc.firstChild.childNodes:
            if isinstance(elem, xml.dom.minidom.Element) and \
                    elem.tagName == 'xacro:property':
                prop_name = elem.attributes['name'].value
                if prop_name in model_dict.keys():
                    elem.attributes['value'].value = float(
                        model_dict[prop_name])
    process_doc(doc)

    # Create temporary URDF path is necessary
    if simulator is None:
        urdf_prefix = Path(xacro_path).stem.split('.', 1)[0]
        fd, urdf_path = tempfile.mkstemp(
            prefix=urdf_prefix, suffix=".urdf")
        os.close(fd)

        if not debug:
            def remove_file_at_exit(file_path=urdf_path):
                try:
                    os.remove(file_path)
                except (PermissionError, FileNotFoundError):
                    pass

            atexit.register(remove_file_at_exit)
    else:
        urdf_path = simulator.robot.urdf_path_orig

    # Write down the new model
    with open(urdf_path, "w") as f:
        doc.writexml(f)

    if simulator is None:
        # Create and initialize backend based to simulate the model
        simulator = Simulator.build(
            urdf_path=urdf_path,
            hardware_path=hardware_path,
            has_freeflyer=False,
            use_theoretical_model=True,
            avoid_instable_collisions=True,
            **simulator_kwargs)

        # Some options must be does right after instantiation, to guarantee the
        # state will have the right dimension and bounds later on, without
        # having to call `reset` once.
        model_options = simulator.robot.get_model_options()

        ### Configure foot table flexibility.
        model_options["dynamics"]["enableFlexibleModel"] = has_flexiblity
        model_options['dynamics']['flexibilityConfig'] = [{
            'frameName': "FootTable",
            'stiffness': np.full(3, fill_value=FLEX_STIFFNESS_DEFAULT),
            'damping': np.full(3, fill_value=FLEX_DAMPING_DEFAULT)}]

        ### Disable joint position bounds since it creates artefact and
        #   numerical instabilities.
        model_options["joints"]["enablePositionLimit"] = True
        model_options["joints"]["enableVelocityLimit"] = True

        simulator.robot.set_model_options(model_options)
    else:
        # Update simulator with new model
        robot_options = simulator.robot.get_options()
        robot = simulator.robot.__class__()
        robot.initialize(simulator.robot.urdf_path_orig,
                         simulator.robot.hardware_path,
                         simulator.robot.mesh_package_dirs[0],
                         simulator.robot.has_freeflyer,
                         avoid_instable_collisions=True,
                         verbose=debug)
        simulator.engine.initialize(robot, None, simulator._callback)
        simulator.robot.set_options(robot_options)

    return simulator


class DoubleTableCartJiminyEnv(BaseJiminyEnv):
    def __init__(self,
                 has_patient: bool = True,
                 has_flexiblity: bool = True,
                 debug: bool = False,
                 **kwargs) -> None:
        # Get the urdf and mesh paths
        data_root_dir = os.path.join(
            resource_filename('gym_jiminy.envs', 'data'),
            "toys_models/double_table_cart")
        self._xacro_path = os.path.join(
            data_root_dir, "double_table_cart.urdf.xacro")
        self._hardware_path = os.path.join(
            data_root_dir, "double_table_cart_hardware.toml")
        self._simulator_kwargs = kwargs
        self.has_patient = has_patient
        self.has_flexiblity = has_flexiblity

        # Instantiate simulator associated with default model
        simulator = _generate_model(
            xacro_path=self._xacro_path,
            hardware_path=self._hardware_path,
            has_flexiblity=self.has_flexiblity,
            model_dict={'has_patient': self.has_patient},
            simulator_kwargs=self._simulator_kwargs,
            debug=debug)

        # Initialize the walker environment
        super().__init__(**{**dict(
            simulator=simulator,
            enforce_bounded_spaces=False,
            step_dt=STEP_DT,
            debug=debug), **kwargs})

        # Coupling force internal parameters
        self._coupling_frames_names = ("RobotCart", "PatientLowerBodyCart")
        self._coupling_state_idx = tuple(
            _find_joints_state_idx(self.robot, name)
            for name in self._coupling_frames_names)
        if self.has_patient:
            self._patient_offset = self.robot.pinocchio_model.frames[
                self.robot.pinocchio_model.getFrameId("RobotStrapsToPatient")
            ].placement.translation[:2]
            self._coupling_joints_idx = tuple(
                self.robot.pinocchio_model.frames[
                    self.robot.pinocchio_model.getFrameId(name)].parent
                for name in self._coupling_frames_names)
            self._coupling_stiffness = COUPLING_STIFFNESS_DEFAULT
            self._coupling_damping = COUPLING_DAMPING_DEFAULT

        # Patient behavior parameters
        if self.has_patient:
            self._patient_frame_name = "PatientUpperBodyCart"
            self._patient_joint_idx = self.robot.pinocchio_model.frames[
                self.robot.pinocchio_model.getFrameId(self._patient_frame_name)
            ].parent
            self._patient_kp = PATIENT_KP_DEFAULT
            self._patient_kd = PATIENT_KD_DEFAULT

        # Foot support area buffers
        self._L_x_F_min = 0.0
        self._L_x_F_max = 0.0
        self._L_y_F = 0.0

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

        # Remove coupling forces, if any
        self.simulator.remove_coupling_forces()

        # Compute updated foot geometry
        foot_pts_list = []
        for i in range(4):
            geom_idx = self.robot.collision_model.getGeometryId(f"Foot_{i}")
            geom_i = self.robot.collision_model.geometryObjects[geom_idx]
            foot_pts_list.append(geom_i.placement.translation[:2])
        foot_pts = np.array(foot_pts_list).T
        self._L_x_F_min = - np.min(foot_pts[0])
        self._L_x_F_max = np.max(foot_pts[0])
        self._L_y_F = (np.max(foot_pts[1]) - np.min(foot_pts[1])) / 2.0

        if self.has_patient:
            # Add coupling force between robot and patient through straps
            self.simulator.add_viscoelastic_coupling_force(
                *self._coupling_frames_names,
                np.full((6,), fill_value=self._coupling_stiffness),
                np.full((6,), fill_value=self._coupling_damping))

            # Add coupling force between patient lower and upper body
            self.simulator.add_viscoelastic_coupling_force(
                "PatientTableTop",
                self._patient_frame_name,
                np.full((6,), fill_value=self._patient_kp),
                np.full((6,), fill_value=self._patient_kd))

        # Get options
        model_options = self.robot.get_model_options()
        engine_options = self.simulator.engine.get_options()

        # Configure joint velocity bounds dynamics
        engine_options = self.engine.get_options()
        engine_options['joints']['boundStiffness'] = 1.0e7
        engine_options['joints']['boundDamping'] = 1.0e2

        engine_options['stepper']['timeout'] = 0.0  # Disable step timeout
        engine_options['stepper']['odeSolver'] = 'runge_kutta_dopri5'
        # engine_options['stepper']['dtMax'] = 1e-3

        # Disable joint position bounds since it creates artefact and numerical
        # instabilities.
        model_options["joints"]["enablePositionLimit"] = True
        model_options["joints"]["enableVelocityLimit"] = True

        # Set options
        self.robot.set_model_options(model_options)
        self.engine.set_options(engine_options)

    def _sample_state(self) -> Tuple[np.ndarray, np.ndarray]:
        # Sample robot cart position randomly, and set the one of patient lower
        # body cart to match straps rest position.
        robot_pos_idx = self._coupling_state_idx[0][0]
        qpos = pin.neutral(self.robot.pinocchio_model)
        qpos[robot_pos_idx] = sample(
            np.array([-self._L_x_F_min, -self._L_y_F]),
            np.array([self._L_x_F_max, self._L_y_F]),
            scale=INIT_POS_REL_MAX, rg=self.rg)
        if self.has_patient:
            patient_pos_idx = self._coupling_state_idx[1][0]
            qpos[patient_pos_idx] = qpos[robot_pos_idx] + self._patient_offset

        # Sample robot cart velocity randomly, and everything else to zero
        robot_vel_idx = self._coupling_state_idx[0][1]
        qvel = np.zeros(self.robot.pinocchio_model.nv)
        qvel[robot_vel_idx] = sample(scale=INIT_VEL_MAX, rg=self.rg)

        # Return rigid state since 'use_theoretical_model'=True
        if self.robot.is_flexible:
            qpos = self.robot.get_rigid_configuration_from_flexible(qpos)
            qvel = self.robot.get_rigid_velocity_from_flexible(qvel)

        return qpos, qvel

    def is_done(self, *args: Any, **kwargs: Any) -> bool:
        # Avoid premature ending because of slightly off initialization
        if self.num_steps < 10:
            return False

        # Initialize done flag
        is_done = False

        # Check if CoP is inside the foot geometry
        F_foot = self.robot.pinocchio_data.f[0].vector
        x_cop, y_cop = -F_foot[4] / F_foot[2], F_foot[3] / F_foot[2]

        is_done = is_done or (
            self._L_x_F_max < x_cop or x_cop < - self._L_x_F_min)
        is_done = is_done or abs(y_cop) > self._L_y_F

        # Check if slipping of the ground
        is_done = is_done or np.any(
            np.abs(F_foot[:2]) > COULOMB_FRICTION_COEFF * F_foot[2])

        return is_done

    def compute_reward(self,  # type: ignore[override]
                       *, info: Dict[str, Any]) -> float:
        """ TODO: Write documentation.

        Non-zero reward reward as long as the termination condition has never
        been reached during the same episode:
            - small positive reward proportional to the inverse of actuator
              forces if step successful
            - large negative reward if step failed
        """
        # pylint: disable=arguments-differ

        reward = 0.0
        if self._num_steps_beyond_done is None:
            reward += 1.0 - np.linalg.norm(
                self._action / self.action_space.high, np.inf) ** 2
        elif self._num_steps_beyond_done == 0:
            reward += -100.0

        return reward


TaskType = Dict[str, Dict[str, Union[float, np.ndarray]]]


class DoubleTableCartJiminyMetaEnv(DoubleTableCartJiminyEnv):
    def __init__(self,
                 has_patient: bool = True,
                 has_flexiblity: bool = True,
                 active_task_features: Optional[Sequence[str]] = None,
                 auto_sampling: bool = True,
                 debug: bool = False,
                 **kwargs: Any) -> None:
        # Backup user arguments
        self.auto_sampling = auto_sampling
        if active_task_features is None:
            active_task_features = ["robot", "flexibility", "trajectory"]
            if has_patient:
                active_task_features += ["patient", "coupling", "behavior"]
        if not has_patient and any(e in active_task_features for e in [
                "patient", "coupling", "behavior"]):
            raise ValueError("Cannot enable patient-specific task feature if "
                             "'has_patient' is false.")
        if not has_flexiblity and "flexibility" in active_task_features:
            raise ValueError("Cannot enable flexibility task feature if "
                             "'has_flexiblity' is false.")

        self.active_task_features = active_task_features

        # Initialize base class
        super().__init__(has_patient, has_flexiblity, debug, **kwargs)

        # Current task buffer
        self._task = self.sample_task()
        fill(self._task, fill_value=0.0)

        # Share trajectory task with observation
        self._observation['trajectory'] = self._task["trajectory"]

    def sample_task(self) -> TaskType:
        # Initialize task dict
        task_dict = OrderedDict()

        # Define sampling method for convenience
        rg = self.rg if hasattr(self, 'rg') else None
        rand = partial(sample, rg=rg)

        # Sample robot model parameters
        if "robot" in self.active_task_features:
            task_dict["robot"] = OrderedDict(
                L_x_F_min=rand(*L_X_F_MIN_RANGE, shape=(1,)),
                L_x_F_max=rand(*L_X_F_MAX_RANGE, shape=(1,)),
                L_y_F=rand(*L_Y_F_RANGE, shape=(1,)),
                I_R_xx=rand(*I_R_XX_RANGE, shape=(1,)),
                I_R_xz=rand(*I_R_XZ_RANGE, shape=(1,)),
                I_R_yy=rand(*I_R_YY_RANGE, shape=(1,)),
                I_R_zz=rand(*I_R_ZZ_RANGE, shape=(1,)),
                z_R=rand(*Z_R_RANGE, shape=(1,)),
                F_x_R_max=rand(*F_X_R_MAX_RANGE, shape=(1,)),
                F_y_R_max=rand(*F_Y_R_MAX_RANGE, shape=(1,)),
                v_x_R_max=rand(*V_X_R_MAX_RANGE, shape=(1,)),
                v_y_R_max=rand(*V_Y_R_MAX_RANGE, shape=(1,)))

        # Sample flexibility dynamics
        if "flexibility" in self.active_task_features:
            task_dict["flexibility"] = OrderedDict(
                stiffness=rand(
                    *FLEX_STIFFNESS_LOG_RANGE, enable_log_scale=True,
                    shape=(3,)),
                damping_ratio=rand(*FLEX_DAMPING_RATIO_RANGE, shape=(3,)))

        if "patient" in self.active_task_features:
            # Sample patient model parameters
            task_dict["patient"] = OrderedDict(
                m_P_l=rand(*M_P_L_RANGE, shape=(1,)),
                x_P=rand(*X_P_RANGE, shape=(1,)),
                I_P_l_xx=rand(*I_P_L_XX_RANGE, shape=(1,)),
                I_P_l_yy=rand(*I_P_L_YY_RANGE, shape=(1,)),
                m_P_u=rand(*M_P_U_RANGE, shape=(1,)),
                I_P_u_xx=rand(*I_P_U_XX_RANGE, shape=(1,)),
                I_P_u_yy=rand(*I_P_U_YY_RANGE, shape=(1,)),
                I_P_u_zz=rand(*I_P_U_ZZ_RANGE, shape=(1,)),
                z_P=rand(*Z_P_RANGE, shape=(1,)),
                L_x_P_min=rand(*L_X_P_MIN_RANGE, shape=(1,)),
                L_x_P_max=rand(*L_X_P_MAX_RANGE, shape=(1,)),
                L_y_P=rand(*L_Y_P_RANGE, shape=(1,)),
                F_x_P_max=rand(*F_X_P_MAX_RANGE, shape=(1,)),
                F_y_P_max=rand(*F_Y_P_MAX_RANGE, shape=(1,)),
                v_x_P_max=rand(*V_X_P_MAX_RANGE, shape=(1,)),
                v_y_P_max=rand(*V_Y_P_MAX_RANGE, shape=(1,)))

        if "coupling" in self.active_task_features:
            # Sample straps dynamics parameters
            task_dict["coupling"] = OrderedDict(
                stiffness=rand(
                    *COUPLING_STIFFNESS_LOG_RANGE, enable_log_scale=True,
                    shape=(1,)),
                damping_ratio=rand(*COUPLING_DAMPING_RATIO_RANGE, shape=(1,)))

        if "behavior" in self.active_task_features:
            # Sample patient dynamics parameters
            task_dict["behavior"] = OrderedDict(
                kp=rand(
                    *PATIENT_KP_LOG_RANGE, enable_log_scale=True, shape=(1,)),
                kd_ratio=rand(*PATIENT_KD_RATIO_RANGE, shape=(1,)))

        if "trajectory" in self.active_task_features:
            # Sample trajectory parameters

            ### Get footprint area
            L_x_F_min = float(task_dict["robot"]["L_x_F_min"])
            L_x_F_max = float(task_dict["robot"]["L_x_F_max"])
            L_x_F = 0.5 * (L_x_F_min + L_x_F_max) - TRAJ_MARGINS[0]
            L_y_F = float(task_dict["robot"]["L_y_F"]) - TRAJ_MARGINS[1]

            ### Sample ellipsoid
            L_x_T = rand(0.0, L_x_F, shape=(1,))
            L_y_T = rand(0.0, L_y_F, shape=(1,))
            beta = rand(scale=np.pi/4.0, shape=(1,))

            ### Compute axis-aligned bounding box
            t_nil = math.atan2(-L_x_T * math.sin(beta), L_y_T * math.cos(beta))
            t_inf = math.atan2(L_x_T * math.cos(beta), L_y_T * math.sin(beta))
            B_x_T = L_x_T * math.sin(t_inf) * math.cos(beta) + \
                    L_y_T * math.cos(t_inf) * math.sin(beta)
            B_y_T = L_y_T * math.cos(t_nil) * math.cos(beta) - \
                    L_x_T * math.sin(t_nil) * math.sin(beta)

            ### Shrink the ellipsoid to fit the footprint area
            if B_y_T - L_y_F > 0:
                L_x_T *= L_y_F / B_y_T
                L_y_T *= L_y_F / B_y_T
                B_x_T *= L_y_F / B_y_T
                B_y_T = L_y_F
            if B_x_T - L_x_F > 0:
                L_x_T *= L_x_F / B_x_T
                L_y_T *= L_x_F / B_x_T
                B_y_T *= L_x_F / B_x_T
                B_x_T = L_x_F

            # Sample trajectory center and velocity
            x_T = 0.5 * (L_x_F_max - L_x_F_min) + rand(
                scale=float(L_x_F - B_x_T), shape=(1,))
            y_T = rand(scale=float(L_y_F - B_y_T), shape=(1,))
            v_T = rand(0.0, TRAJ_V_MAX, shape=(1,))
        else:
            L_x_T = np.array([0.5 * (0.5 * (
                L_X_F_MIN_RANGE[0] + L_X_F_MAX_RANGE[0]) - TRAJ_MARGINS[0])])
            L_y_T = np.array([0.5 * (L_Y_F_RANGE[0] - TRAJ_MARGINS[1])])
            beta = np.array([0.0])
            x_T = np.array([0.5 * (L_X_F_MAX_RANGE[0] - L_X_F_MIN_RANGE[0])])
            y_T = np.array([0.0])
            v_T = np.array([0.5 * TRAJ_V_MAX])

        task_dict["trajectory"] = OrderedDict(
            L_x_T=L_x_T, L_y_T=L_y_T, beta=beta, x_T=x_T, y_T=y_T, v_T=v_T)

        return task_dict

    def sample_tasks(self, n_tasks) -> List[TaskType]:
        return [self.sample_task() for _ in range(n_tasks)]

    def set_task(self, task_dict: TaskType) -> None:
        # Backup task
        set_value(self._task, task_dict)

        # Generate URDF by updating parameters
        if any(k in self.active_task_features for k in ("robot", "patient")):
            model_dict = OrderedDict(has_patient=self.has_patient)
            if "robot" in self.active_task_features:
                model_dict.update(task_dict["robot"])
            if "patient" in self.active_task_features:
                model_dict.update(task_dict["patient"])
            _generate_model(
                self._xacro_path, self._hardware_path, self.has_flexiblity,
                model_dict, self.simulator, debug=self.debug)

        # Compute centroidal dynamics
        pin.ccrba(self.robot.pinocchio_model,
                  self.robot.pinocchio_data,
                  pin.neutral(self.robot.pinocchio_model),
                  np.zeros(self.robot.pinocchio_model.nv))

        # Set patient dynamics parameters
        if "patient" in self.active_task_features:
            self._patient_offset = np.array([
                float(task_dict["patient"]["x_P"]), 0.0])

        # Set coupling dynamics parameters
        if "coupling" in self.active_task_features:
            self._coupling_stiffness = float(
                task_dict["coupling"]["stiffness"])
            coupling_mass = np.mean([
                self.robot.pinocchio_model.inertias[i].mass
                for i in self._coupling_joints_idx])
            coupling_damping_critic = 2.0 * np.sqrt(
                coupling_mass *  self._coupling_stiffness)
            self._coupling_damping = coupling_damping_critic * \
                float(task_dict["coupling"]["damping_ratio"])

        # Set patient behavior parameters
        if "behavior" in self.active_task_features:
            self._patient_kp = float(task_dict["behavior"]["kp"])
            patient_upper_body_mass = self.robot.pinocchio_model.inertias[
                self._patient_joint_idx].mass
            kd_patient_critic = 2.0 * np.sqrt(
                patient_upper_body_mass * self._patient_kp)
            self._patient_kd = float(
                kd_patient_critic * task_dict["behavior"]["kd_ratio"])

        # Set flexibility dynamics
        if "flexibility" in self.active_task_features:
            flex_stiffness = task_dict["flexibility"]["stiffness"]
            flex_damping_critic = 2.0 * np.sqrt(
                np.diag(self.robot.pinocchio_data.Ig.inertia) * flex_stiffness)
            flex_damping = \
                flex_damping_critic * task_dict["flexibility"]["damping_ratio"]
            model_options = self.robot.get_model_options()
            model_options['dynamics']['flexibilityConfig'][0].update({
                'stiffness': flex_stiffness, 'damping': flex_damping})
            self.robot.set_model_options(model_options)

        # Reset the environment if not auto sampling
        if not self.auto_sampling:
            self.reset()

    def get_task(self) -> TaskType:
        return self._task

    def _setup(self) -> None:
        if self.auto_sampling:
            self.set_task(self.sample_task())
        super()._setup()

    def compute_reward(self,  # type: ignore[override]
                       *, info: Dict[str, Any]) -> float:
        # pylint: disable=arguments-differ

        reward = 0.0  #1.0e-1 * super().compute_reward(info=info)
        if self.stepper_state.t < 0.2:
            pass  # Do not return anything during first steps
        elif self._num_steps_beyond_done is None:
            # Get CoP position
            F_foot = self.sensors_data['ForceSensor'][:, 0]
            x_cop, y_cop = -F_foot[4] / F_foot[2], F_foot[3] / F_foot[2]

            # Extract trajectory specification
            x_T = float(self._task["trajectory"]["x_T"])
            y_T = float(self._task["trajectory"]["y_T"])
            L_x_T = float(self._task["trajectory"]["L_x_T"])
            L_y_T = float(self._task["trajectory"]["L_y_T"])
            beta = self._task["trajectory"]["beta"]
            v_T = self._task["trajectory"]["v_T"]

            # Compute closest point on trajectory
            t = self.stepper_state.t
            eps = np.finfo(np.float32).resolution
            R = np.array([[math.cos(beta), -math.sin(beta)],
                          [math.sin(beta),  math.cos(beta)]])
            x_ref, y_ref = np.array([x_T, y_T]) + R @ np.array([
                L_x_T * math.cos(v_T * t) if L_x_T > eps else 0.0,
                L_y_T * math.sin(v_T * t) if L_y_T > eps else 0.0])

            err_rel = np.linalg.norm([
                (x_cop - x_ref) / (L_X_F_MIN_RANGE[1] + L_X_F_MAX_RANGE[1]),
                (y_cop - y_ref) / (2.0 * L_Y_F_RANGE[1])])
            reward += np.clip(1.0 - err_rel ** 2, 1.0e-2, 1.0)

        return reward

    def _refresh_observation_space(self) -> None:
        """Configure the observation of the environment.

        Only the state is observable, while by default, the current time,
        state, and sensors data are available.
        """
        # Call base implementation
        super()._refresh_observation_space()

        # Extract some proxies
        L_x_F_min = L_X_F_MIN_RANGE[1] - TRAJ_MARGINS[0]
        L_x_F_max = L_X_F_MAX_RANGE[1] - TRAJ_MARGINS[0]
        L_x_F = 0.5 * (L_x_F_min + L_x_F_max)
        L_y_F = L_Y_F_RANGE[1] - TRAJ_MARGINS[1]

        # Append trajectory task space to observation space
        self.observation_space.spaces['trajectory'] = spaces.Dict(OrderedDict(
            L_x_T=spaces.Box(
                low=0.0, high=L_x_F, shape=(1,), dtype=np.float32),
            L_y_T=spaces.Box(
                low=0.0, high=L_y_F, shape=(1,), dtype=np.float32),
            beta=spaces.Box(
                low=-np.pi/4.0, high=np.pi/4.0, shape=(1,), dtype=np.float32),
            x_T=spaces.Box(
                low=-L_x_F_min, high=L_x_F_max, shape=(1,), dtype=np.float32),
            y_T=spaces.Box(
                low=-L_y_F, high=L_y_F, shape=(1,), dtype=np.float32),
            v_T=spaces.Box(
                low=0.0, high=TRAJ_V_MAX, shape=(1,), dtype=np.float32)))
