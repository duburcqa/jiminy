## @file

import os
import toml
import copy
import pathlib
import logging
import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Optional, Callable

from . import core as jiminy
from .engine_asynchronous import EngineAsynchronous

from .core import (EncoderSensor as encoder,
                   EffortSensor as effort,
                   ForceSensor as force,
                   ImuSensor as imu)

import pinocchio as pin
from pinocchio.rpy import rpyToMatrix


DEFAULT_UPDATE_RATE = 1000.0


logger = logging.getLogger(__name__)


def generate_hardware_description_file(
        urdf_path : str,
        toml_path : Optional[str] = None,
        default_update_rate : Optional[float] = None):
    """
    @brief     Generate a default hardware description file, based on the
               information grabbed from the URDF when available, using best
               guess otherwise.

    @details   If no Gazebo plugin is available, a single IMU is added on the
               root body, and force sensors are added on every leaf of the
               robot kinematic tree. Otherwise, the definition of the plugins
               in use to infer them.

               Joint fields are parsed to extract the every joints, actuated
               or not. Fixed joints are not considered as actual joints.
               Transmission fields are parsed to determine which one of those
               joints are actuated. If no transmission is found, it is assumed
               that every joint is actuated, with a transmission ratio of 1:1.

               It is assumed that every joints have an encoder attached, as it
               is the case in Gazebo. Every actuated joint have an effort
               sensor attached by default.

               When the default update rate is unspecified, then the default
               sensor update rate is 1KHz if no Gazebo plugin has been found,
               otherwise the highest one among found plugins will be used.

    @remark    Under the hood, it is a configuration in TOML format to be as
               human-friendly as possible to reading and editing it.

    @param     urdf_path    Fullpath of the URDF file.
    @param     toml_path    Fullpath of the hardware description file.
                            Optional: By default, it is the same location than
                            the URDF file, using '.toml' extension.
    @param     default_update_rate    Default update rate of the sensors and
                                      the controller in Hz. It will be used
                                      for sensors whose the update rate is
                                      unspecified. 0.0 for continuous update.
                                      Optional: DEFAULT_UPDATE_RATE if no
                                      Gazebo plugin has been found, the lowest
                                      among the Gazebo plugins otherwise.
    """
    # Read the XML
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Initialize the hardware information
    hardware_info = OrderedDict()
    hardware_info['Global'] = OrderedDict()
    hardware_info['Motor'] = OrderedDict()
    hardware_info['Sensor'] = OrderedDict()

    # Parse the gazebo plugins, if any.
    # Note that it is only useful to extract "advanced" hardware, not basic
    # motors, encoders and effort sensors.
    gazebo_update_rate = None
    gazebo_plugins_found = root.find('gazebo') is not None
    for gazebo_plugin_descr in root.iterfind('gazebo'):
        body_name = gazebo_plugin_descr.get('reference')

        for gazebo_sensor_descr in gazebo_plugin_descr.iterfind('sensor'):
            sensor_info = OrderedDict(body_name=body_name)

            # Extract the sensor name
            sensor_name = gazebo_sensor_descr.get('name')

            # Extract the sensor type
            sensor_type = gazebo_sensor_descr.get('type').casefold()
            if 'imu' in sensor_type:
                sensor_type = imu.type
            elif 'contact' in sensor_type:
                sensor_type = force.type
            else:
                logger.warning(
                    f'Unsupported Gazebo sensor plugin {gazebo_sensor_descr}')
                continue

            # Extract the sensor update period
            update_rate = float(gazebo_sensor_descr.find('./update_rate').text)
            if gazebo_update_rate is None:
                gazebo_update_rate = update_rate
            else:
                if gazebo_update_rate != update_rate:
                    logger.warning("Jiminy does not support sensors with"
                        "different update rate. Using the highest one.")
                    gazebo_update_rate = update_rate

            # Extract the pose of the frame associate with the sensor.
            # Note that it is optional but usually defined since sensors
            # can only be attached to link in Gazebo, not to frame.
            frame_pose = gazebo_sensor_descr.find('./pose')
            if frame_pose is None:
                sensor_info['frame_pose'] = 6 * [0.0]
            else:
                sensor_info['frame_pose'] = list(
                    map(float, frame_pose.text.split()))

            # Add the sensor to the robot's hardware
            hardware_info['Sensor'].setdefault(sensor_type, {}).update(
                {sensor_name: sensor_info})

    # Fallback if no Gazebo plugin has been found
    if not gazebo_plugins_found:
        # Extract the list of parent and child links, excluding the one related
        # to fixed joints, because they are likely not "real" joint.
        parent_links = set()
        child_links = set()
        for joint_descr in root.findall('./joint'):
            if joint_descr.get('type').casefold() != 'fixed':
                parent_links.add(joint_descr.find('./parent').get('link'))
                child_links.add(joint_descr.find('./child').get('link'))

        # Compute the root link and the leaf ones
        root_links = list(parent_links.difference(child_links))
        leaf_links = list(child_links.difference(parent_links))

        # Add IMU sensor to the root link
        for root_link in root_links:
            hardware_info['Sensor'].setdefault(imu.type, {}).update({
                root_link: OrderedDict(
                    body_name=root_link,
                    frame_pose=6 * [0.0]
                )
            })

        # Add force sensors to the leaf links
        for leaf_link in leaf_links:
            hardware_info['Sensor'].setdefault(force.type, {}).update({
                leaf_link: OrderedDict(
                    body_name=root_link,
                    frame_pose=6 * [0.0]
                )
            })

    # Extract the effort sensors.
    # It is done by reading 'transmission' field, that is part of
    # URDF standard, so it should be available on any URDF file.
    transmission_found = root.find('transmission') is not None
    for transmission_descr in root.iterfind('transmission'):
        motor_info = OrderedDict()
        sensor_info = OrderedDict()

        # Extract the motor name
        motor_name = transmission_descr.find('./actuator').get('name')
        sensor_info['motor_name'] = motor_name

        # Extract the associated joint name
        joint_name = transmission_descr.find('./joint').get('name')
        motor_info['joint_name'] = joint_name

        # Extract the transmission ratio (motor / joint)
        ratio = transmission_descr.find('./actuator/mechanicalReduction')
        if ratio is None:
            motor_info['mechanicalReduction'] = 1
        else:
            motor_info['mechanicalReduction'] = float(ratio.text)

        # Extract the armature (rotor) inertia
        armature_inertia = transmission_descr.find(
            './actuator/motorInertia')
        if armature_inertia is None:
            motor_info['rotorInertia'] = 0.0
        else:
            motor_info['rotorInertia'] = float(armature_inertia.text)

        # Add the motor and sensor to the robot's hardware
        hardware_info['Motor'].setdefault('SimpleMotor', {}).update(
            {motor_name: motor_info})
        hardware_info['Sensor'].setdefault(effort.type, {}).update(
            {joint_name: sensor_info})

    # Define default encoder sensors, and default effort sensors if no
    # transmission available.
    for joint_descr in root.iterfind('joint'):
        encoder_info = OrderedDict()

        # Skip fixed joints
        joint_type = joint_descr.get('type').casefold()
        if joint_type == 'fixed':
            continue

        # Extract the joint name
        name = joint_descr.get('name')
        encoder_info['joint_name'] = name

        # Add the sensor to the robot's hardware
        hardware_info['Sensor'].setdefault(encoder.type, {}).update(
            {name: encoder_info})

        # Add motors to robot hardware by default if no transmission found
        if not transmission_found:
            hardware_info['Sensor'].setdefault('SimpleMotor', {}).update(
                {name: OrderedDict(
                    joint_name=name,
                    mechanicalReduction=1.0,
                    rotorInertia=0.0)
            })

    # Handling of default update rate for the controller and the sensors
    if gazebo_update_rate is None:
        if default_update_rate is not None:
            gazebo_update_rate = default_update_rate
        else:
            gazebo_update_rate = DEFAULT_UPDATE_RATE
    hardware_info['Global']['sensorsUpdatePeriod'] = gazebo_update_rate
    hardware_info['Global']['controllerUpdatePeriod'] = gazebo_update_rate

    # Write the sensor description file
    if toml_path is None:
        toml_path = pathlib.Path(urdf_path).with_suffix('.toml')
    with open(toml_path, 'w') as f:
        toml.dump(hardware_info, f)


class BaseJiminyRobot(jiminy.Robot):
    """
    @brief     Base class to instantiate a Jiminy robot based on a standard
               URDF file and Jiminy-specific hardware description file.

    @details   The utility 'generate_hardware_description_file' is provided to
               automatically generate a default hardware description file for
               any given URDF file. URDF file containing Gazebo plugins
               description should not require any further modification as it
               usually includes the information required to fully characterize
               the motors and sensors, along with some of there properties.

               Note that it is assumed that the contact points of the robot
               matches one-by-one the frames of the force sensors. So it is
               not possible to use non-instrumented contact points by default.
               Overload this class if you need finer-grained capability.

    @remark    hardware description files within the same directory and having
               the name than the URDF file will be detected automatically
               without requiring to manually specify its path.
    """
    def __init__(self):
        super().__init__()
        self.robot_options = None

    def initialize(self,
                   urdf_path : str,
                   toml_path : Optional[str] = None,
                   has_freeflyer : bool = True):
        # Initialize the robot without motors nor sensors
        return_code = super().initialize(urdf_path, has_freeflyer)

        if return_code != jiminy.hresult_t.SUCCESS:
            raise ValueError("Impossible to load the URDF file. "
                "Either the file is corrupted or does not exit.")

        # Load the hardware description file
        if toml_path is None:
            toml_path = pathlib.Path(urdf_path).with_suffix('.toml')
        if not os.path.exists(toml_path):
            raise FileNotFoundError("Hardware configuration file not found. "
                "Default file can be generated automatically using "
                "'generate_hardware_description_file' method.")
        hardware_info = toml.load(toml_path)
        global_info = hardware_info.pop('Global')
        motors_info = hardware_info.pop('Motor')
        sensors_info = hardware_info.pop('Sensor')

        # Add the motors to the robot
        for motor_type, motors_descr in motors_info.items():
            for motor_name, motor_descr in motors_descr.items():
                # Create the sensor and attach it
                motor = getattr(jiminy, motor_type)(motor_name)
                self.attach_motor(motor)

                # Initialize the motor
                joint_name = motor_descr.pop('joint_name')
                motor.initialize(joint_name)

                # Set the motor options
                options = motor.get_options()
                option_fields = options.keys()
                for name, value in motor_descr.items():
                    if name not in option_fields:
                        logger.warning(f"'{name}' is not a valid option for "
                            f"the motor {motor_name} of type {motor_type}.")
                    options[name] = value
                motor.set_options(options)

        # Add the sensors to the robot
        for sensor_type, sensors_descr in sensors_info.items():
            for sensor_name, sensor_descr in sensors_descr.items():
                # Create the sensor and attach it
                sensor = getattr(jiminy, sensor_type)(sensor_name)
                self.attach_sensor(sensor)

                # Initialize the sensor
                if sensor_type == encoder.type:
                    joint_name = sensor_descr.pop('joint_name')
                    sensor.initialize(joint_name)
                elif sensor_type == effort.type:
                    motor_name = sensor_descr.pop('motor_name')
                    sensor.initialize(motor_name)
                elif sensor_type in [force.type, imu.type]:
                    # Create the frame and add it to the robot model
                    body_name = sensor_descr.pop('body_name')

                    # Generate a frame name that is intelligible and available
                    i = 0
                    frame_name = sensor_name + "Frame"
                    while self.pinocchio_model.existFrame(frame_name):
                        frame_name = sensor_name + "Frame_%d" % i
                        i += 1

                    # Compute SE3 object representing the frame placement
                    frame_pose_xyzrpy = sensor_descr.pop('frame_pose')
                    frame_trans = np.array(frame_pose_xyzrpy[:3])
                    frame_rot = rpyToMatrix(frame_pose_xyzrpy[3:])
                    frame_placement = pin.SE3(frame_rot, frame_trans)

                    # Add the frame to the robot model
                    self.add_frame(frame_name, body_name, frame_placement)

                    # Initialize the sensor
                    sensor.initialize(frame_name)
                else:
                    raise ValueError(
                        f"Unsupported sensor type {sensor_type}.")

                # Set the sensor options
                options = sensor.get_options()
                option_fields = options.keys()
                for name, value in sensor_descr.items():
                    if name not in option_fields:
                        logger.warning(f"'{name}' is not a valid option for "
                            f"the sensor {sensor_name} of type {sensor_type}.")
                    options[name] = value
                sensor.set_options(options)

        # Add the contact points
        force_sensor_frame_names = [self.get_sensor(force.type, e).frame_name
                                    for e in self.sensors_names[force.type]]
        self.add_contact_points(force_sensor_frame_names)

    def set_model_options(self, model_options):
        super().set_model_options(model_options)
        self.robot_options = copy.deepcopy(model_options)

    def get_model_options(self):
        if self.is_initialized:
            return self.robot_options
        else:
            return super().get_model_options()

    def set_options(self, options):
        super().set_options(options)
        self.robot_options = copy.deepcopy(options["model"])

    def get_options(self):
        options = super().get_options()
        if self.robot_options is not None:
            options["model"] = self.robot_options
        return options


class BaseJiminyController(jiminy.ControllerFunctor):
    """
    @brief     Base class to instantiate a Jiminy controller based on a
               callable function.

    @details   This class is primarily helpful for those who want to
               implement a custom internal dynamics with hysteresis for
               prototyping.
    """
    def __init__(self, compute_command_fn: Callable):
        self.__robot = None
        super().__init__(compute_command_fn, self.internal_dynamics)

    def initialize(self, robot: BaseJiminyRobot):
        self.__robot = robot
        return_code = super().initialize(self.__robot)

        if return_code == jiminy.hresult_t.SUCCESS:
            raise ValueError("Impossible to instantiate the controller. "
                "There is something wrong with the robot.")

    def internal_dynamics(self, t, q, v, sensors_data, uCommand):
        """
        @brief     Internal dynamics of the robot.

        @details   Overload this method to implement a custom internal dynamics
                   for the robot. Note that is results in an overhead of about
                   100% of the simulation in most cases, which is not often not
                   acceptable in production, but still useful for prototyping.

                   One way to solve this issue would be to compile it using
                   CPython.

        @remark   This method is time-continuous as it is designed to implement
                  physical laws.
        """
        pass


class BaseJiminyEngine(EngineAsynchronous):
    def __init__(self,
                 urdf_path: str,
                 toml_path : Optional[str] = None,
                 has_freeflyer : bool = True,
                 use_theoretical_model: bool = False,
                 viewer_backend: Optional[str] = None):
        # Instantiate and initialize the robot
        robot = BaseJiminyRobot()
        robot.initialize(urdf_path, toml_path, has_freeflyer)

        # Instantiate the controller (initialization is managed by the engine)
        controller = BaseJiminyController(self._send_command)

        # Instantiate and initialize the engine
        engine = jiminy.Engine()
        super().__init__(
            robot,
            controller,
            engine,
            use_theoretical_model,
            viewer_backend
        )
