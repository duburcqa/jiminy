import os
import re
import toml
import logging
import pathlib
import tempfile
import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from typing import Optional, Dict, Any

import trimesh

from . import core as jiminy
from .core import (EncoderSensor as encoder,
                   EffortSensor as effort,
                   ContactSensor as contact,
                   ForceSensor as force,
                   ImuSensor as imu)

import hppfcl
import pinocchio as pin
from pinocchio.rpy import rpyToMatrix


DEFAULT_UPDATE_RATE = 1000.0  # [Hz]
DEFAULT_FRICTION_DRY_SLOPE = 0.0


class _DuplicateFilter:
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv


logger = logging.getLogger(__name__)
logger.addFilter(_DuplicateFilter())


def _gcd(a: float,
         b: float,
         rtol: float = 1.0e-05,
         atol: float = 1.0e-08) -> float:
    """Compute the greatest common divisor of two float numbers.
    """
    t = min(abs(a), abs(b))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return a


def _fix_urdf_mesh_path(urdf_path: str,
                        mesh_path: str,
                        output_root_path: Optional[str] = None):
    """Generate an URDF with updated mesh paths.

    :param urdf_path: Full path of the URDF file.
    :param mesh_path: Root path of the meshes.
    :param output_root_path: Root directory of the fixed URDF file.
                             Optional: temporary directory by default.

    :returns: Full path of the fixed URDF file.
    """
    # Extract all the mesh path that are not package path, continue if any
    with open(urdf_path, 'r') as urdf_file:
        urdf_contents = urdf_file.read()
    mesh_tag = "<mesh filename="
    pathlists = {
        filename
        for filename in re.findall(mesh_tag + '"(.*)"', urdf_contents)
        if not filename.startswith('package://')}
    if not pathlists:
        return urdf_path

    # If mesh root path already matching, then nothing to do
    if len(pathlists) > 1:
        if all(path.startswith('.') for path in pathlists):
            mesh_path_orig = '.'
        else:
            mesh_path_orig = os.path.commonpath(pathlists)
    else:
        mesh_path_orig = os.path.dirname(next(iter(pathlists)))
    if mesh_path == mesh_path_orig:
        return urdf_path

    # Create the output directory
    if output_root_path is None:
        output_root_path = tempfile.mkdtemp()
    fixed_urdf_dir = os.path.join(
        output_root_path, "fixed_urdf" + mesh_path.translate(
            str.maketrans({k: '_' for k in '/:'})))
    os.makedirs(fixed_urdf_dir, exist_ok=True)
    fixed_urdf_path = os.path.join(
        fixed_urdf_dir, os.path.basename(urdf_path))

    # Override the root mesh path with the desired one
    urdf_contents = urdf_contents.replace(
        '"'.join((mesh_tag, mesh_path_orig)),
        '"'.join((mesh_tag, mesh_path)))
    with open(fixed_urdf_path, 'w') as f:
        f.write(urdf_contents)

    return fixed_urdf_path


def generate_default_hardware_description_file(
        urdf_path: str,
        hardware_path: Optional[str] = None,
        default_update_rate: Optional[float] = DEFAULT_UPDATE_RATE,
        verbose: bool = True):
    """Generate a default hardware description file, based on the information
    grabbed from the URDF when available, using educated guess otherwise.

    If no IMU sensor is found, a single one is added on the root body of the
    kinematic tree. If no Gazebo plugin is available, collision bodies and
    force sensors are added on every leaf body of the robot. Otherwise, the
    definition of the plugins in use to infer them.

    'joint' fields are parsed to extract every joint, actuated or not. 'fixed'
    joints are not considered as actual joints. Transmission fields are parsed
    to determine which one of those joints are actuated. If no transmission is
    found, it is assumed that every joint is actuated, with a transmission
    ratio of 1:1.

    It is assumed that:

    - every joint has an encoder attached,
    - every actuated joint has an effort sensor attached,
    - every collision body has a force sensor attached
    - for every Gazebo contact sensor, the associated body is added
      to the set of the collision bodies, but it is not the case
      for Gazebo force plugin.

    When the default update rate is unspecified, then the default
    sensor update rate is 1KHz if no Gazebo plugin has been found,
    otherwise the highest one among found plugins will be used.

    .. note::
        It has been primarily designed for robots with freeflyer. The default
        configuration should work out-of-the-box for walking robot, but
        substantial modification may be required for different types of robots
        such as wheeled robots or robotics manipulator arms.

    .. note::
        TOML format as be chosen to make reading and manually editing of the
        file as human-friendly as possible.

    :param urdf_path: Fullpath of the URDF file.
    :param hardware_path: Fullpath of the hardware description file.
                          Optional: By default, it is the same location than
                          the URDF file, using '*_hardware.toml' extension.
    :param default_update_rate: Default update rate of the sensors and the
                                controller in Hz. It will be used for sensors
                                whose the update rate is unspecified. 0.0 for
                                continuous update.
                                Optional: DEFAULT_UPDATE_RATE if no Gazebo
                                plugin has been found, the lowest among the
                                Gazebo plugins otherwise.
    :param verbose: Whether or not to print warnings.
    """
    # Handle verbosity level
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)

    # Read the XML
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Initialize the hardware information
    hardware_info = OrderedDict(
        Global=OrderedDict(
            sensorsUpdatePeriod=1.0/default_update_rate,
            controllerUpdatePeriod=1.0/default_update_rate,
            collisionBodiesNames=[],
            contactFramesNames=[]
        ),
        Motor=OrderedDict(),
        Sensor=OrderedDict()
    )

    # Extract the root link. It is the one having no parent at all.
    links = set()
    for link_descr in root.findall('./link'):
        links.add(link_descr.attrib["name"])
    for joint_descr in root.findall('./joint'):
        links.remove(joint_descr.find('./child').get('link'))
    link_root = next(iter(links))

    # Extract the list of parent and child links, excluding the one related
    # to fixed link not having collision geometry, because they are likely not
    # "real" joint.
    links_parent = set()
    links_child = set()
    for joint_descr in root.findall('./joint'):
        parent_link = joint_descr.find('./parent').get('link')
        child_link = joint_descr.find('./child').get('link')
        if joint_descr.get('type').casefold() != 'fixed' or root.find(
                f"./link[@name='{child_link}']/collision") is not None:
            links_parent.add(parent_link)
            links_child.add(child_link)

    # Determine leaf links. If there is no parent, then use root link instead.
    if links_parent:
        links_leaf = sorted(list(links_child.difference(links_parent)))
    else:
        links_leaf = [link_root]

    # Parse the gazebo plugins, if any.
    # Note that it is only useful to extract "advanced" hardware, not basic
    # motors, encoders and effort sensors.
    gazebo_ground_stiffness = None
    gazebo_ground_damping = None
    gazebo_update_rate = None
    collision_bodies_names = set()
    gazebo_plugins_found = root.find('gazebo') is not None
    for gazebo_plugin_descr in root.iterfind('gazebo'):
        body_name = gazebo_plugin_descr.get('reference')

        # Extract sensors
        for gazebo_sensor_descr in gazebo_plugin_descr.iterfind('sensor'):
            sensor_info = OrderedDict(body_name=body_name)

            # Extract the sensor name
            sensor_name = gazebo_sensor_descr.get('name')

            # Extract the sensor type
            sensor_type = gazebo_sensor_descr.get('type').casefold()
            if 'imu' in sensor_type:
                sensor_type = imu.type
            elif 'contact' in sensor_type:
                collision_bodies_names.add(body_name)
                sensor_type = force.type
            else:
                logger.warning(
                    "Unsupported Gazebo sensor plugin of type "
                    f"'{sensor_type}'")
                continue

            # Extract the sensor update period
            update_rate = float(gazebo_sensor_descr.find('./update_rate').text)
            if gazebo_update_rate is None:
                gazebo_update_rate = update_rate
            else:
                if gazebo_update_rate != update_rate:
                    logger.warning(
                        "Jiminy does not support sensors with different "
                        "update rate. Using greatest common divisor instead.")
                    gazebo_update_rate = _gcd(gazebo_update_rate, update_rate)

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

        # Extract the collision bodies and ground model, then add force
        # sensors. At this point, every force sensor is associated with a
        # collision body.
        if gazebo_plugin_descr.find('kp') is not None:
            # Add a force sensor, if not already in the collision set
            if body_name not in collision_bodies_names:
                hardware_info['Sensor'].setdefault(force.type, {}).update({
                    f"{body_name}Contact": OrderedDict(
                        body_name=body_name,
                        frame_pose=6*[0.0])
                })

            # Add the related body to the collision set
            collision_bodies_names.add(body_name)

            # Update the ground model
            ground_stiffness = float(gazebo_plugin_descr.find('kp').text)
            ground_damping = float(gazebo_plugin_descr.find('kd').text)
            if gazebo_ground_stiffness is None:
                gazebo_ground_stiffness = ground_stiffness
            if gazebo_ground_damping is None:
                gazebo_ground_damping = ground_damping
            if (gazebo_ground_stiffness != ground_stiffness or
                    gazebo_ground_damping != ground_damping):
                raise RuntimeError(
                    "Jiminy does not support contacts with different ground "
                    "models.")

        # Extract plugins not wrapped into a sensor
        for gazebo_plugin_descr in gazebo_plugin_descr.iterfind('plugin'):
            plugin = gazebo_plugin_descr.get('filename')
            if plugin == "libgazebo_ros_force.so":
                body_name = gazebo_plugin_descr.find('bodyName').text
                hardware_info['Sensor'].setdefault(force.type, {}).update({
                    f"{body_name}Wrench": OrderedDict(
                        body_name=body_name,
                        frame_pose=6*[0.0])
                })
            else:
                logger.warning(f"Unsupported Gazebo plugin '{plugin}'")

    # Add IMU sensor to the root link if no Gazebo IMU sensor has been found
    if link_root and imu.type not in hardware_info['Sensor'].keys():
        hardware_info['Sensor'].setdefault(imu.type, {}).update({
            link_root: OrderedDict(
                body_name=link_root,
                frame_pose=6*[0.0])
        })

    # Add force sensors and collision bodies if no Gazebo plugin is available
    if not gazebo_plugins_found:
        for link_leaf in links_leaf:
            # Add a force sensor
            hardware_info['Sensor'].setdefault(force.type, {}).update({
                link_leaf: OrderedDict(
                    body_name=link_leaf,
                    frame_pose=6*[0.0])
            })

            # Add the related body to the collision set if possible
            if root.find(f"./link[@name='{link_leaf}']/collision") is not None:
                collision_bodies_names.add(link_leaf)

    # Specify collision bodies and ground model in global config options
    hardware_info['Global']['collisionBodiesNames'] = \
        sorted(list(collision_bodies_names))
    if gazebo_ground_stiffness is not None:
        hardware_info['Global']['groundStiffness'] = gazebo_ground_stiffness
    if gazebo_ground_damping is not None:
        hardware_info['Global']['groundDamping'] = gazebo_ground_damping

    # Extract joint dynamics properties, namely 'friction' and 'damping'
    joints_options = {}
    for joint_descr in root.findall("./joint"):
        if joint_descr.get('type').casefold() == 'fixed':
            continue
        joint_name = joint_descr.get('name')
        dyn_descr = joint_descr.find('./dynamics')
        if dyn_descr is not None:
            damping = float(dyn_descr.get('damping') or 0.0)
            friction = float(dyn_descr.get('friction') or 0.0)
        else:
            damping, friction = 0.0, 0.0
        joints_options[joint_name] = OrderedDict(
            frictionViscousPositive=-damping,
            frictionViscousNegative=-damping,
            frictionDryPositive=-friction,
            frictionDryNegative=-friction,
            frictionDrySlope=-DEFAULT_FRICTION_DRY_SLOPE)

    # Extract the motors and effort sensors.
    # It is done by reading 'transmission' field, that is part of
    # URDF standard, so it should be available on any URDF file.
    transmission_found = root.find('transmission') is not None
    for transmission_descr in root.iterfind('transmission'):
        # Initialize motor and sensor info
        motor_info, sensor_info = OrderedDict(), OrderedDict()

        # Check that the transmission type is supported
        transmission_name = transmission_descr.get('name')
        transmission_type_descr = transmission_descr.find('./type')
        if transmission_type_descr is not None:
            transmission_type = transmission_type_descr.text
        else:
            transmission_type = transmission_descr.get('type')
        transmission_type = os.path.basename(transmission_type).casefold()
        if transmission_type != 'simpletransmission':
            logger.warning(
                "Jiminy only support SimpleTransmission for now. Skipping"
                f"transmission {transmission_name} of type "
                f"{transmission_type}.")
            continue

        # Extract the motor name
        motor_name = transmission_descr.find('./actuator').get('name')
        sensor_info['motor_name'] = motor_name

        # Extract the associated joint name.
        joint_name = transmission_descr.find('./joint').get('name')
        motor_info['joint_name'] = joint_name

        # Make sure that the joint is revolute
        joint_type = root.find(
            f"./joint[@name='{joint_name}']").get("type").casefold()
        if joint_type not in ["revolute", "continuous", "prismatic"]:
            logger.warning(
                "Jiminy only support 1-dof joint actuators and effort "
                f"sensors. Attached joint cannot of type '{joint_type}'.")
            continue

        # Extract the transmission ratio (motor / joint)
        ratio = transmission_descr.find('./mechanicalReduction')
        if ratio is None:
            motor_info['mechanicalReduction'] = 1
        else:
            motor_info['mechanicalReduction'] = float(ratio.text)

        # Extract the armature (rotor) inertia
        armature = transmission_descr.find('./motorInertia')
        if armature is None:
            motor_info['armature'] = 0.0
        else:
            motor_info['armature'] = \
                float(armature.text) * motor_info['mechanicalReduction'] ** 2

        # Add dynamics property to motor info, if any
        motor_info.update(joints_options.pop(joint_name))

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
        joint_name = joint_descr.get('name')
        encoder_info['joint_name'] = joint_name

        # Add the sensor to the robot's hardware
        hardware_info['Sensor'].setdefault(encoder.type, {}).update(
            {joint_name: encoder_info})

        # Add motors to robot hardware by default if no transmission found
        if not transmission_found:
            hardware_info['Motor'].setdefault('SimpleMotor', {}).update(
                {joint_name: OrderedDict(
                    [('joint_name', joint_name),
                     ('mechanicalReduction', 1.0),
                     ('armature', 0.0),
                     *joints_options.pop(joint_name).items()
                     ])})
            hardware_info['Sensor'].setdefault(effort.type, {}).update(
                {joint_name: OrderedDict(
                    motor_name=joint_name)})

    # Warn if friction model has been defined for non-actuated joints
    if joints_options:
        logger.warning(
            "Jiminy only support friction model for actuated joint.")

    # Specify custom update rate for the controller and the sensors, if any
    if gazebo_update_rate is not None:
        hardware_info['Global']['sensorsUpdatePeriod'] = \
            1.0 / gazebo_update_rate
        hardware_info['Global']['controllerUpdatePeriod'] = \
            1.0 / gazebo_update_rate

    # Write the sensor description file
    if hardware_path is None:
        hardware_path = str(pathlib.Path(
            urdf_path).with_suffix('')) + '_hardware.toml'
    with open(hardware_path, 'w') as f:
        toml.dump(hardware_info, f)


def load_hardware_description_file(
        robot: jiminy.Robot,
        hardware_path: str,
        avoid_instable_collisions: bool = True,
        verbose: bool = True) -> Dict[str, Any]:
    """Load hardware configuration file.

    If no collision geometry is associated with the body requiring collision
    handling, then the visual geometry is used instead, if any. If none is
    available despite at, then a single contact point is added at body frame.

    For now, every mesh used for collision are replaced by the vertices of the
    associated minimum volume bounding box, to avoid numerical instabilities.

    :param robot: Jiminy robot.
    :param hardware_path: Path of Jiminy hardware description toml file.
    :param avoid_instable_collisions: Prevent numerical instabilities by
                                      replacing collision mesh by vertices of
                                      associated minimal volume bounding box,
                                      and primitive box by its vertices.
    :param verbose: Whether or not to print warnings.

    :returns: Unused information available in hardware configuration file.
    """
    # Handle verbosity level
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)

    hardware_info = toml.load(hardware_path)
    extra_info = hardware_info.pop('Global', {})
    motors_info = hardware_info.pop('Motor', {})
    sensors_info = hardware_info.pop('Sensor', {})

    # Extract the list of bodies having visual and collision meshes or
    # primitives.
    geometry_types = {
        geom_type: {'primitive': set(), 'mesh': set()}
        for geom_type in ('collision', 'visual')}
    geometry_specs = {
        geom_type: {
            'primitive': defaultdict(lambda: []),
            'mesh': defaultdict(lambda: [])}
        for geom_type in ('collision', 'visual')}
    for geom_model, geometry_types_i, geometry_specs_i in zip(
            (robot.collision_model, robot.visual_model),
            geometry_types.values(),
            geometry_specs.values()):
        for geometry_object in geom_model.geometryObjects:
            frame_idx = geometry_object.parentFrame
            frame_name = robot.pinocchio_model.frames[frame_idx].name
            mesh_path = geometry_object.meshPath
            is_mesh = any(char in mesh_path for char in ('\\', '/', '.'))
            geom_type = 'mesh' if is_mesh else 'primitive'
            geometry_types_i[geom_type].add(frame_name)
            geometry_specs_i[geom_type][frame_name].append(geometry_object)

    # Checking the collision bodies, to make sure they are associated with
    # supported collision geometries. If not, fixing the issue after
    # throwing a warning.
    collision_bodies_names = extra_info.pop(
        'collisionBodiesNames', [])
    contact_frames_names = extra_info.pop('contactFramesNames', [])
    for body_name in collision_bodies_names.copy():
        # Filter out the different cases.
        # After this filter, we know that their is no collision geometry
        # associated with the body but their is a visual mesh, or there is
        # only collision meshes.
        if body_name in geometry_types['collision']['mesh'] and \
                body_name in geometry_types['collision']['primitive']:
            if not avoid_instable_collisions:
                continue
            logger.warning(
                "Collision body having both primitive and mesh geometries "
                "is not supported. Enabling only primitive collision for "
                "this body.")
            continue
        elif body_name in geometry_types['collision']['primitive']:
            pass
        elif body_name in geometry_types['collision']['mesh']:
            if not avoid_instable_collisions:
                continue
            logger.warning(
                "Collision body associated with mesh geometry is not "
                "supported for now. Replacing it by contact points at the "
                "vertices of the minimal volume bounding box.")
        elif body_name not in geometry_types['visual']['mesh']:
            logger.warning(
                "No visual mesh nor collision geometry associated with "
                f"collision body '{body_name}'. Fallback to adding a single "
                "contact point at body frame.")
            contact_frames_names.append(body_name)
            continue
        else:
            logger.warning(
                "No collision geometry associated with the collision "
                f"body '{body_name}'. Fallback to replacing it by contact "
                "points at the vertices of the minimal volume bounding box of "
                "the available visual meshes.")

        # Check if collision primitive box are available
        collision_boxes_size, collision_boxes_origin = [], []
        for geometry_object in \
                geometry_specs['collision']['primitive'][body_name]:
            geom = geometry_object.geometry
            if isinstance(geom, hppfcl.Box):
                collision_boxes_size.append(2.0 * geom.halfSide)
                collision_boxes_origin.append(geometry_object.placement)

        # Replace the collision boxes by contact points, if any
        if collision_boxes_size:
            if not avoid_instable_collisions:
                continue
            logger.warning(
                "Collision body associated with box geometry is not "
                "numerically stable for now. Replacing it by contact "
                "points at the vertices.")

            for i, (box_size, box_origin) in enumerate(zip(
                    collision_boxes_size, collision_boxes_origin)):
                vertices = [e.flatten() for e in np.meshgrid(*[
                    0.5 * v * np.array([-1.0, 1.0]) for v in box_size])]
                for j, (x, y, z) in enumerate(zip(*vertices)):
                    frame_name = "_".join((
                        body_name, "CollisionBox", str(i), str(j)))
                    vertex_pos_rel = pin.SE3(
                        np.eye(3), np.array([x, y, z]))
                    frame_transform = box_origin.act(vertex_pos_rel)
                    robot.add_frame(frame_name, body_name, frame_transform)
                    contact_frames_names.append(frame_name)
        elif body_name in geometry_types['collision']['primitive']:
            # Do nothing if the primitive is not a box. It should be fine.
            continue

        # Remove the body from the collision detection set
        collision_bodies_names.remove(body_name)

        # Early return if collision box primitives have been replaced
        if collision_boxes_size:
            continue

        # Replace the collision bodies by contact points, falling back to
        # visual bodies if none.
        for geometry_object in (
                geometry_specs['collision']['mesh'][body_name] or
                geometry_specs['visual']['mesh'][body_name]):
            # Extract info from geometry object
            mesh_name = geometry_object.name
            mesh_path = geometry_object.meshPath
            mesh_scale = geometry_object.meshScale
            mesh_origin = geometry_object.placement

            # Replace relative mesh path by absolute one
            if mesh_path.startswith("package://"):
                mesh_path_orig = mesh_path
                for root_dir in robot.mesh_package_dirs:
                    mesh_path = mesh_path_orig.replace(
                        "package:/", root_dir)
                    if os.path.exists(mesh_path):
                        break

            # Compute the minimal volume bounding box, then add new frames to
            # the robot model at its vertices and register contact points at
            # their location.
            try:
                mesh = trimesh.load(mesh_path)
            except ValueError:  # Mesh file is not available
                continue
            box = mesh.bounding_box_oriented
            for i in range(8):
                frame_name = "_".join((mesh_name, "BoundingBox", str(i)))
                frame_transform_rel = pin.SE3(
                    np.eye(3), mesh_scale * np.asarray(box.vertices[i]))
                frame_transform = mesh_origin.act(frame_transform_rel)
                robot.add_frame(frame_name, body_name, frame_transform)
                contact_frames_names.append(frame_name)

    # Add the collision bodies and contact points.
    # Note that it must be done before adding the sensors because
    # Contact sensors requires contact points to be defined.
    # Mesh collisions is not numerically stable for now, so disabling it.
    # Note: Be careful, the order of the contact points is important, it
    # changes the computation of the external forces, which is an iterative
    # algorithm for impulse model, resulting in different simulation
    # results. The order of the element of the set depends of the `hash`
    # method of python, whose seed is randomly generated when starting the
    # interpreter for security reason. As a result, the set must be sorted
    # manually to ensure consistent results.
    robot.add_collision_bodies(
        collision_bodies_names, ignore_meshes=avoid_instable_collisions)
    robot.add_contact_points(sorted(list(set(contact_frames_names))))

    # Add the motors to the robot
    for motor_type, motors_descr in motors_info.items():
        for motor_name, motor_descr in motors_descr.items():
            # Make sure the motor can be instantiated
            joint_name = motor_descr.pop('joint_name')
            if not robot.pinocchio_model.existJointName(joint_name):
                logger.warning(
                    f"'{joint_name}' is not a valid joint name.")
                continue

            # Create the motor and attach it
            motor = getattr(jiminy, motor_type)(motor_name)
            robot.attach_motor(motor)

            # Initialize the motor
            motor.initialize(joint_name)

            # Set the motor options
            options = motor.get_options()
            option_fields = options.keys()
            for name, value in motor_descr.items():
                if name not in option_fields:
                    logger.warning(
                        f"'{name}' is not a valid option for the motor "
                        f"{motor_name} of type {motor_type}.")
                options[name] = value
            options['enableArmature'] = True
            motor.set_options(options)

    # Add the sensors to the robot
    for sensor_type, sensors_descr in sensors_info.items():
        for sensor_name, sensor_descr in sensors_descr.items():
            # Make sure the sensor can be instantiated
            if sensor_type == encoder.type:
                joint_name = sensor_descr.pop('joint_name')
                if not robot.pinocchio_model.existJointName(joint_name):
                    logger.warning(
                        f"'{joint_name}' is not a valid joint name.")
                    continue
            elif sensor_type == effort.type:
                motor_name = sensor_descr.pop('motor_name')
                if motor_name not in robot.motors_names:
                    logger.warning(
                        f"'{motor_name}' is not a valid motor name.")
                    continue

            # Create the sensor and attach it
            sensor = getattr(jiminy, sensor_type)(sensor_name)
            robot.attach_sensor(sensor)

            # Initialize the sensor
            if sensor_type == encoder.type:
                sensor.initialize(joint_name)
            elif sensor_type == effort.type:
                sensor.initialize(motor_name)
            elif sensor_type == contact.type:
                frame_name = sensor_descr.pop('frame_name')
                sensor.initialize(frame_name)
            elif sensor_type in [force.type, imu.type]:
                # Create the frame and add it to the robot model
                frame_name = sensor_descr.pop('frame_name', None)

                # Create a frame if a frame name has been specified.
                # In such a case, the body name must be specified.
                if frame_name is None:
                    # Get the body name
                    body_name = sensor_descr.pop('body_name')

                    # Generate a frame name both intelligible and available
                    i = 0
                    frame_name = "_".join((
                        sensor_name, sensor_type, "Frame"))
                    while robot.pinocchio_model.existFrame(frame_name):
                        frame_name = "_".join((
                            sensor_name, sensor_type, "Frame", str(i)))
                        i += 1

                    # Compute SE3 object representing the frame placement
                    frame_pose_xyzrpy = np.array(
                        sensor_descr.pop('frame_pose'))
                    frame_trans = frame_pose_xyzrpy[:3]
                    frame_rot = rpyToMatrix(frame_pose_xyzrpy[3:])
                    frame_placement = pin.SE3(frame_rot, frame_trans)

                    # Add the frame to the robot model
                    robot.add_frame(frame_name, body_name, frame_placement)

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
                    logger.warning(
                        f"'{name}' is not a valid option for the sensor "
                        f"{sensor_name} of type {sensor_type}.")
                options[name] = value
            sensor.set_options(options)

    return extra_info


class BaseJiminyRobot(jiminy.Robot):
    """Base class to instantiate a Jiminy robot based on a standard URDF file
    and Jiminy-specific hardware description file.

    The utility 'generate_default_hardware_description_file' is provided to
    automatically generate a default hardware description file for any given
    URDF file. URDF file containing Gazebo plugins description should not
    require any further modification as it usually includes the information
    required to fully characterize the motors, sensors, contact points and
    collision bodies, along with some of there properties.

    .. note::
        Overload this class if you need finer-grained capability.

    .. warning::
        Hardware description files within the same directory and having the
        name than the URDF file will be detected automatically without
        requiring to manually specify its path.
    """
    def __init__(self) -> None:
        super().__init__()
        self.extra_info = {}
        self._urdf_path_orig = None

    def initialize(self,
                   urdf_path: str,
                   hardware_path: Optional[str] = None,
                   mesh_path: Optional[str] = None,
                   has_freeflyer: bool = True,
                   avoid_instable_collisions: bool = True,
                   verbose: bool = True) -> None:
        """Initialize the robot.

        :param urdf_path: Path of the URDF file of the robot.
        :param hardware_path: Path of Jiminy hardware description toml file.
                              Optional: Looking for '*_hardware.toml' file in
                              the same folder and with the same name. If not
                              found, then no hardware is added to the robot,
                              which is valid and can be used for display.
        :param mesh_path: Path to the folder containing the URDF meshes. It
                          will overwrite any absolute mesh path.
                          Optional: Env variable 'JIMINY_DATA_PATH' will be
                          used if available.
        :param has_freeflyer: Whether the robot is fixed-based wrt its root
                              link, or can move freely in the world.
        :param avoid_instable_collisions: Prevent numerical instabilities by
                                          replacing collision mesh by vertices
                                          of associated minimal volume bounding
                                          box, primitive box by its vertices,
                                          and primitive sphere by its center.
        :param verbose: Whether or not to print warnings.
        """
        # Backup the original URDF path
        self._urdf_path_orig = urdf_path

        # Fix the URDF mesh paths
        if mesh_path is not None:
            urdf_path = _fix_urdf_mesh_path(urdf_path, mesh_path)

        # Initialize the robot without motors nor sensors
        if mesh_path is not None:
            mesh_root_dirs = [mesh_path]
        else:
            mesh_root_dirs = [os.path.dirname(urdf_path)]
        mesh_env_path = os.environ.get('JIMINY_DATA_PATH', None)
        if mesh_env_path is not None:
            mesh_root_dirs += [mesh_env_path]
        return_code = super().initialize(
            urdf_path, has_freeflyer, mesh_root_dirs)

        if return_code != jiminy.hresult_t.SUCCESS:
            raise ValueError(
                "Impossible to load the URDF file. Either the file is "
                "corrupted or does not exit.")

        # Load the hardware description file if available
        if hardware_path is None:
            hardware_path = str(pathlib.Path(
                self._urdf_path_orig).with_suffix('')) + '_hardware.toml'

        self.hardware_path = hardware_path
        if not os.path.exists(hardware_path):
            if hardware_path:
                logger.warning(
                    "Hardware configuration file not found. Not adding any "
                    "hardware to the robot.\n Default file can be generated "
                    "using 'generate_default_hardware_description_file' "
                    "method.")
            return

        self.extra_info = load_hardware_description_file(
            self, hardware_path, avoid_instable_collisions, verbose)

    def __del__(self) -> None:
        if self.urdf_path != self._urdf_path_orig:
            try:
                os.remove(self.urdf_path)
            except (PermissionError, FileNotFoundError):
                pass
