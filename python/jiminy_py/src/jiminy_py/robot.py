# mypy: disable-error-code="attr-defined, name-defined"
""" TODO: Write documentation.
"""
import os
import re
import logging
import pathlib
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from types import ModuleType
from typing import Optional, Dict, Any, Sequence, Literal, Set, List, get_args

import toml
import numpy as np
import trimesh

import hppfcl
import pinocchio as pin
from pinocchio.rpy import rpyToMatrix  # pylint: disable=import-error

from . import core as jiminy
from .core import (  # pylint: disable=no-name-in-module
    EncoderSensor as encoder,
    EffortSensor as effort,
    ContactSensor as contact,
    ForceSensor as force,
    ImuSensor as imu)


DEFAULT_UPDATE_RATE = 1000.0  # [Hz]
DEFAULT_FRICTION_DRY_SLOPE = 20.0
DEFAULT_ARMATURE = 0.1

EXTENSION_MODULES: Sequence[ModuleType] = ()

GeometryModelType = Literal['collision', 'visual']
GeometryObjectType = Literal['primitive', 'mesh']


class _DuplicateFilter(logging.Filter):
    """ TODO: Write documentation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.msgs: Set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter all messages that have already been logged once.

        All new messages are stored in a singleton buffer.
        """
        if record.msg not in self.msgs:
            self.msgs.add(record.msg)
            return False
        return True


LOGGER = logging.getLogger(__name__)
LOGGER.addFilter(_DuplicateFilter())


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
                        mesh_path_dir: str,
                        output_root_path: Optional[str] = None) -> str:
    """Generate an URDF with updated mesh paths.

    :param urdf_path: Full path of the URDF file.
    :param mesh_path_dir: Root path of the meshes.
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
            mesh_path_dir_orig = '.'
        else:
            mesh_path_dir_orig = os.path.commonpath(list(pathlists))
    else:
        mesh_path_dir_orig = os.path.dirname(next(iter(pathlists)))
    if mesh_path_dir == mesh_path_dir_orig:
        return urdf_path

    # Create the output directory
    if output_root_path is None:
        output_root_path = tempfile.mkdtemp()
    fixed_urdf_dir = os.path.join(
        output_root_path, "fixed_urdf" + mesh_path_dir.translate(
            str.maketrans({k: '_' for k in '/:'})))  # type: ignore[arg-type]
    os.makedirs(fixed_urdf_dir, exist_ok=True)
    fixed_urdf_path = os.path.join(
        fixed_urdf_dir, os.path.basename(urdf_path))

    # Override the root mesh path with the desired one
    urdf_contents = urdf_contents.replace(
        '"'.join((mesh_tag, mesh_path_dir_orig)),
        '"'.join((mesh_tag, mesh_path_dir)))
    with open(fixed_urdf_path, 'w') as f:
        f.write(urdf_contents)

    return fixed_urdf_path


def generate_default_hardware_description_file(
        urdf_path: str,
        hardware_path: Optional[str] = None,
        default_update_rate: float = DEFAULT_UPDATE_RATE,
        verbose: bool = True) -> None:
    r"""Generate a default hardware description file, based on the information
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
                          the URDF file, using '\*_hardware.toml' extension.
    :param default_update_rate: Default update rate of the sensors and the
                                controller in Hz. It will be used for sensors
                                whose the update rate is unspecified. 0.0 for
                                continuous update.
                                Optional: DEFAULT_UPDATE_RATE if no Gazebo
                                plugin has been found, the lowest among the
                                Gazebo plugins otherwise.
    :param verbose: Whether to print warnings.
    """
    # Handle verbosity level
    if verbose:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.ERROR)

    # Read the XML
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Initialize the hardware information
    hardware_info: Dict[str, Dict[str, Any]] = OrderedDict(
        Global=OrderedDict(
            sensorsUpdatePeriod=1.0/default_update_rate,
            controllerUpdatePeriod=1.0/default_update_rate,
            collisionBodiesNames=[],
            contactFramesNames=[]
        ),
        Motor=defaultdict(OrderedDict),
        Sensor=defaultdict(OrderedDict)
    )

    # Extract the root link. It is the one having no parent at all.
    links = set()
    for link_descr in root.findall('./link'):
        links.add(link_descr.attrib["name"])
    for joint_descr in root.findall('./joint'):
        child_obj = joint_descr.find('./child')
        assert child_obj is not None
        links.remove(child_obj.get('link'))
    link_root = next(iter(links))

    # Extract the list of parent and child links, excluding the one related
    # to fixed link not having collision geometry, because they are likely not
    # "real" joint.
    links_parent: Set[str] = set()
    links_child: Set[str] = set()
    for joint_descr in root.findall('./joint'):
        parent_link_obj = joint_descr.find('./parent')
        child_link_obj = joint_descr.find('./child')
        assert parent_link_obj is not None
        assert child_link_obj is not None
        parent_link = parent_link_obj.attrib['link']
        child_link = child_link_obj.attrib['link']
        if joint_descr.attrib['type'].casefold() != 'fixed' or root.find(
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
    gazebo_ground_stiffness: Optional[float] = None
    gazebo_ground_damping: Optional[float] = None
    gazebo_update_rate: Optional[float] = None
    collision_bodies_names: Set[str] = set()
    gazebo_plugins_found = root.find('gazebo') is not None
    for gazebo_plugin_descr in root.iterfind('gazebo'):
        body_name = gazebo_plugin_descr.attrib['reference']

        # Extract sensors
        for gazebo_sensor_descr in gazebo_plugin_descr.iterfind('sensor'):
            sensor_info = OrderedDict(body_name=body_name)

            # Extract the sensor name
            sensor_name = gazebo_sensor_descr.attrib['name']

            # Extract the sensor type
            sensor_type = gazebo_sensor_descr.attrib['type'].casefold()
            if 'imu' in sensor_type:
                sensor_type = imu.type
            elif 'contact' in sensor_type:
                collision_bodies_names.add(body_name)
                sensor_type = force.type
            else:
                LOGGER.warning(
                    "Unsupported Gazebo sensor plugin of type '%s'.",
                    sensor_type)
                continue

            # Extract the sensor update period
            update_rate_obj = gazebo_sensor_descr.find('./update_rate')
            assert update_rate_obj is not None
            assert update_rate_obj.text is not None
            update_rate = float(update_rate_obj.text)
            if gazebo_update_rate is None:
                gazebo_update_rate = update_rate
            elif gazebo_update_rate != update_rate:
                LOGGER.warning(
                    "Jiminy does not support sensors with different "
                    "update rate. Using greatest common divisor instead.")
                gazebo_update_rate = _gcd(gazebo_update_rate, update_rate)

            # Extract the pose of the frame associate with the sensor.
            # Note that it is optional but usually defined since sensors
            # can only be attached to link in Gazebo, not to frame.
            frame_pose_obj = gazebo_sensor_descr.find('./pose')
            if frame_pose_obj is None:
                sensor_info['frame_name'] = sensor_info.pop('body_name')
            else:
                assert frame_pose_obj.text is not None
                sensor_info['frame_pose'] = list(
                    map(float, frame_pose_obj.text.split()))

            # Add the sensor to the robot's hardware
            hardware_info['Sensor'][sensor_type][sensor_name] = sensor_info

        # Extract the collision bodies and ground model, then add force
        # sensors. At this point, every force sensor is associated with a
        # collision body.
        if gazebo_plugin_descr.find('kp') is not None:
            # Add a force sensor, if not already in the collision set
            if body_name not in collision_bodies_names:
                force_sensor_info = hardware_info['Sensor'][force.type]
                force_sensor_info[f"{body_name}Contact"] = OrderedDict(
                    frame_name=body_name)

            # Add the related body to the collision set
            collision_bodies_names.add(body_name)

            # Update the ground model
            kp_obj = gazebo_plugin_descr.find('kp')
            kd_obj = gazebo_plugin_descr.find('kd')
            assert kp_obj is not None and kp_obj.text is not None
            assert kd_obj is not None and kd_obj.text is not None
            ground_stiffness = float(kp_obj.text)
            ground_damping = float(kd_obj.text)
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
            plugin = gazebo_plugin_descr.attrib['filename']
            if plugin == "libgazebo_ros_force.so":
                body_name_obj = gazebo_plugin_descr.find('bodyName')
                assert body_name_obj is not None
                body_name = body_name_obj.text
                force_sensor_info = hardware_info['Sensor'][force.type]
                force_sensor_info[f"{body_name}Wrench"] = OrderedDict(
                    frame_name=body_name)
            else:
                LOGGER.warning("Unsupported Gazebo plugin '%s'", plugin)

    # Add IMU sensor to the root link if no Gazebo IMU sensor has been found
    if link_root and imu.type not in hardware_info['Sensor'].keys():
        hardware_info['Sensor'][imu.type][link_root] = OrderedDict(
            frame_name=link_root)

    # Add force sensors and collision bodies if no Gazebo plugin is available
    if not gazebo_plugins_found:
        for link_leaf in links_leaf:
            # Add a force sensor
            hardware_info['Sensor'][force.type][link_leaf] = OrderedDict(
                frame_name=link_leaf)

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
    joints_options: Dict[str, Dict[str, Any]] = {}
    for joint_descr in root.findall("./joint"):
        if joint_descr.attrib['type'].casefold() == 'fixed':
            continue
        joint_name = joint_descr.attrib['name']
        dyn_descr = joint_descr.find('./dynamics')
        if dyn_descr is not None:
            damping = float(dyn_descr.get('damping',  0.0))
            friction = float(dyn_descr.get('friction', 0.0))
        else:
            damping, friction = 0.0, 0.0
        joints_options[joint_name] = OrderedDict()
        if damping > 0.0:
            joints_options[joint_name].update(
                frictionViscousPositive=-damping,
                frictionViscousNegative=-damping
            )
        if friction > 0.0:
            joints_options[joint_name].update(
                frictionDryPositive=-friction,
                frictionDryNegative=-friction,
                frictionDrySlope=DEFAULT_FRICTION_DRY_SLOPE
            )

    # Extract the motors and effort sensors.
    # It is done by reading 'transmission' field, that is part of
    # URDF standard, so it should be available on any URDF file.
    transmission_found = root.find('transmission') is not None
    for transmission_descr in root.iterfind('transmission'):
        # Assert(s) for type checker
        assert isinstance(transmission_descr, ET.Element)

        # Initialize motor and sensor info
        motor_info: Dict[str, Any] = OrderedDict()
        sensor_info = OrderedDict()

        # Check that the transmission type is supported
        transmission_name = transmission_descr.attrib['name']
        transmission_type_obj = transmission_descr.find('./type')
        if transmission_type_obj is not None:
            transmission_type = transmission_type_obj.text
        else:
            transmission_type = transmission_descr.attrib['type']
        assert transmission_type is not None
        transmission_type = os.path.basename(transmission_type).casefold()
        if transmission_type != 'simpletransmission':
            LOGGER.warning(
                "Jiminy only support SimpleTransmission for now. Skipping"
                "transmission '%s' of type '%s'.", transmission_name,
                transmission_type)
            continue

        # Extract the motor name
        motor_descr = transmission_descr.find('./actuator')
        assert isinstance(motor_descr, ET.Element)
        motor_name = motor_descr.attrib['name']
        sensor_info['motor_name'] = motor_name

        # Extract the associated joint name
        joint_descr = transmission_descr.find('./joint')
        assert isinstance(joint_descr, ET.Element)
        joint_name = joint_descr.attrib['name']
        motor_info['joint_name'] = joint_name

        # Make sure that the joint is revolute
        joint = root.find(f"./joint[@name='{joint_name}']")
        assert joint is not None
        joint_type = joint.attrib['type'].casefold()
        if joint_type not in ("revolute", "continuous", "prismatic"):
            LOGGER.warning(
                "Jiminy only support 1-dof joint actuators and effort "
                "sensors. Attached joint cannot of type '%s'.", joint_type)
            continue

        # Extract the transmission ratio (motor / joint)
        ratio_obj = transmission_descr.find('./mechanicalReduction')
        if ratio_obj is None:
            motor_info['mechanicalReduction'] = 1.0
        else:
            assert ratio_obj.text is not None
            ratio_txt = ratio_obj.text
            motor_info['mechanicalReduction'] = float(ratio_txt)

        # Extract the armature (rotor) inertia
        armature = transmission_descr.find('./motorInertia')
        if armature is None:
            motor_info['armature'] = 0.0
        else:
            armature_txt = armature.text
            assert armature_txt is not None
            motor_info['armature'] = float(armature_txt) * (
                motor_info['mechanicalReduction'] ** 2)

        # Add dynamics property to motor info, if any
        motor_info.update(joints_options.pop(joint_name))

        # Add the motor and sensor to the robot's hardware
        hardware_info['Motor']['SimpleMotor'][motor_name] = motor_info
        hardware_info['Sensor'][effort.type][joint_name] = sensor_info

    # Define default encoder sensors, and default effort sensors if no
    # transmission available.
    for joint_descr in root.iterfind('joint'):
        encoder_info = OrderedDict()

        # Skip fixed joints
        joint_type = joint_descr.attrib['type'].casefold()
        if joint_type == 'fixed':
            continue

        # Extract the joint name
        joint_name = joint_descr.attrib['name']
        encoder_info['joint_name'] = joint_name

        # Add the sensor to the robot's hardware
        hardware_info['Sensor'][encoder.type][joint_name] = encoder_info

        # Add motors to robot hardware by default if no transmission found
        if not transmission_found:
            joint_limit_descr = joint_descr.find('./limit')
            assert joint_limit_descr is not None
            if float(joint_limit_descr.attrib['effort']) == 0.0:
                continue
            hardware_info['Motor']['SimpleMotor'][joint_name] = OrderedDict(
                joint_name=joint_name,
                armature=DEFAULT_ARMATURE,
                **joints_options.pop(joint_name)
            )
            hardware_info['Sensor'][effort.type][joint_name] = OrderedDict(
                motor_name=joint_name)

    # Warn if friction model has been defined for non-actuated joints
    if joints_options:
        LOGGER.warning(
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
    :param verbose: Whether to print warnings.

    :returns: Unused information available in hardware configuration file.
    """
    # Handle verbosity level
    if verbose:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.ERROR)

    hardware_info = toml.load(hardware_path)
    extra_info = hardware_info.pop('Global', {})
    motors_info = hardware_info.pop('Motor', {})
    sensors_info = hardware_info.pop('Sensor', {})

    # Extract the list of bodies having visual and collision meshes or
    # primitives.
    geometry_types: Dict[
            GeometryModelType, Dict[GeometryObjectType, Set[str]]] = {
        geom_type: {'primitive': set(), 'mesh': set()}
        for geom_type in get_args(GeometryModelType)}
    geometry_specs: Dict[GeometryModelType, Dict[GeometryObjectType, Dict[
            GeometryObjectType, List[pin.GeometryObject]]]] = {
        geom_type: {
            'primitive': defaultdict(lambda: []),
            'mesh': defaultdict(lambda: [])}
        for geom_type in get_args(GeometryModelType)}
    for geom_model, geometry_types_i, geometry_specs_i in zip(
            (robot.collision_model, robot.visual_model),
            geometry_types.values(),
            geometry_specs.values()):
        for geometry_object in geom_model.geometryObjects:
            frame_idx = geometry_object.parentFrame
            frame_name = robot.pinocchio_model.frames[frame_idx].name
            mesh_path = geometry_object.meshPath
            is_mesh = any(char in mesh_path for char in ('\\', '/', '.'))
            geom_type: GeometryObjectType = 'mesh' if is_mesh else 'primitive'
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
            LOGGER.warning(
                "Collision body having both primitive and mesh geometries "
                "is not supported. Enabling only primitive collision for "
                "this body.")
            continue
        if body_name in geometry_types['collision']['primitive']:
            pass
        elif body_name in geometry_types['collision']['mesh']:
            if not avoid_instable_collisions:
                continue
            LOGGER.warning(
                "Collision body associated with mesh geometry is not "
                "supported for now. Replacing it by contact points at the "
                "vertices of the minimal volume bounding box.")
        elif body_name not in geometry_types['visual']['mesh']:
            LOGGER.warning(
                "No visual mesh nor collision geometry associated with "
                "collision body '%s'. Fallback to adding a single contact "
                "point at body frame.", body_name)
            contact_frames_names.append(body_name)
            continue
        else:
            LOGGER.warning(
                "No collision geometry associated with the collision body "
                "'%s'. Fallback to replacing it by contact points at the "
                "vertices of the minimal volume bounding box of the available "
                "visual meshes.", body_name)

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
            LOGGER.warning(
                "Collision body associated with box geometry is not "
                "numerically stable for now. Replacing it by contact points "
                "at the vertices.")

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
                LOGGER.warning("'%s' is not a valid joint name.", joint_name)
                continue

            # Create the motor and attach it
            motor = None
            for module in (jiminy, *EXTENSION_MODULES):
                try:
                    motor = getattr(module, motor_type)(motor_name)
                    break
                except AttributeError:
                    pass
            if motor is None:
                raise RuntimeError(
                    f"Cannot instantiate motor of type '{motor_type}'.")
            robot.attach_motor(motor)

            # Initialize the motor
            motor.initialize(joint_name)

            # Set the motor options
            options = motor.get_options()
            option_fields = options.keys()
            for name, value in motor_descr.items():
                if name not in option_fields:
                    LOGGER.warning(
                        "'%s' is not a valid option for the motor '%s' of "
                        "type '%s'.", name, motor_name, motor_type)
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
                    LOGGER.warning(
                        "'%s' is not a valid joint name.", joint_name)
                    continue
            elif sensor_type == effort.type:
                motor_name = sensor_descr.pop('motor_name')
                if motor_name not in robot.motors_names:
                    LOGGER.warning(
                        "'%s' is not a valid motor name.", motor_name)
                    continue

            # Create the sensor and attach it
            for module in (jiminy, *EXTENSION_MODULES):
                try:
                    sensor = getattr(module, sensor_type)(sensor_name)
                    break
                except AttributeError:
                    pass
            else:
                raise RuntimeError(
                    f"Cannot instantiate sensor of type '{sensor_type}'.")
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
                if not frame_name or \
                        not robot.pinocchio_model.existFrame(frame_name):
                    # Get the body name
                    body_name = sensor_descr.pop('body_name')

                    # Generate a frame name both intelligible and available
                    if frame_name is None:
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
                elif 'frame_pose' in sensor_descr.keys():
                    raise ValueError(
                        f"The sensor '{sensor_name}' is attached to the frame "
                        f"'{frame_name}' that already exists whereas a "
                        "specific pose is also requested.")

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
                    LOGGER.warning(
                        "'%s' is not a valid option for the sensor '%s' of "
                        "type '%s'.", name, sensor_name, sensor_type)
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
        self.extra_info: Dict[str, Any] = {}
        self.hardware_path: Optional[str] = None
        self._urdf_path_orig: Optional[str] = None
        super().__init__()

    def initialize(self,
                   urdf_path: str,
                   hardware_path: Optional[str] = None,
                   mesh_path_dir: Optional[str] = None,
                   mesh_package_dirs: Sequence[str] = (),
                   has_freeflyer: bool = True,
                   avoid_instable_collisions: bool = True,
                   load_visual_meshes: bool = False,
                   verbose: bool = True) -> None:
        r"""Initialize the robot.

        :param urdf_path: Path of the URDF file of the robot.
        :param hardware_path: Path of Jiminy hardware description toml file.
                              Optional: Looking for '\*_hardware.toml' file in
                              the same folder and with the same name. If not
                              found, then no hardware is added to the robot,
                              which is valid and can be used for display.
        :param mesh_path_dir: Path to the folder containing the URDF meshes. It
                              will overwrite the common root of all absolute
                              mesh paths.
                              Optional: Env variable 'JIMINY_DATA_PATH' will be
                              used if available.
        :param mesh_package_dirs: Additional search paths for all relative mesh
                                  paths beginning with 'packages://' directive.
                                  'mesh_path_dir' is systematically appended.
        :param has_freeflyer: Whether the robot is fixed-based wrt its root
                              link, or can move freely in the world.
        :param avoid_instable_collisions: Prevent numerical instabilities by
                                          replacing collision mesh by vertices
                                          of associated minimal volume bounding
                                          box, primitive box by its vertices,
                                          and primitive sphere by its center.
        :param load_visual_meshes: Load visual and collision geometries when
                                   creating the robot. It will allow for
                                   dumping standalone log files that are
                                   safe to carry around but larger.
        :param verbose: Whether to print warnings.
        """
        # Backup the original URDF path
        self._urdf_path_orig = urdf_path

        # Fix the URDF mesh paths
        if mesh_path_dir is not None:
            urdf_path = _fix_urdf_mesh_path(urdf_path, mesh_path_dir)

        # Initialize the robot without motors nor sensors
        mesh_package_dirs = list(mesh_package_dirs)
        if mesh_path_dir is not None:
            mesh_package_dirs.append(mesh_path_dir)
        else:
            mesh_package_dirs.append(os.path.dirname(urdf_path))
        mesh_env_path = os.environ.get('JIMINY_DATA_PATH', None)
        if mesh_env_path is not None:
            mesh_package_dirs.append(mesh_env_path)
        return_code = super().initialize(
            urdf_path, has_freeflyer, mesh_package_dirs, load_visual_meshes)

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
                LOGGER.warning(
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
            except (PermissionError, FileNotFoundError, AttributeError):
                pass
