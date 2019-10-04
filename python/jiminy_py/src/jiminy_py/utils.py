#!/usr/bin/env python

import sys
import os
import re
import shutil
import time
from copy import copy
import subprocess
from threading import Thread, Lock
from bisect import bisect_right
import numpy as np
from scipy.interpolate import UnivariateSpline

import jiminy # must be imported before libpinocchio_pywrap to find it

import pinocchio as pnc
from pinocchio.utils import *
from pinocchio.robot_wrapper import RobotWrapper
import libpinocchio_pywrap as pin
from gepetto.corbaserver import Client


## @brief Object that contains the kinematics and dynamics state of the robot at a given time.
class State:
    def __init__(self, q, v, a, t=None, f=None, tau=None, f_ext=None):
        self.t = copy(t)  #< @brief Time.
        self.q = copy(q)  #< @brief Positions of the robot (with freeflyer).
        self.v = copy(v)  #< @brief Velocities of the robot.
        self.a = copy(a)  #< @brief Accelerations of the robot.
        ## @brief   Forces on the different bodies of the robot. It is a dictionary. The keys represent a given foot
        ##          orientation. For each orientation, a dictionary contains the 6D-force for each body.
        if f is None:
            self.f = {}
        else:
            self.f = deepcopy(f)
        ## @brief   Joint torques. It is a dictionary. The keys represent a given foot orientation.
        if tau is None:
            self.tau = {}
        else:
            self.tau = deepcopy(tau)
        ## @brief   External forces represented in the frame of the Henke ankle. It is a dicionary. The keys represent a
        ##          given foot orientation.
        if f_ext is None:
            self.f_ext = {}
        else:
            self.f_ext = deepcopy(f_ext)

    @staticmethod
    def todict(state_list):
        state_dict = dict()
        state_dict['q'] = np.concatenate([s.q for s in state_list], axis=1)
        state_dict['v'] = np.concatenate([s.v for s in state_list], axis=1)
        state_dict['a'] = np.concatenate([s.a for s in state_list], axis=1)
        state_dict['t'] = [s.t for s in state_list]
        state_dict['f'] = [s.f for s in state_list]
        state_dict['tau'] = [s.tau for s in state_list]
        state_dict['f_ext'] = [s.f_ext for s in state_list]
        return state_dict

    @staticmethod
    def fromdict(state_dict):
        default_state_dict = defaultdict(lambda: [None for i in range(state_dict['q'].shape[1])], state_dict)
        state_list = []
        for i in range(state_dict['q'].shape[1]):
            state_list.append(State(default_state_dict['q'][:,[i]],
                                    default_state_dict['v'][:,[i]],
                                    default_state_dict['a'][:,[i]],
                                    default_state_dict['t'][i],
                                    default_state_dict['f'][i],
                                    default_state_dict['tau'][i],
                                    default_state_dict['f_ext'][i]))
        return state_list

    def update(self, q, v, a, t=None, hzd_state=None):
        self.q = copy(q)
        self.v = copy(v)
        self.a = copy(a)
        if t:
            self.t = copy(t)
        if hzd_state:
            self.hzd_state = copy(hzd_state)

    def __repr__(self):
        return "State(q=\n{!r},\nv=\n{!r},\na=\n{!r},\nt=\n{!r},\nf=\n{!r},\nf_ext=\n{!r})".format(
            self.q, self.v, self.a, self.t, self.f, self.f_ext)

def smoothing_filter(time_in,val_in,time_out=None,relabel=None,params=None):
    if time_out is None:
        time_out = time_in
    if params is None:
        params = dict()
        params['mixing_ratio_1'] = 0.12
        params['mixing_ratio_2'] = 0.04
        params['smoothness'] = [0.0,0.0,0.0]
        params['smoothness'][0]  = 5e-4
        params['smoothness'][1]  = 5e-4
        params['smoothness'][2]  = 5e-4

    if relabel is None:
        mix_fit    = [None,None,None]
        mix_fit[0] = lambda t: 0.5*(1+np.sin(1/params['mixing_ratio_1']*((t-time_in[0])/(time_in[-1]-time_in[0]))*np.pi-np.pi/2))
        mix_fit[1] = lambda t: 0.5*(1+np.sin(1/params['mixing_ratio_2']*((t-(1-params['mixing_ratio_2'])*time_in[-1])/(time_in[-1]-time_in[0]))*np.pi+np.pi/2))
        mix_fit[2] = lambda t: 1

        val_fit = []
        for jj in range(val_in.shape[0]):
            val_fit_jj = []
            for kk in range(len(params['smoothness'])):
                val_fit_jj.append(UnivariateSpline(time_in, val_in[jj], s=params['smoothness'][kk]))
            val_fit.append(val_fit_jj)

        time_out_mixing = [None, None, None]
        time_out_mixing_ind = [None, None, None]
        time_out_mixing_ind[0] = time_out < time_out[-1]*params['mixing_ratio_1']
        time_out_mixing[0] = time_out[time_out_mixing_ind[0]]
        time_out_mixing_ind[1] = time_out > time_out[-1]*(1-params['mixing_ratio_2'])
        time_out_mixing[1] = time_out[time_out_mixing_ind[1]]
        time_out_mixing_ind[2] = np.logical_and(np.logical_not(time_out_mixing_ind[0]), np.logical_not(time_out_mixing_ind[1]))
        time_out_mixing[2] = time_out[time_out_mixing_ind[2]]

        val_out = np.zeros((val_in.shape[0],len(time_out)))
        for jj in range(val_in.shape[0]):
            for kk in range(len(time_out_mixing)):
                val_out[jj,time_out_mixing_ind[kk]] = \
                   (1 - mix_fit[kk](time_out_mixing[kk])) * val_fit[jj][kk](time_out_mixing[kk]) + \
                        mix_fit[kk](time_out_mixing[kk])  * val_fit[jj][-1](time_out_mixing[kk])
    else:
        time_tmp   = np.concatenate([time_in[:-1]-time_in[-1],time_in,time_in[1:]+time_in[-1]])
        val_in_tmp = np.concatenate([relabel.dot(val_in[:,:-1]),val_in,relabel.dot(val_in[:,1:])], axis=1)
        val_out = np.zeros((val_in.shape[0],len(time_out)))
        for jj in range(val_in_tmp.shape[0]):
            f = UnivariateSpline(time_tmp, val_in_tmp[jj], s=params['smoothness'][-1])
            val_out[jj] = f(time_out)

    return val_out

def load_csv_log(csv_log_path):
    return np.genfromtxt(csv_log_path, delimiter=',')

def delete_nodes_viewer(*nodes_path):
    client = Client()
    for node_path in nodes_path:
        if node_path in client.gui.getNodeList():
            client.gui.deleteNode(node_path, True)

# Fix bug if root_joint=None
def initFromURDF(self, filename, package_dirs=None, root_joint=None, verbose=False, meshLoader=None):
    if root_joint is None:
        model = pin.buildModelFromUrdf(filename)
    else:
        model = pin.buildModelFromUrdf(filename, root_joint)

    if "buildGeomFromUrdf" not in dir(pin):
        collision_model = None
        visual_model = None
        if verbose:
            print('Info: the Geometry Module has not been compiled with Pinocchio. No geometry model and data have been built.')
    else:
        if verbose and "removeCollisionPairs" not in dir(pin) and meshLoader is not None:
            print('Info: Pinocchio was compiled without hpp-fcl. meshLoader is ignored.')
        def _buildGeomFromUrdf (model, filename, geometryType, meshLoader, dirs=None):
            if "removeCollisionPairs" not in dir(pin):
                if dirs:
                    return pin.buildGeomFromUrdf(model, filename, dirs, geometryType)
                else:
                    return pin.buildGeomFromUrdf(model, filename, geometryType)
            else:
                if dirs:
                    return pin.buildGeomFromUrdf(model, filename, dirs, geometryType, meshLoader)
                else:
                    return pin.buildGeomFromUrdf(model, filename, geometryType, meshLoader)

        if package_dirs is None:
            collision_model = _buildGeomFromUrdf(model, filename, pin.GeometryType.COLLISION,meshLoader)
            visual_model = _buildGeomFromUrdf(model, filename, pin.GeometryType.VISUAL, meshLoader)
        else:
            if not all(isinstance(item, str) for item in package_dirs):
                raise Exception('The list of package directories is wrong. At least one is not a string')
            else:
                collision_model = _buildGeomFromUrdf(model, filename, pin.GeometryType.COLLISION, meshLoader,
                                                        dirs = utils.fromListToVectorOfString(package_dirs))
                visual_model = _buildGeomFromUrdf(model, filename, pin.GeometryType.VISUAL, meshLoader,
                                                        dirs = utils.fromListToVectorOfString(package_dirs))


    RobotWrapper.__init__(self, model=model, collision_model=collision_model, visual_model=visual_model)

def extract_state_from_simulation_log(log_header, log_data, urdf_path, pinocchio_model=None, has_freeflyer=True):
    # Extract time, joint positions and velocities evolution from log.
    # Note that the quaternion angular velocity vectors are expressed
    # it body frame rather than world frame.

    t = log_data[:,log_header.index('Global.Time')]
    qe = log_data[:,np.array(['currentFreeFlyerPosition' in field
                              or 'currentPosition' in field for field in log_header])].T
    dqe = log_data[:,np.array(['currentFreeFlyerVelocity' in field
                               or 'currentVelocity' in field for field in log_header])].T
    ddqe = log_data[:,np.array(['currentFreeFlyerAcceleration' in field
                                or 'currentAcceleration' in field for field in log_header])].T

    # Create state sequence
    evolution_robot = []
    for i in range(len(t)):
        evolution_robot.append(State(qe[:,[i]], dqe[:,[i]], ddqe[:,[i]], t[i]))

    return {"evolution_robot": evolution_robot,
            "urdf": urdf_path,
            "has_freeflyer": has_freeflyer,
            "pinocchio_model": pinocchio_model}

## @brief   Display robot evolution in Gepetto viewer.
lock = Lock()
def play_trajectories(trajectory_data, xyz_offset=None, urdf_rgba=None,
                      speed_ratio=1.0, window_name='python-pinocchio', scene_name='world'):
    # Load robots in gepetto viewer
    robots = []
    for i in range(len(trajectory_data)):
        rb = RobotWrapper()
        robot_name = scene_name + '/' + "robot_" + str(i)
        delete_nodes_viewer(scene_name + '/' + robot_name)
        alpha = 1.0
        urdf_path = trajectory_data[i]["urdf"]
        if (urdf_rgba is not None and urdf_rgba[i] is not None):
            alpha = urdf_rgba[i][3]
            urdf_path = get_colorized_urdf(urdf_path, urdf_rgba[i][:3])
        pinocchio_model = trajectory_data[i]["pinocchio_model"]
        if (pinocchio_model is None):
            has_freeflyer = trajectory_data[i]["has_freeflyer"]
            if (has_freeflyer):
                root_joint = pnc.JointModelFreeFlyer()
            else:
                root_joint = None
            initFromURDF(rb, urdf_path, root_joint=root_joint)
        else:
            collision_model = pin.buildGeomFromUrdf(pinocchio_model, urdf_path, [], pin.GeometryType.COLLISION)
            visual_model = pin.buildGeomFromUrdf(pinocchio_model, urdf_path, [], pin.GeometryType.VISUAL)
            rb.__init__(model=pinocchio_model, collision_model=collision_model, visual_model=visual_model)
        client = Client()
        if not scene_name in client.gui.getSceneList():
            client.gui.createSceneWithFloor(scene_name)
        if not window_name in client.gui.getWindowList():
            window_id = client.gui.createWindow(window_name)
            client.gui.addSceneToWindow(scene_name, window_id)
            client.gui.createGroup(scene_name + '/' + scene_name)
            client.gui.addLandmark(scene_name + '/' + scene_name, 0.1)
        rb.initDisplay(window_name, scene_name, loadModel=False)
        rb.loadDisplayModel(robot_name)
        client.gui.setFloatProperty(scene_name + '/' + robot_name, "Transparency", 1-alpha)
        if (xyz_offset is not None and xyz_offset[i] is not None):
            q = trajectory_data[i]["evolution_robot"][0].q.copy() # Make sure to use a copy to avoid altering the original data
            q[:3] += xyz_offset[i]
        else:
            q = trajectory_data[i]["evolution_robot"][0].q
        rb.display(q)
        robots.append(rb)

    if (xyz_offset is None):
        xyz_offset = [None for i in range(len(trajectory_data))]

    # Animate the robot
    def display_robot(rb, evolution_robot, speed_ratio, xyz_offset=None):
        global lock # Share the same lock on each thread (python 2 does not support `nonlocal` keyword)

        t = [s.t for s in evolution_robot]
        i = 0
        init_time = time.time()
        while i < len(evolution_robot):
            s = evolution_robot[i]
            if (xyz_offset is not None):
                q = s.q.copy() # Make sure to use a copy to avoid altering the original data
                q[:3] += xyz_offset
            else:
                q = s.q
            with lock: # It is necessary to use lock since corbaserver does not support multiple connection simultaneously.
                rb.display(q)
            t_simu = (time.time() - init_time) * speed_ratio
            i = bisect_right(t, t_simu)
            if t_simu < s.t:
                time.sleep(s.t - t_simu)

    threads = []
    for i in range(len(trajectory_data)):
        threads.append(Thread(target=display_robot, args=(robots[i], trajectory_data[i]["evolution_robot"], speed_ratio, xyz_offset[i])))
    for i in range(len(trajectory_data)):
        threads[i].start()
    for i in range(len(trajectory_data)):
        threads[i].join()

def get_colorized_urdf(urdf_path, rgb):
    color_string = "%.3f_%.3f_%.3f_1.0" % rgb
    color_tag = "<color rgba=\"%.3f %.3f %.3f 1.0\"" % rgb # don't close tag with '>', in order to handle <color/> and <color></color>
    colorized_tmp_path = os.path.join("/tmp", "colorized_urdf_rgba_" + color_string)
    colorized_urdf_path = os.path.join(colorized_tmp_path, os.path.basename(urdf_path))
    if not os.path.exists(colorized_tmp_path):
        os.makedirs(colorized_tmp_path)

    with open(urdf_path, "r") as urdf_file:
        colorized_contents = urdf_file.read()

    for mesh_fullpath in re.findall('<mesh filename="(.*)"', colorized_contents):
        colorized_mesh_fullpath = os.path.join(colorized_tmp_path, mesh_fullpath[1:])
        colorized_mesh_path = os.path.dirname(colorized_mesh_fullpath)
        if not os.access(colorized_mesh_path, os.F_OK):
            os.makedirs(colorized_mesh_path)
        shutil.copy2(mesh_fullpath, colorized_mesh_fullpath)
        colorized_contents = colorized_contents.replace('"' + mesh_fullpath + '"',
                                                        '"' + colorized_mesh_fullpath + '"', 1)
    colorized_contents = re.sub("<color rgba=\"[\d. ]*\"", color_tag, colorized_contents)

    with open(colorized_urdf_path, "w") as colorized_urdf_file:
        colorized_urdf_file.write(colorized_contents)

    return colorized_urdf_path

def get_gepetto_client(open_if_needed=False):
    try:
        return Client(), None
    except:
        try:
            return Client(), None
        except:
            if (open_if_needed):
                FNULL = open(os.devnull, 'w')
                proc = subprocess.Popen(['gepetto-gui'],
                                       shell=False,
                                       stdout=FNULL,
                                       stderr=FNULL)
                time.sleep(1)
                try:
                    return Client(), proc
                except:
                    try:
                        return Client(), proc
                    except:
                        print("Impossible to open Gepetto-viewer")

    return None, None

def _getViewerNodeName(rb, geometry_object, geometry_type):
    if geometry_type is pin.GeometryType.VISUAL:
        return rb.viewerVisualGroupName + '/' + geometry_object.name
    elif geometry_type is pin.GeometryType.COLLISION:
        return rb.viewerCollisionGroupName + '/' + geometry_object.name

def _updateGeometryPlacements(rb, data, visual=False):
    if visual:
        geom_model = rb.visual_model
        geom_data = rb.visual_data
    else:
        geom_model = rb.collision_model
        geom_data = rb.collision_data

    pin.updateGeometryPlacements(rb.model, data, geom_model, geom_data)

def update_gepetto_viewer(rb, data, client):
    if rb.display_collisions:
        client.gui.applyConfigurations(
            [_getViewerNodeName(rb, collision,pin.GeometryType.COLLISION)
             for collision in rb.collision_model.geometryObjects ],
            [pin.se3ToXYZQUATtuple(rb.collision_data.oMg[rb.collision_model.getGeometryId(collision.name)])
             for collision in rb.collision_model.geometryObjects ]
        )

    if rb.display_visuals:
        _updateGeometryPlacements(rb, data, visual=True)
        client.gui.applyConfigurations(
            [_getViewerNodeName(rb, visual,pin.GeometryType.VISUAL)
                for visual in rb.visual_model.geometryObjects ],
            [pin.se3ToXYZQUATtuple(rb.visual_data.oMg[rb.visual_model.getGeometryId(visual.name)])
                for visual in rb.visual_model.geometryObjects ]
        )

    client.gui.refresh()
