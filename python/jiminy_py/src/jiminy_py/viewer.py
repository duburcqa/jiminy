#!/usr/bin/env python

## @file jiminy_py/viewer.py

import os
import re
import time
import shutil
import tempfile
import subprocess
import numpy as np
from bisect import bisect_right
from threading import Thread, Lock
from PIL import Image

import pinocchio as pnc
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio import libpinocchio_pywrap as pin
from pinocchio import Quaternion, SE3, se3ToXYZQUAT
from pinocchio.rpy import rpyToMatrix

# Determine if Gepetto-Viewer is available
try:
    import gepetto as _gepetto
    is_gepetto_available = True
except ImportError:
    is_gepetto_available = False


def sleep(dt):
    ''' 
        @brief   Function to provide cross-plateform time sleep with maximum accuracy.
    
        @details Use this method with cautious since it relies on busy looping principle instead of system scheduler.
                 As a result, it wastes a lot more resources than time.sleep. However, it is the only way to ensure  
                 accurate delay on a non-real-time systems such as Windows 10.
    '''
    _ = time.perf_counter() + dt
    while time.perf_counter() < _:
        pass


class Viewer:
    backend = None
    port_forwarding = None
    _backend_obj = None
    _backend_exception = None
    _backend_proc = None
    ## Unique threading.Lock for every simulation.
    # It is required for parallel rendering since corbaserver does not support multiple connection simultaneously.
    _lock = Lock()

    def __init__(self, robot, use_theoretical_model=False,
                 urdf_rgba=None, robot_index=0,
                 backend=None, window_name='python-pinocchio', scene_name='world'):
        # Backup some user arguments
        self.urdf_path = robot.urdf_path
        self.scene_name = scene_name
        self.window_name = window_name
        self.use_theoretical_model = use_theoretical_model

        # Extract the right Pinocchio model
        if self.use_theoretical_model:
            self.pinocchio_model = robot.pinocchio_model_th
            self.pinocchio_data = robot.pinocchio_data_th
        else:
            self.pinocchio_model = robot.pinocchio_model
            self.pinocchio_data = robot.pinocchio_data

        # Select the desired backend
        if backend is None:
            if Viewer.backend is None:
                if Viewer._is_notebook() or not is_gepetto_available:
                    backend = 'meshcat'
                else:
                    backend = 'gepetto-gui'
            else:
                backend = Viewer.backend

        # Update the backend currently running, if any
        if (Viewer.backend != backend) and \
           (Viewer._backend_obj is not None or \
            Viewer._backend_proc is not None):
            Viewer.close()
            print("Different backend already running. Closing it...")
        Viewer.backend = backend

        # Check if the backend is still available, if any
        if Viewer._backend_obj is not None and Viewer._backend_proc is not None:
            if Viewer._backend_proc.poll() is not None:
                Viewer._backend_obj = None
                Viewer._backend_proc = None
                Viewer._backend_exception = None

        # Access the current backend or create one if none is available
        try:
            if (Viewer.backend == 'gepetto-gui'):
                import omniORB
                Viewer._backend_exception = omniORB.CORBA.COMM_FAILURE
                if Viewer._backend_obj is None:
                    Viewer._backend_obj, Viewer._backend_proc = \
                        Viewer._get_gepetto_client(True)
                if  Viewer._backend_obj is not None:
                    self._client = Viewer._backend_obj.gui
                else:
                    raise RuntimeError("Impossible to open Gepetto-viewer.")
            else:
                from pinocchio.visualize import MeshcatVisualizer
                from pinocchio.shortcuts import createDatas

                if Viewer._backend_obj is None:
                    Viewer._create_meshcat_backend()
                    if Viewer._is_notebook():
                        Viewer.display_jupyter_cell()
                    else:
                        Viewer._backend_obj.open()

                self._client = MeshcatVisualizer(self.pinocchio_model, None, None)
                self._client.viewer = Viewer._backend_obj
        except:
            raise RuntimeError("Impossible to load backend.")

        # Create a RobotWrapper
        robot_name = "robot_" + str(robot_index)
        if (Viewer.backend == 'gepetto-gui'):
            Viewer._delete_gepetto_nodes_viewer(scene_name + '/' + robot_name)
            if (urdf_rgba is not None):
                alpha = urdf_rgba[3]
                self.urdf_path = Viewer._get_colorized_urdf(self.urdf_path, urdf_rgba[:3])
            else:
                alpha = 1.0
        collision_model = pin.buildGeomFromUrdf(self.pinocchio_model, self.urdf_path,
                                                os.environ.get('JIMINY_MESH_PATH', []),
                                                pin.GeometryType.COLLISION)
        visual_model = pin.buildGeomFromUrdf(self.pinocchio_model, self.urdf_path,
                                             os.environ.get('JIMINY_MESH_PATH', []),
                                             pin.GeometryType.VISUAL)
        self._rb = RobotWrapper(model=self.pinocchio_model,
                                collision_model=collision_model,
                                visual_model=visual_model)
        if not self.use_theoretical_model:
            self._rb.data = robot.pinocchio_data
        self.pinocchio_data = self._rb.data

        # Load robot in the backend viewer
        if (Viewer.backend == 'gepetto-gui'):
            if not scene_name in self._client.getSceneList():
                self._client.createSceneWithFloor(scene_name)
            if not window_name in self._client.getWindowList():
                self._window_id = self._client.createWindow(window_name)
                self._client.addSceneToWindow(scene_name, self._window_id)
                self._client.createGroup(scene_name + '/' + scene_name)
                self._client.addLandmark(scene_name + '/' + scene_name, 0.1)
            else:
                self._window_id = int(np.where([name == window_name
                                            for name in self._client.getWindowList()])[0][0])
            self._rb.initViewer(windowName=window_name, sceneName=scene_name, loadModel=False)
            self._rb.loadViewerModel(robot_name)
            self._client.setFloatProperty(scene_name + '/' + robot_name,
                                          'Transparency', 1 - alpha)
        else:
            self._client.collision_model = collision_model
            self._client.visual_model = visual_model
            self._client.data, self._client.collision_data, self._client.visual_data = \
                createDatas(self.pinocchio_model, collision_model, visual_model)
            self._client.loadViewerModel(rootNodeName=robot_name, color=urdf_rgba)
            self._rb.viz = self._client

    @staticmethod
    def _create_meshcat_backend():
        import meshcat
        from contextlib import redirect_stdout

        with redirect_stdout(None):
            Viewer._backend_obj = meshcat.Visualizer()
            Viewer._backend_proc = Viewer._backend_obj.window.server_proc

    @staticmethod
    def reset_port_forwarding(port_forwarding=None):
        Viewer.port_forwarding = port_forwarding

    @staticmethod
    def display_jupyter_cell(height=600, width=900, force_create_backend=False):
        if Viewer.backend == 'meshcat' and Viewer._is_notebook():
            from IPython.core.display import HTML, display as ipython_display

            if Viewer._backend_obj is None:
                if force_create_backend:
                    Viewer._create_meshcat_backend()
                else:
                    raise ValueError("No meshcat backend available and 'force_create_backend' is set to False.")

            viewer_url = Viewer._backend_obj.url()
            if Viewer.port_forwarding is not None:
                url_port_pattern = '(?<=:)[0-9]+(?=/)'
                port_localhost = int(re.search(url_port_pattern, viewer_url).group())
                if port_localhost in Viewer.port_forwarding.keys():
                    viewer_url = re.sub(url_port_pattern, str(Viewer.port_forwarding[port_localhost]), viewer_url)
                else:
                    print("Port forwarding defined but no port mapping associated with {port_localhost}.")

            jupyter_html = f'\n<div style="height: {height}px; width: {width}px; overflow-x: auto; overflow-y: hidden; resize: both">\
                             \n<iframe src="{viewer_url}" style="width: 100%; height: 100%; border: none">\
                             </iframe>\n</div>\n'

            ipython_display(HTML(jupyter_html))
        else:
            raise ValueError("Display in a Jupyter cell is only available using 'meshcat' backend and within a Jupyter notebook.")

    @staticmethod
    def close():
        if Viewer._backend_proc is not None:
            if Viewer._backend_proc.poll() is not None:
                Viewer._backend_proc.terminate()
            Viewer._backend_proc = None
        Viewer._backend_obj = None
        Viewer._backend_exception = None

    @staticmethod
    def _is_notebook():
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type, if any
        except NameError:
            return False      # Probably standard Python interpreter

    @staticmethod
    def _get_colorized_urdf(urdf_path, rgb, custom_mesh_search_path=None):
        """
        @brief      Generate a unique colorized URDF.

        @remark     Multiple identical URDF model of different colors can be
                    loaded in Gepetto-viewer this way.

        @param[in]  urdf_path     Full path of the URDF file
        @param[in]  rgb           RGB code defining the color of the model. It is the same for each link.

        @return     Full path of the colorized URDF file.
        """

        color_string = "%.3f_%.3f_%.3f_1.0" % rgb
        color_tag = "<color rgba=\"%.3f %.3f %.3f 1.0\"" % rgb # don't close tag with '>', in order to handle <color/> and <color></color>
        colorized_tmp_path = os.path.join("/tmp", "colorized_urdf_rgba_" + color_string)
        colorized_urdf_path = os.path.join(colorized_tmp_path, os.path.basename(urdf_path))
        if not os.path.exists(colorized_tmp_path):
            os.makedirs(colorized_tmp_path)

        with open(urdf_path, 'r') as urdf_file:
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

        with open(colorized_urdf_path, 'w') as colorized_urdf_file:
            colorized_urdf_file.write(colorized_contents)

        return colorized_urdf_path

    @staticmethod
    def _get_gepetto_client(open_if_needed=False):
        """
        @brief      Get a pointer to the running process of Gepetto-Viewer.

        @details    This method can be used to open a new process if necessary.
        .
        @param[in]  open_if_needed      Whether a new process must be opened if
                                        no running process is found.
                                        Optional: False by default

        @return     A pointer to the running Gepetto-viewer Client and its PID.
        """

        try:
            from gepetto.corbaserver.client import Client
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
                    time.sleep(1.0)
                    try:
                        return Client(), proc
                    except:
                        try:
                            return Client(), proc
                        except:
                            print("Impossible to open Gepetto-viewer")

        return None, None

    def _delete_gepetto_nodes_viewer(self, *nodes_path):
        """
        @brief      Delete a 'node' in Gepetto-viewer.

        @remark     Be careful, one must specify the full path of a node, including
                    all parent group, but without the window name, ie
                    'scene_name/robot_name' to delete the robot.

        @param[in]  nodes_path     Full path of the node to delete
        """

        if Viewer.backend == 'gepetto-gui':
            for node_path in nodes_path:
                if node_path in self._client.getNodeList():
                    self._client.deleteNode(node_path, True)

    def _getViewerNodeName(self, geometry_object, geometry_type):
        """
        @brief      Get the full path of a node associated with a given geometry
                    object and geometry type.

        @remark     This is a hidden function that is not automatically imported
                    using 'from wdc_jiminy_py import *'. It is not intended to
                    be called manually.

        @param[in]  geometry_object     Geometry object from which to get the node
        @param[in]  geometry_type       Geometry type. It must be either
                                        pin.GeometryType.VISUAL or pin.GeometryType.COLLISION
                                        for display and collision, respectively.

        @return     Full path of the associated node.
        """
        if geometry_type is pin.GeometryType.VISUAL:
            return self._rb.viz.viewerVisualGroupName + '/' + geometry_object.name
        elif geometry_type is pin.GeometryType.COLLISION:
            return self._rb.viz.viewerCollisionGroupName + '/' + geometry_object.name

    def _updateGeometryPlacements(self, visual=False):
        """
        @brief      Update the generalized position of a geometry object.

        @remark     This is a hidden function that is not automatically imported
                    using 'from wdc_jiminy_py import *'. It is not intended to
                    be called manually.

        @param[in]  visual      Wether it is a visual or collision update
        """
        if visual:
            geom_model = self._rb.visual_model
            geom_data = self._rb.visual_data
        else:
            geom_model = self._rb.collision_model
            geom_data = self._rb.collision_data

        pin.updateGeometryPlacements(self.pinocchio_model,
                                     self.pinocchio_data,
                                     geom_model, geom_data)

    def setCameraTransform(self, translation, rotation):
        # translation : [Px, Py, Pz],
        # rotation : [Roll, Pitch, Yaw]

        R_pnc = rpyToMatrix(np.array(rotation))
        if Viewer.backend == 'gepetto-gui':
            T_pnc = np.array(translation)
            T_R = SE3(R_pnc, T_pnc)
            self._client.setCameraTransform(self._window_id, se3ToXYZQUAT(T_R).tolist())
        else:
            import meshcat.transformations as tf
            # Transformation of the camera object
            T_meshcat = tf.translation_matrix(translation)
            self._client.viewer["/Cameras/default/rotated/<object>"].set_transform(T_meshcat)
            # Orientation of the camera object
            Q_pnc = Quaternion(R_pnc).coeffs()
            Q_meshcat = np.roll(Q_pnc, shift=1)
            R_meshcat = tf.quaternion_matrix(Q_meshcat)
            self._client.viewer["/Cameras/default"].set_transform(R_meshcat)

    def captureFrame(self):
        if Viewer.backend == 'gepetto-gui':
            png_path = next(tempfile._get_candidate_names())
            self._client.captureFrame(self._window_id, png_path)
            rgb_array = np.array(Image.open(png_path))[:, :, :-1]
            os.remove(png_path)
            return rgb_array
        else:
            raise RuntimeError("Screen capture through Python only available using 'gepetto-gui' backend.")


    def refresh(self):
        """
        @brief      Refresh the configuration of Robot in the viewer.
        """

        if self.use_theoretical_model:
            raise RuntimeError("'Refresh' method only available if 'use_theoretical_model'=False.")

        if Viewer.backend == 'gepetto-gui':
            if self._rb.displayCollisions:
                self._client.applyConfigurations(
                    [self._getViewerNodeName(collision, pin.GeometryType.COLLISION)
                    for collision in self._rb.collision_model.geometryObjects],
                    [pin.se3ToXYZQUATtuple(self._rb.collision_data.oMg[\
                        self._rb.collision_model.getGeometryId(collision.name)])
                    for collision in self._rb.collision_model.geometryObjects]
                )

            if self._rb.displayVisuals:
                self._updateGeometryPlacements(visual=True)
                self._client.applyConfigurations(
                    [self._getViewerNodeName(visual, pin.GeometryType.VISUAL)
                        for visual in self._rb.visual_model.geometryObjects],
                    [pin.se3ToXYZQUATtuple(self._rb.visual_data.oMg[\
                        self._rb.visual_model.getGeometryId(visual.name)])
                        for visual in self._rb.visual_model.geometryObjects]
                )

            self._client.refresh()
        else:
            self._updateGeometryPlacements(visual=True)
            for visual in self._rb.visual_model.geometryObjects:
                T = self._rb.visual_data.oMg[\
                    self._rb.visual_model.getGeometryId(visual.name)].homogeneous
                self._client.viewer[\
                    self._getViewerNodeName(visual, pin.GeometryType.VISUAL)].set_transform(T)

    def display(self, evolution_robot, speed_ratio, xyz_offset=None):
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
            with Viewer._lock: # It is necessary to use Lock since corbaserver does not support multiple connection simultaneously.omniORB
                try:
                    self._rb.display(q)
                except Viewer._backend_exception:
                    break
            t_simu = (time.time() - init_time) * speed_ratio
            i = bisect_right(t, t_simu)
            sleep(s.t - t_simu)


def play_trajectories(trajectory_data, xyz_offset=None, urdf_rgba=None, speed_ratio=1.0,
                      backend=None, window_name='python-pinocchio', scene_name='world',
                      close_backend=None):
    """!
    @brief      Display robot evolution in Gepetto-viewer at stable speed.

    @remark     The speed is independent of the machine, and more
                specifically of CPU power.

    @param[in]  trajectory_data     Trajectory dictionary with keys:
                                    'evolution_robot': list of State object of increasing time
                                    'robot': jiminy robot (None if omitted)
                                    'use_theoretical_model':  whether the theoretical or actual model must be used
    @param[in]  xyz_offset          Constant translation of the root joint in world frame (1D numpy array)
    @param[in]  urdf_rgba           RGBA code defining the color of the model. It is the same for each link.
                                    Optional: Original colors of each link. No alpha.
    @param[in]  speed_ratio         Speed ratio of the simulation
    @param[in]  window_name         Name of the Gepetto-viewer's window in which to display the robot.
                                    Optional: Common default name if omitted.
    @param[in]  scene_name          Name of the Gepetto-viewer's scene in which to display the robot.
                                    Optional: Common default name if omitted.
    """

    if (close_backend is None):
        # Close backend if it was not available beforehand
        close_backend = Viewer._backend_obj is None

    # Load robots in gepetto viewer
    robots = []
    for i in range(len(trajectory_data)):
        robot = trajectory_data[i]['robot']
        use_theoretical_model = trajectory_data[i]['use_theoretical_model']
        robot = Viewer(robot, use_theoretical_model=use_theoretical_model,
                       urdf_rgba=urdf_rgba[i] if urdf_rgba is not None else None, robot_index=i,
                       backend=backend, window_name=window_name, scene_name=scene_name)

        if (xyz_offset is not None and xyz_offset[i] is not None):
            q = trajectory_data[i]['evolution_robot'][0].q.copy()
            q[:3] += xyz_offset[i]
        else:
            q = trajectory_data[i]['evolution_robot'][0].q
        try:
            robot._rb.display(q)
        except Viewer._backend_exception:
            break
        robots.append(robot)

    if (xyz_offset is None):
        xyz_offset = len(trajectory_data) * (None,)

    threads = []
    for i in range(len(trajectory_data)):
        threads.append(Thread(target=robots[i].display,
                              args=(trajectory_data[i]['evolution_robot'],
                                    speed_ratio, xyz_offset[i])))
    for i in range(len(trajectory_data)):
        threads[i].start()
    for i in range(len(trajectory_data)):
        threads[i].join()

    if close_backend:
        Viewer.close()
