import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym

from panda3d.core import VBase4, Point3, Vec3
from jiminy_py.viewer import Viewer
import pinocchio as pin

Viewer.close()
#Viewer.connect_backend("panda3d-sync")
env = gym.make("gym_jiminy.envs:atlas-pid", viewer_kwargs={"backend": "panda3d-sync"})
env.reset(seed=0)
env.step(env.action)
#env.render()
env.simulator.render(return_rgb_array=True)

env.viewer.add_marker("sphere",
                      shape="sphere",
                      pose=(np.array((1.7, 0.0, 1.5)), None),
                      color="red",
                      radius=0.1,
                      always_foreground=False)

Viewer.add_camera("rgb", height=200, width=150, is_depthmap=False)
Viewer.add_camera("depth", height=128, width=128, is_depthmap=True)
Viewer.set_camera_transform(
    position=[2.5, -1.4, 1.6],  # [3.0, 0.0, 0.0],
    rotation=[1.35, 0.0, 0.8],  # [np.pi/2, 0.0, np.pi/2]
    camera_name="depth")

frame_index = env.robot.pinocchio_model.getFrameId("head")
frame_pose = env.robot.pinocchio_data.oMf[frame_index]
# Viewer._backend_obj.gui.set_camera_transform(
#     pos=frame_pose.translation + np.array([0.0, 0.0, 0.0]),
#     quat=pin.Quaternion(frame_pose.rotation @ pin.rpy.rpyToMatrix(0.0, 0.0, -np.pi/2)).coeffs(),
#     camera_name="rgb")
Viewer.set_camera_transform(
    position=frame_pose.translation + np.array([0.0, 0.0, 0.0]),
    rotation=pin.rpy.matrixToRpy(frame_pose.rotation @ pin.rpy.rpyToMatrix(np.pi/2, 0.0, -np.pi/2)),
    camera_name="rgb")

lens = Viewer._backend_obj.render.find("user_camera_depth").node().get_lens()
# proj = lens.get_projection_mat_inv()
# buffer = Viewer._backend_obj._user_buffers["depth"]
# buffer.trigger_copy()
# Viewer._backend_obj.graphics_engine.render_frame()
# texture = buffer.get_texture()
# tex_peeker = texture.peek()
# pixel = VBase4()
# tex_peeker.lookup(pixel, 0.5, 0.5)  # (y, x normalized coordinates, from top-left to bottom-right)
# depth_rel = 2.0 * pixel[0] - 1.0   # map range [0.0 (near), 1.0 (far)] to [-1.0, 1.0]
# point = Point3()
# #lens.extrude_depth(Point3(0.0, 0.0, depth_rel), point)
# # proj.xform_point_general(Point3(0.0, 0.0, pixel[0]))
# # depth = point[1]
# depth = 1.0 / (proj[2][3] * depth_rel + proj[3][3])
# print(depth)

rgb_array = Viewer.capture_frame(camera_name="rgb")
depth_array = Viewer.capture_frame(camera_name="depth")
# depth_normalized_array = lens.near / (lens.far - (lens.far - lens.near) * depth_array)
depth_true_array = lens.near / (1.0 - (1.0 - lens.near / lens.far) * depth_array)
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax1.imshow(rgb_array)
ax2 = fig.add_subplot(122)
ax2.imshow(depth_true_array, cmap=plt.cm.binary)
for ax in (ax1, ax2):
    ax.axis('off')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
fig.tight_layout(pad=1.0)
plt.show(block=False)
