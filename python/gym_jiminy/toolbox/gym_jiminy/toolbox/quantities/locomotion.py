"""Quantities mainly relevant for locomotion tasks on floating-base robots.
"""
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from jiminy_py.core import (  # pylint: disable=no-name-in-module
    multi_array_copyto)
import pinocchio as pin

from gym_jiminy.common.bases import (
    InterfaceJiminyEnv, InterfaceQuantity, AbstractQuantity, QuantityEvalMode)
from gym_jiminy.common.quantities import (
    BaseOdometryPose, ZeroMomentPoint, translate_position_odom)

from ..math import ConvexHull2D


@dataclass(unsafe_hash=True)
class ProjectedSupportPolygon(AbstractQuantity[ConvexHull2D]):
    """Projected support polygon of the robot.

    The projected support polygon is defined as the 2D convex hull of all the
    candidate contact points. It has the major advantage to be agnostic to the
    contact state, unlike the true support polygon. This criterion can be
    viewed as an anticipation of future impact. This decouples vertical foot
    landing and timings from horizontal foot placement in world plane, which
    makes it easier to derive additive reward components.

    .. note::
        This quantity is only supported for robots with specified contact
        points but no collision bodies for now.
    """

    reference_frame: pin.ReferenceFrame
    """Whether to compute the projected support polygon in local frame
    specified by the odometry pose of floating base of the robot or the frame
    located on the position of the floating base with axes kept aligned with
    world frame.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 *,
                 reference_frame: pin.ReferenceFrame = pin.LOCAL,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param reference_frame: Whether to compute the support polygon in local
                                odometry frame, aka 'pin.LOCAL', or aligned
                                with world axes, aka 'pin.LOCAL_WORLD_ALIGNED'.
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Make sure at requested reference frame is supported
        if reference_frame not in (pin.LOCAL, pin.LOCAL_WORLD_ALIGNED):
            raise ValueError("Reference frame must be either 'pin.LOCAL' or "
                             "'pin.LOCAL_WORLD_ALIGNED'.")

        # Backup some user argument(s)
        self.reference_frame = reference_frame

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                odom_pose=(BaseOdometryPose, dict(mode=mode))),
            mode=mode,
            auto_refresh=False)

        # Stacked position in world plane of all the candidate contact points
        self._candidate_xy_batch = np.array([])

        # Define proxy for views of individual position vectors in the stack
        self._candidate_xy_views: Tuple[np.ndarray, ...] = ()

        # Define proxy for positions in world plane of candidate contact points
        self._candidate_xy_refs: Tuple[np.ndarray, ...] = ()

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Make sure that the robot has contact points but no collision bodies.
        # This must be done at runtime as it may change dynamically.
        if self.robot.collision_body_names:
            raise RuntimeError(
                "Robots having collision bodies are not supported for now.")
        if not self.robot.contact_frame_names:
            raise RuntimeError(
                "Only robots having at least one contact are supported.")

        # Get all groups of contact points having the same parent body
        contact_joint_indices: List[int] = []
        contact_positions: List[List[np.ndarray]] = []
        for frame_index in self.robot.contact_frame_indices:
            frame = self.robot.pinocchio_model.frames[frame_index]
            joint_index = frame.parent
            try:
                contact_index = contact_joint_indices.index(joint_index)
            except ValueError:
                contact_index = len(contact_joint_indices)
                contact_joint_indices.append(joint_index)
                contact_positions.append([])
            transform = self.robot.pinocchio_data.oMf[frame_index]
            contact_positions[contact_index].append(transform.translation)

        # Filter out candidate contact points that will never be part of the
        # projected support polygon no matter what, then gather their position
        # in world plane in a single list.
        # One can show that computing of 3D convex hull of the 3D volume before
        # computing the 2D convex hull of its projection on a given plan yields
        # to the exact same result. This 2-steps process is advantageous over,
        # as the first step can be done only once, and computing the 2D convex
        # hull will be faster if there are least points to consider.
        # Note that this procedure is only applicable for fixed 3D volume. This
        # implies that the convex hull of each contact group must be computed
        # separately rather than all at once.
        candidate_xy_refs: List[np.ndarray] = []
        for positions in contact_positions:
            convhull = ConvexHull(np.stack(positions, axis=0))
            candidate_indices = set(
                range(len(positions))).intersection(convhull.vertices)
            candidate_xy_refs += (
                positions[j][:2] for j in candidate_indices)
        self._candidate_xy_refs = tuple(candidate_xy_refs)

        # Allocate memory for stacked position of candidate contact points.
        # Note that Fortran memory layout (row-major) is used for speed up
        # because it preserves contiguity when copying frame data, and because
        # the `ConvexHull2D` would perform one extra copy otherwise.
        self._candidate_xy_batch = np.empty(
            (len(self._candidate_xy_refs), 2), order='C')

        # Refresh proxies
        self._candidate_xy_views = tuple(self._candidate_xy_batch)

    def refresh(self) -> ConvexHull2D:
        # Copy all translations in contiguous buffer
        multi_array_copyto(self._candidate_xy_views, self._candidate_xy_refs)

        # Translate candidate contact points from world to local odometry frame
        if self.reference_frame == pin.LOCAL:
            translate_position_odom(self._candidate_xy_batch,
                                    self.odom_pose.get(),
                                    self._candidate_xy_batch)

        # Compute the 2D convex hull in world plane
        return ConvexHull2D(self._candidate_xy_batch)


@dataclass(unsafe_hash=True)
class StabilityMarginProjectedSupportPolygon(InterfaceQuantity[float]):
    """Signed distance of the Zero Moment Point (ZMP) from the borders of the
    projected support polygon on world plane.

    The distance is positive if the ZMP lies inside the support polygon,
    negative otherwise.

    To keep balance on flat ground, it is sufficient to maintain the ZMP inside
    the projected support polygon at all time. In addition, the larger the
    distance from the borders of the support polygon, the more "stable" the
    robot. This means that it can withstand larger external impulse forces in
    any direction before starting tiling around an edge of the convex hull.

    .. seealso::
        For an primer about the most common stability assessment criteria for
        legged robots, see "Learning and Optimization of the Locomotion with an
        Exoskeleton for Paraplegic People", A. Duburcq, 2022, Chap.2.2.2, p.33.

    .. note::
        This quantity is only supported for robots with specified contact
        points but no collision bodies for now.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Backup some user argument(s)
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                support_polygon=(ProjectedSupportPolygon, dict(
                    reference_frame=pin.LOCAL_WORLD_ALIGNED,
                    mode=mode)),
                zmp=(ZeroMomentPoint, dict(
                    reference_frame=pin.LOCAL_WORLD_ALIGNED,
                    mode=mode))),
            auto_refresh=False)

    def refresh(self) -> float:
        support_polygon, zmp = self.support_polygon.get(), self.zmp.get()
        return - support_polygon.get_distance_to_point(zmp).item()
