"""Implementation of a stateless kinematic deformation estimator block
compatible with gym_jiminy reinforcement learning pipeline environment design.
"""
from collections import OrderedDict
from collections.abc import Mapping
from typing import List, Dict, Sequence, Tuple, Optional

import numpy as np
import numba as nb
import gymnasium as gym

import jiminy_py.core as jiminy
from jiminy_py.core import (  # pylint: disable=no-name-in-module
    EncoderSensor, ImuSensor, get_frame_indices)

import pinocchio as pin

from ..bases import BaseAct, BaseObs, BaseObserverBlock, InterfaceJiminyEnv
from ..utils import (DataNested,
                     quat_to_rpy,
                     matrices_to_quat,
                     quat_multiply,
                     compute_tilt_from_quat,
                     swing_from_vector)


# FIXME: Enabling cache causes segfault on Apple Silicon
@nb.jit(nopython=True, cache=False, inline='always')
def _compute_orientation_error(obs_imu_quats: np.ndarray,
                               obs_imu_indices: Tuple[int, ...],
                               inv_obs_imu_quats: np.ndarray,
                               kin_imu_rots: Tuple[np.ndarray, ...],
                               kin_imu_quats: np.ndarray,
                               dev_imu_quats: np.ndarray,
                               ignore_twist: bool) -> None:
    """Compute total deviation of observed IMU data wrt to theoretical model in
    world frame.
    """
    # Re-order IMU data
    for i, imu_index in enumerate(obs_imu_indices):
        inv_obs_imu_quats[:, i] = obs_imu_quats[:, imu_index]

    # Compute orientation error
    if ignore_twist:
        # Compute swing of deviation between observed and kinematic IMU data:
        # 1. Extract observed IMU tilt
        # 2. Apply kinematic IMU rotation on it
        # 3. Reconstruct swing from tilt of deviation
        # This approach is equivalent to removing the twist from the exact
        # deviation directly, but much faster as it does not require computing
        # the observed IMU quaternions nor to compose rotations.

        # Extract observed tilt: w_R_obs.T @ e_z
        obs_imu_tilts = np.stack(compute_tilt_from_quat(inv_obs_imu_quats), 1)

        # Apply theoretical kinematic IMU rotation on observed tilt.
        # The result can be interpreted as the tilt error between observed
        # and theoretical kinematic IMU orientation in world frame, ie:
        # w_tilt_err = (w_R_kin @ w_R_obs.T) @ e_z
        for i, kin_imu_rot in enumerate(kin_imu_rots):
            obs_imu_tilts[i] = kin_imu_rot @ obs_imu_tilts[i]

        # Compute "smallest" rotation that can explain the tilt error.
        swing_from_vector(obs_imu_tilts.T, dev_imu_quats)

        # Conjugate quaternion of IMU deviation
        dev_imu_quats[-1] *= -1
    else:
        # Convert kinematic IMU rotation matrices to quaternions
        matrices_to_quat(kin_imu_rots, kin_imu_quats)

        # Conjugate quaternion of IMU orientation
        inv_obs_imu_quats[-1] *= -1

        # Compute one-by-one difference between observed and theoretical
        # kinematic IMU orientations.
        quat_multiply(kin_imu_quats,
                      inv_obs_imu_quats,
                      dev_imu_quats)


@nb.jit(nopython=True, cache=True, inline='always')
def _compute_deformation_from_deviation(kin_flex_rots: Tuple[np.ndarray, ...],
                                        kin_flex_quats: np.ndarray,
                                        is_flex_flipped: np.ndarray,
                                        is_chain_orphan: Tuple[bool, bool],
                                        dev_imu_quats: np.ndarray,
                                        inv_child_dev_imu_quats: np.ndarray,
                                        parent_dev_imu_quats: np.ndarray,
                                        deformation_flex_quats: np.ndarray
                                        ) -> None:
    """Cast observed IMU deviations in world frame into deformations at their
    corresponding flexibility frame.
    """
    # Convert kinematic flexibility rotation matrices to quaternions
    matrices_to_quat(kin_flex_rots, kin_flex_quats)

    # Translate observed IMU deviation in world frame to their respective
    # parent and child flex frames.
    is_flex_parent_orphan, is_flex_child_orphan = is_chain_orphan
    if is_flex_parent_orphan:
        quat_multiply(
            dev_imu_quats, kin_flex_quats[:, 1:], parent_dev_imu_quats[:, 1:])
        if is_flex_child_orphan:
            quat_multiply(dev_imu_quats,
                          kin_flex_quats[:, :-1],
                          inv_child_dev_imu_quats[:, :-1])
        else:
            quat_multiply(
                dev_imu_quats, kin_flex_quats, inv_child_dev_imu_quats)
    else:
        nflexs = len(kin_flex_rots)
        quat_multiply(
            dev_imu_quats[:, :nflexs], kin_flex_quats, parent_dev_imu_quats)
        if is_flex_child_orphan:
            quat_multiply(dev_imu_quats[:, 1:],
                          kin_flex_quats[:, :-1],
                          inv_child_dev_imu_quats[:, :-1])
        else:
            quat_multiply(
                dev_imu_quats[:, 1:], kin_flex_quats, inv_child_dev_imu_quats)

    if is_flex_parent_orphan:
        parent_dev_imu_quats[:, 0] = kin_flex_quats[:, 0]
    if is_flex_child_orphan:
        inv_child_dev_imu_quats[:, -1] = kin_flex_quats[:, -1]

    inv_child_dev_imu_quats[-1] *= -1

    # Project observed total deformation on flexibility frames
    quat_multiply(inv_child_dev_imu_quats,
                  parent_dev_imu_quats,
                  deformation_flex_quats)

    # Conjugate deformation quaternion if triplet (parent IMU, flex, child IMU)
    # is reversed wrt to the standard joint ordering from URDF.
    deformation_flex_quats[-1][is_flex_flipped] *= -1


# FIXME: Enabling cache causes segfault on Apple Silicon
@nb.jit(nopython=True, cache=False)
def flexibility_estimator(obs_imu_quats: np.ndarray,
                          obs_imu_indices: Tuple[int, ...],
                          inv_obs_imu_quats: np.ndarray,
                          kin_imu_rots: Tuple[np.ndarray, ...],
                          kin_imu_quats: np.ndarray,
                          kin_flex_rots: Tuple[np.ndarray, ...],
                          kin_flex_quats: np.ndarray,
                          is_flex_flipped: np.ndarray,
                          is_chain_orphan: Tuple[bool, bool],
                          dev_imu_quats: np.ndarray,
                          inv_child_dev_imu_quats: np.ndarray,
                          parent_dev_imu_quats: np.ndarray,
                          deformation_flex_quats: np.ndarray,
                          ignore_twist: bool) -> None:
    """Compute the local deformation at an arbitrary set of flexibility points
    that are presumably responsible for most of the whole deformation of the
    mechanical structure.

    .. warning::
        The local deformation at each flexibility frame must be observable, ie
        the flexibility and IMU frames interleave with each others in a unique
        and contiguous sub-chain in theoretical kinematic tree of the robot.

    :param obs_imu_quats: Orientation estimates of an unordered arbitrary set
                          of IMUs as a 2D array whose first dimension gathers
                          the 4 quaternion coordinates [qx, qy, qz, qw] while
                          the second corresponds to N independent IMU data.
    :param obs_imu_indices: M-tuple of ordered IMU indices of interest for
                            which the total deviation will be computed.
    :param inv_obs_imu_quats: Pre-allocated memory for storing the inverse of
                              the orientation estimates for an ordered subset
                              of the IMU data `obs_imu_quats` according to
                              `obs_imu_indices`.
    :param kin_imu_rots: Tuple of M kinematic frame orientations corresponding
                         to the ordered subset of IMUs `obs_imu_indices`, for
                         the configuration of the theoretical robot model.
    :param kin_imu_quats: Pre-allocated memory for storing the kinematic frame
                          orientations of the ordered subset of IMUs of
                          interest as a 2D array whose first dimension gathers
                          the 4 quaternion coordinates while the second
                          corresponds to the M independent IMUs.
    :param kin_flex_rots: Tuple of K kinematic frame orientations for all the
                          flexibility points that interleaves with the ordered
                          subset of IMUs of interest in the kinematic tree.
    :param kin_flex_quats: Pre-allocated memory for storing the kinematic frame
                           orientations of the flexibility points that
                           interleaves with the ordered subset of IMUs of
                           interest as a 2D array whose first dimension gathers
                           the 4 quaternion coordinates while the second
                           corresponds to the K independent flexibility points.
    :param is_flex_flipped: Whether local deformation estimates for each
                            flexibility point must be inverted to be consistent
                            with standard URDF convention as 1D boolean array.
    :param is_chain_orphan: 2-Tuple stating whether first and last flexibility
                            point is orphan respectively, ie only a single IMU
                            is available for estimating its local deformation.
    :param dev_imu_quats: Total deviation of th observed IMU data wrt to the
                          theoretical model in world frame for the ordered
                          subset of IMUs of interest, as a 2D array whose first
                          dimension gathers the 4 quaternion coordinates while
                          the second corresponds to the M independent IMUs.
    :param inv_child_dev_imu_quats:
        Total deviation observed IMU data in child flexibility frame as a 2D
        array whose first dimension gathers the 4 quaternion coordinates while
        the second corresponds to the K independent flexibility frames.
    :param parent_dev_imu_quats:
        Total deviation observed IMU data in parent flexibility frame as a 2D
        array whose first dimension gathers the 4 quaternion coordinates while
        the second corresponds to the K independent flexibility frames.
    :param deformation_flex_quats:
        Pre-allocated memory for storing the local deformation estimates for
        each flexibility point flexibility points as a 2D array whose first
        dimension gathers the 4 quaternion coordinates while the second
        corresponds to the K independent flexibility points.
    :param ignore_twist: Whether to ignore the twist of the orientation
                         estimates of the ordered subset of IMUs of interest,
                         and incidentally the twist of deformation at the
                         flexibility points.
    """
    # Compute error between observed and theoretical kinematic IMU orientation
    _compute_orientation_error(obs_imu_quats,
                               obs_imu_indices,
                               inv_obs_imu_quats,
                               kin_imu_rots,
                               kin_imu_quats,
                               dev_imu_quats,
                               ignore_twist)

    # Project observed total deformation on flexibility frames
    _compute_deformation_from_deviation(kin_flex_rots,
                                        kin_flex_quats,
                                        is_flex_flipped,
                                        is_chain_orphan,
                                        dev_imu_quats,
                                        inv_child_dev_imu_quats,
                                        parent_dev_imu_quats,
                                        deformation_flex_quats)


def get_flexibility_imu_frame_chains(
        pinocchio_model: pin.Model,
        flex_joint_names: Sequence[str],
        imu_frame_names: Sequence[str]) -> Sequence[Tuple[
            Sequence[str], Sequence[Optional[str]], Sequence[bool]]]:
    """Extract the minimal set of contiguous sub-chains in kinematic tree of a
    given model that goes through all the specified flexibility and IMU frames.

    :param pinocchio_model: Model from which to extract sub-chains.
    :param flex_joint_names: Unordered sequence of joint names that must be
                             considered as associated with flexibility frames.
    :param imu_frame_names: Unordered sequence of frame names that must be
                             considered as associated with IMU frames.
    """
    # Determine the leaf joints of the kinematic tree, ie all the joints that
    # are not parent of any other joint.
    parents = pinocchio_model.parents
    leaf_joint_indices = set(range(len(parents))) - set(parents)

    # Compute the support of each leaf joint, ie the sub-chain going from each
    # leaf to the root joint.
    supports = []
    for joint_index in leaf_joint_indices:
        support = []
        while True:
            support.append(joint_index)
            if joint_index == 0:
                break
            joint_index = parents[joint_index]
        supports.append(support)

    # Deduce all the kinematic chains.
    # For each support, check if there is a match in any other chain.
    # The first match (in order) must be the only one to be considered.
    # It always exists, as a root joint is shared by all supports.
    chains = []
    for i, support_1 in enumerate(supports):
        for support_2 in supports[(i + 1):]:
            for joint_index in support_1:
                if joint_index in support_2:
                    break
            support_left = support_1[:(support_1.index(joint_index) + 1)]
            support_right = support_2[:support_2.index(joint_index)]
            chains.append([*support_left, *support_right[::-1]])

    # Special case when there is a chain in the kinematic tree
    if not chains:
        chains.append(supports[0])

    # Determine the parent joint of all flexibility and IMU frames
    flex_joint_map, imu_joint_map = ({
        pinocchio_model.frames[index].parent: name
        for name, index in zip(
            frame_names, get_frame_indices(pinocchio_model, frame_names))}
        for frame_names in (flex_joint_names, imu_frame_names))
    flex_joint_indices, imu_joint_indices = (
        set(joint_map.keys())
        for joint_map in (flex_joint_map, imu_joint_map))

    # Make sure that the robot has no IMU on its root joint if fixed-based
    root_type = jiminy.get_joint_type(pinocchio_model, 1)
    if root_type != jiminy.JointModelType.FREE:
        if 1 in imu_joint_indices:
            raise ValueError(
                "There must not be an IMU frame attached to the root joint of "
                "the robot if it has a fixed based (no freeflyer).")
        if not imu_joint_indices.issuperset(leaf_joint_indices):
            raise ValueError(
                "There must be an IMU frame attached to all the leaf joints "
                "of the robot if it has a fixed based (no freeflyer).")

    # Remove all joints not being flex or having an IMU attached
    flex_imu_joint_chains_all = []
    flex_or_imu_joint_indices = flex_joint_indices | imu_joint_indices
    for chain in chains:
        flex_imu_chain = list(
            i for i in chain if i in flex_or_imu_joint_indices)
        if len(flex_imu_chain) > 1:
            flex_imu_joint_chains_all.append(flex_imu_chain)

    # Remove redundant chains, ie the subchains of some other
    flex_imu_joint_chains = []
    for i, chain_i in enumerate(flex_imu_joint_chains_all):
        for chain_j in flex_imu_joint_chains_all:
            is_subchain = False
            for i in range(0, len(chain_j) - len(chain_i)):
                if chain_i == chain_j[:len(chain_i)]:
                    is_subchain = True
                    break
            if is_subchain:
                break
        else:
            flex_imu_joint_chains.append(chain_i)

    # Duplicate flexibility and IMU frames sharing the same joint indices.
    # Go through each chain and check that IMU and flex joints interleaves.
    duplicated_flex_imu_joint_chains = []
    flex_and_imu_joint_indices = flex_joint_indices & imu_joint_indices
    for chain in flex_imu_joint_chains_all:
        is_imu_joint = False
        duplicated_chain = []
        for joint_index in chain:
            duplicated_chain.append(joint_index)
            if joint_index in flex_and_imu_joint_indices:
                duplicated_chain.append(joint_index)
            else:
                if (is_imu_joint and joint_index in flex_joint_indices) or (
                        not is_imu_joint and joint_index in imu_joint_indices):
                    is_imu_joint = not is_imu_joint
                    continue
                raise ValueError("Flexibility and IMU frames must interleave.")
        duplicated_flex_imu_joint_chains.append(duplicated_chain)

    # Extract triplets (parent IMU, flex, child IMU), where either the parent
    # or child IMU is None for orphan flexibility frames.
    flex_imu_joint_triplets_all: List[
        Tuple[Optional[int], int, Optional[int]]] = []
    imu_and_not_flex_joint_indices = imu_joint_indices - flex_joint_indices
    for chain in duplicated_flex_imu_joint_chains:
        start_index = 0
        if chain[0] not in imu_and_not_flex_joint_indices:
            flex_imu_joint_triplets_all.append(
                (None, *chain[:2]))  # type: ignore[arg-type]
            i = 2
        for j in range(start_index, len(chain) - 2, 2):
            flex_imu_joint_triplets_all.append(
                chain[j:(j + 3)])  # type: ignore[arg-type]
        if chain[-1] not in imu_and_not_flex_joint_indices:
            flex_imu_joint_triplets_all.append(
                (*chain[-2:], None))  # type: ignore[arg-type]

    # Remove redundant triplet, ie associated with the same flexibility frame.
    # Complete triplet must be preferred over orphan triplet.
    flex_imu_joint_triplets = []
    flex_joint_triplets = []
    is_orphan_triplets = []
    for triplet in flex_imu_joint_triplets_all:
        parent_imu_joint, flex_joint, child_imu_joint = triplet
        is_orphan = parent_imu_joint is None or child_imu_joint is None
        if flex_joint not in flex_joint_triplets:
            flex_imu_joint_triplets.append(triplet)
            flex_joint_triplets.append(flex_joint)
            is_orphan_triplets.append(is_orphan)
        elif not is_orphan:
            triplet_index = flex_joint_triplets.index(flex_joint)
            if is_orphan_triplets[triplet_index]:
                flex_imu_joint_triplets[triplet_index] = triplet
                flex_joint_triplets[triplet_index] = flex_joint

    # Concatenate triplets back in frame chains.
    # Note that computations can be vectorized for each independent chain.
    flex_imu_name_chains = []  # [(flexs, imus, flipped), ...]
    imu_grp: List[Optional[str]] = []
    child_joint_imu_prev: Optional[int] = -1
    for triplet in flex_imu_joint_triplets:
        parent_imu_joint, flex_joint, child_imu_joint = triplet
        if parent_imu_joint is None:
            assert child_imu_joint is not None
            is_flipped = flex_joint >= child_imu_joint
        elif child_imu_joint is None:
            assert parent_imu_joint is not None
            is_flipped = parent_imu_joint >= flex_joint
        else:
            is_flipped = triplet != sorted(
                triplet)  # type: ignore[type-var]
        if child_joint_imu_prev != parent_imu_joint:
            flex_grp: List[str] = []
            imu_grp = [imu_joint_map.get(parent_imu_joint, None)]
            is_flipped_grp: List[bool] = []
            flex_imu_name_chains.append((flex_grp, imu_grp, is_flipped_grp))
        child_joint_imu_prev = child_imu_joint
        flex_grp.append(flex_joint_map[flex_joint])
        imu_grp.append(imu_joint_map.get(child_imu_joint, None))
        is_flipped_grp.append(is_flipped)

    return flex_imu_name_chains


class DeformationEstimator(BaseObserverBlock[
        Dict[str, np.ndarray], np.ndarray, BaseObs, BaseAct]):
    """Compute the local deformation at an arbitrary set of flexibility points
    that are presumably responsible for most of the whole deformation of the
    mechanical structure.

    The number of IMU sensors and flexibility frames must be consistent:
        * If the robot has no freeflyer, there must be as many IMU sensors as
          flexibility frames (0), ie

            *---o---x---o---x---o---x
                        |
                        |
                        x---o---x

        * Otherwise, it can either have one more IMU than flexibility frames
          (1), the same number (2), or up to one less IMU per branch in the
          kinematic tree (3).

            (1) x---o---x---o---x---o---x
                            |
                            |
                            x---o---x

            (2) +---o---x---o---x---o---x
                            |
                            |
                            x---o---x

            (3) +---o---x---o---x---o---+
                            |
                            |
                            x---o---+

    *: Fixed base, +: leaf frame, x: IMU frame, o: flexibility frame

    (1): The pose of the freeflyer is ignored when estimating the deformation
         at the flexibility frames in local frame. Mathematically, it is the
         same as (0) when considering a virtual IMU with fixed orientation to
         identity for the root joint.

    (2): One has to compensate for the missing IMU by providing instead the
         configuration of the freeflyer. More precisely, one should ensure that
         the orientation of the parent frame of the orphan flexibility frame
         matches the reality for the theoretical configuration. This usually
         requires making some assumptions to guess to pose of the frame that is
         not directly observable. Any discrepancy will be aggregated to the
         estimated deformation for the orphan flexibility frame specifically
         since both cannot be distinguished. This issue typically happens when
         there is no IMUs in the feet of a legged robot. In such a case, there
         is no better option than assuming that one of the active contact
         bodies remains flat on the ground. If the twist of the IMUs are
         ignored, then the twist of the contact body does not matter either,
         otherwise it must be set appropriately by the user to get a
         meaningless estimate for the twist of the deformation. If it cannot be
         observed by some exteroceptive sensor such as vision, then the most
         reasonable assumption is to suppose that it matches the twist of the
         IMU coming right after in the kinematic tree. This way, they will
         cancel out each other without adding bias to the twist of the orphan
         flexibility frame.

    (3): This case is basically the same as (2), with the addition that only
         the deformation of one of the orphan flexibility frames can be
         estimated at once, namely the one whose parent frame is declared as
         having known orientation. The other ones will be set to identity. For
         a legged robot, this corresponds to one of the contact bodies, usually
         the one holding most of the total weight.
    .. warning::
        (2) and (3) are not supported for now, as it requires using one
        additional observation layer responsible for estimating the theoretical
        configuration of the robot including its freeflyer, along with the name
        of the reference frame, ie the one having known orientation.

    .. note::
        The feature space of this observer is the same as `MahonyFilter`. See
        documentation for details.

    .. seealso::
        Matthieu Vigne, Antonio El Khoury, Marine Pétriaux, Florent Di Meglio,
        and Nicolas Petit "MOVIE: a Velocity-aided IMU Attitude Estimator for
        Observing and Controlling Multiple Deformations on Legged Robots" IEEE
        Robotics and Automation Letters, Institute of Electrical and
        Electronics Engineers, 2022, 7 (2):
        https://hal.science/hal-03511198/document
    """
    def __init__(self,
                 name: str,
                 env: InterfaceJiminyEnv[BaseObs, BaseAct],
                 *,
                 imu_frame_names: Sequence[str],
                 flex_frame_names: Sequence[str],
                 ignore_twist: bool = True,
                 nested_imu_key: Sequence[str] = (
                    "features", "mahony_filter", "quat"),
                 compute_rpy: bool = True,
                 update_ratio: int = 1) -> None:
        """
        .. warning::
            The user-specified ordering of the flexibility frames will not be
            honored. Indeed, they are reordered to be consistent with the
            kinematic tree according to URDF standard.

        :param name: Name of the block.
        :param env: Environment to connect with.
        :param imu_frame_names: Unordered sequence of IMU frame names that must
                                be used to estimate the local deformation at
                                all flexibility frames.
        :param flex_frame_names: Unordered sequence of flexibility frame names.
                                 It does not have to match the flexibility
                                 frames of the true dynamic model, which is
                                 only known in simulation. It is up to the user
                                 to choose them appropriately. Ideally, they
                                 must be chosen so as to explain as much as
                                 possible the effect of the deformation on
                                 kinematic quantities, eg the vertical position
                                 of the end effectors, considering the number
                                 and placement of the IMUs at hand.
        :param ignore_twist: Whether to ignore the twist of the IMU quaternion
                             estimate. If so, then the twist of the deformation
                             will not be estimated either.
        :param nested_imu_key: Nested key from environment observation mapping
                               to the IMU quaternion estimates. Their ordering
                               must be consistent with the true IMU sensors of
                               the robot.
        :param compute_rpy: Whether to compute the Yaw-Pitch-Roll Euler angles
                            representation for the 3D orientation of the IMU,
                            in addition to the quaternion representation.
                            Optional: False by default.
        :param update_ratio: Ratio between the update period of the observer
                             and the one of the subsequent observer. -1 to
                             match the simulation timestep of the environment.
                             Optional: `1` by default.
        """
        # Make sure that the list of IMU and flexibility frames are not empty
        if not imu_frame_names or not flex_frame_names:
            raise RuntimeError(
                "Please specify at least one IMU and one deformation point.")

        # Sanitize user argument(s)
        imu_frame_names, flex_frame_names = map(
            list, (imu_frame_names, flex_frame_names))

        # Backup some of the user-argument(s)
        self.compute_rpy = compute_rpy
        self.ignore_twist = ignore_twist

        # Create flexible dynamic model.
        # Dummy physical parameters are specified as they have no effect on
        # kinematic computations.
        model = jiminy.Model()
        pinocchio_model_th = env.robot.pinocchio_model_th
        if env.robot.has_freeflyer:
            pinocchio_model_th = pin.buildReducedModel(
                pinocchio_model_th, [1], pin.neutral(pinocchio_model_th))
        model.initialize(pinocchio_model_th)
        options = model.get_options()
        options["dynamics"]["enableFlexibility"] = True
        for frame_name in flex_frame_names:
            options["dynamics"]["flexibilityConfig"].append(
                {
                    "frameName": frame_name,
                    "stiffness": np.ones(3),
                    "damping": np.ones(3),
                    "inertia": np.ones(3),
                }
            )
        model.set_options(options)

        # Backup theoretical pinocchio model without floating base
        self.pinocchio_model_th = model.pinocchio_model_th.copy()
        self.pinocchio_data_th = model.pinocchio_data_th.copy()

        # Extract contiguous chains of flexibility and IMU frames for which
        # computations can be vectorized. It also stores the information of
        # whether or not the sign of the deformation must be reversed to be
        # consistent with standard convention.
        flexibility_joint_names = model.flexibility_joint_names
        flex_imu_frame_names_chains = get_flexibility_imu_frame_chains(
            model.pinocchio_model, flexibility_joint_names, imu_frame_names)

        # Replace actual flex joint name by corresponding rigid frame
        self.flex_imu_frame_names_chains = []
        for flex_frame_names_, imu_frame_names_, is_flipped in (
                flex_imu_frame_names_chains):
            flex_frame_names_ = [
                flex_frame_names[flexibility_joint_names.index(name)]
                for name in flex_frame_names_]
            self.flex_imu_frame_names_chains.append(
                (flex_frame_names_, imu_frame_names_, is_flipped))

        # Check if a freeflyer estimator is required
        if model.has_freeflyer:
            for _, imu_frame_names_, _ in self.flex_imu_frame_names_chains:
                if None in imu_frame_names_:
                    raise NotImplementedError(
                        "Freeflyer estimator is not supported for now.")

        # Backup flexibility frame names
        self.flexibility_frame_names = [
            name for flex_frame_names, _, _ in self.flex_imu_frame_names_chains
            for name in flex_frame_names]

        # Define flexibility and IMU frame orientation proxies for fast access.
        # Note that they will be initialized in `_setup` method.
        self._kin_flex_rots: List[Tuple[np.ndarray, ...]] = []
        self._kin_imu_rots: List[Tuple[np.ndarray, ...]] = []

        # Pre-allocate memory for intermediary computations
        self._is_chain_orphan = []
        self._is_flex_flipped = []
        self._kin_flex_quats = []
        self._inv_obs_imu_quats = []
        self._kin_imu_quats = []
        self._dev_imu_quats = []
        self._dev_parent_imu_quats = []
        self._dev_child_imu_quats = []
        for flex_frame_names_, imu_frame_names_, is_flipped in (
                self.flex_imu_frame_names_chains):
            num_flexs = len(flex_frame_names_)
            num_imus = len(tuple(filter(None, imu_frame_names_)))
            self._kin_flex_quats.append(np.empty((4, num_flexs)))
            self._kin_imu_quats.append(np.empty((4, num_imus)))
            self._inv_obs_imu_quats.append(np.empty((4, num_imus)))
            self._dev_imu_quats.append(np.empty((4, num_imus)))
            dev_parent_imu_quats = np.empty((4, num_flexs))
            is_parent_orphan = imu_frame_names_[0] is None
            is_child_orphan = imu_frame_names_[-1] is None
            self._dev_parent_imu_quats.append(dev_parent_imu_quats)
            dev_child_imu_quats = np.empty((4, num_flexs))
            self._dev_child_imu_quats.append(dev_child_imu_quats)
            self._is_flex_flipped.append(np.array(is_flipped))
            self._is_chain_orphan.append((is_parent_orphan, is_child_orphan))

        # Define IMU and encoder measurement proxy for fast access
        obs_imu_quats: DataNested = env.observation
        for key in nested_imu_key:
            assert isinstance(obs_imu_quats, Mapping)
            obs_imu_quats = obs_imu_quats[key]
        assert isinstance(obs_imu_quats, np.ndarray)
        self._obs_imu_quats = obs_imu_quats

        # Get mapping from IMU frame to index
        imu_frame_map: Dict[str, int] = {}
        for sensor in env.robot.sensors[ImuSensor.type]:
            assert isinstance(sensor, ImuSensor)
            imu_frame_map[sensor.frame_name] = sensor.index

        # Make sure that the robot has one encoder per mechanical joint
        encoder_sensors = env.robot.sensors[EncoderSensor.type]
        if len(encoder_sensors) < len(model.mechanical_joint_indices):
            raise ValueError(
                "The robot must have one encoder per mechanical joints.")

        # Extract mapping from encoders to theoretical configuration.
        # Note that revolute unbounded joints are not supported for now.
        self.encoder_to_position_map = [-1,] * env.robot.nmotors
        for sensor in env.robot.sensors[EncoderSensor.type]:
            assert isinstance(sensor, EncoderSensor)
            joint_index = self.pinocchio_model_th.getJointId(sensor.joint_name)
            joint = self.pinocchio_model_th.joints[joint_index]
            joint_type = jiminy.get_joint_type(joint)
            if joint_type == jiminy.JointModelType.ROTARY_UNBOUNDED:
                raise ValueError(
                    "Revolute unbounded joints are not supported for now.")
            self.encoder_to_position_map[sensor.index] = joint.idx_q

        # Extract measured motor / joint positions for fast access.
        # Note that they will be initialized in `_setup` method.
        self.encoder_data = np.array([])

        # Ratio to translate encoder data to joint side
        self.encoder_to_joint_ratio = np.array([])

        # Buffer storing the theoretical configuration
        self._q_th = pin.neutral(self.pinocchio_model_th)

        # Whether the observer has been compiled already
        self._is_compiled = False

        # Initialize the observer
        super().__init__(name, env, update_ratio)

        # Define some proxies for fast access
        self._quat = self.observation["quat"]
        if self.compute_rpy:
            self._rpy = self.observation["rpy"]
        else:
            self._rpy = np.array([])

        # Define chunks associated with each independent flexibility-imu chain
        self._deformation_flex_quats, self._obs_imu_indices = [], []
        flex_start_index = 0
        for flex_frame_names_, imu_frame_names_, _ in (
                self.flex_imu_frame_names_chains):
            flex_last_index = flex_start_index + len(flex_frame_names_)
            self._deformation_flex_quats.append(
                self._quat[:, flex_start_index:flex_last_index])
            flex_start_index = flex_last_index
            imu_frame_names_filtered = tuple(filter(None, imu_frame_names_))
            imu_indices = tuple(
                imu_frame_map[name] for name in imu_frame_names_filtered)
            self._obs_imu_indices.append(imu_indices)

    def _initialize_observation_space(self) -> None:
        nflex = sum(
            len(flex_frame_names)
            for flex_frame_names, _, _ in self.flex_imu_frame_names_chains)
        observation_space: Dict[str, gym.Space] = OrderedDict()
        observation_space["quat"] = gym.spaces.Box(
            low=np.full((4, nflex), -1.0 - 1e-9),
            high=np.full((4, nflex), 1.0 + 1e-9),
            dtype=np.float64)
        if self.compute_rpy:
            high = np.array([np.pi, np.pi/2, np.pi]) + 1e-9
            observation_space["rpy"] = gym.spaces.Box(
                low=-high[:, np.newaxis].repeat(nflex, axis=1),
                high=high[:, np.newaxis].repeat(nflex, axis=1),
                dtype=np.float64)
        self.observation_space = gym.spaces.Dict(observation_space)

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

        # Refresh the theoretical model of the robot.
        # Even if the robot may change, the theoretical model of the robot is
        # not supposed to change in a way that would break this observer.
        pinocchio_model_th = self.env.robot.pinocchio_model_th
        if self.env.robot.has_freeflyer:
            pinocchio_model_th = pin.buildReducedModel(
                pinocchio_model_th, [1], pin.neutral(pinocchio_model_th))
        self.pinocchio_model_th = pinocchio_model_th
        self.pinocchio_data_th = self.pinocchio_model_th.createData()

        # Fix initialization of the observation to be valid quaternions
        self._quat[-1] = 1.0

        # Refresh flexibility and IMU frame orientation proxies
        self._kin_flex_rots.clear()
        self._kin_imu_rots.clear()
        for flex_frame_names, imu_frame_names_, _ in (
                self.flex_imu_frame_names_chains):
            imu_frame_names = list(filter(None, imu_frame_names_))
            kin_flex_rots, kin_imu_rots = (tuple(
                self.pinocchio_data_th.oMf[frame_index].rotation
                for frame_index in get_frame_indices(
                    self.pinocchio_model_th, frame_names))
                for frame_names in (flex_frame_names, imu_frame_names))
            self._kin_flex_rots.append(kin_flex_rots)
            self._kin_imu_rots.append(kin_imu_rots)

        # Refresh measured motor position proxy
        self.encoder_data, _ = self.env.sensor_measurements[EncoderSensor.type]

        # Refresh mechanical reduction ratio
        encoder_to_joint_ratio = []
        for sensor in self.env.robot.sensors[EncoderSensor.type]:
            try:
                motor = self.env.robot.motors[sensor.motor_index]
                motor_options = motor.get_options()
                mechanical_reduction = motor_options["mechanicalReduction"]
                encoder_to_joint_ratio.append(1.0 / mechanical_reduction)
            except IndexError:
                encoder_to_joint_ratio.append(1.0)
        self.encoder_to_joint_ratio = np.array(encoder_to_joint_ratio)

        # Call `refresh_observation` manually to pre-compile it if necessary
        if not self._is_compiled:
            self.refresh_observation(self.env.observation)
            self._is_compiled = True

    @property
    def fieldnames(self) -> Dict[str, List[List[str]]]:
        fieldnames: Dict[str, List[List[str]]] = {}
        fieldnames["quat"] = [
            [f"{name}.Quat{e}" for name in self.flexibility_frame_names]
            for e in ("x", "y", "z", "w")]
        if self.compute_rpy:
            fieldnames["rpy"] = [
                [".".join((name, e)) for name in self.flexibility_frame_names]
                for e in ("Roll", "Pitch", "Yaw")]
        return fieldnames

    def refresh_observation(self, measurement: BaseObs) -> None:
        # Translate encoder data at joint level
        joint_positions = self.encoder_to_joint_ratio * self.encoder_data

        # Update the configuration of the theoretical model of the robot
        self._q_th[self.encoder_to_position_map] = joint_positions

        # Update kinematic quantities according to the estimated configuration.
        # FIXME: Compute frame placement only for relevant IMUs.
        pin.framesForwardKinematics(
            self.pinocchio_model_th, self.pinocchio_data_th, self._q_th)

        # Estimate all the deformations in their local frame.
        # It loops over each flexibility-imu chain independently.
        for args in zip(
                self._obs_imu_indices,
                self._inv_obs_imu_quats,
                self._kin_imu_rots,
                self._kin_imu_quats,
                self._kin_flex_rots,
                self._kin_flex_quats,
                self._is_flex_flipped,
                self._is_chain_orphan,
                self._dev_imu_quats,
                self._dev_child_imu_quats,
                self._dev_parent_imu_quats,
                self._deformation_flex_quats):
            flexibility_estimator(
                self._obs_imu_quats, *args, self.ignore_twist)

        # Compute the RPY representation if requested
        if self.compute_rpy:
            quat_to_rpy(self._quat, self._rpy)
