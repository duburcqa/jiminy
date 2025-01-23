#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/hardware/abstract_motor.h"
#include "jiminy/core/constraints/abstract_constraint.h"
#include "jiminy/core/robot/robot.h"
#include "jiminy/core/robot/pinocchio_overload_algorithms.h"
#include "jiminy/core/io/memory_device.h"
#include "jiminy/core/io/serialization.h"
#include "jiminy/core/utilities/pinocchio.h"
#include "jiminy/core/utilities/random.h"

#define HPP_FCL_SKIP_EIGEN_BOOST_SERIALIZATION
#include "hpp/fcl/serialization/collision_object.h"  // `serialize<hpp::fcl::CollisionGeometry>`
#undef HPP_FCL_SKIP_EIGEN_BOOST_SERIALIZATION

#define NO_IMPORT_ARRAY
#include "jiminy/python/fwd.h"
#include "jiminy/python/utilities.h"
#include "jiminy/python/helpers.h"

#include <boost/optional.hpp>


namespace jiminy::python
{
    namespace bp = boost::python;

    pinocchio::GeometryModel buildGeometryModelFromUrdf(const pinocchio::Model & model,
                                                        const std::string & filename,
                                                        const int & typePy,
                                                        const bp::object & packageDirsPy,
                                                        bool loadMeshes,
                                                        bool makeMeshesConvex)
    {
        /* Note that enum bindings interoperability is buggy, so that `pin.GeometryType` is not
           properly converted from Python to C++ automatically in some cases. */
        pinocchio::GeometryModel geometryModel;
        const auto type = static_cast<pinocchio::GeometryType>(typePy);
        auto packageDirs = convertFromPython<std::vector<std::string>>(packageDirsPy);
        return ::jiminy::buildGeometryModelFromUrdf(
            model, filename, type, packageDirs, loadMeshes, makeMeshesConvex);
    }

    bp::tuple buildMultipleModelsFromUrdf(const std::string & urdfPath,
                                          bool hasFreeflyer,
                                          const bp::object & packageDirsPy,
                                          bool buildVisualModel,
                                          bool loadVisualMeshes)
    {
        /* Note that enum bindings interoperability is buggy, so that `pin.GeometryType` is not
           properly converted from Python to C++ automatically in some cases. */
        pinocchio::Model model;
        pinocchio::GeometryModel collisionModel;
        pinocchio::GeometryModel visualModel;
        std::optional<std::reference_wrapper<pinocchio::GeometryModel>> visualModelOptional =
            std::nullopt;
        if (buildVisualModel)
        {
            visualModelOptional = visualModel;
        }
        auto packageDirs = convertFromPython<std::vector<std::string>>(packageDirsPy);
        ::jiminy::buildMultipleModelsFromUrdf(urdfPath,
                                              hasFreeflyer,
                                              packageDirs,
                                              model,
                                              collisionModel,
                                              visualModelOptional,
                                              loadVisualMeshes);
        if (buildVisualModel)
        {
            return bp::make_tuple(model, collisionModel, visualModel);
        }
        return bp::make_tuple(model, collisionModel);
    }

    bp::object saveRobotToBinary(const std::shared_ptr<jiminy::Robot> & robot)
    {
        std::string data = saveToBinary(robot, true);
        return bp::object(bp::handle<>(PyBytes_FromStringAndSize(data.c_str(), data.size())));
    }

    std::shared_ptr<jiminy::Robot> buildRobotFromBinary(const std::string & data,
                                                        const bp::object & meshPathDirPy,
                                                        const bp::object & packageDirsPy)
    {
        std::optional<std::string> meshPathDir = std::nullopt;
        if (!meshPathDirPy.is_none())
        {
            meshPathDir = bp::extract<std::string>(meshPathDirPy);
        }
        auto meshPackageDirs = convertFromPython<std::vector<std::string>>(packageDirsPy);

        std::shared_ptr<jiminy::Robot> robot;
        loadFromBinary(robot, data, meshPathDir, meshPackageDirs);
        return robot;
    }

    hpp::fcl::CollisionGeometryPtr_t buildHeightmapFromBinary(const std::string & data)
    {
        hpp::fcl::CollisionGeometryPtr_t heightmap;
        loadFromBinary(heightmap, data);
        return heightmap;
    }

    np::ndarray solveJMinvJtv(
        pinocchio::Data & data, const np::ndarray & vPy, bool updateDecomposition)
    {
        bp::object objPy;
        const int32_t nDims = vPy.get_nd();
        assert(nDims < 3 && "The number of dimensions of 'v' cannot exceed 2.");
        if (nDims == 1)
        {
            const Eigen::VectorXd v = convertFromPython<Eigen::VectorXd>(vPy);
            const Eigen::VectorXd x =
                pinocchio_overload::solveJMinvJtv<Eigen::VectorXd>(data, v, updateDecomposition);
            objPy = convertToPython(x, true);
        }
        else
        {
            const Eigen::MatrixXd v = convertFromPython<Eigen::MatrixXd>(vPy);
            const Eigen::MatrixXd x =
                pinocchio_overload::solveJMinvJtv<Eigen::MatrixXd>(data, v, updateDecomposition);
            objPy = convertToPython(x, true);
        }
        return bp::extract<np::ndarray>(objPy);
    }

    template<typename T>
    void EigenMapTransposeAssign(char * dstData, char * srcData, npy_intp rows, npy_intp cols)
    {
        using EigenMapType = Eigen::Map<MatrixX<T>>;
        EigenMapType dst(reinterpret_cast<T *>(dstData), rows, cols);
        EigenMapType src(reinterpret_cast<T *>(srcData), cols, rows);
        dst = src.transpose();
    }

    void arrayCopyTo(PyObject * dstPy, PyObject * srcPy)
    {
        // Organize code path to run as fast as possible for the most common use-cases
        // 1. 1D or 2D arrays for which neither type casting nor broadcasting is necessary
        // 2. Assigning a scalar value to a N-D array, incl. type casting if necessary
        // 3. Discontinuous 1D or N-D array assignment without type casting nor broadcasting
        // 4. Deal with all other cases by calling generic Numpy routine `PyArray_CopyInto`

        // Re-interpret source and destination as a Numpy array, not just a plain Python object
        PyArrayObject * dstPyArray = reinterpret_cast<PyArrayObject *>(dstPy);
        PyArrayObject * srcPyArray = reinterpret_cast<PyArrayObject *>(srcPy);

        // Make sure that 'dst' is a valid array and is writable, raises an exception otherwise
        if (!PyArray_Check(dstPy))
        {
            JIMINY_THROW(std::invalid_argument, "'dst' must have type 'np.ndarray'.");
        }

        // Make sure that 'dst' is writable
        const int dstPyFlags = PyArray_FLAGS(dstPyArray);
        if (!(dstPyFlags & NPY_ARRAY_WRITEABLE))
        {
            JIMINY_THROW(std::invalid_argument, "'dst' must be writable.");
        }

        // Return early if destination is empty
        if (PyArray_SIZE(dstPyArray) < 1)
        {
            return;
        }

        // Check if the source is a numpy array
        const bool isSrcArray = PyArray_Check(srcPy);

        // Source and destination are both numpy arrays
        if (isSrcArray)
        {
            // Get shape of source and destination arrays
            const int dstNdim = PyArray_NDIM(dstPyArray);
            const int srcNdim = PyArray_NDIM(srcPyArray);
            const npy_intp * const dstShape = PyArray_SHAPE(dstPyArray);
            const npy_intp * const srcShape = PyArray_SHAPE(srcPyArray);

            // N-Dim arrays but no broadcasting nor casting required. Easy enough to handle.
            if (dstNdim == srcNdim && PyArray_CompareLists(dstShape, srcShape, dstNdim) &&
                PyArray_EquivArrTypes(dstPyArray, srcPyArray))
            {
                // Extract data pointers
                char * dstPyData = PyArray_BYTES(dstPyArray);
                char * srcPyData = PyArray_BYTES(srcPyArray);

                // Check if source and destination are the referring to the same memory location
                if ((dstPyData == srcPyData) &&
                    (PyArray_BASE(dstPyArray) == PyArray_BASE(srcPyArray)))
                {
                    return;
                }

                // Get memory layout of source and destination arrays
                const npy_intp itemsize = PyArray_ITEMSIZE(dstPyArray);
                const int srcPyFlags = PyArray_FLAGS(srcPyArray);
                const int commonPyFlags = dstPyFlags & srcPyFlags;

                // Same memory layout. Straightforward regardless the dimensionality.
                if ((commonPyFlags & NPY_ARRAY_ALIGNED) &&
                    (commonPyFlags & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)))
                {
                    Eigen::Map<MatrixX<char>> dst(dstPyData, itemsize, PyArray_SIZE(dstPyArray));
                    Eigen::Map<MatrixX<char>> src(srcPyData, itemsize, PyArray_SIZE(srcPyArray));
                    dst = src;
                    return;
                }

                /* Different memory layout in 2D.
                   TODO: Extend to support any number of dims by operating on flattened view. */
                if ((dstNdim == 2) &&
                    (dstPyFlags & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) &&
                    (srcPyFlags & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)))
                {
                    /* Using Eigen once again to avoid slow element-wise copy assignment.
                       Note that only the width of the scalar type matters, not the actual type. */
                    npy_intp rows, cols;
                    if (dstPyFlags & NPY_ARRAY_C_CONTIGUOUS)
                    {
                        rows = dstShape[1];
                        cols = dstShape[0];
                    }
                    else
                    {
                        rows = dstShape[0];
                        cols = dstShape[1];
                    }
                    switch (itemsize)
                    {
                    case 8:
                        EigenMapTransposeAssign<uint64_t>(dstPyData, srcPyData, rows, cols);
                        return;
                    case 4:
                        EigenMapTransposeAssign<uint32_t>(dstPyData, srcPyData, rows, cols);
                        return;
                    case 2:
                        EigenMapTransposeAssign<uint16_t>(dstPyData, srcPyData, rows, cols);
                        return;
                    case 1:
                        EigenMapTransposeAssign<uint8_t>(dstPyData, srcPyData, rows, cols);
                        return;
                    default:
                        break;
                    }
                }

                // Falling back to slow element-wise strided ND-array copy assignment
                int i = 0;
                npy_intp coord[NPY_MAXDIMS];
                const npy_intp * const dstStrides = PyArray_STRIDES(dstPyArray);
                const npy_intp * const srcStrides = PyArray_STRIDES(srcPyArray);
                memset(coord, 0, dstNdim * sizeof(npy_intp));
                while (i < dstNdim)
                {
                    char * _dstPyData = dstPyData;
                    char * _srcPyData = srcPyData;
                    for (int j = 0; j < dstShape[0]; ++j)
                    {
                        memcpy(_dstPyData, _srcPyData, itemsize);
                        _dstPyData += dstStrides[0];
                        _srcPyData += srcStrides[0];
                    }
                    for (i = 1; i < dstNdim; ++i)
                    {
                        if (++coord[i] == dstShape[i])
                        {
                            coord[i] = 0;
                            dstPyData -= (dstShape[i] - 1) * dstStrides[i];
                            srcPyData -= (dstShape[i] - 1) * srcStrides[i];
                        }
                        else
                        {
                            dstPyData += dstStrides[i];
                            srcPyData += srcStrides[i];
                            break;
                        }
                    }
                }
                return;
            }
        }

        // Specialization to fill with scalar
        if (!isSrcArray || PyArray_IsScalar(srcPy, Generic) || (PyArray_SIZE(srcPyArray) == 1))
        {
            // Extract built-in scalar value to promote setitem fast path
            if (isSrcArray)
            {
                srcPy = PyArray_GETITEM(srcPyArray, PyArray_BYTES(srcPyArray));
            }

            /* Eigen does a much better job than element-wise copy assignment in this scenario.
               Ensuring copy and casting are both slow as they allocate new array, so avoiding
               using them entirely if possible and falling back to default routine otherwise. */
            if ((dstPyFlags & NPY_ARRAY_ALIGNED) &&
                (dstPyFlags & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)))
            {
                // Convert src scalar data to raw bytes with dst dtype, casting if necessary
                const npy_intp itemsize = PyArray_ITEMSIZE(dstPyArray);
                char * dstPyData = PyArray_BYTES(dstPyArray);
                PyArray_SETITEM(dstPyArray, dstPyData, srcPy);
                Eigen::Map<MatrixX<char>> dst(dstPyData, itemsize, PyArray_SIZE(dstPyArray));
                dst.rightCols(dst.cols() - 1).colwise() = dst.col(0);
                return;
            }

            // Too complicated to deal with it manually. Falling back to default routine.
            if (PyArray_FillWithScalar(dstPyArray, srcPy) < 0)
            {
                JIMINY_THROW(std::runtime_error, "Impossible to copy from 'src' to 'dst'.");
            }
            return;
        }

        // Falling back to default routine if too complicated to deal with it manually
        if (PyArray_CopyInto(dstPyArray, srcPyArray) < 0)
        {
            JIMINY_THROW(std::runtime_error, "Impossible to copy from 'src' to 'dst'.");
        }
        return;
    }

    void multiArrayCopyTo(PyObject * dstPy, PyObject * srcPy)
    {
        // Wrap the input arguments as tuple or list if not already the case
        PyObject * dstSeqPy = PySequence_Fast(dstPy, "'dst' must be a sequence or an iterable.");
        if (dstSeqPy == nullptr)
        {
            throw bp::error_already_set();
        }
        PyObject * srcSeqPy = PySequence_Fast(srcPy, "'src' must be a sequence or an iterable.");
        if (srcSeqPy == nullptr)
        {
            Py_DECREF(dstSeqPy);
            throw bp::error_already_set();
        }

        // Make sure that source and destination have the same length
        const Py_ssize_t dstSize = PySequence_Fast_GET_SIZE(dstSeqPy);
        const Py_ssize_t srcSize = PySequence_Fast_GET_SIZE(srcSeqPy);
        if (dstSize != srcSize)
        {
            Py_DECREF(dstSeqPy);
            Py_DECREF(srcSeqPy);
            JIMINY_THROW(std::runtime_error, "Length mismatch between 'src' and 'dst'.");
        }

        // Loop over all pairs one-by-one
        PyObject ** dstItemsPy = PySequence_Fast_ITEMS(dstSeqPy);
        PyObject ** srcItemsPy = PySequence_Fast_ITEMS(srcSeqPy);
        for (uint32_t i = 0; i < dstSize; ++i)
        {
            arrayCopyTo(dstItemsPy[i], srcItemsPy[i]);
        }

        // Release memory
        Py_DECREF(dstSeqPy);
        Py_DECREF(srcSeqPy);
    }

    void exposeHelpers()
    {
        bp::def("build_geom_from_urdf",
                &buildGeometryModelFromUrdf,
                (bp::arg("pinocchio_model"),
                 "urdf_filename",
                 "geom_type",
                 bp::arg("mesh_package_dirs") = bp::list(),
                 bp::arg("load_meshes") = true,
                 bp::arg("make_meshes_convex") = false));

        bp::def("build_models_from_urdf",
                &buildMultipleModelsFromUrdf,
                (bp::arg("urdf_path"),
                 "has_freeflyer",
                 bp::arg("mesh_package_dirs") = bp::list(),
                 bp::arg("build_visual_model") = false,
                 bp::arg("load_visual_meshes") = false));

        bp::def("save_robot_to_binary", &saveRobotToBinary, (bp::arg("robot")));

        bp::def("load_robot_from_binary",
                &buildRobotFromBinary,
                (bp::arg("data"),
                 bp::arg("mesh_dir_path") = bp::object(),
                 bp::arg("mesh_package_dirs") = bp::list()));

        bp::def("load_heightmap_from_binary", &buildHeightmapFromBinary, bp::arg("data"));

        bp::def("get_joint_type", &getJointType, bp::arg("joint_model"));
        bp::def(
            "get_joint_type", &getJointTypeFromIndex, (bp::arg("pinocchio_model"), "joint_index"));
        bp::def(
            "get_joint_indices", &getJointIndices, (bp::arg("pinocchio_model"), "joint_names"));
        bp::def("get_joint_position_first_index",
                &getJointPositionFirstIndex,
                (bp::arg("pinocchio_model"), "joint_name"));
        bp::def("is_position_valid",
                &isPositionValid,
                (bp::arg("pinocchio_model"),
                 "position",
                 bp::arg("tol_abs") = std::numeric_limits<float>::epsilon()));

        bp::def("get_frame_indices",
                &getFrameIndices,
                bp::return_value_policy<result_converter<true>>(),
                (bp::arg("pinocchio_model"), "frame_names"));
        bp::def("get_joint_indices",
                &getFrameIndices,
                bp::return_value_policy<result_converter<true>>(),
                (bp::arg("pinocchio_model"), "joint_names"));

        bp::def("array_copyto", &arrayCopyTo, (bp::arg("dst"), "src"));
        bp::def("multi_array_copyto", &multiArrayCopyTo, (bp::arg("dst"), "src"));

        // Do NOT use EigenPy to-python converter as it considers arrays with 1 column as vectors
        bp::def("interpolate_positions",
                &interpolatePositions,
                bp::return_value_policy<result_converter<true>>(),
                (bp::arg("pinocchio_model"), "times_in", "positions_in", "times_out"));

        bp::def("aba",
                &pinocchio_overload::aba<double,
                                         0,
                                         pinocchio::JointCollectionDefaultTpl,
                                         Eigen::VectorXd,
                                         Eigen::VectorXd,
                                         Eigen::VectorXd,
                                         pinocchio::Force>,
                bp::return_value_policy<result_converter<false>>(),
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v", "u", "fext"),
                "Compute ABA with external forces, store the result in Data::ddq and return it.");
        bp::def(
            "rnea",
            &pinocchio_overload::rnea<double,
                                      0,
                                      pinocchio::JointCollectionDefaultTpl,
                                      Eigen::VectorXd,
                                      Eigen::VectorXd,
                                      Eigen::VectorXd>,
            bp::return_value_policy<result_converter<false>>(),
            (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v", "a"),
            "Compute the RNEA without external forces, store the result in Data and return it.");
        bp::def("rnea",
                &pinocchio_overload::rnea<double,
                                          0,
                                          pinocchio::JointCollectionDefaultTpl,
                                          Eigen::VectorXd,
                                          Eigen::VectorXd,
                                          Eigen::VectorXd,
                                          pinocchio::Force>,
                bp::return_value_policy<result_converter<false>>(),
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v", "a", "fext"),
                "Compute the RNEA with external forces, store the result in Data and return it.");
        bp::def("crba",
                &pinocchio_overload::
                    crba<double, 0, pinocchio::JointCollectionDefaultTpl, Eigen::VectorXd>,
                bp::return_value_policy<result_converter<false>>(),
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", bp::arg("fastmath") = false),
                "Computes CRBA, store the result in Data and return it.");
        bp::def("computeKineticEnergy",
                &pinocchio_overload::computeKineticEnergy<double,
                                                          0,
                                                          pinocchio::JointCollectionDefaultTpl,
                                                          Eigen::VectorXd,
                                                          Eigen::VectorXd>,
                (bp::arg("pinocchio_model"),
                 "pinocchio_data",
                 "q",
                 "v",
                 bp::arg("update_kinematics") = true),
                "Computes the forward kinematics and the kinematic energy of the model for the "
                "given joint configuration and velocity given as input. "
                "The result is accessible through data.kinetic_energy.");

        bp::def("computeJMinvJt",
                &pinocchio_overload::computeJMinvJt<Eigen::MatrixXd>,
                bp::return_value_policy<result_converter<false>>(),
                (bp::arg("pinocchio_model"),
                 "pinocchio_data",
                 "J",
                 bp::arg("update_decomposition") = true));
        bp::def("solveJMinvJtv",
                &solveJMinvJtv,
                (bp::arg("pinocchio_data"), "v", bp::arg("update_decomposition") = true));
    }
}
