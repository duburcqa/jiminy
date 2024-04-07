#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/hardware/abstract_motor.h"
#include "jiminy/core/constraints/abstract_constraint.h"
#include "jiminy/core/robot/robot.h"
#include "jiminy/core/robot/pinocchio_overload_algorithms.h"
#include "jiminy/core/io/memory_device.h"
#include "jiminy/core/utilities/pinocchio.h"
#include "jiminy/core/utilities/random.h"

#include <boost/optional.hpp>

#include "pinocchio/bindings/python/fwd.hpp"

#include "jiminy/python/utilities.h"
#include "jiminy/python/helpers.h"


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
            THROW_ERROR(std::invalid_argument, "'dst' must have type 'np.ndarray'.");
        }

        // Make sure that 'dst' is writable
        const int dstPyFlags = PyArray_FLAGS(dstPyArray);
        if (!(dstPyFlags & NPY_ARRAY_WRITEABLE))
        {
            THROW_ERROR(std::invalid_argument, "'dst' must be writable.");
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
                // Get memory layout of source and destination arrays
                char * dstPyData = PyArray_BYTES(dstPyArray);
                char * srcPyData = PyArray_BYTES(srcPyArray);
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
                THROW_ERROR(std::runtime_error, "Impossible to copy from 'src' to 'dst'.");
            }
            return;
        }

        // Falling back to default routine if too complicated to deal with it manually
        if (PyArray_CopyInto(dstPyArray, srcPyArray) < 0)
        {
            THROW_ERROR(std::runtime_error, "Impossible to copy from 'src' to 'dst'.");
        }
        return;
    }

    void multiArrayCopyTo(PyObject * dstPy, PyObject * srcPy)
    {
        PyObject * dstSeqPy = PySequence_Fast(dstPy, "Only tuple or list is supported for 'dst'.");
        PyObject * srcSeqPy = PySequence_Fast(srcPy, "Only tuple or list is supported for 'src'.");
        const Py_ssize_t dstSize = PySequence_Fast_GET_SIZE(dstSeqPy);
        const Py_ssize_t srcSize = PySequence_Fast_GET_SIZE(srcSeqPy);
        if (dstSize != srcSize)
        {
            THROW_ERROR(std::runtime_error, "Length mismatch between 'src' and 'dst'.");
        }

        PyObject ** dstItemsPy = PySequence_Fast_ITEMS(dstPy);
        PyObject ** srcItemsPy = PySequence_Fast_ITEMS(srcPy);
        for (uint32_t i = 0; i < dstSize; ++i)
        {
            arrayCopyTo(dstItemsPy[i], srcItemsPy[i]);
        }
    }

    void exposeHelpers()
    {
        // clang-format off
        bp::def("build_geom_from_urdf", &buildGeometryModelFromUrdf,
                                        (bp::arg("pinocchio_model"), "urdf_filename", "geom_type",
                                         bp::arg("mesh_package_dirs") = bp::list(),
                                         bp::arg("load_meshes") = true,
                                         bp::arg("make_meshes_convex") = false));

        bp::def("build_models_from_urdf", &buildMultipleModelsFromUrdf,
                                          (bp::arg("urdf_path"), "has_freeflyer",
                                           bp::arg("mesh_package_dirs") = bp::list(),
                                           bp::arg("build_visual_model") = false,
                                           bp::arg("load_visual_meshes") = false));

        bp::def("get_joint_type", &getJointTypeFromIndex,
                                  (bp::arg("pinocchio_model"), "joint_index"));
        bp::def("get_joint_indices", &getJointIndices,
                                     (bp::arg("pinocchio_model"), "joint_names"));
        bp::def("get_joint_position_first_index", &getJointPositionFirstIndex,
                                          (bp::arg("pinocchio_model"), "joint_name"));
        bp::def("is_position_valid", &isPositionValid,
                                     (bp::arg("pinocchio_model"), "position", bp::arg("tol_abs") = std::numeric_limits<float>::epsilon()));

        bp::def("get_frame_indices", &getFrameIndices,
                                     bp::return_value_policy<result_converter<true>>(),
                                     (bp::arg("pinocchio_model"), "frame_names"));
        bp::def("get_joint_indices", &getFrameIndices,
                                     bp::return_value_policy<result_converter<true>>(),
                                     (bp::arg("pinocchio_model"), "joint_names"));

        bp::def("array_copyto", &arrayCopyTo, (bp::arg("dst"), "src"));
        bp::def("multi_array_copyto", &multiArrayCopyTo, (bp::arg("dst"), "src"));

        // Do not use EigenPy To-Python converter because it considers matrices with 1 column as vectors
        bp::def("interpolate_positions", &interpolatePositions,
                                         bp::return_value_policy<result_converter<true>>(),
                                         (bp::arg("pinocchio_model"), "times_in", "positions_in", "times_out"));

        bp::def("aba",
                &pinocchio_overload::aba<
                    double, 0, pinocchio::JointCollectionDefaultTpl, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, pinocchio::Force>,
                bp::return_value_policy<result_converter<false>>(),
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v", "u", "fext"),
                "Compute ABA with external forces, store the result in Data::ddq and return it.");
        bp::def("rnea",
                &pinocchio_overload::rnea<
                    double, 0, pinocchio::JointCollectionDefaultTpl, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>,
                bp::return_value_policy<result_converter<false>>(),
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v", "a"),
                "Compute the RNEA without external forces, store the result in Data and return it.");
        bp::def("rnea",
                &pinocchio_overload::rnea<
                    double, 0, pinocchio::JointCollectionDefaultTpl, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, pinocchio::Force>,
                bp::return_value_policy<result_converter<false>>(),
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v", "a", "fext"),
                "Compute the RNEA with external forces, store the result in Data and return it.");
        bp::def("crba",
                &pinocchio_overload::crba<
                    double, 0, pinocchio::JointCollectionDefaultTpl, Eigen::VectorXd>,
                bp::return_value_policy<result_converter<false>>(),
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", bp::arg("fast_math") = false),
                "Computes CRBA, store the result in Data and return it.");
        bp::def("computeKineticEnergy",
                &pinocchio_overload::computeKineticEnergy<
                    double, 0, pinocchio::JointCollectionDefaultTpl, Eigen::VectorXd, Eigen::VectorXd>,
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v"),
                "Computes the forward kinematics and the kinematic energy of the model for the "
                "given joint configuration and velocity given as input. "
                "The result is accessible through data.kinetic_energy.");

        bp::def("computeJMinvJt",
                &pinocchio_overload::computeJMinvJt<Eigen::MatrixXd>,
                bp::return_value_policy<result_converter<false>>(),
                (bp::arg("pinocchio_model"), "pinocchio_data", "J", bp::arg("update_decomposition") = true));
        bp::def("solveJMinvJtv", &solveJMinvJtv,
                (bp::arg("pinocchio_data"), "v", bp::arg("update_decomposition") = true));
        // clang-format on
    }
}
