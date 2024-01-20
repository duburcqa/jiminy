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

    JointModelType getJointTypeFromIdx(const pinocchio::Model & model, std::size_t jointIdx)
    {
        JointModelType jointType = JointModelType::UNSUPPORTED;
        ::jiminy::getJointTypeFromIdx(model, jointIdx, jointType);
        return jointType;
    }

    Eigen::Index getJointPositionIdx(const pinocchio::Model & model, const std::string & name)
    {
        Eigen::Index jointPositionFirstIdx = model.nq;
        ::jiminy::getJointPositionIdx(model, name, jointPositionFirstIdx);
        return jointPositionFirstIdx;
    }

    bool isPositionValid(const pinocchio::Model & model, const Eigen::VectorXd & position)
    {
        bool isValid;
        ::jiminy::isPositionValid(
            model, position, isValid, Eigen::NumTraits<double>::dummy_precision());
        return isValid;
    }

    Eigen::MatrixXd interpolate(const pinocchio::Model & modelIn,
                                const Eigen::VectorXd & timesIn,
                                const Eigen::MatrixXd & positionsIn,
                                const Eigen::VectorXd & timesOut)
    {
        Eigen::MatrixXd positionOut;
        ::jiminy::interpolate(modelIn, timesIn, positionsIn, timesOut, positionOut);
        return positionOut;
    }

    pinocchio::GeometryModel buildGeomFromUrdf(const pinocchio::Model & model,
                                               const std::string & filename,
                                               const int & typePy,
                                               const bp::list & packageDirsPy,
                                               bool loadMeshes,
                                               bool makeMeshesConvex)
    {
        /* Note that enum bindings interoperability is buggy, so that `pin.GeometryType` is not
           properly converted from Python to C++ automatically in some cases. */
        pinocchio::GeometryModel geometryModel;
        const auto type = static_cast<pinocchio::GeometryType>(typePy);
        auto packageDirs = convertFromPython<std::vector<std::string>>(packageDirsPy);
        ::jiminy::buildGeomFromUrdf(
            model, filename, type, geometryModel, packageDirs, loadMeshes, makeMeshesConvex);
        return geometryModel;
    }

    bp::tuple buildModelsFromUrdf(const std::string & urdfPath,
                                  bool hasFreeflyer,
                                  const bp::list & packageDirsPy,
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
        ::jiminy::buildModelsFromUrdf(urdfPath,
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
        const int32_t nDims = vPy.get_nd();
        assert(nDims < 3 && "The number of dimensions of 'v' cannot exceed 2.");
        if (nDims == 1)
        {
            const Eigen::VectorXd v = convertFromPython<Eigen::VectorXd>(vPy);
            const Eigen::VectorXd x =
                pinocchio_overload::solveJMinvJtv<Eigen::VectorXd>(data, v, updateDecomposition);
            return bp::extract<np::ndarray>(convertToPython(x, true));
        }
        else
        {
            const Eigen::MatrixXd v = convertFromPython<Eigen::MatrixXd>(vPy);
            const Eigen::MatrixXd x =
                pinocchio_overload::solveJMinvJtv<Eigen::MatrixXd>(data, v, updateDecomposition);
            return bp::extract<np::ndarray>(convertToPython(x, true));
        }
    }

    void arrayCopyTo(PyObject * dstPy, PyObject * srcPy)
    {
        // Making sure that 'dst' is a valid array and is writable, raises an exception otherwise
        if (!PyArray_Check(dstPy))
        {
            throw std::runtime_error("'dst' must have type 'np.ndarray'.");
        }
        PyArrayObject * dstPyArray = reinterpret_cast<PyArrayObject *>(dstPy);
        const int dstPyFlags = PyArray_FLAGS(dstPyArray);
        if (!(dstPyFlags & NPY_ARRAY_WRITEABLE))
        {
            throw std::runtime_error("'dst' must be writable.");
        }

        // Dedicated path to fill with scalar
        const npy_intp itemsize = PyArray_ITEMSIZE(dstPyArray);
        char * dstPyData = PyArray_BYTES(dstPyArray);
        if (!PyArray_Check(srcPy) || PyArray_IsScalar(srcPy, Generic) ||
            (PyArray_SIZE(reinterpret_cast<PyArrayObject *>(srcPy)) == 1))
        {
            /* Eigen does a much better job than element-wise copy assignment in this scenario.
               Note that only the width of the scalar type matters here, not the actual type.
               Ensure copy and casting are both slow as they allocate new array, so avoiding
               using them entirely if possible and falling back to default routine otherwise. */
            if ((itemsize == 8) && (dstPyFlags & NPY_ARRAY_ALIGNED) &&
                (dstPyFlags & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)))
            {
                // Convert src scalar data to raw bytes with dst dtype, casting if necessary
                bool isSuccess = false;
                double srcPyScalar;
                if (PyArray_Check(srcPy))
                {
                    PyArrayObject * srcPyArray = reinterpret_cast<PyArrayObject *>(srcPy);
                    if (PyArray_EquivArrTypes(dstPyArray, srcPyArray))
                    {
                        srcPyScalar = *reinterpret_cast<double *>(PyArray_DATA(srcPyArray));
                        isSuccess = true;
                    }
                    else
                    {
                        int dstPyTypeNum = PyArray_TYPE(dstPyArray);
                        PyArray_Descr * srcPyDtype = PyArray_DESCR(srcPyArray);
                        if (!PyTypeNum_ISEXTENDED(dstPyTypeNum) &&
                            !PyTypeNum_ISEXTENDED(srcPyDtype->type_num))
                        {
                            auto srcToDstCastFunc = PyArray_GetCastFunc(srcPyDtype, dstPyTypeNum);
                            srcToDstCastFunc(
                                PyArray_DATA(srcPyArray), &srcPyScalar, 1, NULL, NULL);
                            isSuccess = true;
                        }
                    }
                }
                else if (PyArray_IsScalar(srcPy, Generic))
                {
                    PyArray_CastScalarToCtype(srcPy, &srcPyScalar, PyArray_DESCR(dstPyArray));
                    isSuccess = true;
                }
                else if (PyFloat_Check(srcPy) || PyLong_Check(srcPy))
                {
                    int dstPyTypeNum = PyArray_TYPE(dstPyArray);
                    PyArray_Descr * srcPyDtype = PyArray_DescrFromObject(srcPy, NULL);
                    if (!PyTypeNum_ISEXTENDED(dstPyTypeNum))
                    {
                        auto srcToDstCastFunc = PyArray_GetCastFunc(srcPyDtype, dstPyTypeNum);
                        if (PyFloat_Check(srcPy))
                        {
                            srcPyScalar = PyFloat_AsDouble(srcPy);
                        }
                        else
                        {
                            auto srcPyBuiltin = std::make_unique<long>(PyLong_AsLong(srcPy));
                            srcPyScalar = *reinterpret_cast<double *>(srcPyBuiltin.get());
                        }
                        if (srcToDstCastFunc != NULL)
                        {
                            srcToDstCastFunc(&srcPyScalar, &srcPyScalar, 1, NULL, NULL);
                            isSuccess = true;
                        }
                    }
                    Py_DECREF(srcPyDtype);
                }

                // Copy scalar bytes to destination if available
                if (isSuccess)
                {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> dst(
                        reinterpret_cast<double *>(dstPyData), PyArray_SIZE(dstPyArray));
                    dst.setConstant(srcPyScalar);
                    return;
                }
            }
            // Too complicated to deal with it manually. Falling back to default routine.
            if (PyArray_FillWithScalar(dstPyArray, srcPy) < 0)
            {
                throw std::runtime_error("Impossible to copy from 'src' to 'dst'.");
            }
            return;
        }

        // Check if too complicated to deal with it manually. Falling back to default routine.
        PyArrayObject * srcPyArray = reinterpret_cast<PyArrayObject *>(srcPy);
        const int dstNdim = PyArray_NDIM(dstPyArray);
        const int srcNdim = PyArray_NDIM(srcPyArray);
        const npy_intp * const dstShape = PyArray_SHAPE(dstPyArray);
        const npy_intp * const srcShape = PyArray_SHAPE(srcPyArray);
        const int srcPyFlags = PyArray_FLAGS(srcPyArray);
        const int commonPyFlags = dstPyFlags & srcPyFlags;
        if (dstNdim != srcNdim || !PyArray_CompareLists(dstShape, srcShape, dstNdim) ||
            !(commonPyFlags & NPY_ARRAY_ALIGNED) || !PyArray_EquivArrTypes(dstPyArray, srcPyArray))
        {
            if (PyArray_CopyInto(dstPyArray, srcPyArray) < 0)
            {
                throw std::runtime_error("Impossible to copy from 'src' to 'dst'.");
            }
            return;
        }

        // Multi-dimensional array but no broadcasting nor casting required. Easy enough to handle.
        char * srcPyData = PyArray_BYTES(srcPyArray);
        if (commonPyFlags & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))
        {
            /* Fast specialization if both dst and src are jointly C- or F-contiguous.
               Note that converting arrays to Eigen matrices would leverage SIMD-vectorized
               assignment, which is faster than `memcpy`. Yet, instantiating the mapping is tricky
               because of memory alignment issues and dtype handling, so let's keep it simple. The
               slowdown should be marginal for small-size arrays (size < 50). */
            memcpy(dstPyData, srcPyData, PyArray_NBYTES(dstPyArray));
        }
        else if ((dstNdim == 2) && (itemsize == 8) &&
                 (dstPyFlags & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) &&
                 (srcPyFlags & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)))
        {
            /* Using Eigen once again to avoid slow element-wise copy assignment.
               TODO: Extend to support any number of dims by working on flattened view. */
            using EigenMapType = Eigen::Map<Eigen::MatrixXd>;
            if (dstPyFlags & NPY_ARRAY_C_CONTIGUOUS)
            {
                EigenMapType dst(reinterpret_cast<double *>(dstPyData), dstShape[1], dstShape[0]);
                EigenMapType src(reinterpret_cast<double *>(srcPyData), dstShape[0], dstShape[1]);
                dst = src.transpose();
            }
            else
            {
                EigenMapType dst(reinterpret_cast<double *>(dstPyData), dstShape[0], dstShape[1]);
                EigenMapType src(reinterpret_cast<double *>(srcPyData), dstShape[1], dstShape[0]);
                dst = src.transpose();
            }
        }
        else
        {
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
        }
    }

    void exposeHelpers()
    {
        // clang-format off
        bp::def("build_geom_from_urdf", &buildGeomFromUrdf,
                                        (bp::arg("pinocchio_model"), "urdf_filename", "geom_type",
                                         bp::arg("mesh_package_dirs") = bp::list(),
                                         bp::arg("load_meshes") = true,
                                         bp::arg("make_meshes_convex") = false));

        bp::def("build_models_from_urdf", &buildModelsFromUrdf,
                                          (bp::arg("urdf_path"), "has_freeflyer",
                                           bp::arg("mesh_package_dirs") = bp::list(),
                                           bp::arg("build_visual_model") = false,
                                           bp::arg("load_visual_meshes") = false));

        bp::def("get_joint_type", &getJointTypeFromIdx,
                                  (bp::arg("pinocchio_model"), "joint_idx"));
        bp::def("get_joint_position_idx", &getJointPositionIdx,
                                          (bp::arg("pinocchio_model"), "joint_name"));
        bp::def("is_position_valid", &isPositionValid,
                                     (bp::arg("pinocchio_model"), "position"));

        bp::def("array_copyto", &arrayCopyTo, (bp::arg("dst"), "src"));

        // Do not use EigenPy To-Python converter because it considers matrices with 1 column as vectors
        bp::def("interpolate", &interpolate,
                               (bp::arg("pinocchio_model"), "times_in", "positions_in", "times_out"),
                               bp::return_value_policy<result_converter<true>>());

        bp::def("aba",
                &pinocchio_overload::aba<
                    double, 0, pinocchio::JointCollectionDefaultTpl, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, pinocchio::Force>,
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v", "u", "fext"),
                "Compute ABA with external forces, store the result in Data::ddq and return it.",
                bp::return_value_policy<result_converter<false>>());
        bp::def("rnea",
                &pinocchio_overload::rnea<
                    double, 0, pinocchio::JointCollectionDefaultTpl, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>,
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v", "a"),
                "Compute the RNEA without external forces, store the result in Data and return it.",
                bp::return_value_policy<result_converter<false>>());
        bp::def("rnea",
                &pinocchio_overload::rnea<
                    double, 0, pinocchio::JointCollectionDefaultTpl, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, pinocchio::Force>,
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v", "a", "fext"),
                "Compute the RNEA with external forces, store the result in Data and return it.",
                bp::return_value_policy<result_converter<false>>());
        bp::def("crba",
                &pinocchio_overload::crba<
                    double, 0, pinocchio::JointCollectionDefaultTpl, Eigen::VectorXd>,
                (bp::arg("pinocchio_model"), "pinocchio_data", "q"),
                "Computes CRBA, store the result in Data and return it.",
                bp::return_value_policy<result_converter<false>>());
        bp::def("computeKineticEnergy",
                &pinocchio_overload::computeKineticEnergy<
                    double, 0, pinocchio::JointCollectionDefaultTpl, Eigen::VectorXd, Eigen::VectorXd>,
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v"),
                "Computes the forward kinematics and the kinematic energy of the model for the "
                "given joint configuration and velocity given as input. "
                "The result is accessible through data.kinetic_energy.");

        bp::def("computeJMinvJt",
                &pinocchio_overload::computeJMinvJt<Eigen::MatrixXd>,
                (bp::arg("pinocchio_model"), "pinocchio_data", "J", bp::arg("update_decomposition") = true));
        bp::def("solveJMinvJtv", &solveJMinvJtv,
                (bp::arg("pinocchio_data"), "v", bp::arg("update_decomposition") = true));
        // clang-format on
    }
}
