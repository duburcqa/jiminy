#include "jiminy/core/robot/abstract_sensor.h"
#include "jiminy/core/robot/abstract_motor.h"
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


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    uint32_t getRandomSeed(void)
    {
        uint32_t seed;
        ::jiminy::getRandomSeed(seed);  // Cannot fail since random number generators are initialized when imported
        return seed;
    }

    joint_t getJointTypeFromIdx(pinocchio::Model const & model,
                                int32_t          const & idIn)
    {
        joint_t jointType = joint_t::NONE;
        ::jiminy::getJointTypeFromIdx(model, idIn, jointType);
        return jointType;
    }

    int32_t getJointPositionIdx(pinocchio::Model const & model,
                                std::string      const & name)
    {
        int32_t jointPositionFirstIdx = model.nq;
        ::jiminy::getJointPositionIdx(model, name, jointPositionFirstIdx);
        return jointPositionFirstIdx;
    }

    bool_t isPositionValid(pinocchio::Model const & model,
                           vectorN_t        const & position)
    {
        bool_t isValid;
        ::jiminy::isPositionValid(
            model, position, isValid, Eigen::NumTraits<float64_t>::dummy_precision());
        return isValid;
    }

    matrixN_t interpolate(pinocchio::Model const & modelIn,
                          vectorN_t        const & timesIn,
                          matrixN_t        const & positionsIn,
                          vectorN_t        const & timesOut)
    {
        matrixN_t positionOut;
        ::jiminy::interpolate(modelIn, timesIn, positionsIn, timesOut, positionOut);
        return positionOut;
    }

    pinocchio::GeometryModel buildGeomFromUrdf(pinocchio::Model const & model,
                                               std::string const & filename,
                                               int const & typePy,
                                               bp::list const & packageDirsPy,
                                               bool_t const & loadMeshes,
                                               bool_t const & makeMeshesConvex)
    {
        /* Note that enum bindings interoperability is buggy, so that `pin.GeometryType`
           is not properly converted from Python to C++ automatically in some cases. */
        pinocchio::GeometryModel geometryModel;
        auto const type = static_cast<pinocchio::GeometryType>(typePy);
        auto packageDirs = convertFromPython<std::vector<std::string> >(packageDirsPy);
        ::jiminy::buildGeomFromUrdf(model,
                                    filename,
                                    type,
                                    geometryModel,
                                    packageDirs,
                                    loadMeshes,
                                    makeMeshesConvex);
        return geometryModel;
    }

    bp::tuple buildModelsFromUrdf(std::string const & urdfPath,
                                  bool_t const & hasFreeflyer,
                                  bp::list const & packageDirsPy,
                                  bool_t const & buildVisualModel,
                                  bool_t const & loadVisualMeshes)
    {
        /* Note that enum bindings interoperability is buggy, so that `pin.GeometryType`
           is not properly converted from Python to C++ automatically in some cases. */
        pinocchio::Model model;
        pinocchio::GeometryModel collisionModel;
        pinocchio::GeometryModel visualModel;
        std::optional<std::reference_wrapper<pinocchio::GeometryModel> > visualModelOptional = std::nullopt;
        if (buildVisualModel)
        {
            visualModelOptional = visualModel;
        }
        auto packageDirs = convertFromPython<std::vector<std::string> >(packageDirsPy);
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

    np::ndarray solveJMinvJtv(pinocchio::Data & data,
                              np::ndarray const & vPy,
                              bool_t const & updateDecomposition)
    {
        int32_t const nDims = vPy.get_nd();
        assert(nDims < 3 && "The number of dimensions of 'v' cannot exceed 2.");
        if (nDims == 1)
        {
            vectorN_t const v = convertFromPython<vectorN_t>(vPy);
            vectorN_t const x = pinocchio_overload::solveJMinvJtv<vectorN_t>(data, v, updateDecomposition);
            return bp::extract<np::ndarray>(convertToPython(x, true));
        }
        else
        {
            matrixN_t const v = convertFromPython<matrixN_t>(vPy);
            matrixN_t const x = pinocchio_overload::solveJMinvJtv<matrixN_t>(data, v, updateDecomposition);
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
        int const dstPyFlags = PyArray_FLAGS(dstPyArray);
        if (!(dstPyFlags & NPY_ARRAY_WRITEABLE))
        {
            throw std::runtime_error("'dst' must be writable.");
        }

        // Dedicated path to fill with scalar
        npy_intp const itemsize = PyArray_ITEMSIZE(dstPyArray);
        char * dstPyData = PyArray_BYTES(dstPyArray);
        if (!PyArray_Check(srcPy) || PyArray_IsScalar(srcPy, Generic)
         || (PyArray_SIZE(reinterpret_cast<PyArrayObject *>(srcPy)) == 1))
        {
            /* Eigen does a much better job than element-wise copy assignment in this scenario.
               Note that only the width of the scalar type matters here, not the actual type.
               Ensure copy and casting are both slow as they allocate new array, so avoiding
               using them entirely if possible and falling back to default routine otherwise. */
            if ((itemsize == 8) && (dstPyFlags & NPY_ARRAY_ALIGNED)
             && (dstPyFlags & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)))
            {
                // Convert src scalar data to raw bytes with dst dtype, casting if necessary
                bool_t isSuccess = false;
                float64_t srcPyScalar;
                if (PyArray_Check(srcPy))
                {
                    PyArrayObject * srcPyArray = reinterpret_cast<PyArrayObject *>(srcPy);
                    if (PyArray_EquivArrTypes(dstPyArray, srcPyArray))
                    {
                        srcPyScalar = *reinterpret_cast<float64_t *>(PyArray_DATA(srcPyArray));
                        isSuccess = true;
                    }
                    else
                    {
                        int dstPyTypeNum = PyArray_TYPE(dstPyArray);
                        PyArray_Descr * srcPyDtype = PyArray_DESCR(srcPyArray);
                        if (!PyTypeNum_ISEXTENDED(dstPyTypeNum) && !PyTypeNum_ISEXTENDED(srcPyDtype->type_num))
                        {
                            auto srcToDstCastFunc = PyArray_GetCastFunc(srcPyDtype, dstPyTypeNum);
                            srcToDstCastFunc(PyArray_DATA(srcPyArray), &srcPyScalar, 1, NULL, NULL);
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
                            srcPyScalar = *reinterpret_cast<float64_t *>(srcPyBuiltin.get());
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
                    Eigen::Map<Eigen::Matrix<float64_t, Eigen::Dynamic, 1> > dst(
                        reinterpret_cast<float64_t *>(dstPyData), PyArray_SIZE(dstPyArray));
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
        int const dstNdim = PyArray_NDIM(dstPyArray);
        int const srcNdim = PyArray_NDIM(srcPyArray);
        npy_intp const * const dstShape = PyArray_SHAPE(dstPyArray);
        npy_intp const * const srcShape = PyArray_SHAPE(srcPyArray);
        int const srcPyFlags = PyArray_FLAGS(srcPyArray);
        int const commonPyFlags = dstPyFlags & srcPyFlags;
        if (dstNdim != srcNdim || !PyArray_CompareLists(dstShape, srcShape, dstNdim)
         || !(commonPyFlags & NPY_ARRAY_ALIGNED) || !PyArray_EquivArrTypes(dstPyArray, srcPyArray))
        {
            if (PyArray_CopyInto(dstPyArray, srcPyArray) < 0)
            {
                throw std::runtime_error("Impossible to copy from 'src' to 'dst'.");
            }
            return;
        }

        // Multi-dimensional array but no broadcasting nor casting required. Easy enough to handle it.
        char * srcPyData = PyArray_BYTES(srcPyArray);
        if (commonPyFlags & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))
        {
            /* Fast specialization if both dst and src are jointly C- or F-contiguous.
               Note that converting arrays to Eigen matrices would leverage SIMD-vectorized
               assignment, which is faster than `memcpy`. Yet, instantiating the mapping is
               tricky because of memory alignment issues and dtype handling, so let's keep
               it simple. The slowdown should be marginal for small-size arrays (size < 50). */
            memcpy(dstPyData, srcPyData, PyArray_NBYTES(dstPyArray));
        }
        else if ((dstNdim == 2) && (itemsize == 8)
         && (dstPyFlags & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))
         && (srcPyFlags & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)))
        {
            /* Using Eigen once again to avoid slow element-wise copy assignment.
               TODO: Extend to support any number of dims by working on flattened view. */
            using EigenMapType = Eigen::Map<matrixN_t>;
            if (dstPyFlags & NPY_ARRAY_C_CONTIGUOUS)
            {
                EigenMapType dst(reinterpret_cast<float64_t *>(dstPyData), dstShape[1], dstShape[0]);
                EigenMapType src(reinterpret_cast<float64_t *>(srcPyData), dstShape[0], dstShape[1]);
                dst = src.transpose();
            }
            else
            {
                EigenMapType dst(reinterpret_cast<float64_t *>(dstPyData), dstShape[0], dstShape[1]);
                EigenMapType src(reinterpret_cast<float64_t *>(srcPyData), dstShape[1], dstShape[0]);
                dst = src.transpose();
            }
        }
        else
        {
            // Falling back to slow element-wise strided ND-array copy assignment
            int i = 0;
            npy_intp coord[NPY_MAXDIMS];
            npy_intp const * const dstStrides = PyArray_STRIDES(dstPyArray);
            npy_intp const * const srcStrides = PyArray_STRIDES(srcPyArray);
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

    void exposeHelpers(void)
    {
        bp::def("get_random_seed", bp::make_function(&getRandomSeed,
                                   bp::return_value_policy<bp::return_by_value>()));
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
                               bp::return_value_policy<result_converter<true> >());

        bp::def("aba",
                &pinocchio_overload::aba<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t, vectorN_t, vectorN_t, pinocchio::Force>,
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v", "u", "fext"),
                "Compute ABA with external forces, store the result in Data::ddq and return it.",
                bp::return_value_policy<result_converter<false> >());
        bp::def("rnea",
                &pinocchio_overload::rnea<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t, vectorN_t, vectorN_t>,
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v", "a"),
                "Compute the RNEA without external forces, store the result in Data and return it.",
                bp::return_value_policy<result_converter<false> >());
        bp::def("rnea",
                &pinocchio_overload::rnea<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t, vectorN_t, vectorN_t, pinocchio::Force>,
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v", "a", "fext"),
                "Compute the RNEA with external forces, store the result in Data and return it.",
                bp::return_value_policy<result_converter<false> >());
        bp::def("crba",
                &pinocchio_overload::crba<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t>,
                (bp::arg("pinocchio_model"), "pinocchio_data", "q"),
                "Computes CRBA, store the result in Data and return it.",
                bp::return_value_policy<result_converter<false> >());
        bp::def("computeKineticEnergy",
                &pinocchio_overload::computeKineticEnergy<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t, vectorN_t>,
                (bp::arg("pinocchio_model"), "pinocchio_data", "q", "v"),
                "Computes the forward kinematics and the kinematic energy of the model for the "
                "given joint configuration and velocity given as input. "
                "The result is accessible through data.kinetic_energy.");

        bp::def("computeJMinvJt",
                &pinocchio_overload::computeJMinvJt<matrixN_t>,
                (bp::arg("pinocchio_model"), "pinocchio_data", "J", bp::arg("update_decomposition") = true));
        bp::def("solveJMinvJtv", &solveJMinvJtv,
                (bp::arg("pinocchio_data"), "v", bp::arg("update_decomposition") = true));
    }

}  // End of namespace python.
}  // End of namespace jiminy.
