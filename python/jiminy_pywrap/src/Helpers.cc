#include "jiminy/core/robot/AbstractSensor.h"
#include "jiminy/core/robot/AbstractMotor.h"
#include "jiminy/core/constraints/AbstractConstraint.h"
#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/robot/PinocchioOverloadAlgorithms.h"
#include "jiminy/core/io/MemoryDevice.h"
#include "jiminy/core/utilities/Pinocchio.h"
#include "jiminy/core/utilities/Random.h"

#include <boost/optional.hpp>

#include "pinocchio/bindings/python/fwd.hpp"

#include "jiminy/python/Utilities.h"
#include "jiminy/python/Helpers.h"


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

        bp::def("interpolate", &interpolate,
                               (bp::arg("pinocchio_model"), "times_in", "positions_in", "times_out"));

        bp::def("aba",
                &pinocchio_overload::aba<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t, vectorN_t, vectorN_t, pinocchio::Force>,
                bp::args("pinocchio_model", "pinocchio_data", "q", "v", "u", "fext"),
                "Compute ABA with external forces, store the result in Data::ddq and return it.",
                bp::return_value_policy<result_converter<false> >());
        bp::def("rnea",
                &pinocchio_overload::rnea<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t, vectorN_t, vectorN_t>,
                bp::args("pinocchio_model", "pinocchio_data", "q", "v", "a"),
                "Compute the RNEA without external forces, store the result in Data and return it.",
                bp::return_value_policy<result_converter<false> >());
        bp::def("rnea",
                &pinocchio_overload::rnea<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t, vectorN_t, vectorN_t, pinocchio::Force>,
                bp::args("pinocchio_model", "pinocchio_data", "q", "v", "a", "fext"),
                "Compute the RNEA with external forces, store the result in Data and return it.",
                bp::return_value_policy<result_converter<false> >());
        bp::def("crba",
                &pinocchio_overload::crba<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t>,
                bp::args("pinocchio_model", "pinocchio_data", "q"),
                "Computes CRBA, store the result in Data and return it.",
                bp::return_value_policy<result_converter<false> >());
        bp::def("computeKineticEnergy",
                &pinocchio_overload::computeKineticEnergy<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t, vectorN_t>,
                bp::args("pinocchio_model", "pinocchio_data", "q", "v"),
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
