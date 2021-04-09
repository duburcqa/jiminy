#include "jiminy/core/robot/AbstractSensor.h"
#include "jiminy/core/robot/AbstractMotor.h"
#include "jiminy/core/robot/AbstractConstraint.h"
#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/robot/PinocchioOverloadAlgorithms.h"
#include "jiminy/core/utilities/Pinocchio.h"
#include "jiminy/core/utilities/Random.h"

#include "jiminy/python/Utilities.h"
#include "jiminy/python/Helpers.h"

#include <boost/python.hpp>
#include <eigenpy/eigenpy.hpp>  // Required to have access to eigenpy from python converters


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    joint_t getJointTypeFromIdx(pinocchio::Model const & model,
                                int32_t          const & idIn)
    {
        joint_t jointType = joint_t::NONE;
        ::jiminy::getJointTypeFromIdx(model, idIn, jointType);
        return jointType;
    }

    bool_t isPositionValid(pinocchio::Model const & model,
                           vectorN_t        const & position)
    {
        bool_t isValid;
        hresult_t returnCode = ::jiminy::isPositionValid(
            model, position, isValid, Eigen::NumTraits<float64_t>::dummy_precision());
        if (returnCode != hresult_t::SUCCESS)
        {
            return false;
        }
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

    void resetRandomGenerators(bp::object const & seedPy)
    {
        std::optional<uint32_t> seed = std::nullopt;
        if (!seedPy.is_none())
        {
            seed = bp::extract<uint32_t>(seedPy);
        }
        ::jiminy::resetRandomGenerators(seed);
    }

    pinocchio::GeometryModel buildGeomFromUrdf(pinocchio::Model const & model,
                                               std::string const & filename,
                                               pinocchio::GeometryType const & type,
                                               std::vector<std::string> const & packageDirs,
                                               bool_t loadMeshes)
    {
        pinocchio::GeometryModel geometryModel;
        buildGeom(model, filename, type, geometryModel, packageDirs, loadMeshes);
        return geometryModel;
    }

    void exposeHelpers(void)
    {
        bp::def("reset_random_generator", &resetRandomGenerators, (bp::arg("seed") = bp::object()));

        bp::def("buildGeomFromUrdf", &buildGeomFromUrdf,
                                     (bp::arg("pinocchio_model"), "urdf_filename", "geom_type",
                                      bp::arg("package_dirs") = std::vector<std::string>(),
                                      bp::arg("load_meshes") = true));

        bp::def("get_joint_type", &getJointTypeFromIdx,
                                  (bp::arg("pinocchio_model"), "joint_idx"));
        bp::def("is_position_valid", &isPositionValid,
                                     (bp::arg("pinocchio_model"), "position"));

        bp::def("interpolate", &interpolate,
                               (bp::arg("pinocchio_model"), "times_in", "positions_in", "times_out"));

        bp::def("aba",
                &pinocchio_overload::aba<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t, vectorN_t, vectorN_t, pinocchio::Force>,
                bp::args("pinocchio_model", "pinocchio_data", "q", "v", "u", "fext"),
                "Compute ABA with external forces, store the result in Data::ddq and return it.",
                bp::return_value_policy<bp::return_by_value>());
        bp::def("rnea",
                &pinocchio_overload::aba<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t, vectorN_t, vectorN_t, pinocchio::Force>,
                bp::args("pinocchio_model", "pinocchio_data", "q", "v", "a", "fext"),
                "Compute the RNEA with external forces, store the result in Data and return it.",
                bp::return_value_policy<bp::return_by_value>());
        bp::def("crba",
                &pinocchio_overload::crba<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t>,
                bp::args("pinocchio_model", "pinocchio_data", "q"),
                "Computes CRBA, store the result in Data and return it.",
                bp::return_value_policy<bp::return_by_value>());
        bp::def("computeKineticEnergy",
                &pinocchio_overload::kineticEnergy<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t, vectorN_t>,
                bp::args("pinocchio_model", "pinocchio_data", "q", "v", "update_kinematics"),
                "Computes the forward kinematics and the kinematic energy of the model for the "
                "given joint configuration and velocity given as input. "
                "The result is accessible through data.kinetic_energy.");

        bp::class_<RandomPerlinProcess,
                   std::shared_ptr<RandomPerlinProcess>,
                   boost::noncopyable>("RandomPerlinProcess",
                   bp::init<float64_t const &, uint32_t const &>(
                   (bp::arg("self"), "wavelength", bp::arg("num_octaves") = 6U)))
            .def("__call__", &RandomPerlinProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", &RandomPerlinProcess::reset)
            .add_property("wavelength", bp::make_function(&RandomPerlinProcess::getWavelength,
                                        bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("num_octaves", bp::make_function(&RandomPerlinProcess::getNumOctaves,
                                         bp::return_value_policy<bp::copy_const_reference>()));

        bp::class_<PeriodicPerlinProcess,
                   std::shared_ptr<PeriodicPerlinProcess>,
                   boost::noncopyable>("PeriodicPerlinProcess",
                   bp::init<float64_t const &, float64_t const &, uint32_t const &>(
                   (bp::arg("self"), "wavelength", "period", bp::arg("num_octaves") = 6U)))
            .def("__call__", &PeriodicPerlinProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", &PeriodicPerlinProcess::reset)
            .add_property("wavelength", bp::make_function(&PeriodicPerlinProcess::getWavelength,
                                        bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("period", bp::make_function(&PeriodicPerlinProcess::getPeriod,
                                    bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("num_octaves", bp::make_function(&PeriodicPerlinProcess::getNumOctaves,
                                         bp::return_value_policy<bp::copy_const_reference>()));

        bp::class_<PeriodicGaussianProcess,
                   std::shared_ptr<PeriodicGaussianProcess>,
                   boost::noncopyable>("PeriodicGaussianProcess",
                   bp::init<float64_t const &, float64_t const &>(
                   bp::args("self", "wavelength", "period")))
            .def("__call__", &PeriodicGaussianProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", &PeriodicGaussianProcess::reset)
            .add_property("wavelength", bp::make_function(&PeriodicGaussianProcess::getWavelength,
                                        bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("period", bp::make_function(&PeriodicGaussianProcess::getPeriod,
                                    bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("dt", bp::make_function(&PeriodicGaussianProcess::getDt,
                                bp::return_value_policy<bp::copy_const_reference>()));

        bp::class_<PeriodicFourierProcess,
                   std::shared_ptr<PeriodicFourierProcess>,
                   boost::noncopyable>("PeriodicFourierProcess",
                   bp::init<float64_t const &, float64_t const &>(
                   bp::args("self", "wavelength", "period")))
            .def("__call__", &PeriodicFourierProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", &PeriodicFourierProcess::reset)
            .add_property("wavelength", bp::make_function(&PeriodicFourierProcess::getWavelength,
                                        bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("period", bp::make_function(&PeriodicFourierProcess::getPeriod,
                                    bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("dt", bp::make_function(&PeriodicFourierProcess::getDt,
                                bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("num_harmonics", bp::make_function(&PeriodicFourierProcess::getNumHarmonics,
                                           bp::return_value_policy<bp::copy_const_reference>()));
    }

}  // End of namespace python.
}  // End of namespace jiminy.
