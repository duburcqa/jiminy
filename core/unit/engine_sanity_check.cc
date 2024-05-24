/// \brief Test the sanity of the simulation engine on a double inverted pendulum.
///
/// \details The tests in this file verify that the behavior of a simulated system matches
///          real-world physics, and that no memory is allocated by Eigen during a simulation.
#include <filesystem>

#include <gtest/gtest.h>

#define EIGEN_RUNTIME_NO_MALLOC

#include "jiminy/core/fwd.h"
#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/hardware/fwd.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/hardware/basic_motors.h"
#include "jiminy/core/control/controller_functor.h"
#include "jiminy/core/engine/engine.h"


using namespace jiminy;

inline constexpr double TOLERANCE = 1e-9;


/// \brief Controller sending zero torque to the motors.
void computeCommand(double /* t */,
                    const Eigen::VectorXd & /* q */,
                    const Eigen::VectorXd & /* v */,
                    const SensorMeasurementTree & /* sensorData */,
                    Eigen::VectorXd & /* command */)
{
}

/// \brief Internal dynamics of the system (friction, ...)
void internalDynamics(double /* t */,
                      const Eigen::VectorXd & /* q */,
                      const Eigen::VectorXd & /* v */,
                      const SensorMeasurementTree & /* sensorData */,
                      Eigen::VectorXd & /* uCustom */)
{
}


TEST(EngineSanity, EnergyConservation)
{
    // Verify that when sending zero torque to a system, its energy remains constant

    // Double pendulum model
    const std::filesystem::path dataDirPath(UNIT_TEST_DATA_DIR);
    const auto urdfPath = dataDirPath / "double_pendulum_rigid.urdf";

    // All joints actuated
    std::vector<std::string> motorJointNames{"PendulumJoint", "SecondPendulumJoint"};
    auto robot = std::make_shared<Robot>();
    robot->initialize(urdfPath.string(), false);
    for (const std::string & jointName : motorJointNames)
    {
        auto motor = std::make_shared<SimpleMotor>(jointName);
        robot->attachMotor(motor);
        motor->initialize(jointName);
    }

    // Get all robot options
    GenericConfig robotOptions = robot->getOptions();

    // Disable position limits
    GenericConfig & modelOptions = boost::get<GenericConfig>(robotOptions.at("model"));
    GenericConfig & jointsOptions = boost::get<GenericConfig>(modelOptions.at("joints"));
    boost::get<bool>(jointsOptions.at("positionLimitFromUrdf")) = false;
    boost::get<Eigen::VectorXd>(jointsOptions.at("positionLimitMin")) =
        Eigen::Vector2d::Constant(-INF);
    boost::get<Eigen::VectorXd>(jointsOptions.at("positionLimitMax")) =
        Eigen::Vector2d::Constant(+INF);

    // Disable velocity and torque limits
    GenericConfig & motorsOptions = boost::get<GenericConfig>(robotOptions.at("motors"));
    for (auto & motorOptionsItem : motorsOptions)
    {
        GenericConfig & motorOptions = boost::get<GenericConfig>(motorOptionsItem.second);
        boost::get<bool>(motorOptions.at("enableVelocityLimit")) = false;
        boost::get<bool>(motorOptions.at("enableEffortLimit")) = false;
    }

    // Set all robot options
    robot->setOptions(robotOptions);

    // Instantiate the controller
    robot->setController(
        std::make_shared<FunctionalController<>>(computeCommand, internalDynamics));

    // Create engine
    Engine engine{};
    engine.addRobot(robot);

    // Configure engine: High accuracy + Continuous-time integration + telemetry
    GenericConfig simuOptions = engine.getDefaultEngineOptions();
    {
        GenericConfig & stepperOptions = boost::get<GenericConfig>(simuOptions.at("stepper"));
        boost::get<double>(stepperOptions.at("tolAbs")) = TOLERANCE * 1.0e-2;
        boost::get<double>(stepperOptions.at("tolRel")) = TOLERANCE * 1.0e-2;
    }
    {
        GenericConfig & telemetryOptions = boost::get<GenericConfig>(simuOptions.at("telemetry"));
        boost::get<bool>(telemetryOptions.at("enableEnergy")) = true;
    }
    engine.setOptions(simuOptions);

    // Run simulation
    Eigen::VectorXd q0 = Eigen::VectorXd::Zero(2);
    q0(0) = 1.0;
    Eigen::VectorXd v0 = Eigen::VectorXd::Zero(2);
    double tf = 10.0;

    // Run simulation
    engine.reset();
    engine.start(q0, v0);
    Eigen::internal::set_is_malloc_allowed(false);
    engine.step(tf);
    engine.stop();
    Eigen::internal::set_is_malloc_allowed(true);

    // Get system energy
    std::shared_ptr<const LogData> logDataPtr = engine.getLog();
    const Eigen::VectorXd timesCont = getLogVariable(*logDataPtr, "Global.Time");
    ASSERT_DOUBLE_EQ(timesCont[timesCont.size() - 1], tf);
    const Eigen::VectorXd energyCont = getLogVariable(*logDataPtr, "energy");
    ASSERT_GT(energyCont.size(), 0);

    // Check that energy is constant
    const double deltaEnergyCont = energyCont.maxCoeff() - energyCont.minCoeff();
    ASSERT_NEAR(0.0, deltaEnergyCont, TOLERANCE);

    // Configure engine: Default accuracy + Discrete-time simulation
    simuOptions = engine.getDefaultEngineOptions();
    {
        GenericConfig & stepperOptions = boost::get<GenericConfig>(simuOptions.at("stepper"));
        boost::get<double>(stepperOptions.at("sensorsUpdatePeriod")) = 1.0e-3;
        boost::get<double>(stepperOptions.at("controllerUpdatePeriod")) = 1.0e-3;
    }
    {
        GenericConfig & telemetryOptions = boost::get<GenericConfig>(simuOptions.at("telemetry"));
        boost::get<bool>(telemetryOptions.at("enableEnergy")) = true;
    }
    engine.setOptions(simuOptions);

    // Run simulation
    engine.reset();
    engine.start(q0, v0);
    Eigen::internal::set_is_malloc_allowed(false);
    engine.step(tf);
    engine.stop();
    Eigen::internal::set_is_malloc_allowed(true);

    // Get system energy
    logDataPtr = engine.getLog();
    const Eigen::VectorXd timesDisc = getLogVariable(*logDataPtr, "Global.Time");
    ASSERT_DOUBLE_EQ(timesDisc[timesDisc.size() - 1], tf);
    const Eigen::VectorXd energyDisc = getLogVariable(*logDataPtr, "energy");
    ASSERT_GT(energyDisc.size(), 0);

    // Check that energy is constant
    const double deltaEnergyDisc = energyDisc.maxCoeff() - energyDisc.minCoeff();
    ASSERT_NEAR(0.0, deltaEnergyDisc, TOLERANCE);
}
