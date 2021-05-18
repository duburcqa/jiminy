// Test the sanity of the simulation engine.
// The tests in this file verify that the behavior of a simulated system matches
// real-world physics, and that no memory is allocated by Eigen during a simulation.
// The test system is a double inverted pendulum.
#include <gtest/gtest.h>

#define EIGEN_RUNTIME_NO_MALLOC

#include "jiminy/core/engine/Engine.h"
#include "jiminy/core/robot/BasicMotors.h"
#include "jiminy/core/control/ControllerFunctor.h"
#include "jiminy/core/utilities/Helpers.h"
#include "jiminy/core/Types.h"


using namespace jiminy;

float64_t const TOLERANCE = 1e-9;


// Controller sending zero torque to the motors.
void controllerZeroTorque(float64_t        const & /* t */,
                          vectorN_t        const & /* q */,
                          vectorN_t        const & /* v */,
                          sensorsDataMap_t const & /* sensorData */,
                          vectorN_t              & /* command */)
{
    // Empty on purpose
}

// Internal dynamics of the system (friction, ...)
void internalDynamics(float64_t        const & /* t */,
                      vectorN_t        const & /* q */,
                      vectorN_t        const & /* v */,
                      sensorsDataMap_t const & /* sensorData */,
                      vectorN_t              & /* uCustom */)
{
    // Empty on purpose
}

bool_t callback(float64_t const & /* t */,
                vectorN_t const & /* q */,
                vectorN_t const & /* v */)
{
    return true;
}


TEST(EngineSanity, EnergyConservation)
{
    // Verify that when sending zero torque to a system, its energy remains constant

    // Double pendulum model
    std::string const dataDirPath(UNIT_TEST_DATA_DIR);
    auto const urdfPath = dataDirPath + "/double_pendulum_rigid.urdf";

    // All joints actuated.
    std::vector<std::string> motorJointNames{"PendulumJoint", "SecondPendulumJoint"};

    auto robot = std::make_shared<Robot>();
    robot->initialize(urdfPath, false);
    for (std::string const & jointName : motorJointNames)
    {
        auto motor = std::make_shared<SimpleMotor>(jointName);
        robot->attachMotor(motor);
        motor->initialize(jointName);
    }

    // Disable velocity and position limits
    configHolder_t modelOptions = robot->getModelOptions();
    boost::get<bool_t>(boost::get<configHolder_t>(modelOptions.at("joints")).at("enablePositionLimit")) = false;
    boost::get<bool_t>(boost::get<configHolder_t>(modelOptions.at("joints")).at("enableVelocityLimit")) = false;
    robot->setModelOptions(modelOptions);

    // Disable torque limits
    configHolder_t motorsOptions = robot->getMotorsOptions();
    for (auto & options : motorsOptions)
    {
        configHolder_t & motorOptions = boost::get<configHolder_t>(options.second);
        boost::get<bool_t>(motorOptions.at("enableCommandLimit")) = false;
    }
    robot->setMotorsOptions(motorsOptions);

    auto controller = std::make_shared<
        ControllerFunctor<decltype(controllerZeroTorque),
                          decltype(internalDynamics)>
    >(controllerZeroTorque, internalDynamics);
    controller->initialize(robot);

    // Create engine
    auto engine = std::make_shared<Engine>();
    engine->initialize(robot, controller, callback);

    // Configure engine: High accuracy + Continuous-time integration
    configHolder_t simuOptions = engine->getDefaultEngineOptions();
    boost::get<float64_t>(boost::get<configHolder_t>(simuOptions.at("stepper")).at("tolAbs")) = TOLERANCE * 1.0e-2;
    boost::get<float64_t>(boost::get<configHolder_t>(simuOptions.at("stepper")).at("tolRel")) = TOLERANCE * 1.0e-2;
    engine->setOptions(simuOptions);

    // Run simulation
    vectorN_t q0 = vectorN_t::Zero(2);
    q0(0) = 1.0;
    vectorN_t v0 = vectorN_t::Zero(2);
    float64_t tf = 10.0;

    // Run simulation
    engine->reset();
    engine->start(q0, v0);
    Eigen::internal::set_is_malloc_allowed(false);
    engine->step(tf);
    engine->stop();
    Eigen::internal::set_is_malloc_allowed(true);

    // Get system energy
    std::vector<std::string> header;
    matrixN_t data;
    engine->getLogData(header, data);
    auto timeCont = getLogFieldValue("Global.Time", header, data);
    ASSERT_DOUBLE_EQ(timeCont[timeCont.size()-1], tf);
    auto energyCont = getLogFieldValue("HighLevelController.energy", header, data);
    ASSERT_GT(energyCont.size(), 0);

    // Check that energy is constant
    float64_t const deltaEnergyCont = energyCont.maxCoeff() - energyCont.minCoeff();
    ASSERT_NEAR(0.0, deltaEnergyCont, TOLERANCE);

    // Configure engine: Default accuracy + Discrete-time simulation
    simuOptions = engine->getDefaultEngineOptions();
    boost::get<float64_t>(boost::get<configHolder_t>(simuOptions.at("stepper")).at("sensorsUpdatePeriod")) = 1.0e-3;
    boost::get<float64_t>(boost::get<configHolder_t>(simuOptions.at("stepper")).at("controllerUpdatePeriod")) = 1.0e-3;
    engine->setOptions(simuOptions);

    // Run simulation
    engine->reset();
    engine->start(q0, v0);
    Eigen::internal::set_is_malloc_allowed(false);
    engine->step(tf);
    engine->stop();
    Eigen::internal::set_is_malloc_allowed(true);

    // Get system energy
    engine->getLogData(header, data);
    auto timeDisc = getLogFieldValue("Global.Time", header, data);
    ASSERT_DOUBLE_EQ(timeDisc[timeDisc.size()-1], tf);
    auto energyDisc = getLogFieldValue("HighLevelController.energy", header, data);
    ASSERT_GT(energyDisc.size(), 0);

    // Check that energy is constant
    float64_t const deltaEnergyDisc = energyDisc.maxCoeff() - energyDisc.minCoeff();
    ASSERT_NEAR(0.0, deltaEnergyDisc, TOLERANCE);

    // Don't try simulation with Euler integrator, this scheme is not precise enough to keep energy constant.
}
