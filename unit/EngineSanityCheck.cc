// Test the sanity of the simulation engine.
// The tests in this file verify that the behavior of a simulated system matches
// real-world physics.
// The test system is a double inverted pendulum.
#include <gtest/gtest.h>

#include "jiminy/core/engine/Engine.h"
#include "jiminy/core/robot/BasicMotors.h"
#include "jiminy/core/control/ControllerFunctor.h"
#include "jiminy/core/Utilities.h"
#include "jiminy/core/Types.h"


using namespace jiminy;


// Controller sending zero torque to the motors.
void controllerZeroTorque(float64_t                   const & t,
                          Eigen::Ref<vectorN_t const> const & q,
                          Eigen::Ref<vectorN_t const> const & v,
                          sensorsDataMap_t            const & sensorData,
                          vectorN_t                         & u)
{
    u.setZero();
}

// Internal dynamics of the system (friction, ...)
void internalDynamics(float64_t        const & t,
                      vectorN_t        const & q,
                      vectorN_t        const & v,
                      sensorsDataMap_t const & sensorData,
                      vectorN_t              & u)
{
    u.setZero();
}

bool_t callback(float64_t const & t,
                vectorN_t const & q,
                vectorN_t const & v)
{
    return true;
}


TEST(EngineSanity, EnergyConservation)
{
    // Verify that when sending zero torque to a system, its energy remains constant

    // Double pendulum model
    std::string urdfPath = "data/double_pendulum_rigid.urdf";
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

    // Disable velocity and position limits.
    configHolder_t modelOptions = robot->getModelOptions();
    boost::get<bool_t>(boost::get<configHolder_t>(modelOptions.at("joints")).at("enablePositionLimit")) = false;
    boost::get<bool_t>(boost::get<configHolder_t>(modelOptions.at("joints")).at("enableVelocityLimit")) = false;
    robot->setModelOptions(modelOptions);

    // Disable torque limits.
    configHolder_t motorsOptions = robot->getMotorsOptions();
    for (auto & options : motorsOptions)
    {
        configHolder_t & motorOptions = boost::get<configHolder_t>(options.second);
        boost::get<bool_t>(motorOptions.at("enableTorqueLimit")) = false;
    }
    robot->setMotorsOptions(motorsOptions);

    auto controller = std::make_shared<
        ControllerFunctor<decltype(controllerZeroTorque),
                          decltype(internalDynamics)>
    >(controllerZeroTorque, internalDynamics);
    controller->initialize(robot.get());

    // Continuous simulation
    auto engine = std::make_shared<Engine>();
    engine->initialize(robot, controller, callback);

    // Run simulation
    vectorN_t x0 = vectorN_t::Zero(4);
    x0(0) = 1.0;
    float64_t tf = 10.0;

    // Run simulation
    engine->simulate(tf, x0);

    // Get system energy.
    std::vector<std::string> header;
    matrixN_t data;
    engine->getLogData(header, data);
    auto energyCont = getLogFieldValue("HighLevelController.energy", header, data);
    ASSERT_GT(energyCont.size(), 0);

    // Check that energy is constant.
    float64_t const deltaEnergyCont = energyCont.maxCoeff() - energyCont.minCoeff();
    ASSERT_NEAR(0.0, deltaEnergyCont, std::numeric_limits<float64_t>::epsilon());

    // Discrete-time simulation
    configHolder_t simuOptions = engine->getOptions();
    boost::get<float64_t>(boost::get<configHolder_t>(simuOptions.at("stepper")).at("sensorsUpdatePeriod")) = 1.0e-3;
    boost::get<float64_t>(boost::get<configHolder_t>(simuOptions.at("stepper")).at("controllerUpdatePeriod")) = 1.0e-3;
    engine->setOptions(simuOptions);
    engine->simulate(tf, x0);

    // Get system energy.
    engine->getLogData(header, data);
    auto energyDisc = getLogFieldValue("HighLevelController.energy", header, data);
    ASSERT_GT(energyDisc.size(), 0);

    // Check that energy is constant.
    float64_t const deltaEnergyDisc = energyDisc.maxCoeff() - energyDisc.minCoeff();
    ASSERT_NEAR(0.0, deltaEnergyDisc, std::numeric_limits<float64_t>::epsilon());

    // Don't try simulation with Euler integrator, this scheme is not precise enough to keep energy constant.
}
