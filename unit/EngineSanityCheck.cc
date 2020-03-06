// Test the sanity of the simulation engine.
// The tests in this file verify that the behavior of a simulated system matches
// real-world physics.
// The test system is a double inverted pendulum.
#include <pinocchio/fwd.hpp>
#include <sys/types.h>
#include <pwd.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <string>
#include <gtest/gtest.h>

#include "jiminy/core/Types.h"
#include "jiminy/core/Utilities.h"
#include "jiminy/core/Engine.h"
#include "jiminy/core/ControllerFunctor.h"

using namespace jiminy;

// Controller sending zero torque to the motors.
void controllerZeroTorque(float64_t const & t,
                          vectorN_t const & q,
                          vectorN_t const & v,
                          sensorsDataMap_t const & sensorData,
                          vectorN_t & u)
{
    u.setZero();
}

// Internal dynamics of the system (friction, ...)
void internalDynamics(float64_t const & t,
                      vectorN_t const & q,
                      vectorN_t const & v,
                      sensorsDataMap_t const & sensorData,
                      vectorN_t       & u)
{
    u.setZero();
}

bool callback(float64_t const & t,
              vectorN_t const & x)
{
    return true;
}


TEST(EngineSanity, EnergyConservation)
{
    // Verify that when sending zero torque to a system, its energy remains constant.

    // Load double pendulum model.
    std::string urdfPath = "data/double_pendulum_rigid.urdf";
    // All joints actuated.
    std::vector<std::string> jointNames;
    jointNames.push_back("PendulumJoint");
    jointNames.push_back("SecondPendulumJoint");

    std::shared_ptr<Model> model = std::make_shared<Model>();
    // Disable velocity and position limits.
    configHolder_t mdlOptions = model->getOptions();
    boost::get<bool>(boost::get<configHolder_t>(mdlOptions.at("joints")).at("enablePositionLimit")) = false;
    boost::get<bool>(boost::get<configHolder_t>(mdlOptions.at("joints")).at("enableVelocityLimit")) = false;
    boost::get<bool>(boost::get<configHolder_t>(mdlOptions.at("joints")).at("enableTorqueLimit")) = false;
    model->setOptions(mdlOptions);

    model->initialize(urdfPath, false);
    model->addMotors(jointNames);

    auto controller = std::make_shared<ControllerFunctor<decltype(controllerZeroTorque),
                                                         decltype(internalDynamics)> >(controllerZeroTorque, internalDynamics);
    controller->initialize(model);

    // Continuous simulation
    Engine engine;
    engine.initialize(model, controller, callback);

    // Run simulation
    vectorN_t x0 = vectorN_t::Zero(4);
    x0(0) = 1.0;
    float64_t tf = 10.0;

    // Run simulation
    engine.simulate(x0, tf);

    // Get system energy.
    std::vector<std::string> header;
    matrixN_t data;
    engine.getLogData(header, data);
    vectorN_t energy = Engine::getLogFieldValue("HighLevelController.energy", header, data);
    ASSERT_GT(energy.size(), 0);

    // Ignore first sample where energy is zero.
    vectorN_t energyCrop = energy.tail(energy.size() - 1);
    // Check that energy is constant.
    float64_t deltaEnergy = energyCrop.maxCoeff() - energyCrop.minCoeff();
    ASSERT_NEAR(0.0, std::abs(deltaEnergy), std::numeric_limits<float64_t>::epsilon());

    // Discrete-time simulation
    configHolder_t simuOptions = engine.getDefaultOptions();
    boost::get<float64_t>(boost::get<configHolder_t>(simuOptions.at("stepper")).at("sensorsUpdatePeriod")) = 1.0e-3;
    boost::get<float64_t>(boost::get<configHolder_t>(simuOptions.at("stepper")).at("controllerUpdatePeriod")) = 1.0e-3;
    engine.setOptions(simuOptions);
    engine.setState(x0);
    engine.simulate(x0, tf);

    engine.getLogData(header, data);
    energy = Engine::getLogFieldValue("HighLevelController.energy", header, data);
    ASSERT_GT(energy.size(), 0);
    energyCrop = energy.tail(energy.size() - 1);
    deltaEnergy = energyCrop.maxCoeff() - energyCrop.minCoeff();
    ASSERT_NEAR(0.0, std::abs(deltaEnergy), std::numeric_limits<float64_t>::epsilon());

    // Don't try simulation with Euler integrator, this scheme is not precise enough to keep energy constant.
}

