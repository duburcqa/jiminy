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
void controllerZeroTorque(float64_t const & t, vectorN_t const & q, vectorN_t const & v, vectorN_t & u)
{
    u.setZero();
}

// Internal dynamics of the system (friction, ...)
void internalDynamics(float64_t const & t,
                      vectorN_t const & q,
                      vectorN_t const & v,
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
    // No contact point.
    std::vector<std::string> contacts;
    // All joints actuated.
    std::vector<std::string> jointNames;
    jointNames.push_back("PendulumJoint");
    jointNames.push_back("SecondPendulumJoint");

    Model model;
    model.initialize(urdfPath, contacts, jointNames, false);

    ControllerFunctor<decltype(controllerZeroTorque), decltype(internalDynamics)> controller(controllerZeroTorque, internalDynamics);
    controller.initialize(model);

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
    vectorN_t energy = engine.getLogFieldValue("HighLevelController.energy");
    ASSERT_GT(energy.size(), 0);

    // Ignore first sample where energy is zero.
    vectorN_t energyCrop = energy.tail(energy.size() - 1);
    // Check that energy is constant.
    float64_t deltaEnergy = energyCrop.maxCoeff() - energyCrop.minCoeff();
    engine.writeLogBinary("/tmp/blackbox/log.data");
    ASSERT_NEAR(0.0, std::abs(deltaEnergy), std::numeric_limits<float64_t>::epsilon());

    // Discrete-time simulation
    configHolder_t simuOptions = engine.getDefaultOptions();
    boost::get<float64_t>(boost::get<configHolder_t>(simuOptions.at("stepper")).at("sensorsUpdatePeriod")) = 1.0e-3;
    boost::get<float64_t>(boost::get<configHolder_t>(simuOptions.at("stepper")).at("controllerUpdatePeriod")) = 1.0e-3;
    engine.setOptions(simuOptions);
    engine.reset(x0);
    engine.simulate(x0, tf);

    energy = engine.getLogFieldValue("HighLevelController.energy");
    ASSERT_GT(energy.size(), 0);
    energyCrop = energy.tail(energy.size() - 1);
    deltaEnergy = energyCrop.maxCoeff() - energyCrop.minCoeff();
    ASSERT_NEAR(0.0, std::abs(deltaEnergy), std::numeric_limits<float64_t>::epsilon());

    // Don't try simulation with Euler integrator, this scheme is not precise enough to keep energy constant.
}

