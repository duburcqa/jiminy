/// \brief A simple test case: simulation of a double inverted pendulum without contact forces.
///
/// \details This simulation checks the overall simulator sanity (i.e. conservation of energy) and
///          genericity (supporting a system that is not a humanoid robot).

#include <iostream>
#include <filesystem>

#include "jiminy/core/engine/engine.h"
#include "jiminy/core/hardware/basic_motors.h"
#include "jiminy/core/control/controller_functor.h"
#include "jiminy/core/io/file_device.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/types.h"


using namespace jiminy;

void computeCommand(const float64_t & /* t */,
                    const Eigen::VectorXd & /* q */,
                    const Eigen::VectorXd & /* v */,
                    const sensorsDataMap_t & /* sensorsData */,
                    Eigen::VectorXd & /* command */)
{
    // No controller: energy should be preserved
}

void internalDynamics(const float64_t & /* t */,
                      const Eigen::VectorXd & /* q */,
                      const Eigen::VectorXd & /* v */,
                      const sensorsDataMap_t & /* sensorsData */,
                      Eigen::VectorXd & /* uCustom */)
{
}

bool_t callback(
    const float64_t & /* t */, const Eigen::VectorXd & /* q */, const Eigen::VectorXd & /* v */)
{
    return true;
}

int main(int /* argc */, char_t * /* argv */[])
{
    // =====================================================================
    // ==================== Extract the user paramaters ====================
    // =====================================================================

    // Set URDF and log output.
    const std::filesystem::path filePath(__FILE__);
    const auto jiminySrcPath = filePath.parent_path().parent_path().parent_path().parent_path();
    const auto dataPath = jiminySrcPath / "data/toys_models";
    const auto urdfPath = dataPath / "double_pendulum/double_pendulum.urdf";
    const auto outputDirPath = std::filesystem::temp_directory_path();

    // =====================================================================
    // ============ Instantiate and configure the simulation ===============
    // =====================================================================

    // Instantiate timer
    Timer timer;

    timer.tic();

    // Instantiate and configuration the robot
    std::vector<std::string> motorJointNames{"SecondPendulumJoint"};

    auto robot = std::make_shared<Robot>();
    configHolder_t modelOptions = robot->getModelOptions();
    configHolder_t & jointsOptions = boost::get<configHolder_t>(modelOptions.at("joints"));
    boost::get<bool_t>(jointsOptions.at("positionLimitFromUrdf")) = true;
    boost::get<bool_t>(jointsOptions.at("velocityLimitFromUrdf")) = true;
    robot->setModelOptions(modelOptions);
    robot->initialize(urdfPath.string(), false, {dataPath.string()});
    for (const std::string & jointName : motorJointNames)
    {
        auto motor = std::make_shared<SimpleMotor>(jointName);
        robot->attachMotor(motor);
        motor->initialize(jointName);
    }

    // Instantiate and configuration the controller
    auto controller =
        std::make_shared<ControllerFunctor<decltype(computeCommand), decltype(internalDynamics)>>(
            computeCommand, internalDynamics);
    controller->initialize(robot);

    // Instantiate and configuration the engine
    auto engine = std::make_shared<Engine>();
    configHolder_t simuOptions = engine->getOptions();
    configHolder_t & telemetryOptions = boost::get<configHolder_t>(simuOptions.at("telemetry"));
    boost::get<bool_t>(telemetryOptions.at("isPersistent")) = true;
    boost::get<bool_t>(telemetryOptions.at("enableConfiguration")) = true;
    boost::get<bool_t>(telemetryOptions.at("enableVelocity")) = true;
    boost::get<bool_t>(telemetryOptions.at("enableAcceleration")) = true;
    boost::get<bool_t>(telemetryOptions.at("enableForceExternal")) = false;
    boost::get<bool_t>(telemetryOptions.at("enableCommand")) = true;
    boost::get<bool_t>(telemetryOptions.at("enableMotorEffort")) = true;
    boost::get<bool_t>(telemetryOptions.at("enableEnergy")) = true;
    configHolder_t & worldOptions = boost::get<configHolder_t>(simuOptions.at("world"));
    boost::get<Eigen::VectorXd>(worldOptions.at("gravity"))[2] = -9.81;
    configHolder_t & stepperOptions = boost::get<configHolder_t>(simuOptions.at("stepper"));
    boost::get<std::string>(stepperOptions.at("odeSolver")) = std::string("runge_kutta_dopri5");
    boost::get<float64_t>(stepperOptions.at("tolRel")) = 1.0e-5;
    boost::get<float64_t>(stepperOptions.at("tolAbs")) = 1.0e-4;
    boost::get<float64_t>(stepperOptions.at("dtMax")) = 3.0e-3;
    boost::get<float64_t>(stepperOptions.at("dtRestoreThresholdRel")) = 0.2;
    boost::get<uint32_t>(stepperOptions.at("iterMax")) = 100000U;  // -1 to disable
    boost::get<float64_t>(stepperOptions.at("timeout")) = -1;      // -1 to disable
    boost::get<float64_t>(stepperOptions.at("sensorsUpdatePeriod")) = 1.0e-3;
    boost::get<float64_t>(stepperOptions.at("controllerUpdatePeriod")) = 1.0e-3;
    boost::get<bool_t>(stepperOptions.at("logInternalStepperSteps")) = false;
    boost::get<uint32_t>(stepperOptions.at("randomSeed")) = 0U;  // `time(nullptr)` for random seed
    configHolder_t & contactsOptions = boost::get<configHolder_t>(simuOptions.at("contacts"));
    boost::get<std::string>(contactsOptions.at("model")) = std::string("spring_damper");
    boost::get<float64_t>(contactsOptions.at("stiffness")) = 1.0e6;
    boost::get<float64_t>(contactsOptions.at("damping")) = 2000.0;
    boost::get<float64_t>(contactsOptions.at("friction")) = 5.0;
    boost::get<float64_t>(contactsOptions.at("transitionEps")) = 0.001;
    boost::get<float64_t>(contactsOptions.at("transitionVelocity")) = 0.01;
    engine->setOptions(simuOptions);
    engine->initialize(robot, controller, callback);

    timer.toc();

    // =====================================================================
    // ======================= Run the simulation ==========================
    // =====================================================================

    // Prepare options
    Eigen::VectorXd q0 = Eigen::VectorXd::Zero(2);
    q0[1] = 0.1;
    Eigen::VectorXd v0 = Eigen::VectorXd::Zero(2);
    const float64_t tf = 3.0;

    // Run simulation
    timer.tic();
    engine->simulate(tf, q0, v0);
    timer.toc();
    std::cout << "Simulation time: " << (timer.dt * 1.0e3) << "ms" << std::endl;

    // Write the log file
    std::vector<std::string> fieldnames;
    std::shared_ptr<const logData_t> logData;
    engine->getLog(logData);
    std::cout << logData->timestamps.size() << " log points" << std::endl;
    std::cout << engine->getStepperState().iter << " internal integration steps" << std::endl;
    timer.tic();
    engine->writeLog((outputDirPath / "log.data").string(), "binary");
    timer.toc();
    std::cout << "Write log binary: " << (timer.dt * 1.0e3) << "ms" << std::endl;
    timer.tic();
    engine->writeLog((outputDirPath / "log.hdf5").string(), "hdf5");
    timer.toc();
    std::cout << "Write log HDF5: " << (timer.dt * 1.0e3) << "ms" << std::endl;

    return 0;
}
