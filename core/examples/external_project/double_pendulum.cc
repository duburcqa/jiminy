/// \brief A simple test case: simulation of a double inverted pendulum without contact forces.
///
/// \details This simulation checks the overall simulator sanity (i.e. conservation of energy) and
///          genericity (supporting systems that are not legged robots).

#include <iostream>
#include <filesystem>

#include "jiminy/core/fwd.h"
#include "jiminy/core/hardware/fwd.h"
#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/io/file_device.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/hardware/basic_sensors.h"
#include "jiminy/core/hardware/basic_motors.h"
#include "jiminy/core/control/controller_functor.h"
#include "jiminy/core/engine/engine.h"


using namespace jiminy;


void computeCommand(double t,
                    const Eigen::VectorXd & q,
                    const Eigen::VectorXd & v,
                    const SensorMeasurementTree & sensorMeasurements,
                    Eigen::VectorXd & command)
{
    // No controller: energy should be preserved
}

void internalDynamics(double t,
                      const Eigen::VectorXd & q,
                      const Eigen::VectorXd & v,
                      const SensorMeasurementTree & sensorMeasurements,
                      Eigen::VectorXd & uCustom)
{
}

bool callback()
{
    return true;
}

int main(int argc, char * argv[])
{
    // =====================================================================
    // ==================== Extract the user paramaters ====================
    // =====================================================================

    // Set URDF and log output.
    const std::filesystem::path filePath(__FILE__);
    const auto urdfPath = filePath.parent_path() / "double_pendulum.urdf";
    const auto outputDirPath = std::filesystem::temp_directory_path();
    std::cout << "Output directory: " << outputDirPath << std::endl;

    // =====================================================================
    // ============ Instantiate and configure the simulation ===============
    // =====================================================================

    // Instantiate timer
    Timer timer;

    // Instantiate and configuration the robot
    auto robot = std::make_shared<Robot>();
    GenericConfig modelOptions = robot->getModelOptions();
    GenericConfig & jointsOptions = boost::get<GenericConfig>(modelOptions.at("joints"));
    boost::get<bool>(jointsOptions.at("positionLimitFromUrdf")) = true;
    robot->setModelOptions(modelOptions);
    robot->initialize(urdfPath.string(), false, {});

    // Attach motor and encoder to the robot
    auto motor = std::make_shared<SimpleMotor>("motor");
    robot->attachMotor(motor);
    motor->initialize("SecondPendulumJoint");
    GenericConfig motorOptions = motor->getOptions();
    boost::get<bool>(motorOptions.at("velocityLimitFromUrdf")) = true;
    motor->setOptions(motorOptions);
    auto sensor = std::make_shared<EncoderSensor>("encoder");
    robot->attachSensor(sensor);
    sensor->initialize("SecondPendulumJoint");

    // Print encoder sensor index
    std::cout << "Encoder sensor index: " << sensor->getIndex() << std::endl;

    // Instantiate the controller
    auto controller = std::make_shared<FunctionalController<>>(computeCommand, internalDynamics);
    robot->setController(controller);

    // Instantiate the engine
    Engine engine{};
    engine.addRobot(robot);
    std::cout << "Initialization: " << timer.toc<std::milli>() << "ms" << std::endl;

    // =====================================================================
    // ======================= Run the simulation ==========================
    // =====================================================================

    // Prepare options
    Eigen::VectorXd q0 = Eigen::VectorXd::Zero(2);
    q0[1] = 0.1;
    Eigen::VectorXd v0 = Eigen::VectorXd::Zero(2);
    const double tf = 3.0;

    // Run simulation
    timer.tic();
    engine.simulate(tf, q0, v0, std::nullopt, false, callback);
    std::cout << "Simulation: " << timer.toc<std::milli>() << "ms" << std::endl;

    // Print final encoder data for debug
    std::cout << "Final encoder data: "
              << controller->sensorMeasurements_[EncoderSensor::type_].getAll().transpose()
              << std::endl;

    // Write the log file
    std::vector<std::string> fieldnames;
    std::shared_ptr<const LogData> logDataPtr = engine.getLog();
    std::cout << logDataPtr->times.size() << " log points" << std::endl;
    std::cout << engine.getStepperState().iter << " internal integration steps" << std::endl;
    timer.tic();
    engine.writeLog((outputDirPath / "log.data").string(), "binary");
    std::cout << "Write log binary: " << timer.toc<std::milli>() << "ms" << std::endl;
    engine.writeLog((outputDirPath / "log.hdf5").string(), "hdf5");
    std::cout << "Write log HDF5: " << timer.toc<std::milli>() << "ms" << std::endl;

    return 0;
}
