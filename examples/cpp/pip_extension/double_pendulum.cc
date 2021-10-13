// A simple test case: simulation of a double inverted pendulum.
// There are no contact forces.
// This simulation checks the overall simulator sanity (i.e. conservation of energy) and genericity (working
// with something that is not an exoskeleton).

#include <iostream>

#include "jiminy/core/engine/Engine.h"
#include "jiminy/core/robot/BasicMotors.h"
#include "jiminy/core/control/ControllerFunctor.h"
#include "jiminy/core/io/FileDevice.h"
#include "jiminy/core/utilities/Helpers.h"
#include "jiminy/core/Types.h"


// `filesystem` is experimental for gcc<=8.1
#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = std::experimental::filesystem;
}
#endif


using namespace jiminy;


void computeCommand(float64_t        const & t,
                    vectorN_t        const & q,
                    vectorN_t        const & v,
                    sensorsDataMap_t const & sensorsData,
                    vectorN_t              & command)
{
    // No controller: energy should be preserved
}

void internalDynamics(float64_t        const & t,
                      vectorN_t        const & q,
                      vectorN_t        const & v,
                      sensorsDataMap_t const & sensorsData,
                      vectorN_t              & uCustom)
{
    // Empty on purpose
}

bool_t callback(float64_t const & t,
                vectorN_t const & q,
                vectorN_t const & v)
{
    return true;
}

int main(int argc, char_t * argv[])
{
    // =====================================================================
    // ==================== Extract the user paramaters ====================
    // =====================================================================

    // Set URDF and log output.
    std::filesystem::path const filePath(__FILE__);
    auto const urdfPath = filePath.parent_path() / "double_pendulum.urdf";
    auto const outputDirPath = std::filesystem::temp_directory_path();

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
    boost::get<bool_t>(boost::get<configHolder_t>(modelOptions.at("joints")).at("positionLimitFromUrdf")) = true;
    boost::get<bool_t>(boost::get<configHolder_t>(modelOptions.at("joints")).at("velocityLimitFromUrdf")) = true;
    robot->setModelOptions(modelOptions);
    robot->initialize(urdfPath.string(), false, {});
    for (std::string const & jointName : motorJointNames)
    {
        auto motor = std::make_shared<SimpleMotor>(jointName);
        robot->attachMotor(motor);
        motor->initialize(jointName);
    }

    // Instantiate the controller
    auto controller = std::make_shared<ControllerFunctor<
        decltype(computeCommand), decltype(internalDynamics)> >(computeCommand, internalDynamics);
    controller->initialize(robot);

    // Instantiate the engine
    auto engine = std::make_shared<Engine>();
    engine->initialize(robot, controller, callback);

    timer.toc();

    // =====================================================================
    // ======================= Run the simulation ==========================
    // =====================================================================

    // Prepare options
    Eigen::VectorXd q0 = Eigen::VectorXd::Zero(2);
    q0[1] = 0.1;
    Eigen::VectorXd v0 = Eigen::VectorXd::Zero(2);
    float64_t const tf = 3.0;

    // Run simulation
    timer.tic();
    engine->simulate(tf, q0, v0);
    timer.toc();
    std::cout << "Simulation time: " << (timer.dt * 1.0e3) << "ms" << std::endl;

    // Write the log file
    std::vector<std::string> header;
    matrixN_t log;
    engine->getLogData(header, log);
    std::cout << log.rows() << " log points" << std::endl;
    std::cout << engine->getStepperState().iter << " internal integration steps" << std::endl;
    timer.tic();
    engine->writeLog((outputDirPath / "log.data").string(), "binary");
    timer.toc();
    std::cout << "Write log binary: " << (timer.dt * 1.0e3) << "ms" << std::endl;
    timer.tic();
    engine->writeLog((outputDirPath / "log.csv").string(), "csv");
    timer.toc();
    std::cout << "Write log CSV: " << (timer.dt * 1.0e3) << "ms" << std::endl;
    timer.tic();
    engine->writeLog((outputDirPath / "log.hdf5").string(), "hdf5");
    timer.toc();
    std::cout << "Write log HDF5: " << (timer.dt * 1.0e3) << "ms" << std::endl;

    return 0;
}
