#include <fstream>
#include <exception>

#include "jiminy/core/constraints/abstract_constraint.h"
#include "jiminy/core/constraints/joint_constraint.h"
#include "jiminy/core/hardware/abstract_motor.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/control/abstract_controller.h"
#include "jiminy/core/control/controller_functor.h"
#include "jiminy/core/io/file_device.h"
#include "jiminy/core/io/serialization.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/utilities/pinocchio.h"
#include "jiminy/core/utilities/json.h"

#include "jiminy/core/robot/robot.h"

#include <boost/serialization/nvp.hpp>


namespace jiminy
{
    Robot::Robot(const std::string & name) noexcept :
    name_{name},
    motorSharedStorage_{std::make_shared<MotorSharedStorage>()}
    {
        // Initialize options
        robotOptionsGeneric_ = getDefaultRobotOptions();
        setOptions(getOptions());
    }

    Robot::Robot(const Robot & other) :
    Model()
    {
        *this = other;
    }

    Robot & Robot::operator=(const Robot & other)
    {
        // Serialize the user-specified robot
        std::stringstream sstream;
        {
            stateful_binary_oarchive oa(sstream);
            const Robot * const otherPtr = &other;
            oa << boost::serialization::make_nvp("px", otherPtr);
        }

        // De-serialize the user-specified robot then move assign to this robot
        {
            stateful_binary_iarchive ia(sstream);
            Robot * robotPtr;
            ia >> boost::serialization::make_nvp("px", robotPtr);
            *this = std::move(*robotPtr);
        }

        return *this;
    }

    Robot::~Robot()
    {
        // Detach all the motors and sensors
        detachSensors();
        detachMotors();
    }

    template<typename... Args>
    static void initializeImpl(Robot & robot, Args... args)
    {
        // Make sure that no simulation is already running
        if (robot.getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Robot already locked, probably because a simulation is running. "
                         "Please stop it before calling 'initialize'.");
        }

        // Detach all the motors and sensors
        robot.detachSensors();
        robot.detachMotors();

        /* Delete the current model and generate a new one.
           Note that is also refresh all proxies automatically. */
        robot.Model::initialize(args...);

        // Initialize default controller
        robot.setController({});
    }

    void Robot::initialize(const std::string & urdfPath,
                           bool hasFreeflyer,
                           const std::vector<std::string> & meshPackageDirs,
                           bool loadVisualMeshes)
    {
        initializeImpl(*this, urdfPath, hasFreeflyer, meshPackageDirs, loadVisualMeshes);
    }

    void Robot::initialize(const pinocchio::Model & pinocchioModel,
                           const std::optional<pinocchio::GeometryModel> & collisionModel,
                           const std::optional<pinocchio::GeometryModel> & visualModel)
    {
        initializeImpl(*this, pinocchioModel, collisionModel, visualModel);
    }

    const std::string & Robot::getName() const
    {
        return name_;
    }

    void Robot::reset(const uniform_random_bit_generator_ref<uint32_t> & g)
    {
        // Reset telemetry flag
        isTelemetryConfigured_ = false;

        // Make sure that the robot is initialized
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        // Make sure that the robot is not locked
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Robot already locked, probably because a simulation is running. "
                         "Please stop it before calling 'reset'.");
        }

        // Reset model
        Model::reset(g);

        // Reset motors
        if (!motors_.empty())
        {
            (*motors_.begin())->resetAll();
        }

        // Reset sensors
        for (auto & sensorGroupItem : sensors_)
        {
            if (!sensorGroupItem.second.empty())
            {
                (*sensorGroupItem.second.begin())->resetAll(g());
            }
        }

        // Reset controller
        if (controller_)
        {
            controller_->reset();
        }
    }

    void Robot::configureTelemetry(std::shared_ptr<TelemetryData> telemetryData)
    {
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot is initialized.");
        }

        isTelemetryConfigured_ = false;
        telemetryData_ = telemetryData;
        try
        {
            // Configure hardware telemetry
            const GenericConfig & telemetryOptions =
                boost::get<GenericConfig>(robotOptionsGeneric_.at("telemetry"));
            for (const auto & [sensorType, sensorGroup] : sensors_)
            {
                for (const auto & sensor : sensorGroup)
                {
                    if (boost::get<bool>(telemetryOptions.at(sensorType)))
                    {
                        sensor->configureTelemetry(telemetryData_, name_);
                    }
                }
            }

            // Configure controller telemetry
            controller_->configureTelemetry(telemetryData_, name_);
        }
        catch (...)
        {
            telemetryData_.reset();
            throw;
        }

        isTelemetryConfigured_ = true;
    }

    void Robot::attachMotor(std::shared_ptr<AbstractMotorBase> motor)
    {
        // The robot must be initialized
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        // No simulation must be running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Robot already locked, probably because a simulation is running. "
                         "Please stop it before adding motors.");
        }

        const std::string & motorName = motor->getName();
        auto motorIt = std::find_if(motors_.begin(),
                                    motors_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt != motors_.end())
        {
            JIMINY_THROW(std::logic_error,
                         "Another motor with name '",
                         motorName,
                         "' is already attached.");
        }

        // Define robot notification method, responsible for updating the robot if
        // necessary after changing the motor parameters, for example the armature.
        auto notifyRobot = [robot_ = std::weak_ptr<Robot>(shared_from_this())](
                               AbstractMotorBase & motorIn, bool hasChanged) mutable
        {
            // Make sure the robot still exists
            auto robot = robot_.lock();
            if (!robot)
            {
                JIMINY_THROW(std::runtime_error,
                             "Robot has been deleted. Impossible to notify motor update.");
            }

            /* Trigger extended model regeneration.
               * The whole robot must be reset as joint and frame indices would be corrupted.
               * Skip reset if nothing has changed from this motor point of view. This is necessary
                 to prevent infinite recursive reset loop. */
            if (hasChanged)
            {
                robot->reset(std::random_device{});
            }

            // Update rotor inertia
            const std::string & mechanicaljointName = motorIn.getJointName();
            std::string inertiaJointName = mechanicaljointName;
            const Eigen::Index motorVelocityIndex = motorIn.getJointVelocityIndex();
            robot->pinocchioModel_.rotorInertia[motorVelocityIndex] += motorIn.getArmature();
        };
        // Attach the motor
        motor->attach(shared_from_this(), notifyRobot, motorSharedStorage_.get());

        // Add the motor to the registry
        motors_.push_back(motor);
        motorsWeakConstRef_.push_back(motor);

        // Append log telemetry motor command fieldnames
        logCommandFieldnames_.push_back(toString(JOINT_PREFIX_BASE, "Command", motorName));
    }

    void Robot::detachMotor(const std::string & motorName, bool triggerReset)
    {
        // The robot must be initialized
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        // No simulation must be running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Robot already locked, probably because a simulation is running. "
                         "Please stop it before removing motors.");
        }

        // Get motor if it exists
        auto motorIt = std::find_if(motors_.begin(),
                                    motors_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motors_.end())
        {
            JIMINY_THROW(std::logic_error, "No motor with name '", motorName, "' is attached.");
        }

        // Hold the motor alive before removing it
        std::shared_ptr<AbstractMotorBase> motor = *motorIt;

        // Remove the motor from registry
        motorsWeakConstRef_.erase(motorsWeakConstRef_.begin() + motor->getIndex());
        motors_.erase(motorIt);

        // Remove log telemetry motor command fieldname
        logCommandFieldnames_.erase(logCommandFieldnames_.begin() + motor->getIndex());

        // Detach the motor
        motor->detach();

        // Skip reset and refresh if postponed
        if (triggerReset)
        {
            // Early return if the model has been deleted already
            try
            {
                (void)shared_from_this();
            }
            catch (std::bad_weak_ptr &)
            {
                return;
            }

            // Trigger extended model regeneration
            reset(std::random_device{});
        }
    }

    void Robot::detachMotor(const std::string & motorName)
    {
        // Detach motor
        detachMotor(motorName, true);
    }

    template<typename T>
    static std::vector<std::string>
    getHardwareNames(const std::vector<std::shared_ptr<T>> & hardwares)
    {
        std::vector<std::string> hardwareNames;
        hardwareNames.reserve(hardwares.size());
        std::transform(hardwares.begin(),
                       hardwares.end(),
                       std::back_inserter(hardwareNames),
                       [](const std::shared_ptr<T> & hardware) { return hardware->getName(); });
        return hardwareNames;
    }

    void Robot::detachMotors(std::vector<std::string> motorNames)
    {
        // Get the name of all motors currently attache to the robot
        std::vector<std::string> allMotorNames = getHardwareNames(motors_);

        // Remove all sensors if none is specified
        if (motorNames.empty())
        {
            if (!allMotorNames.empty())
            {
                detachMotors(allMotorNames);
            }
            return;
        }

        // Make sure that no motor names are duplicates
        if (checkDuplicates(motorNames))
        {
            JIMINY_THROW(std::invalid_argument, "Duplicated motor names found.");
        }

        // Make sure that every motor name exist
        if (!checkInclusion(allMotorNames, motorNames))
        {
            JIMINY_THROW(std::invalid_argument, "At least one of the motor names does not exist.");
        }

        /* Detach motors one-by-one, and reset proxies at the very end only.
           Beware the motor names must be stored locally because the list
           of motors will be updated in-place. */
        const std::size_t nMotors = motorNames.size();
        for (std::size_t i = 0; i < nMotors; ++i)
        {
            detachMotor(motorNames[i], i == (nMotors - 1));
        }
    }

    void Robot::attachSensor(std::shared_ptr<AbstractSensorBase> sensor)
    {
        // The robot must be initialized
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        // No simulation must be running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Robot already locked, probably because a simulation is running. "
                         "Please stop it before removing motors.");
        }

        // Attached sensors' names must be unique, even if their type is different.
        const std::string & sensorName = sensor->getName();
        const std::string & sensorType = sensor->getType();
        SensorTree::const_iterator sensorGroupIt;
        sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt != sensors_.end())
        {
            auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                         sensorGroupIt->second.end(),
                                         [&sensorName](const auto & elem)
                                         { return (elem->getName() == sensorName); });
            if (sensorIt != sensorGroupIt->second.end())
            {
                JIMINY_THROW(std::invalid_argument,
                             "A sensor with the same type and name already exists.");
            }
        }

        // Create a new sensor data holder if necessary
        GenericConfig & telemetryOptions =
            boost::get<GenericConfig>(robotOptionsGeneric_.at("telemetry"));
        if (sensorGroupIt == sensors_.end())
        {
            sensorSharedStorageMap_.emplace(sensorType, std::make_shared<SensorSharedStorage>());
            telemetryOptions[sensorType] = true;  // Enable telemetry by default
        }

        // Attach the sensor
        sensor->attach(shared_from_this(), sensorSharedStorageMap_[sensorType].get());

        // Create the sensor and add it to its group
        sensors_[sensorType].push_back(sensor);
        sensorsWeakConstRef_[sensorType].push_back(sensor);
    }

    void Robot::detachSensor(const std::string & sensorType, const std::string & sensorName)
    {
        // The robot must be initialized
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        // No simulation must be running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Robot already locked, probably because a simulation is running. "
                         "Please stop it before removing motors.");
        }

        // FIXME: remove explicit conversion to `std::string` when moving to C++20
        auto sensorGroupIt = sensors_.find(std::string{sensorType});
        if (sensorGroupIt == sensors_.end())
        {
            JIMINY_THROW(std::invalid_argument,
                         "None of the attached sensors has type '",
                         sensorType,
                         "'.");
        }

        SensorVector::iterator sensorIt;
        sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                sensorGroupIt->second.end(),
                                [&sensorName](const auto & elem)
                                { return (elem->getName() == sensorName); });
        if (sensorIt == sensorGroupIt->second.end())
        {
            JIMINY_THROW(std::invalid_argument,
                         "None of the attached sensors of type '",
                         sensorType,
                         "' has name '",
                         sensorName,
                         "'.");
        }

        // Hold the sensor alive before removing it
        std::shared_ptr<AbstractSensorBase> sensor = *sensorIt;

        // Remove the sensor from its group
        WeakSensorVector & sensorGroupWeakConstRef = sensorsWeakConstRef_[sensorType];
        sensorGroupWeakConstRef.erase(sensorGroupWeakConstRef.begin() + sensor->getIndex());
        sensorGroupIt->second.erase(sensorIt);

        // Detach the sensor
        sensor->detach();

        // Remove the sensor group if there is no more sensors left
        GenericConfig & telemetryOptions =
            boost::get<GenericConfig>(robotOptionsGeneric_.at("telemetry"));
        if (sensorGroupIt->second.empty())
        {
            sensors_.erase(sensorType);
            sensorsWeakConstRef_.erase(sensorType);
            sensorSharedStorageMap_.erase(sensorType);
            telemetryOptions.erase(sensorType);
        }
    }

    void Robot::detachSensors(const std::string & sensorType)
    {
        if (!sensorType.empty())
        {
            auto sensorGroupIt = sensors_.find(sensorType);
            if (sensorGroupIt == sensors_.end())
            {
                JIMINY_THROW(std::invalid_argument,
                             "None of the attached sensors has type '",
                             sensorType,
                             "'.");
            }

            std::vector<std::string> sensorGroupNames = getHardwareNames(sensorGroupIt->second);
            for (const std::string & sensorName : sensorGroupNames)
            {
                detachSensor(sensorType, sensorName);
            }
        }
        else
        {
            std::vector<std::string> sensorsTypesNames;
            sensorsTypesNames.reserve(sensors_.size());
            std::transform(sensors_.begin(),
                           sensors_.end(),
                           std::back_inserter(sensorsTypesNames),
                           [](const auto & pair) -> std::string { return pair.first; });
            for (const std::string & sensorTypeName : sensorsTypesNames)
            {
                detachSensors(sensorTypeName);
            }
        }
    }

    void Robot::setController(const std::shared_ptr<AbstractController> & controller)
    {
        // The robot must be initialized
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        // No simulation must be running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Robot already locked, probably because a simulation is running. "
                         "Please stop it before removing motors.");
        }

        // Reset controller to default if none was specified
        if (!controller)
        {
            auto noop = [](double /* t */,
                           const Eigen::VectorXd & /* q */,
                           const Eigen::VectorXd & /* v */,
                           const SensorMeasurementTree & /* sensorMeasurements */,
                           Eigen::VectorXd & /* out */)
            {
                // Empty on purpose
            };
            controller_ = std::make_shared<FunctionalController<>>(noop, noop);
            controller_->initialize(shared_from_this());
            return;
        }

        // Unbind the old controller to allow for initialization of the new controller
        controller_.reset();

        try
        {
            // Initialize the new controller for this robot
            controller->initialize(shared_from_this());

            // Set the controller
            controller_ = controller;
        }
        catch (...)
        {
            // Reset controller to default before throwing exception in case of failure
            setController({});
            throw;
        }
    }

    std::shared_ptr<AbstractController> Robot::getController()
    {
        return controller_;
    }

    std::weak_ptr<const AbstractController> Robot::getController() const
    {
        return std::const_pointer_cast<const AbstractController>(controller_);
    }

    void Robot::initializeExtendedModel()
    {
        // Call base implementation
        Model::initializeExtendedModel();

        // Add backlash joint for each motor
        for (const auto & motor : motors_)
        {
            // Extract some proxies
            const std::string & jointName = motor->getJointName();
            std::string backlashName = jointName;
            backlashName += BACKLASH_JOINT_SUFFIX;

            // Check if constraint a joint bounds constraint exist
            const bool hasConstraint =
                constraints_.exist(backlashName, ConstraintNodeType::BOUNDS_JOINTS);

            // Skip adding joint if no backlash
            const double backlash = motor->getBacklash();
            if (backlash < EPS)
            {
                // Remove joint bounds constraint if any
                if (hasConstraint)
                {
                    removeConstraint(backlashName, ConstraintNodeType::BOUNDS_JOINTS);
                }

                continue;
            }

            // Add joint bounds constraint if necessary
            if (!hasConstraint)
            {
                std::shared_ptr<AbstractConstraintBase> constraint =
                    std::make_shared<JointConstraint>(backlashName);
                addConstraint(backlashName, constraint, ConstraintNodeType::BOUNDS_JOINTS);
            }

            // Add backlash joint to the model
            addBacklashJointAfterMechanicalJoint(pinocchioModel_, jointName, backlashName);
            backlashJointNames_.push_back(backlashName);

            // Update position limits in model
            const Eigen::Index positionIndex =
                getJointPositionFirstIndex(pinocchioModel_, backlashName);
            pinocchioModel_.lowerPositionLimit[positionIndex] = -backlash / 2.0;
            pinocchioModel_.upperPositionLimit[positionIndex] = backlash / 2.0;
        }
    }

    std::shared_ptr<AbstractMotorBase> Robot::getMotor(const std::string & motorName)
    {
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        auto motorIt = std::find_if(motors_.begin(),
                                    motors_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motors_.end())
        {
            JIMINY_THROW(
                std::invalid_argument, "None of the attached motors has name '", motorName, "'.");
        }
        return *motorIt;
    }

    std::weak_ptr<const AbstractMotorBase> Robot::getMotor(const std::string & motorName) const
    {
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        auto motorIt = std::find_if(motors_.begin(),
                                    motors_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motors_.end())
        {
            JIMINY_THROW(
                std::invalid_argument, "None of the attached motors has name '", motorName, "'.");
        }
        return std::const_pointer_cast<const AbstractMotorBase>(*motorIt);
    }

    const Robot::MotorVector & Robot::getMotors()
    {
        return motors_;
    }

    const Robot::WeakMotorVector & Robot::getMotors() const
    {
        return motorsWeakConstRef_;
    }

    std::shared_ptr<AbstractSensorBase> Robot::getSensor(const std::string & sensorType,
                                                         const std::string & sensorName)
    {
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        auto sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt == sensors_.end())
        {
            JIMINY_THROW(std::invalid_argument,
                         "None of the attached sensors has type '",
                         sensorType,
                         "'.");
        }

        auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                     sensorGroupIt->second.end(),
                                     [&sensorName](const auto & elem)
                                     { return (elem->getName() == sensorName); });
        if (sensorIt == sensorGroupIt->second.end())
        {
            JIMINY_THROW(std::invalid_argument,
                         "None of the attached sensors of type '",
                         sensorType,
                         "' has name '",
                         sensorName,
                         "'.");
        }

        return *sensorIt;
    }

    std::weak_ptr<const AbstractSensorBase> Robot::getSensor(const std::string & sensorType,
                                                             const std::string & sensorName) const
    {
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        auto sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt == sensors_.end())
        {
            JIMINY_THROW(std::invalid_argument,
                         "None of the attached sensors has type '",
                         sensorType,
                         "'.");
        }

        auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                     sensorGroupIt->second.end(),
                                     [&sensorName](const auto & elem)
                                     { return (elem->getName() == sensorName); });
        if (sensorIt == sensorGroupIt->second.end())
        {
            JIMINY_THROW(std::invalid_argument,
                         "None of the attached sensors of type '",
                         sensorType,
                         "' has name '",
                         sensorName,
                         "'.");
        }

        return std::const_pointer_cast<const AbstractSensorBase>(*sensorIt);
    }

    const Robot::SensorTree & Robot::getSensors()
    {
        return sensors_;
    }

    const Robot::WeakSensorTree & Robot::getSensors() const
    {
        return sensorsWeakConstRef_;
    }

    void Robot::setOptions(const GenericConfig & robotOptions)
    {
        // Make sure that no simulation is running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Robot already locked, probably because a simulation is running. "
                         "Please stop it setting options.");
        }

        // Set base model options
        const GenericConfig & modelOptions = boost::get<GenericConfig>(robotOptions.at("model"));
        setModelOptions(modelOptions);

        // Set motor options
        bool areMotorsOptionsShared{true};
        const GenericConfig & motorsOptions = boost::get<GenericConfig>(robotOptions.at("motors"));
        for (const auto & motor : motors_)
        {
            auto motorOptionsIt = motorsOptions.find(motor->getName());
            if (motorOptionsIt == motorsOptions.end())
            {
                if (areMotorsOptionsShared)
                {
                    motor->setOptionsAll(motorsOptions);
                    break;
                }
            }
            else
            {
                motor->setOptions(boost::get<GenericConfig>(motorOptionsIt->second));
                areMotorsOptionsShared = false;
            }
        }

        // Set sensor options
        const GenericConfig & sensorsOptions =
            boost::get<GenericConfig>(robotOptions.at("sensors"));
        for (const auto & [sensorType, sensorGroup] : sensors_)
        {
            // FIXME: remove explicit conversion to `std::string` when moving to C++20
            auto sensorGroupOptionsIt = sensorsOptions.find(std::string{sensorType});
            if (sensorGroupOptionsIt != sensorsOptions.end())
            {
                const GenericConfig & sensorGroupOptions =
                    boost::get<GenericConfig>(sensorGroupOptionsIt->second);
                for (const auto & sensor : sensorGroup)
                {
                    const std::string & sensorName = sensor->getName();
                    auto sensorOptionsIt = sensorGroupOptions.find(sensorName);
                    if (sensorOptionsIt != sensorGroupOptions.end())
                    {
                        sensor->setOptions(boost::get<GenericConfig>(sensorOptionsIt->second));
                    }
                    else
                    {
                        JIMINY_THROW(std::invalid_argument,
                                     "None of the attached sensors of type '",
                                     sensorType,
                                     "' has name '",
                                     sensorName,
                                     "'.");
                    }
                }
            }
            else
            {
                JIMINY_THROW(std::invalid_argument,
                             "None of the attached sensors has type '",
                             sensorType,
                             "'.");
            }
        }

        // Set controller options if any
        if (controller_)
        {
            const GenericConfig & controllerOptions =
                boost::get<GenericConfig>(robotOptions.at("controller"));
            controller_->setOptions(controllerOptions);
        }

        // Update inherited polymorphic accessor
        deepUpdate(robotOptionsGeneric_, robotOptions);
    }

    const GenericConfig & Robot::getOptions() const noexcept
    {
        /* Return options without refreshing all options if and only if the robot has not been
           unlock since the last time they were considered valid. */
        if (areRobotOptionsRefreshed_ && getIsLocked())
        {
            return robotOptionsGeneric_;
        }

        // Refresh model options
        robotOptionsGeneric_["model"] = getModelOptions();

        // Refresh motors options
        GenericConfig & motorsOptions = boost::get<GenericConfig>(robotOptionsGeneric_["motors"]);
        motorsOptions.clear();
        for (const auto & motor : motors_)
        {
            motorsOptions[motor->getName()] = motor->getOptions();
        }

        // Refresh sensor options
        GenericConfig & sensorsOptions =
            boost::get<GenericConfig>(robotOptionsGeneric_["sensors"]);
        sensorsOptions.clear();
        for (const auto & [sensorType, sensorGroup] : sensors_)
        {
            GenericConfig sensorGroupOptions;
            for (const auto & sensor : sensorGroup)
            {
                sensorGroupOptions[sensor->getName()] = sensor->getOptions();
            }
            sensorsOptions[sensorType] = sensorGroupOptions;
        }

        // Refresh controller options
        GenericConfig & controllerOptions =
            boost::get<GenericConfig>(robotOptionsGeneric_["controller"]);
        controllerOptions.clear();
        if (controller_)
        {
            controllerOptions = controller_->getOptions();
        }

        // Options are now considered "valid"
        areRobotOptionsRefreshed_ = true;

        return robotOptionsGeneric_;
    }

    void Robot::setModelOptions(const GenericConfig & modelOptions)
    {
        Model::setOptions(modelOptions);
    }
    const GenericConfig & Robot::getModelOptions() const noexcept
    {
        return Model::getOptions();
    }

    bool Robot::getIsTelemetryConfigured() const
    {
        return isTelemetryConfigured_;
    }

    void Robot::computeMotorEfforts(double t,
                                    const Eigen::VectorXd & q,
                                    const Eigen::VectorXd & v,
                                    const Eigen::VectorXd & a,
                                    const Eigen::VectorXd & command)
    {
        if (!motors_.empty())
        {
            (*motors_.begin())->computeEffortAll(t, q, v, a, command);
        }
    }

    const Eigen::VectorXd & Robot::getMotorEfforts() const
    {
        static const Eigen::VectorXd motorsEffortsEmpty;

        if (!motors_.empty())
        {
            return (*motors_.begin())->getAll();
        }

        return motorsEffortsEmpty;
    }

    double Robot::getMotorEffort(const std::string & motorName) const
    {
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        auto motorIt = std::find_if(motors_.begin(),
                                    motors_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motors_.end())
        {
            JIMINY_THROW(
                std::logic_error, "No motor with name '", motorName, "' attached to the robot.");
        }

        return (*motorIt)->get();
    }

    void Robot::computeSensorMeasurements(double t,
                                          const Eigen::VectorXd & q,
                                          const Eigen::VectorXd & v,
                                          const Eigen::VectorXd & a,
                                          const Eigen::VectorXd & uMotor,
                                          const ForceVector & fExternal)
    {
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        for (const auto & sensorGroupItem : sensors_)
        {
            if (!sensorGroupItem.second.empty())
            {
                (*sensorGroupItem.second.begin())->setAll(t, q, v, a, uMotor, fExternal);
            }
        }
    }

    SensorMeasurementTree Robot::getSensorMeasurements() const
    {
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        SensorMeasurementTree data;
        auto sensorGroupIt = sensors_.cbegin();
        auto sensorsSharedIt = sensorSharedStorageMap_.cbegin();
        for (; sensorGroupIt != sensors_.cend(); ++sensorGroupIt, ++sensorsSharedIt)
        {
            auto & [sensorType, sensorGroup] = *sensorGroupIt;
            SensorMeasurementTree::mapped_type sensorsMeasurementsStack(
                &sensorsSharedIt->second->measurements_);
            for (const auto & sensor : sensorGroup)
            {
                sensorsMeasurementsStack.insert(
                    {sensor->getName(), sensor->getIndex(), sensor->get()});
            }
            data.emplace(sensorType, std::move(sensorsMeasurementsStack));
        }
        return data;
    }

    Eigen::Ref<const Eigen::VectorXd> Robot::getSensorMeasurement(
        const std::string & sensorType, const std::string & sensorName) const
    {
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        auto sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt == sensors_.end())
        {
            JIMINY_THROW(
                std::logic_error, "No sensor of type '", sensorType, "' attached to the robot.");
        }

        auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                     sensorGroupIt->second.end(),
                                     [&sensorName](const auto & elem)
                                     { return (elem->getName() == sensorName); });
        if (sensorIt == sensorGroupIt->second.end())
        {
            JIMINY_THROW(std::logic_error,
                         "No sensor of type '",
                         sensorType,
                         "' with name '",
                         sensorName,
                         "' attached to the robot.");
        }

        return (*sensorIt)->get();
    }

    void Robot::updateTelemetry()
    {
        // Update hardware telemetry
        for (const auto & sensorGroupItem : sensors_)
        {
            if (!sensorGroupItem.second.empty())
            {
                (*sensorGroupItem.second.begin())->updateTelemetryAll();
            }
        }

        // Update controller telemetry
        controller_->updateTelemetry();
    }

    const std::vector<std::string> & Robot::getLogCommandFieldnames() const
    {
        return logCommandFieldnames_;
    }

    std::unique_ptr<LockGuardLocal> Robot::getLock()
    {
        // Make sure that the robot is not already locked
        if (mutexLocal_->isLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Robot already locked. Please release it first prior requesting lock.");
        }

        // Make sure that the options are not already considered valid as it was impossible to
        // guarantee it before locking the robot.
        areRobotOptionsRefreshed_ = false;

        // Return lock
        return std::make_unique<LockGuardLocal>(*mutexLocal_);
    }

    bool Robot::getIsLocked() const
    {
        return mutexLocal_->isLocked();
    }

    Eigen::Index Robot::nmotors() const
    {
        return nmotors_;
    }
}
