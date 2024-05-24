#include <numeric>

#include "jiminy/core/robot/robot.h"
#include "jiminy/core/utilities/random.h"


namespace jiminy
{
    inline constexpr uint8_t DELAY_MIN_BUFFER_RESERVE{20U};
    inline constexpr uint8_t DELAY_MAX_BUFFER_EXCEED{100U};

    // ========================== AbstractSensorBase ==============================

    template<typename DerivedType>
    void AbstractSensorBase::set(const Eigen::MatrixBase<DerivedType> & value)
    {
        if (!isAttached_)
        {
            JIMINY_THROW(bad_control_flow, "Sensor not attached to any robot.");
        }

        auto robot = robot_.lock();
        if (!robot || robot->getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Robot is locked, probably because a simulation is running. "
                         "Please stop it before setting sensor value manually.");
        }

        get() = value;
    }

    // ========================== AbstractSensorTpl ===============================

    template<typename T>
    AbstractSensorTpl<T>::~AbstractSensorTpl()
    {
        // Detach the sensor before deleting it if necessary
        if (isAttached_)
        {
            detach();
        }
    }

    template<typename T>
    void AbstractSensorTpl<T>::attach(std::weak_ptr<const Robot> robot,
                                      SensorSharedStorage * sharedStorage)
    {
        // Make sure the sensor is not already attached
        if (isAttached_)
        {
            JIMINY_THROW(
                bad_control_flow,
                "Sensor already attached to a robot. Please 'detach' method before attaching it.");
        }

        // Make sure the robot still exists
        if (robot.expired())
        {
            JIMINY_THROW(bad_control_flow, "Robot pointer expired or unset.");
        }

        // Copy references to the robot and shared data
        robot_ = robot;
        sharedStorage_ = sharedStorage;

        // Define the sensor index
        sensorIndex_ = sharedStorage_->num_;

        // Make sure the shared data buffers are properly pre-allocated
        if (sharedStorage_->times_.empty())
        {
            sharedStorage_->times_.assign(1, 0.0);
            sharedStorage_->data_.resize(1);
        }

        // Add a column for the sensor to the shared data buffers
        for (Eigen::MatrixXd & data : sharedStorage_->data_)
        {
            data.conservativeResize(getSize(), sharedStorage_->num_ + 1);
            data.rightCols<1>().setZero();
        }
        sharedStorage_->measurements_.conservativeResize(getSize(), sharedStorage_->num_ + 1);
        sharedStorage_->measurements_.rightCols<1>().setZero();

        // Add the sensor to the shared memory
        sharedStorage_->sensors_.push_back(this);
        ++sharedStorage_->num_;

        // Update the flag
        isAttached_ = true;
    }

    template<typename T>
    void AbstractSensorTpl<T>::detach()
    {
        // Delete the part of the shared memory associated with the sensor

        if (!isAttached_)
        {
            JIMINY_THROW(bad_control_flow, "Sensor not attached to any robot.");
        }

        // Remove associated col in the shared data buffers
        if (sensorIndex_ < sharedStorage_->num_ - 1)
        {
            const std::size_t sensorShift = sharedStorage_->num_ - sensorIndex_ - 1;
            for (Eigen::MatrixXd & data : sharedStorage_->data_)
            {
                /* Aliasing is NOT an issue when shifting left/up the columns/rows of matrices.
                   This holds true regardless if the matrix is row- and column-major. Yet, it is
                   necessary to make an intermediary copy when shifting right or down! */
                data.middleCols(sensorIndex_, sensorShift) = data.rightCols(sensorShift);
            }
            sharedStorage_->measurements_.middleCols(sensorIndex_, sensorShift) =
                sharedStorage_->measurements_.rightCols(sensorShift);
        }
        for (Eigen::MatrixXd & data : sharedStorage_->data_)
        {
            data.conservativeResize(Eigen::NoChange, sharedStorage_->num_ - 1);
        }
        sharedStorage_->measurements_.conservativeResize(Eigen::NoChange,
                                                         sharedStorage_->num_ - 1);

        // Shift the sensor indices
        for (std::size_t i = sensorIndex_ + 1; i < sharedStorage_->num_; ++i)
        {
            AbstractSensorTpl<T> * sensor =
                static_cast<AbstractSensorTpl<T> *>(sharedStorage_->sensors_[i]);
            --sensor->sensorIndex_;
        }

        // Remove the sensor from the shared memory
        sharedStorage_->sensors_.erase(sharedStorage_->sensors_.begin() + sensorIndex_);
        --sharedStorage_->num_;

        // Clear the references to the robot and shared data
        robot_.reset();
        sharedStorage_ = nullptr;

        // Unset the sensor index
        sensorIndex_ = -1;

        // Update the flag
        isAttached_ = false;
    }

    template<typename T>
    void AbstractSensorTpl<T>::resetAll(uint32_t seed)
    {
        // Make sure the sensor is attached to a robot
        if (!isAttached_)
        {
            JIMINY_THROW(bad_control_flow, "Sensor not attached to any robot.");
        }

        // Make sure all the sensors are attached to a robot and initialized
        for (AbstractSensorBase * sensor : sharedStorage_->sensors_)
        {
            if (!sensor->isAttached_)
            {
                JIMINY_THROW(bad_control_flow,
                             "Sensor '",
                             sensor->name_,
                             "' of type '",
                             type_,
                             "' not attached to any robot.");
            }
            if (!sensor->isInitialized_)
            {
                JIMINY_THROW(bad_control_flow,
                             "Sensor '",
                             sensor->name_,
                             "' of type '",
                             type_,
                             "' not initialized.");
            }
        }

        // Make sure the robot still exists
        if (robot_.expired())
        {
            JIMINY_THROW(bad_control_flow, "Robot has been deleted. Impossible to reset sensors.");
        }

        // Make sure that no simulation is already running
        auto robot = robot_.lock();
        if (robot && robot->getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Robot already locked, probably because a simulation is running. "
                         "Please stop it before resetting sensors.");
        }

        // Clear the shared data buffers
        sharedStorage_->times_.assign(1, 0.0);
        sharedStorage_->data_.resize(1);
        sharedStorage_->data_[0].setZero();
        sharedStorage_->measurements_.setZero();

        // Compute max delay
        sharedStorage_->delayMax_ = std::accumulate(sharedStorage_->sensors_.begin(),
                                                    sharedStorage_->sensors_.end(),
                                                    0.0,
                                                    [](double value, AbstractSensorBase * sensor)
                                                    {
                                                        const double delay =
                                                            sensor->baseSensorOptions_->delay +
                                                            sensor->baseSensorOptions_->jitter;
                                                        return std::max(delay, value);
                                                    });

        // Generate high-entropy seed sequence from the provided initial seed
        std::seed_seq seq{seed};
        std::vector<std::uint32_t> seeds(sharedStorage_->num_);
        seq.generate(seeds.begin(), seeds.end());

        // Reset sensor-specific state
        std::vector<AbstractSensorBase *>::iterator sensorIt = sharedStorage_->sensors_.begin();
        std::vector<std::uint32_t>::const_iterator seedIt = seeds.cbegin();
        for (; sensorIt != sharedStorage_->sensors_.end(); ++sensorIt, ++seedIt)
        {
            AbstractSensorBase & sensor = *(*sensorIt);

            // Reset the internal random number generator
            sensor.generator_.seed(*seedIt);

            // Refresh proxies that are robot-dependent
            sensor.refreshProxies();

            // Reset the telemetry state
            sensor.isTelemetryConfigured_ = false;
        }
    }

    template<typename T>
    void AbstractSensorTpl<T>::setOptionsAll(const GenericConfig & sensorOptions)
    {
        if (!isAttached_)
        {
            JIMINY_THROW(bad_control_flow, "Sensor not attached to any robot.");
        }

        for (AbstractSensorBase * sensor : sharedStorage_->sensors_)
        {
            sensor->setOptions(sensorOptions);
        }
    }

    template<typename T>
    std::size_t AbstractSensorTpl<T>::getIndex() const
    {
        return sensorIndex_;
    }

    template<typename T>
    const std::string & AbstractSensorTpl<T>::getType() const
    {
        return type_;
    }

    template<typename T>
    const std::vector<std::string> & AbstractSensorTpl<T>::getFieldnames() const
    {
        return fieldnames_;
    }

    template<typename T>
    std::size_t AbstractSensorTpl<T>::getSize() const
    {
        return fieldnames_.size();
    }

    template<typename T>
    std::string AbstractSensorTpl<T>::getTelemetryName() const
    {
        return addCircumfix(name_, getType(), {}, TELEMETRY_FIELDNAME_DELIMITER);
    }

    template<typename T>
    Eigen::Ref<const Eigen::VectorXd> AbstractSensorTpl<T>::get() const
    {
        static Eigen::VectorXd dataDummy = Eigen::VectorXd::Zero(fieldnames_.size());
        if (isAttached_)
        {
            return sharedStorage_->measurements_.col(sensorIndex_);
        }
        return dataDummy;
    }

    template<typename T>
    inline Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, 1, true> AbstractSensorTpl<T>::get()
    {
        // No guard, since this method is not public
        return sharedStorage_->measurements_.col(sensorIndex_);
    }

    template<typename T>
    inline Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, 1, true> AbstractSensorTpl<T>::data()
    {
        // No guard, since this method is not public
        return sharedStorage_->data_.back().col(sensorIndex_);
    }

    template<typename T>
    void AbstractSensorTpl<T>::interpolateData()
    {
        // Make sure that data is available
        if (sharedStorage_->times_.empty())
        {
            throw std::logic_error("No data to interpolate.");
        }

        // Sample the delay uniformly
        const double delay =
            baseSensorOptions_->delay +
            uniform(generator_, 0.0F, static_cast<float>(baseSensorOptions_->jitter));

        // Get time at which to fetch sensor data
        double timeDesired = sharedStorage_->times_.back() - delay;

        // Floating-point comparison is every sensitive to rounding errors. This is an issue when
        // the sensor delay exactly matches the sensor update period and Zero-Order Hold (ZOH)
        // interpolation is being used, as it would translate in a seemingly noisy sensor signal.
        // To prevent it, the desired time is slightly shifted before calling bisect. This promotes
        // picking an index that is always on the same side by introducing bias in the comparison.
        if (baseSensorOptions_->delayInterpolationOrder == 0)
        {
            timeDesired += STEPPER_MIN_TIMESTEP;
        }

        /* Determine the position of the closest right element.
           Bisection method can be used since times are sorted. */
        auto bisectLeft = [&]() -> std::ptrdiff_t
        {
            std::ptrdiff_t left = 0;
            std::ptrdiff_t right = sharedStorage_->times_.size() - 1;
            std::ptrdiff_t mid = 0;

            if (timeDesired >= sharedStorage_->times_.back())
            {
                return right;
            }
            else if (timeDesired < sharedStorage_->times_.front())
            {
                return -1;
            }

            while (left < right)
            {
                mid = (left + right) / 2;
                if (timeDesired < sharedStorage_->times_[mid])
                {
                    right = mid;
                }
                else if (timeDesired > sharedStorage_->times_[mid])
                {
                    left = mid + 1;
                }
                else
                {
                    return mid;
                }
            }

            if (timeDesired < sharedStorage_->times_[mid])
            {
                return mid - 1;
            }
            else
            {
                return mid;
            }
        };

        const int64_t idxLeft = bisectLeft();
        if (timeDesired >= 0.0 &&
            idxLeft + 1 < static_cast<int64_t>(sharedStorage_->times_.size()))
        {
            if (idxLeft < 0)
            {
                JIMINY_THROW(std::runtime_error, "No data old enough is available.");
            }
            else if (baseSensorOptions_->delayInterpolationOrder == 0)
            {
                get() = sharedStorage_->data_[idxLeft].col(sensorIndex_);
            }
            else if (baseSensorOptions_->delayInterpolationOrder == 1)
            {
                // FIXME: Linear interpolation is not valid on Lie algebra
                const double ratio =
                    (timeDesired - sharedStorage_->times_[idxLeft]) /
                    (sharedStorage_->times_[idxLeft + 1] - sharedStorage_->times_[idxLeft]);
                auto dataNext = sharedStorage_->data_[idxLeft + 1].col(sensorIndex_);
                auto dataPrev = sharedStorage_->data_[idxLeft].col(sensorIndex_);
                get() = dataPrev + ratio * (dataNext - dataPrev);
            }
            else
            {
                JIMINY_THROW(not_implemented_error,
                             "`delayInterpolationOrder` must be either 0 or 1.");
            }
        }
        else
        {
            if (baseSensorOptions_->delay > EPS || baseSensorOptions_->jitter > EPS)
            {
                // Return the oldest value since the buffer is not fully initialized yet
                auto it = std::find_if(sharedStorage_->times_.begin(),
                                       sharedStorage_->times_.end(),
                                       [](double t) -> bool { return t > 0; });
                if (it != sharedStorage_->times_.end())
                {
                    std::ptrdiff_t index = std::distance(sharedStorage_->times_.begin(), it);
                    index = std::max(std::ptrdiff_t(0), index - 1);
                    get() = sharedStorage_->data_[index].col(sensorIndex_);
                }
                else
                {
                    get() = sharedStorage_->data_.back().col(sensorIndex_);
                }
            }
            else
            {
                // Return the most recent value available
                get() = sharedStorage_->data_.back().col(sensorIndex_);
            }
        }
    }

    template<typename T>
    void AbstractSensorTpl<T>::measureDataAll()
    {
        for (AbstractSensorBase * sensor : sharedStorage_->sensors_)
        {
            // Compute the real value at current time, namely taking into account the sensor delay
            sensor->interpolateData();

            // Skew the data with white noise and bias
            sensor->measureData();
        }
    }

    template<typename T>
    void AbstractSensorTpl<T>::setAll(double t,
                                      const Eigen::VectorXd & q,
                                      const Eigen::VectorXd & v,
                                      const Eigen::VectorXd & a,
                                      const Eigen::VectorXd & uMotor,
                                      const ForceVector & fExternal)
    {
        if (!isAttached_)
        {
            JIMINY_THROW(bad_control_flow, "Sensor not attached to any robot.");
        }

        /* Make sure at least the requested delay plus the maximum time step is available to handle
           the case where the solver goes back in time. Even though it makes the buffer much larger
           than necessary as the actual maximum step is given by `engineOptions_->stepper.dtMax`,
           it is not a big deal since `rotate`, `pop_front`, `push_back` have O(1) complexity. */
        const double timeMin = t - sharedStorage_->delayMax_ - SIMULATION_MAX_TIMESTEP;

        // Internal buffer memory management
        if (t + EPS > sharedStorage_->times_.back())
        {
            const std::size_t bufferSize = sharedStorage_->times_.size();
            if (timeMin > sharedStorage_->times_.front())
            {
                // Remove some unecessary extra elements if appropriate
                if (bufferSize > 1U + DELAY_MAX_BUFFER_EXCEED &&
                    timeMin > sharedStorage_->times_[DELAY_MAX_BUFFER_EXCEED])
                {
                    sharedStorage_->times_.erase_begin(DELAY_MAX_BUFFER_EXCEED);
                    sharedStorage_->data_.erase_begin(DELAY_MAX_BUFFER_EXCEED);
                    sharedStorage_->times_.rset_capacity(bufferSize + DELAY_MIN_BUFFER_RESERVE);
                    sharedStorage_->data_.rset_capacity(bufferSize + DELAY_MIN_BUFFER_RESERVE);
                }

                // Rotate the internal buffer
                sharedStorage_->times_.rotate(sharedStorage_->times_.begin() + 1U);
                sharedStorage_->data_.rotate(sharedStorage_->data_.begin() + 1U);
            }
            else
            {
                // Increase capacity if required
                if (sharedStorage_->times_.full())
                {
                    sharedStorage_->times_.rset_capacity(bufferSize + DELAY_MIN_BUFFER_RESERVE);
                    sharedStorage_->data_.rset_capacity(bufferSize + DELAY_MIN_BUFFER_RESERVE);
                }

                /* Push back new buffer.

                   Note that it is a copy of the last value. This is important for `data()` to
                   always provide the last true value instead of some initialized memory. The
                   previous value is used for the quaternion of IMU sensors to choice the right
                   value that ensures its continuity over time amond to two possible choices. */
                sharedStorage_->times_.push_back(INF);
                sharedStorage_->data_.push_back(sharedStorage_->data_.back());
            }
        }
        else
        {
            /* Remove the extra last elements if for some reason the solver went back in time.
               It happens when integration fails for ode solvers relying on try_step mechanism. */
            while (t + EPS < sharedStorage_->times_.back() && sharedStorage_->times_.size() > 1)
            {
                sharedStorage_->times_.pop_back();
                sharedStorage_->data_.pop_back();
            }
        }
        sharedStorage_->times_.back() = t;

        // Update the last real data buffer
        for (AbstractSensorBase * sensor : sharedStorage_->sensors_)
        {
            sensor->set(t, q, v, a, uMotor, fExternal);
        }

        // Compute the measurement data
        measureDataAll();
    }

    template<typename T>
    void AbstractSensorTpl<T>::updateTelemetryAll()
    {
        for (AbstractSensorBase * sensor : sharedStorage_->sensors_)
        {
            sensor->updateTelemetry();
        }
    }
}
