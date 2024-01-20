#include <numeric>

#include "jiminy/core/robot/robot.h"
#include "jiminy/core/utilities/random.h"


namespace jiminy
{
    inline constexpr uint8_t DELAY_MIN_BUFFER_RESERVE{20U};
    inline constexpr uint8_t DELAY_MAX_BUFFER_EXCEED{100U};

    // ========================== AbstractSensorBase ==============================

    template<typename DerivedType>
    hresult_t AbstractSensorBase::set(const Eigen::MatrixBase<DerivedType> & value)
    {
        if (!isAttached_)
        {
            PRINT_ERROR("Sensor not attached to any robot.");
            return hresult_t::ERROR_GENERIC;
        }

        auto robot = robot_.lock();
        if (!robot || robot->getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before setting sensor value manually.");
            return hresult_t::ERROR_GENERIC;
        }

        get() = value;
        return hresult_t::SUCCESS;
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
    hresult_t AbstractSensorTpl<T>::attach(std::weak_ptr<const Robot> robot,
                                           SensorSharedDataHolder_t * sharedHolder)
    {
        // Make sure the sensor is not already attached
        if (isAttached_)
        {
            PRINT_ERROR(
                "Sensor already attached to a robot. Please 'detach' method before attaching it.");
            return hresult_t::ERROR_GENERIC;
        }

        // Make sure the robot still exists
        if (robot.expired())
        {
            PRINT_ERROR("Robot pointer expired or unset.");
            return hresult_t::ERROR_GENERIC;
        }

        // Copy references to the robot and shared data
        robot_ = robot;
        sharedHolder_ = sharedHolder;

        // Define the sensor index
        sensorIdx_ = sharedHolder_->num_;

        // Make sure the shared data buffers are properly pre-allocated
        if (sharedHolder_->time_.empty())
        {
            sharedHolder_->time_.assign(1, 0.0);
            sharedHolder_->data_.resize(1);
        }

        // Add a column for the sensor to the shared data buffers
        for (Eigen::MatrixXd & data : sharedHolder_->data_)
        {
            data.conservativeResize(getSize(), sharedHolder_->num_ + 1);
            data.rightCols<1>().setZero();
        }
        sharedHolder_->dataMeasured_.conservativeResize(getSize(), sharedHolder_->num_ + 1);
        sharedHolder_->dataMeasured_.rightCols<1>().setZero();

        // Add the sensor to the shared memory
        sharedHolder_->sensors_.push_back(this);
        ++sharedHolder_->num_;

        // Update the flag
        isAttached_ = true;

        return hresult_t::SUCCESS;
    }

    template<typename T>
    hresult_t AbstractSensorTpl<T>::detach()
    {
        // Delete the part of the shared memory associated with the sensor

        if (!isAttached_)
        {
            PRINT_ERROR("Sensor not attached to any robot.");
            return hresult_t::ERROR_GENERIC;
        }

        // Remove associated col in the shared data buffers
        if (sensorIdx_ < sharedHolder_->num_ - 1)
        {
            const std::size_t sensorShift = sharedHolder_->num_ - sensorIdx_ - 1;
            for (Eigen::MatrixXd & data : sharedHolder_->data_)
            {
                /* Aliasing is NOT an issue when shifting left/up the columns/rows of matrices.
                   This holds true regardless if the matrix is row- and column-major. Yet, it is
                   necessary to make an intermediary copy when shifting right or down! */
                data.middleCols(sensorIdx_, sensorShift) = data.rightCols(sensorShift);
            }
            sharedHolder_->dataMeasured_.middleCols(sensorIdx_, sensorShift) =
                sharedHolder_->dataMeasured_.rightCols(sensorShift);
        }
        for (Eigen::MatrixXd & data : sharedHolder_->data_)
        {
            data.conservativeResize(Eigen::NoChange, sharedHolder_->num_ - 1);
        }
        sharedHolder_->dataMeasured_.conservativeResize(Eigen::NoChange, sharedHolder_->num_ - 1);

        // Shift the sensor indices
        for (std::size_t i = sensorIdx_ + 1; i < sharedHolder_->num_; ++i)
        {
            AbstractSensorTpl<T> * sensor =
                static_cast<AbstractSensorTpl<T> *>(sharedHolder_->sensors_[i]);
            --sensor->sensorIdx_;
        }

        // Remove the sensor from the shared memory
        sharedHolder_->sensors_.erase(sharedHolder_->sensors_.begin() + sensorIdx_);
        --sharedHolder_->num_;

        // Clear the references to the robot and shared data
        robot_.reset();
        sharedHolder_ = nullptr;

        // Unset the Id
        sensorIdx_ = -1;

        // Update the flag
        isAttached_ = false;

        return hresult_t::SUCCESS;
    }

    template<typename T>
    hresult_t AbstractSensorTpl<T>::resetAll(uint32_t seed)
    {
        // Make sure all the sensors are attached to a robot
        for (AbstractSensorBase * sensor : sharedHolder_->sensors_)
        {
            if (!sensor->isAttached_)
            {
                PRINT_ERROR("Sensor '",
                            sensor->name_,
                            "' of type '",
                            type_,
                            "' not attached to any robot.");
                return hresult_t::ERROR_GENERIC;
            }
        }

        // Make sure the robot still exists
        if (robot_.expired())
        {
            PRINT_ERROR("Robot has been deleted. Impossible to reset the sensors.");
            return hresult_t::ERROR_GENERIC;
        }

        // Clear the shared data buffers
        sharedHolder_->time_.assign(1, 0.0);
        sharedHolder_->data_.resize(1);
        sharedHolder_->data_[0].setZero();
        sharedHolder_->dataMeasured_.setZero();

        // Compute max delay
        sharedHolder_->delayMax_ = std::accumulate(sharedHolder_->sensors_.begin(),
                                                   sharedHolder_->sensors_.end(),
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
        std::vector<std::uint32_t> seeds(sharedHolder_->num_);
        seq.generate(seeds.begin(), seeds.end());

        // Reset sensor-specific state
        std::vector<AbstractSensorBase *>::iterator sensorIt = sharedHolder_->sensors_.begin();
        std::vector<std::uint32_t>::const_iterator seedIt = seeds.cbegin();
        for (; sensorIt != sharedHolder_->sensors_.end(); ++sensorIt, ++seedIt)
        {
            AbstractSensorBase & sensor = *(*sensorIt);

            // Reset the internal random number generator
            sensor.generator_.seed(*seedIt);

            // Refresh proxies that are robot-dependent
            sensor.refreshProxies();

            // Reset the telemetry state
            sensor.isTelemetryConfigured_ = false;
        }

        return hresult_t::SUCCESS;
    }

    template<typename T>
    hresult_t AbstractSensorTpl<T>::setOptionsAll(const GenericConfig & sensorOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isAttached_)
        {
            PRINT_ERROR("Sensor not attached to any robot.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        for (AbstractSensorBase * sensor : sharedHolder_->sensors_)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = sensor->setOptions(sensorOptions);
            }
        }

        return returnCode;
    }

    template<typename T>
    std::size_t AbstractSensorTpl<T>::getIdx() const
    {
        return sensorIdx_;
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
        if (areFieldnamesGrouped_)
        {
            return addCircumfix(name_, getType(), {}, TELEMETRY_FIELDNAME_DELIMITER);
        }
        else
        {
            return name_;
        }
    }

    template<typename T>
    Eigen::Ref<const Eigen::VectorXd> AbstractSensorTpl<T>::get() const
    {
        static Eigen::VectorXd dataDummy = Eigen::VectorXd::Zero(fieldnames_.size());
        if (isAttached_)
        {
            return sharedHolder_->dataMeasured_.col(sensorIdx_);
        }
        return dataDummy;
    }

    template<typename T>
    inline Eigen::Ref<Eigen::VectorXd> AbstractSensorTpl<T>::get()
    {
        // No guard, since this method is not public
        return sharedHolder_->dataMeasured_.col(sensorIdx_);
    }

    template<typename T>
    inline Eigen::Ref<Eigen::VectorXd> AbstractSensorTpl<T>::data()
    {
        // No guard, since this method is not public
        return sharedHolder_->data_.back().col(sensorIdx_);
    }

    template<typename T>
    hresult_t AbstractSensorTpl<T>::interpolateData()
    {
        assert(sharedHolder_->time_.size() > 0 && "No data to interpolate.");

        // Sample the delay uniformly
        const double delay =
            baseSensorOptions_->delay +
            uniform(generator_, 0.0F, static_cast<float>(baseSensorOptions_->jitter));

        // Get time at which to fetch sensor data
        double timeDesired = sharedHolder_->time_.back() - delay;

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
            std::ptrdiff_t right = sharedHolder_->time_.size() - 1;
            std::ptrdiff_t mid = 0;

            if (timeDesired >= sharedHolder_->time_.back())
            {
                return right;
            }
            else if (timeDesired < sharedHolder_->time_.front())
            {
                return -1;
            }

            while (left < right)
            {
                mid = (left + right) / 2;
                if (timeDesired < sharedHolder_->time_[mid])
                {
                    right = mid;
                }
                else if (timeDesired > sharedHolder_->time_[mid])
                {
                    left = mid + 1;
                }
                else
                {
                    return mid;
                }
            }

            if (timeDesired < sharedHolder_->time_[mid])
            {
                return mid - 1;
            }
            else
            {
                return mid;
            }
        };

        const int64_t idxLeft = bisectLeft();
        if (timeDesired >= 0.0 && idxLeft + 1 < static_cast<int64_t>(sharedHolder_->time_.size()))
        {
            if (idxLeft < 0)
            {
                PRINT_ERROR("No data old enough is available.");
                return hresult_t::ERROR_GENERIC;
            }
            else if (baseSensorOptions_->delayInterpolationOrder == 0)
            {
                get() = sharedHolder_->data_[idxLeft].col(sensorIdx_);
            }
            else if (baseSensorOptions_->delayInterpolationOrder == 1)
            {
                // FIXME: Linear interpolation is not valid on Lie algebra
                const double ratio =
                    (timeDesired - sharedHolder_->time_[idxLeft]) /
                    (sharedHolder_->time_[idxLeft + 1] - sharedHolder_->time_[idxLeft]);
                auto dataNext = sharedHolder_->data_[idxLeft + 1].col(sensorIdx_);
                auto dataPrev = sharedHolder_->data_[idxLeft].col(sensorIdx_);
                get() = dataPrev + ratio * (dataNext - dataPrev);
            }
            else
            {
                PRINT_ERROR("`delayInterpolationOrder` must be either 0 or 1.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }
        else
        {
            if (baseSensorOptions_->delay > EPS || baseSensorOptions_->jitter > EPS)
            {
                // Return the oldest value since the buffer is not fully initialized yet
                auto it = std::find_if(sharedHolder_->time_.begin(),
                                       sharedHolder_->time_.end(),
                                       [](double t) -> bool { return t > 0; });
                if (it != sharedHolder_->time_.end())
                {
                    std::ptrdiff_t idx = std::distance(sharedHolder_->time_.begin(), it);
                    idx = std::max(std::ptrdiff_t(0), idx - 1);
                    get() = sharedHolder_->data_[idx].col(sensorIdx_);
                }
                else
                {
                    get() = sharedHolder_->data_.back().col(sensorIdx_);
                }
            }
            else
            {
                // Return the most recent value available
                get() = sharedHolder_->data_.back().col(sensorIdx_);
            }
        }

        return hresult_t::SUCCESS;
    }

    template<typename T>
    hresult_t AbstractSensorTpl<T>::measureDataAll()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        for (AbstractSensorBase * sensor : sharedHolder_->sensors_)
        {
            // Compute the real value at current time, namely taking into account the sensor delay
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = sensor->interpolateData();
            }

            // Skew the data with white noise and bias
            if (returnCode == hresult_t::SUCCESS)
            {
                sensor->measureData();
            }
        }

        return returnCode;
    }

    template<typename T>
    hresult_t AbstractSensorTpl<T>::setAll(double t,
                                           const Eigen::VectorXd & q,
                                           const Eigen::VectorXd & v,
                                           const Eigen::VectorXd & a,
                                           const Eigen::VectorXd & uMotor,
                                           const ForceVector & fExternal)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isAttached_)
        {
            PRINT_ERROR("Sensor not attached to any robot.");
            return hresult_t::ERROR_GENERIC;
        }

        /* Make sure at least the requested delay plus the maximum time step is available to handle
           the case where the solver goes back in time. Even though it makes the buffer much larger
           than necessary as the actual maximum step is given by `engineOptions_->stepper.dtMax`,
           it is not a big deal since `rotate`, `pop_front`, `push_back` have O(1) complexity. */
        const double timeMin = t - sharedHolder_->delayMax_ - SIMULATION_MAX_TIMESTEP;

        // Internal buffer memory management
        if (t + EPS > sharedHolder_->time_.back())
        {
            const std::size_t bufferSize = sharedHolder_->time_.size();
            if (timeMin > sharedHolder_->time_.front())
            {
                // Remove some unecessary extra elements if appropriate
                if (bufferSize > 1U + DELAY_MAX_BUFFER_EXCEED &&
                    timeMin > sharedHolder_->time_[DELAY_MAX_BUFFER_EXCEED])
                {
                    sharedHolder_->time_.erase_begin(DELAY_MAX_BUFFER_EXCEED);
                    sharedHolder_->data_.erase_begin(DELAY_MAX_BUFFER_EXCEED);
                    sharedHolder_->time_.rset_capacity(bufferSize + DELAY_MIN_BUFFER_RESERVE);
                    sharedHolder_->data_.rset_capacity(bufferSize + DELAY_MIN_BUFFER_RESERVE);
                }

                // Rotate the internal buffer
                sharedHolder_->time_.rotate(sharedHolder_->time_.begin() + 1U);
                sharedHolder_->data_.rotate(sharedHolder_->data_.begin() + 1U);
            }
            else
            {
                // Increase capacity if required
                if (sharedHolder_->time_.full())
                {
                    sharedHolder_->time_.rset_capacity(bufferSize + DELAY_MIN_BUFFER_RESERVE);
                    sharedHolder_->data_.rset_capacity(bufferSize + DELAY_MIN_BUFFER_RESERVE);
                }

                /* Push back new buffer.

                   Note that it is a copy of the last value. This is important for `data()` to
                   always provide the last true value instead of some initialized memory. The
                   previous value is used for the quaternion of IMU sensors to choice the right
                   value that ensures its continuity over time amond to two possible choices. */
                sharedHolder_->time_.push_back(INF);
                sharedHolder_->data_.push_back(sharedHolder_->data_.back());
            }
        }
        else
        {
            /* Remove the extra last elements if for some reason the solver went back in time.
               It happens when integration fails for ode solvers relying on try_step mechanism. */
            while (t + EPS < sharedHolder_->time_.back() && sharedHolder_->time_.size() > 1)
            {
                sharedHolder_->time_.pop_back();
                sharedHolder_->data_.pop_back();
            }
        }
        sharedHolder_->time_.back() = t;

        // Update the last real data buffer
        for (AbstractSensorBase * sensor : sharedHolder_->sensors_)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = sensor->set(t, q, v, a, uMotor, fExternal);
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Compute the measurement data
            returnCode = measureDataAll();
        }

        return returnCode;
    }

    template<typename T>
    void AbstractSensorTpl<T>::updateTelemetryAll()
    {
        for (AbstractSensorBase * sensor : sharedHolder_->sensors_)
        {
            sensor->updateTelemetry();
        }
    }
}
