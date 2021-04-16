#include <numeric>

#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Constants.h"


namespace jiminy
{
    template <typename T>
    AbstractSensorTpl<T>::AbstractSensorTpl(std::string const & name) :
    AbstractSensorBase(name),
    sensorIdx_(-1),
    sharedHolder_(nullptr)
    {
        // Empty
    }

    template <typename T>
    AbstractSensorTpl<T>::~AbstractSensorTpl(void)
    {
        // Detach the sensor before deleting it if necessary
        if (isAttached_)
        {
            detach();
        }
    }

    template <typename T>
    hresult_t AbstractSensorTpl<T>::attach(std::weak_ptr<Robot const> robot,
                                           SensorSharedDataHolder_t * sharedHolder)
    {
        // Make sure the sensor is not already attached
        if (isAttached_)
        {
            PRINT_ERROR("Sensor already attached to a robot. Please 'detach' method before attaching it.");
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

        // Add a column for the sensor to the shared data buffers
        for (matrixN_t & data : sharedHolder_->data_)
        {
            data.conservativeResize(Eigen::NoChange, sharedHolder_->num_ + 1);
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

    template <typename T>
    hresult_t AbstractSensorTpl<T>::detach(void)
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
            int32_t sensorShift = sharedHolder_->num_ - sensorIdx_ - 1;
            for (matrixN_t & data : sharedHolder_->data_)
            {
                data.middleCols(sensorIdx_, sensorShift) =
                    data.middleCols(sensorIdx_ + 1, sensorShift).eval();
            }
            sharedHolder_->dataMeasured_.middleCols(sensorIdx_, sensorShift) =
                sharedHolder_->dataMeasured_.middleCols(sensorIdx_ + 1, sensorShift).eval();
        }
        for (matrixN_t & data : sharedHolder_->data_)
        {
            data.conservativeResize(Eigen::NoChange, sharedHolder_->num_ - 1);
        }
        sharedHolder_->dataMeasured_.conservativeResize(Eigen::NoChange, sharedHolder_->num_ - 1);

        // Shift the sensor indices
        for (int32_t i = sensorIdx_ + 1; i < sharedHolder_->num_; ++i)
        {
            AbstractSensorTpl<T> * sensor =
                static_cast<AbstractSensorTpl<T> *>(sharedHolder_->sensors_[i]);
            --sensor->sensorIdx_;
        }

        // Remove the sensor from the shared memory
        sharedHolder_->sensors_.erase(sharedHolder_->sensors_.begin() + sensorIdx_);
        --sharedHolder_->num_;

        // Update delayMax_ proxy
        if (sharedHolder_->delayMax_ < baseSensorOptions_->delay + EPS)
        {
            sharedHolder_->delayMax_ = 0.0;
            for (AbstractSensorBase * sensor : sharedHolder_->sensors_)
            {
                sharedHolder_->delayMax_ = std::max(sharedHolder_->delayMax_,
                                                    sensor->baseSensorOptions_->delay);
            }
        }

        // Clear the references to the robot and shared data
        robot_.reset();
        sharedHolder_ = nullptr;

        // Unset the Id
        sensorIdx_ = -1;

        // Update the flag
        isAttached_ = false;

        return hresult_t::SUCCESS;
    }

    template <typename T>
    hresult_t AbstractSensorTpl<T>::resetAll(void)
    {
        // Make sure the sensor is attached to a robot
        if (!isAttached_)
        {
            PRINT_ERROR("Sensor not attached to any robot.");
            return hresult_t::ERROR_GENERIC;
        }

        // Make sure the robot still exists
        if (robot_.expired())
        {
            PRINT_ERROR("Robot has been deleted. Impossible to reset the sensors.");
            return hresult_t::ERROR_GENERIC;
        }

        // Clear the shared data buffers
        sharedHolder_->time_.resize(2);
        std::fill(sharedHolder_->time_.begin(), sharedHolder_->time_.end(), -1);
        sharedHolder_->time_.back() = 0.0;
        sharedHolder_->data_.resize(2);
        for (matrixN_t & data : sharedHolder_->data_)
        {
            data = matrixN_t::Zero(getSize(), sharedHolder_->num_);  // Do NOT use setZero since the size may be ill-defined
        }
        sharedHolder_->dataMeasured_.setZero();

        // Compute max delay
        sharedHolder_->delayMax_ = std::accumulate(
            sharedHolder_->sensors_.begin(), sharedHolder_->sensors_.end(), 0.0,
            [](float64_t const & value, AbstractSensorBase * sensor)
            {
                return std::max(sensor->baseSensorOptions_->delay, value);
            });

        // Update sensor scope information
        for (AbstractSensorBase * sensor : sharedHolder_->sensors_)
        {
            // Refresh proxies that are robot-dependent
            sensor->refreshProxies();

            // Reset the telemetry state
            sensor->isTelemetryConfigured_ = false;
        }

        return hresult_t::SUCCESS;
    }

    template <typename T>
    hresult_t AbstractSensorTpl<T>::setOptionsAll(configHolder_t const & sensorOptions)
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

    template <typename T>
    int32_t const & AbstractSensorTpl<T>::getIdx(void) const
    {
        return sensorIdx_;
    }

    template <typename T>
    std::string const & AbstractSensorTpl<T>::getType(void) const
    {
        return type_;
    }

    template <typename T>
    std::vector<std::string> const & AbstractSensorTpl<T>::getFieldnames(void) const
    {
        return fieldNames_;
    }

    template <typename T>
    uint32_t AbstractSensorTpl<T>::getSize(void) const
    {
        return fieldNames_.size();
    }
    template <typename T>
    std::string AbstractSensorTpl<T>::getTelemetryName(void) const
    {
        if (areFieldnamesGrouped_)
        {
            return getType() + TELEMETRY_FIELDNAME_DELIMITER + name_;
        }
        else
        {
            return name_;
        }
    }

    template <typename T>
    inline Eigen::Ref<vectorN_t const> AbstractSensorTpl<T>::get(void) const
    {
        static vectorN_t dataDummy = vectorN_t::Zero(fieldNames_.size());
        if (sharedHolder_)
        {
            return sharedHolder_->dataMeasured_.col(sensorIdx_);
        }
        return dataDummy;
    }

    template <typename T>
    inline Eigen::Ref<vectorN_t> AbstractSensorTpl<T>::get(void)
    {
        // No guard, since this method is not public
        return sharedHolder_->dataMeasured_.col(sensorIdx_);
    }

    template <typename T>
    inline Eigen::Ref<vectorN_t> AbstractSensorTpl<T>::data(void)
    {
        // No guard, since this method is not public
        return sharedHolder_->data_.back().col(sensorIdx_);
    }

    template <typename T>
    hresult_t AbstractSensorTpl<T>::interpolateData(void)
    {
        // Add STEPPER_MIN_TIMESTEP to timeDesired to avoid float comparison issues
        float64_t const timeDesired = sharedHolder_->time_.back() - baseSensorOptions_->delay + STEPPER_MIN_TIMESTEP;

        /* Determine the position of the closest right element.
        Bisection method can be used since times are sorted. */
        auto bisectLeft =
            [&](void) -> int32_t
            {
                int32_t left = 0;
                int32_t right = sharedHolder_->time_.size() - 1;
                int32_t mid = 0;

                if (timeDesired >= sharedHolder_->time_.back())
                {
                    return right;
                }
                else if (timeDesired < sharedHolder_->time_.front())
                {
                    return -1;
                }

                while(left < right)
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

        int32_t const idxLeft = bisectLeft();
        if (timeDesired >= 0.0 && uint32_t(idxLeft + 1) < sharedHolder_->time_.size())
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
                get() = 1 / (sharedHolder_->time_[idxLeft + 1] - sharedHolder_->time_[idxLeft]) *
                        ((timeDesired - sharedHolder_->time_[idxLeft]) * sharedHolder_->data_[idxLeft + 1].col(sensorIdx_) +
                        (sharedHolder_->time_[idxLeft + 1] - timeDesired) * sharedHolder_->data_[idxLeft].col(sensorIdx_));
            }
            else
            {
                PRINT_ERROR("The delayInterpolationOrder must be either 0 or 1 so far.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }
        else
        {
            if (baseSensorOptions_->delay > EPS)
            {
                // Return the oldest value since the buffer is not fully initialized yet
                auto it = std::find_if(sharedHolder_->time_.begin(), sharedHolder_->time_.end(),
                                       [] (float64_t const & t)
                                       {
                                           return t > 0;
                                       });
                if (it != sharedHolder_->time_.end())
                {
                    int32_t ind = std::distance(sharedHolder_->time_.begin(), it);
                    ind = std::max(0, ind - 1);
                    get() = sharedHolder_->data_[ind].col(sensorIdx_);
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

    template <typename T>
    hresult_t AbstractSensorTpl<T>::generateMeasurementAll(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        for (AbstractSensorBase * sensor : sharedHolder_->sensors_)
        {
            // Compute the real value at current time, namely taking into account the sensor delay
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = sensor->interpolateData();
            }

            // Shew the data with white noise and bias
            if (returnCode == hresult_t::SUCCESS)
            {
                sensor->skewMeasurement();
            }
        }

        return returnCode;
    }

    template <typename T>
    hresult_t AbstractSensorTpl<T>::setAll(float64_t const & t,
                                           vectorN_t const & q,
                                           vectorN_t const & v,
                                           vectorN_t const & a,
                                           vectorN_t const & uMotor)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isAttached_)
        {
            PRINT_ERROR("Sensor not attached to any robot.");
            return hresult_t::ERROR_GENERIC;
        }

        /* Make sure at least the requested delay plus the maximum time step
           is available to handle the case where the solver goes back in time.
           Even though it can make the buffer quite large irrelevantly since
           the actual maximum step is given by engineOptions_->stepper.dtMax,
           it is not a big deal in practice since `rotate`, `pop_front`, and
           `push_back` have O(1) complexity. */
        float64_t const timeMin = t - sharedHolder_->delayMax_ - SIMULATION_MAX_TIMESTEP;

        // Internal buffer memory management
        if (t + EPS > sharedHolder_->time_.back())
        {
            if (sharedHolder_->time_[0] < 0.0 || timeMin > sharedHolder_->time_[1])
            {
                // Remove some unecessary extra elements if appropriate
                if (sharedHolder_->time_.size() > 2U + DELAY_MAX_BUFFER_EXCEED
                && timeMin > sharedHolder_->time_[2U + DELAY_MAX_BUFFER_EXCEED])
                {
                    for (uint8_t i=0; i < 1 + DELAY_MAX_BUFFER_EXCEED; ++i)
                    {
                        sharedHolder_->time_.pop_front();
                        sharedHolder_->data_.pop_front();
                    }

                    sharedHolder_->time_.rset_capacity(sharedHolder_->time_.size() + DELAY_MIN_BUFFER_RESERVE);
                    sharedHolder_->data_.rset_capacity(sharedHolder_->data_.size() + DELAY_MIN_BUFFER_RESERVE);
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
                    sharedHolder_->time_.rset_capacity(sharedHolder_->time_.size() + 1U + DELAY_MIN_BUFFER_RESERVE);
                    sharedHolder_->data_.rset_capacity(sharedHolder_->data_.size() + 1U + DELAY_MIN_BUFFER_RESERVE);
                }

                // Push back new empty buffer
                sharedHolder_->time_.push_back(-1);
                sharedHolder_->data_.push_back(matrixN_t::Zero(getSize(), sharedHolder_->num_));
            }
        }
        else
        {
            /* Remove the extra last elements if for some reason the solver went back in time.
               It happens when an iteration fails using ode solvers relying on try_step mechanism. */
            while(t + EPS < sharedHolder_->time_.back() && sharedHolder_->time_.size() > 2)
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
                returnCode = sensor->set(t, q, v, a, uMotor);
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Compute the measurement data
            returnCode = generateMeasurementAll();
        }

        return returnCode;
    }

    template <typename T>
    void AbstractSensorTpl<T>::updateTelemetryAll(void)
    {
        for (AbstractSensorBase * sensor : sharedHolder_->sensors_)
        {
            sensor->updateTelemetry();
        }
    }
}
