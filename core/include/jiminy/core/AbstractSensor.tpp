#include "jiminy/core/Engine.h" // MIN_TIME_STEP and MAX_TIME_STEP
#include "jiminy/core/Model.h"
#include "jiminy/core/Utilities.h"

namespace jiminy
{
    extern float64_t const MIN_TIME_STEP;
    extern float64_t const MAX_TIME_STEP;

    template <typename T>
    AbstractSensorTpl<T>::AbstractSensorTpl(Model       const & model,
                                            std::shared_ptr<SensorSharedDataHolder_t> const & sharedHolder,
                                            std::string const & name) :
    AbstractSensorBase(model, name),
    sharedHolder_(sharedHolder),
    sensorId_(sharedHolder_->num_)
    {
        // Initialize the options
        setOptions(getDefaultOptions());

        // Add the sensor to the data holder
        ++sharedHolder_->num_;
        sharedHolder_->sensors_.push_back(this);

        // Generate a new data buffer taking into account the new sensor
        clearDataBuffer();
    }

    template <typename T>
    AbstractSensorTpl<T>::~AbstractSensorTpl(void)
    {
        // Remove associated col in the global data buffer
        if(sensorId_ < sharedHolder_->num_ - 1)
        {
            int8_t sensorShift = sharedHolder_->num_ - sensorId_ - 1;
            for (matrixN_t & data : sharedHolder_->data_)
            {
                data.middleCols(sensorId_, sensorShift) =
                    data.middleCols(sensorId_ + 1, sensorShift).eval();
            }
        }
        for (matrixN_t & data : sharedHolder_->data_)
        {
            data.conservativeResize(Eigen::NoChange, sharedHolder_->num_ - 1);
        }

        // Shift the sensor ids
        for (uint8_t i = sensorId_ + 1; i < sharedHolder_->num_; i++)
        {
            AbstractSensorTpl<T> * sensor =
                static_cast<AbstractSensorTpl<T> *>(sharedHolder_->sensors_[i]);
            --sensor->sensorId_;
        }

        // Remove the deprecated elements of the global containers
        sharedHolder_->sensors_.erase(sharedHolder_->sensors_.begin() + sensorId_);

        // Update the total number of sensors left
        --sharedHolder_->num_;

        // Update delayMax_ proxy if necessary
        if (sharedHolder_->delayMax_ < baseSensorOptions_->delay + EPS)
        {
            sharedHolder_->delayMax_ = 0.0;
            for (AbstractSensorBase * sensor : sharedHolder_->sensors_)
            {
                sharedHolder_->delayMax_ = std::max(sharedHolder_->delayMax_,
                                                    sensor->baseSensorOptions_->delay);
            }
        }

        // Generate a new data buffer taking into account the new sensor
        clearDataBuffer();
    }

    template <typename T>
    void AbstractSensorTpl<T>::reset(void)
    {
        // Clear the data buffer
        clearDataBuffer();

        // Refresh proxies that are model-dependent
        refreshProxies();

        // Reset the telemetry state
        isTelemetryConfigured_ = false;
    }

    template <typename T>
    result_t AbstractSensorTpl<T>::setOptions(configHolder_t const & sensorOptions)
    {
        AbstractSensorBase::setOptions(sensorOptions);
        sharedHolder_->delayMax_ = std::max(sharedHolder_->delayMax_, baseSensorOptions_->delay);
        return result_t::SUCCESS;
    }

    template <typename T>
    result_t AbstractSensorTpl<T>::setOptionsAll(configHolder_t const & sensorOptions)
    {
        result_t returnCode = result_t::SUCCESS;

        for (AbstractSensorBase * sensor : sharedHolder_->sensors_)
        {
            if (returnCode == result_t::SUCCESS)
            {
                returnCode = sensor->setOptions(sensorOptions);
            }
        }

        return returnCode;
    }

    template <typename T>
    uint8_t const & AbstractSensorTpl<T>::getId(void) const
    {
        return sensorId_;
    }

    template <typename T>
    std::string const & AbstractSensorTpl<T>::getType(void) const
    {
        return type_;
    }

    template <typename T>
    std::vector<std::string> const & AbstractSensorTpl<T>::getFieldNames(void) const
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
        if (areFieldNamesGrouped_)
        {
            return getType() + "." + name_;
        }
        else
        {
            return name_;
        }
    }

    template <typename T>
    inline vectorN_t const * AbstractSensorTpl<T>::get(void)
    {
        return &data_;
    }

    template <typename T>
    matrixN_t AbstractSensorTpl<T>::getAll(void)
    {
        matrixN_t data(getSize(), sharedHolder_->num_);
        for (AbstractSensorBase * sensor : sharedHolder_->sensors_)
        {
            uint8_t const & sensorId = static_cast<AbstractSensorTpl<T> *>(sensor)->sensorId_;
            data.col(sensorId) = *sensor->get();
        }
        return data;
    }

    template <typename T>
    Eigen::Ref<vectorN_t> AbstractSensorTpl<T>::data(void)
    {
        return sharedHolder_->data_.back().col(sensorId_);
    }

    template <typename T>
    result_t AbstractSensorTpl<T>::updateDataBuffer(void)
    {
        // Add 1e-9 to timeDesired to avoid float comparison issues (std::numeric_limits<float64_t>::epsilon() is not enough)
        float64_t const timeDesired = sharedHolder_->time_.back() - baseSensorOptions_->delay + 1e-9;

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

        int32_t const inputIndexLeft = bisectLeft();
        data_ = vectorN_t::Zero(getSize());
        if (timeDesired >= 0.0 && uint32_t(inputIndexLeft + 1) < sharedHolder_->time_.size())
        {
            if (inputIndexLeft < 0)
            {
                std::cout << "Error - AbstractSensorTpl<T>::updateDataBuffer - No data old enough is available." << std::endl;
                return result_t::ERROR_GENERIC;
            }
            else if (baseSensorOptions_->delayInterpolationOrder == 0)
            {
                data_ = sharedHolder_->data_[inputIndexLeft].col(sensorId_);
            }
            else if (baseSensorOptions_->delayInterpolationOrder == 1)
            {
                data_ = 1 / (sharedHolder_->time_[inputIndexLeft + 1] - sharedHolder_->time_[inputIndexLeft]) *
                    ((timeDesired - sharedHolder_->time_[inputIndexLeft]) * sharedHolder_->data_[inputIndexLeft + 1].col(sensorId_) +
                    (sharedHolder_->time_[inputIndexLeft + 1] - timeDesired) * sharedHolder_->data_[inputIndexLeft].col(sensorId_));
            }
            else
            {
                std::cout << "Error - AbstractSensorTpl<T>::updateDataBuffer - The delayInterpolationOrder must be either 0 or 1 so far." << std::endl;
                return result_t::ERROR_BAD_INPUT;
            }
        }
        else
        {
            if (sharedHolder_->time_[0] >= 0.0
            || baseSensorOptions_->delay < std::numeric_limits<float64_t>::epsilon())
            {
                // Return the most recent value
                data_ = sharedHolder_->data_.back().col(sensorId_);
            }
            else
            {
                // Return Zero since the sensor is not fully initialized yet
                data_ = sharedHolder_->data_.front().col(sensorId_);
            }
        }

        return result_t::SUCCESS;
    }

    template <typename T>
    void AbstractSensorTpl<T>::clearDataBuffer(void)
    {
        sharedHolder_->time_.resize(2);
        std::fill(sharedHolder_->time_.begin(), sharedHolder_->time_.end(), -1);
        sharedHolder_->time_.back() = 0;
        sharedHolder_->data_.resize(2);
        for (matrixN_t & data : sharedHolder_->data_)
        {
            data = matrixN_t::Zero(getSize(), sharedHolder_->num_); // Do not use setZero since the size is ill-defined
        }
        data_ = vectorN_t::Zero(getSize());
    }

    template <typename T>
    result_t AbstractSensorTpl<T>::setAll(float64_t const & t,
                                          vectorN_t const & q,
                                          vectorN_t const & v,
                                          vectorN_t const & a,
                                          vectorN_t const & u)
    {
        result_t returnCode = result_t::SUCCESS;

        /* Make sure at least the requested delay plus the maximum time step
           is available to handle the case where the solver goes back in time */
        float64_t const timeMin = t - sharedHolder_->delayMax_ - MAX_TIME_STEP;

        // Internal buffer memory management
        if (t + std::numeric_limits<float64_t>::epsilon() > sharedHolder_->time_.back())
        {
            if (sharedHolder_->time_[0] < 0 || timeMin > sharedHolder_->time_[1])
            {
                // Remove some unecessary extra elements if appropriate
                if (sharedHolder_->time_.size() > 2 + MAX_DELAY_BUFFER_EXCEED
                && timeMin > sharedHolder_->time_[2 + MAX_DELAY_BUFFER_EXCEED])
                {
                    for (uint8_t i=0; i < 1 + MAX_DELAY_BUFFER_EXCEED; i ++)
                    {
                        sharedHolder_->time_.pop_front();
                        sharedHolder_->data_.pop_front();
                    }

                    sharedHolder_->time_.rset_capacity(sharedHolder_->time_.size() + MIN_DELAY_BUFFER_RESERVE);
                    sharedHolder_->data_.rset_capacity(sharedHolder_->data_.size() + MIN_DELAY_BUFFER_RESERVE);
                }

                // Rotate the internal buffer
                sharedHolder_->time_.rotate(sharedHolder_->time_.begin() + 1);
                sharedHolder_->data_.rotate(sharedHolder_->data_.begin() + 1);
            }
            else
            {
                // Increase capacity if required
                if(sharedHolder_->time_.full())
                {
                    sharedHolder_->time_.rset_capacity(sharedHolder_->time_.size() + 1 + MIN_DELAY_BUFFER_RESERVE);
                    sharedHolder_->data_.rset_capacity(sharedHolder_->data_.size() + 1 + MIN_DELAY_BUFFER_RESERVE);
                }

                // Push back new empty buffer (Do NOT initialize it for efficiency)
                sharedHolder_->time_.push_back();
                sharedHolder_->data_.push_back();
                sharedHolder_->data_.back().resize(getSize(), sharedHolder_->num_);
            }
        }
        else
        {
            /* Remove the extra last elements if for some reason the solver went back in time.
                It happens when an iteration fails using ode solvers relying on try_step mechanism. */
            while(t + std::numeric_limits<float64_t>::epsilon() < sharedHolder_->time_.back()
            && sharedHolder_->time_.size() > 2)
            {
                sharedHolder_->time_.pop_back();
                sharedHolder_->data_.pop_back();
            }
        }
        sharedHolder_->time_.back() = t;

        // Compute the sensors' output
        for (AbstractSensorBase * sensor : sharedHolder_->sensors_)
        {
            if (returnCode == result_t::SUCCESS)
            {
                // Compute the true value
                returnCode = sensor->set(t, q, v, a, u);
            }

            if (returnCode == result_t::SUCCESS)
            {
                // Add white noise
                if (baseSensorOptions_->noiseStd.size())
                {
                    data(sensor) += randVectorNormal(sensor->baseSensorOptions_->noiseStd);
                }
                if (sensor->baseSensorOptions_->bias.size())
                {
                    data(sensor) += sensor->baseSensorOptions_->bias;
                }
            }

            if (returnCode == result_t::SUCCESS)
            {
                // Update data buffer
                returnCode = updateDataBuffer(sensor);
            }
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