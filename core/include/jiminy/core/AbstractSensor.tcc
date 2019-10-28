#include "jiminy/core/Model.h"
#include "jiminy/core/Engine.h" // Required to get access to MIN_TIME_STEP and MAX_TIME_STEP

namespace jiminy
{
    extern float64_t const MIN_TIME_STEP;
    extern float64_t const MAX_TIME_STEP;

    template <typename T>
    float64_t AbstractSensorTpl<T>::delayMax_(0);

    template <typename T>
    AbstractSensorTpl<T>::AbstractSensorTpl(Model                               const & model,
                                            std::shared_ptr<SensorDataHolder_t> const & dataHolder,
                                            std::string                         const & name) :
    AbstractSensorBase(model, name),
    dataHolder_(dataHolder),
    sensorId_(dataHolder_->num_)
    {
        // Add the sensor to the data holder
        ++dataHolder_->num_;
        dataHolder_->sensors_.push_back(this);
        dataHolder_->counters_.push_back(1);

        // Reset the sensors' internal state
        AbstractSensorTpl<T>::reset();
    }

    template <typename T>
    AbstractSensorTpl<T>::~AbstractSensorTpl(void)
    {
        --dataHolder_->counters_[sensorId_];
        if (!dataHolder_->counters_[sensorId_])
        {
            // Remove associated col in the global data buffer
            if(sensorId_ < dataHolder_->num_ - 1)
            {
                for (matrixN_t & data : dataHolder_->data_)
                {
                    data.block(0, sensorId_, getSize(), dataHolder_->num_ - sensorId_ - 1) =
                        data.block(0, sensorId_ + 1, getSize(), dataHolder_->num_ - sensorId_ - 1).eval(); // eval to avoid aliasing
                }
            }
            for (matrixN_t & data : dataHolder_->data_)
            {
                data.resize(Eigen::NoChange, dataHolder_->num_ - 1);
            }

            // Shift the sensor ids
            for (uint32_t i=sensorId_ + 1; i < dataHolder_->num_; i++)
            {
                AbstractSensorTpl<T> * sensor = static_cast<AbstractSensorTpl<T> *>(dataHolder_->sensors_[i]);
                --sensor->sensorId_;
            }

            // Remove the deprecated elements of the global containers
            dataHolder_->sensors_.erase(dataHolder_->sensors_.begin() + sensorId_);
            dataHolder_->counters_.erase(dataHolder_->counters_.begin() + sensorId_);

            // Update the total number of sensors left
            --dataHolder_->num_;
        }

        // Reset the sensors' internal state
        reset();
    }

    template <typename T>
    void AbstractSensorTpl<T>::reset(void)
    {
        dataHolder_->time_.resize(2);
        std::fill(dataHolder_->time_.begin(), dataHolder_->time_.end(), -1);
        dataHolder_->time_.back() = 0;
        dataHolder_->data_.resize(2);
        for (matrixN_t & data : dataHolder_->data_)
        {
            data = matrixN_t::Zero(getSize(), dataHolder_->num_); // Do not use setZero since the size is ill-defined
        }
        data_ = vectorN_t::Zero(getSize());

        // Reset the telemetry state
        isTelemetryConfigured_ = false;
    }

    template <typename T>
    void AbstractSensorTpl<T>::setOptions(configHolder_t const & sensorOptions)
    {
        AbstractSensorBase::setOptions(sensorOptions);
        delayMax_ = std::max(delayMax_, sensorOptions_->delay); // No need to loop over all sensors
    }

    template <typename T>
    void AbstractSensorTpl<T>::setOptionsAll(configHolder_t const & sensorOptions)
    {
        for (AbstractSensorBase * sensor : dataHolder_->sensors_)
        {
            sensor->setOptions(sensorOptions);
        }
    }

    template <typename T>
    std::string const & AbstractSensorTpl<T>::getType(void) const
    {
        return type_;
    }

    template <typename T>
    std::vector<std::string> const & AbstractSensorTpl<T>::getFieldNames(void) const
    {
        if(sensorOptions_->rawData)
        {
            return fieldNamesPreProcess_;
        }
        else
        {
            return fieldNamesPostProcess_;
        }
    }

    template <typename T>
    uint32_t AbstractSensorTpl<T>::getSize(void) const
    {
        if(sensorOptions_->rawData)
        {
            return fieldNamesPreProcess_.size();
        }
        else
        {
            return fieldNamesPostProcess_.size();
        }
    }

    template <typename T>
    result_t AbstractSensorTpl<T>::get(Eigen::Ref<vectorN_t> data)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isDataUpToDate_)
        {
            // Add 1e-9 to timeDesired to avoid float comparison issues (std::numeric_limits<float64_t>::epsilon() is not enough)
            float64_t const timeDesired = dataHolder_->time_.back() - sensorOptions_->delay + 1e-9;

            /* Determine the position of the closest right element.
               Bisection method can be used since times are sorted. */
            auto bisectLeft =
                [&](void) -> int32_t
                {
                    int32_t left = 0;
                    int32_t right = dataHolder_->time_.size() - 1;
                    int32_t mid = 0;

                    if (timeDesired >= dataHolder_->time_.back())
                    {
                        return right;
                    }
                    else if (timeDesired < dataHolder_->time_.front())
                    {
                        return -1;
                    }

                    while(left < right)
                    {
                        mid = (left + right) / 2;
                        if (timeDesired < dataHolder_->time_[mid])
                        {
                            right = mid;
                        }
                        else if (timeDesired > dataHolder_->time_[mid])
                        {
                            left = mid + 1;
                        }
                        else
                        {
                            return mid;
                        }
                    }

                    if (timeDesired < dataHolder_->time_[mid])
                    {
                        return mid - 1;
                    }
                    else
                    {
                        return mid;
                    }
                };

            int32_t const inputIndexLeft = bisectLeft();
            if (timeDesired >= 0.0 && uint32_t(inputIndexLeft + 1) < dataHolder_->time_.size())
            {
                if (inputIndexLeft < 0)
                {
                    std::cout << "Error - AbstractSensorTpl<T>::get - No data old enough is available." << std::endl;
                    returnCode = result_t::ERROR_GENERIC;
                }
                else if (sensorOptions_->delayInterpolationOrder == 0)
                {
                    data_ = dataHolder_->data_[inputIndexLeft].col(sensorId_);
                }
                else if (sensorOptions_->delayInterpolationOrder == 1)
                {
                    data_ = 1 / (dataHolder_->time_[inputIndexLeft + 1] - dataHolder_->time_[inputIndexLeft]) *
                        ((timeDesired - dataHolder_->time_[inputIndexLeft]) * dataHolder_->data_[inputIndexLeft + 1].col(sensorId_) +
                        (dataHolder_->time_[inputIndexLeft + 1] - timeDesired) * dataHolder_->data_[inputIndexLeft].col(sensorId_));
                }
                else
                {
                    std::cout << "Error - AbstractSensorTpl<T>::get - The delayInterpolationOrder must be either 0 or 1 so far." << std::endl;
                    returnCode = result_t::ERROR_BAD_INPUT;
                }
            }
            else
            {
                if (dataHolder_->time_[0] >= 0.0 || sensorOptions_->delay < std::numeric_limits<float64_t>::epsilon())
                {
                    // Return the most recent value
                    data_ = dataHolder_->data_.back().col(sensorId_);
                }
                else
                {
                    // Return Zero since the sensor is not fully initialized yet
                    data_ = dataHolder_->data_.front().col(sensorId_);
                }
            }
        }

        if (returnCode != result_t::SUCCESS)
        {
            data_ = vectorN_t::Zero(getSize());
        }
        else
        {
            data = data_;
            isDataUpToDate_ = true;
        }

        return returnCode;
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
    matrixN_t::ColXpr AbstractSensorTpl<T>::data(void)
    {
        return dataHolder_->data_.back().col(sensorId_);
    }

    template <typename T>
    result_t AbstractSensorTpl<T>::getAll(matrixN_t & data)
    {
        result_t returnCode = result_t::SUCCESS;

        data.resize(dataHolder_->data_[0].rows(), dataHolder_->num_);
        for (AbstractSensorBase * sensor : dataHolder_->sensors_)
        {
            if (returnCode == result_t::SUCCESS)
            {
                float64_t sensorId = static_cast<AbstractSensorTpl<T> *>(sensor)->sensorId_;
                returnCode = sensor->get(data.col(sensorId));
            }
        }

        return returnCode;
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
        float64_t const timeMin = t - delayMax_ - MAX_TIME_STEP;

        // Internal buffer memory management
        if (t + std::numeric_limits<float64_t>::epsilon() > dataHolder_->time_.back())
        {
            if (dataHolder_->time_[0] < 0 || timeMin > dataHolder_->time_[1])
            {
                // Remove some unecessary extra elements if appropriate
                if (dataHolder_->time_.size() > 2 + MAX_DELAY_BUFFER_EXCEED
                && timeMin > dataHolder_->time_[2 + MAX_DELAY_BUFFER_EXCEED])
                {
                    for (uint8_t i=0; i < 1 + MAX_DELAY_BUFFER_EXCEED; i ++)
                    {
                        dataHolder_->time_.pop_front();
                        dataHolder_->data_.pop_front();
                    }

                    dataHolder_->time_.rset_capacity(dataHolder_->time_.size() + MIN_DELAY_BUFFER_RESERVE);
                    dataHolder_->data_.rset_capacity(dataHolder_->data_.size() + MIN_DELAY_BUFFER_RESERVE);
                }

                // Rotate the internal buffer
                dataHolder_->time_.rotate(dataHolder_->time_.begin() + 1);
                dataHolder_->data_.rotate(dataHolder_->data_.begin() + 1);
            }
            else
            {
                // Increase capacity if required
                if(dataHolder_->time_.full())
                {
                    dataHolder_->time_.rset_capacity(dataHolder_->time_.size() + 1 + MIN_DELAY_BUFFER_RESERVE);
                    dataHolder_->data_.rset_capacity(dataHolder_->data_.size() + 1 + MIN_DELAY_BUFFER_RESERVE);
                }

                // Push back new empty buffer (Do NOT initialize it for efficiency)
                dataHolder_->time_.push_back();
                dataHolder_->data_.push_back();
                dataHolder_->data_.back().resize(dataHolder_->data_[0].rows(), dataHolder_->num_);
            }
        }
        else
        {
            /* Remove the extra last elements if for some reason the solver went back in time.
                It happens when an iteration fails using ode solvers relying on try_step mechanism. */
            while(t + std::numeric_limits<float64_t>::epsilon() < dataHolder_->time_.back() && dataHolder_->time_.size() > 2)
            {
                dataHolder_->time_.pop_back();
                dataHolder_->data_.pop_back();
            }
        }
        dataHolder_->time_.back() = t;

        // Compute the sensors' output
        for (AbstractSensorBase * sensor : dataHolder_->sensors_)
        {
            // Compute the true value
            if (returnCode == result_t::SUCCESS)
            {
                returnCode = sensor->set(t, q, v, a, u);
            }

            // Add white noise
            if (returnCode == result_t::SUCCESS)
            {
                if (sensorOptions_->noiseStd.size())
                {
                    sensor->data() += randVectorNormal(sensor->sensorOptions_->noiseStd);
                }
                if (sensor->sensorOptions_->bias.size())
                {
                    sensor->data() += sensor->sensorOptions_->bias;
                }
                sensor->isDataUpToDate_ = false;
            }
        }

        return returnCode;
    }

    template <typename T>
    void AbstractSensorTpl<T>::updateTelemetryAll(void)
    {
        for (AbstractSensorBase * sensor : dataHolder_->sensors_)
        {
            sensor->updateTelemetry();
        }
    }
}