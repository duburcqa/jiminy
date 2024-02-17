#ifndef JIMINY_FORWARD_SENSOR_H
#define JIMINY_FORWARD_SENSOR_H

#include "jiminy/core/fwd.h"

#include <boost/multi_index/hashed_index.hpp>     // `boost::multi_index::hashed_unique`
#include <boost/multi_index/member.hpp>           // `boost::multi_index::member`
#include <boost/multi_index/ordered_index.hpp>    // `boost::multi_index::ordered_unique`
#include <boost/multi_index/sequenced_index.hpp>  // `boost::multi_index::sequenced`
#include <boost/multi_index/tag.hpp>              // `boost::multi_index::tag`
#include <boost/multi_index_container.hpp>        // `boost::multi_index::multi_index_container`


namespace jiminy
{
    struct JIMINY_DLLAPI IndexByIndex
    {
    };
    struct JIMINY_DLLAPI IndexByName
    {
    };

    namespace internal
    {
        struct SensorMeasurementItem
        {
            std::string name;
            std::size_t index;
            Eigen::Ref<const Eigen::VectorXd> value;
        };
    }

    /// \brief Lightweight non-owning read-only accessor to the measurements of multiple sensors.
    ///
    /// \details One can either retrieve the measurement of each individual sensor, or all at once
    ///          stacked in contiguous Eigen Matrix.
    struct SensorMeasurementStack :
    public boost::multi_index::multi_index_container<
        internal::SensorMeasurementItem,
        boost::multi_index::indexed_by<
            boost::multi_index::ordered_unique<
                boost::multi_index::tag<IndexByIndex>,
                boost::multi_index::member<internal::SensorMeasurementItem,
                                           std::size_t,
                                           &internal::SensorMeasurementItem::index>,
                std::less<std::size_t>  // Ordering by ascending order
                >,
            boost::multi_index::hashed_unique<
                boost::multi_index::tag<IndexByName>,
                boost::multi_index::member<internal::SensorMeasurementItem,
                                           std::string,
                                           &internal::SensorMeasurementItem::name>>,
            boost::multi_index::sequenced<>>>
    {
    public:
        explicit SensorMeasurementStack(
            const Eigen::MatrixXd * sharedMeasurementsPtr = nullptr) noexcept :
        multi_index_container(),
        sharedMeasurementsPtr_{sharedMeasurementsPtr}
        {
        }

        /// \brief Returning data associated with all sensors at once.
        ///
        /// \warning It is up to the sure to make sure that the data are up-to-date.
        inline const Eigen::MatrixXd & getAll() const
        {
            if (sharedMeasurementsPtr_)
            {
                if (size() != static_cast<std::size_t>(sharedMeasurementsPtr_->cols()))
                {
                    throw std::logic_error(
                        "Number of sensors inconsistent with shared measurements.");
                }
                return *sharedMeasurementsPtr_;
            }
            else
            {
                // Get sensors data size
                Eigen::Index dataSize = 0;
                if (size() > 0)
                {
                    dataSize = this->cbegin()->value.size();
                }

                // Resize internal buffer if needed
                data_.resize(Eigen::NoChange, dataSize);

                // Set internal buffer by copying sensor data sequentially
                for (const auto & sensor : *this)
                {
                    if (sensor.value.size() != dataSize)
                    {
                        throw std::logic_error(
                            "Cannot get all data at once for heterogeneous sensors.");
                    }
                    data_.row(sensor.index) = sensor.value;
                }

                return data_;
            }
        }

    private:
        const Eigen::MatrixXd * const sharedMeasurementsPtr_;
        /// \brief Internal buffer used in absence of shared buffer.
        ///
        /// \details Especially useful if sensors data are not stored in a contiguous buffer
        //           in the first place, which is likely to be the case when allocated from
        ///          Python, or when emulating sensor data from log files.
        mutable Eigen::MatrixXd data_{};
    };

    using SensorMeasurementTree = std::unordered_map<std::string, SensorMeasurementStack>;
}

#endif  // JIMINY_FORWARD_SENSOR_H
