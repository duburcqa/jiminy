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
    // Sensor data holder
    namespace details
    {
        struct SensorDataItem
        {
            std::string name;
            std::size_t idx;
            Eigen::Ref<const Eigen::VectorXd> value;
        };
    }

    struct JIMINY_DLLAPI IndexByIndex
    {
    };
    struct JIMINY_DLLAPI IndexByName
    {
    };

    struct SensorDataTypeMap :
    public boost::multi_index::multi_index_container<
        details::SensorDataItem,
        boost::multi_index::indexed_by<
            boost::multi_index::ordered_unique<
                boost::multi_index::tag<IndexByIndex>,
                boost::multi_index::
                    member<details::SensorDataItem, std::size_t, &details::SensorDataItem::idx>,
                std::less<std::size_t>  // Ordering by ascending order
                >,
            boost::multi_index::hashed_unique<
                boost::multi_index::tag<IndexByName>,
                boost::multi_index::
                    member<details::SensorDataItem, std::string, &details::SensorDataItem::name>>,
            boost::multi_index::sequenced<>>>
    {
    public:
        explicit SensorDataTypeMap(const Eigen::MatrixXd * sharedDataPtr = nullptr) noexcept :
        multi_index_container(),
        sharedDataPtr_{sharedDataPtr}
        {
        }

        /// @brief Returning data associated with all sensors at once.
        ///
        /// @warning It is up to the sure to make sure that the data are up-to-date.
        inline const Eigen::MatrixXd & getAll() const
        {
            if (sharedDataPtr_)
            {
                assert((size() == static_cast<std::size_t>(sharedDataPtr_->cols())) &&
                       "Shared data inconsistent with sensors.");
                return *sharedDataPtr_;
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
                    assert(sensor.value.size() == dataSize &&
                           "Cannot get all data at once for heterogeneous sensors.");
                    data_.row(sensor.idx) = sensor.value;
                }

                return data_;
            }
        }

    private:
        const Eigen::MatrixXd * const sharedDataPtr_;
        /* Internal buffer if no shared memory available.
           It is useful if the sensors data is not contiguous in the first place,
           which is likely to be the case when allocated from Python, or when
           re-generating sensor data from log files. */
        mutable Eigen::MatrixXd data_{};
    };

    using SensorsDataMap = std::unordered_map<std::string, SensorDataTypeMap>;
}

#endif  // JIMINY_FORWARD_SENSOR_H
