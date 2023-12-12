#ifndef JIMINY_TELEMETRY_CLIENT_CLASS_H
#define JIMINY_TELEMETRY_CLIENT_CLASS_H

#include <string>
#include <memory>
#include <variant>
#include <unordered_map>

#include "jiminy/core/fwd.h"


namespace jiminy
{
    class TelemetryData;

    const std::string DEFAULT_TELEMETRY_NAMESPACE("Uninitialized Object");

    /// \brief Class to inherit for sending telemetry data.
    class JIMINY_DLLAPI TelemetrySender
    {
    public:
        DISABLE_COPY(TelemetrySender)

    public:
        template<typename T>
        using telemetry_data_pair_t = std::pair<const T * const, T * const>;

        template<typename... T>
        using telemetry_data_registry_t = std::vector<std::variant<telemetry_data_pair_t<T>...>>;

        explicit TelemetrySender();
        explicit TelemetrySender(TelemetrySender &&) = default;
        ~TelemetrySender() = default;

        /// \brief Configure the object.
        ///
        /// \remarks Should only be used when default constructor is called for delayed
        ///          configuration. Should be set before registering any entry.
        ///
        /// \param[in] telemetryDataInstance Shared pointer to the telemetry instance.
        /// \param[in] objectName Name of the object.
        void configureObject(std::shared_ptr<TelemetryData> telemetryDataInstance,
                             const std::string & objectName);

        /// \brief Register a new variable to the telemetry.
        ///
        /// \details A variable must be registered to be taken into account by the telemetry. The
        ///          user is responsible for  managing its lifetime and updating it in-place. The
        ///          telemetry sender will fetch its value when calling 'updateValues'.
        ///
        /// \param[in] fieldname Name of the field to record in the telemetry.
        /// \param[in] value Pointer to the newly recorded field.
        template<typename T>
        hresult_t registerVariable(const std::string & fieldname, const T * value);

        template<typename Derived>
        hresult_t registerVariable(const std::vector<std::string> & fieldnames,
                                   const Eigen::MatrixBase<Derived> & values);

        /// \brief Add an invariant header entry in the log file.
        ///
        /// \param[in] invariantName Name of the invariant.
        /// \param[in] value Value of the invariant.
        hresult_t registerConstant(const std::string & invariantName, const std::string & value);

        /// \brief Update all registered variables in the telemetry buffer.
        void updateValues();

        /// \brief The number of registered entries.
        uint32_t getLocalNumEntries() const;

        /// \brief The object name.
        const std::string & getObjectName() const;

    protected:
        /// \brief Name of the logged object.
        std::string objectName_;

    private:
        std::shared_ptr<TelemetryData> telemetryData_;
        /// \brief Associate each variable pointer provided by the user to their reserved position
        ///        in the contiguous storage of telemetry data.
        telemetry_data_registry_t<float64_t, int64_t> bufferPosition_;
    };
}

#include "jiminy/core/telemetry/telemetry_sender.hxx"

#endif  // JIMINY_TELEMETRY_CLIENT_CLASS_H
