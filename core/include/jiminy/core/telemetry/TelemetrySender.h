///////////////////////////////////////////////////////////////////////////////
///
/// \brief       Contains the telemetry sender interface.
///
/// \details     This file contains the telemetry sender class that will be the parent
///              of all classes that needed to send telemetry data.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_TELEMETRY_CLIENT_CLASS_H
#define JIMINY_TELEMETRY_CLIENT_CLASS_H

#include <string>
#include <unordered_map>

#include "jiminy/core/Types.h"


namespace jiminy
{
    class TelemetryData;

    std::string const DEFAULT_TELEMETRY_NAMESPACE("Uninitialized Object");

    ////////////////////////////////////////////////////////////////////////
    /// \class TelemetrySender
    /// \brief Class to inherit if you want to send telemetry data.
    ////////////////////////////////////////////////////////////////////////
    class TelemetrySender
    {
    public:
        template<typename T>
        using telemetry_data_pair_t = std::pair<T const * const, T * const>;

        template<typename ... T>
        using telemetry_data_registry_t = std::vector<std::variant<telemetry_data_pair_t<T>... > >;

        explicit TelemetrySender(void);
        ~TelemetrySender(void) = default;

        ////////////////////////////////////////////////////////////////////////
        /// \brief     Configure the object.
        ///
        /// \param[in]  telemetryDataInstance Shared pointer to the telemetry instance.
        /// \param[in]  objectName            Name of the object.
        ///
        /// \remark  Should only be used when default constructor is called for
        ///          later configuration. Should be set before registering any entry.
        ///////////////////////////////////////////////////////////////////////
        void configureObject(std::shared_ptr<TelemetryData> telemetryDataInstance,
                             std::string const & objectName);

        ////////////////////////////////////////////////////////////////////////
        /// \brief      Register a new variable to the telemetry system..
        ///
        /// \details    A variable must be registered to be taken into account by the
        ///             telemetry system. The user is responsible for  managing its
        ///             lifetime and updating it in-place. The telemetry sender will
        ///             fetch its value when calling 'updateValues'.
        ///
        /// \param[in]  fieldname   Name of the field to record in the telemetry system.
        /// \param[in]  value       Pointer to the newly recorded field.
        ////////////////////////////////////////////////////////////////////////
        template<typename T>
        hresult_t registerVariable(std::string const & fieldname,
                                   T           const * value);

        template<typename Derived>
        hresult_t registerVariable(std::vector<std::string>   const & fieldnames,
                                   Eigen::MatrixBase<Derived> const & values);

        ////////////////////////////////////////////////////////////////////////
        /// \brief     Add an invariant header entry in the log file.
        ///
        /// \param[in] invariantName  Name of the invariant.
        /// \param[in] value          Value of the invariant.
        ///////////////////////////////////////////////////////////////////////
        hresult_t registerConstant(std::string const & invariantName,
                                   std::string const & value);

        ////////////////////////////////////////////////////////////////////////
        /// \brief      Update all registered variables in the telemetry buffer.
        ////////////////////////////////////////////////////////////////////////
        void updateValues(void);

        ////////////////////////////////////////////////////////////////////////
        /// \brief     Get the number of registered entries.
        ///
        /// \return    The number of registered entries.
        ///////////////////////////////////////////////////////////////////////
        uint32_t getLocalNumEntries(void) const;

        ////////////////////////////////////////////////////////////////////////
        /// \brief     Get the object name.
        ///
        /// \return    The object name.
        ///////////////////////////////////////////////////////////////////////
        std::string const & getObjectName(void) const;

    protected:
        std::string objectName_;  ///< Name of the logged object.

    private:
        std::shared_ptr<TelemetryData> telemetryData_;
        /// \brief Associate each variable pointer provided by the user to their
        ///        reserved position in the contiguous storage of telemetry data.
        telemetry_data_registry_t<float64_t, int64_t> bufferPosition_;
    };
} // End of jiminy namespace

#include "TelemetrySender.tpp"

#endif  //  JIMINY_TELEMETRY_CLIENT_CLASS_H
