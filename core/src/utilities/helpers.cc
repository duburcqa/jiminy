#ifndef _WIN32
#    include <pwd.h>
#    include <unistd.h>
#    include <getopt.h>
#else
#    include <stdlib.h>
#    include <stdio.h>
#endif

#include "jiminy/core/telemetry/fwd.h"  // `LogData`
#include "jiminy/core/utilities/helpers.h"


namespace jiminy
{
    // ******************************* Local Mutex/Lock mechanism ****************************** //

    MutexLocal::~MutexLocal()
    {
        *isLocked_ = false;
    }

    bool MutexLocal::isLocked() const noexcept
    {
        return *isLocked_;
    }

    LockGuardLocal::LockGuardLocal(MutexLocal & mutexLocal) noexcept :
    mutexFlag_{mutexLocal.isLocked_}
    {
        *mutexFlag_ = true;
    }

    LockGuardLocal::~LockGuardLocal()
    {
        *mutexFlag_ = false;
    }

    // ***************************************** Timer ***************************************** //

    Timer::Timer() noexcept
    {
        tic();
    }

    void Timer::tic() noexcept
    {
        t0_ = clock::now();
    }

    // **************************** IO file and Directory utilities **************************** //

    std::string getUserDirectory()
    {
#ifndef _WIN32
        struct passwd * pw = getpwuid(getuid());
        if (pw)
        {
            return {pw->pw_dir};
        }
        return {};
#else
        return {getenv("USERPROFILE")};
#endif
    }

    // ********************************** Telemetry utilities ********************************** //

    bool endsWith(const std::string & str, const std::string & substr)
    {
        const size_t strSize = str.length();
        const size_t substrSize = substr.length();
        if (strSize >= substrSize)
        {
            return str.compare(strSize - substrSize, substrSize, substr) == 0;
        }
        return false;
    }

    std::string addCircumfix(std::string fieldname,
                             const std::string_view & prefix,
                             const std::string_view & suffix,
                             const std::string_view & delimiter)
    {
        if (!prefix.empty())
        {
            fieldname = toString(prefix, delimiter, fieldname);
        }
        if (!suffix.empty())
        {
            fieldname = toString(fieldname, delimiter, suffix);
        }
        return fieldname;
    }

    std::vector<std::string> addCircumfix(const std::vector<std::string> & fieldnamesIn,
                                          const std::string_view & prefix,
                                          const std::string_view & suffix,
                                          const std::string_view & delimiter)
    {
        std::vector<std::string> fieldnames;
        fieldnames.reserve(fieldnamesIn.size());
        std::transform(fieldnamesIn.begin(),
                       fieldnamesIn.end(),
                       std::back_inserter(fieldnames),
                       [&prefix, &suffix, &delimiter](const std::string & name) -> std::string
                       { return addCircumfix(name, prefix, suffix, delimiter); });
        return fieldnames;
    }

    std::string removeSuffix(std::string fieldname, const std::string & suffix)
    {
        if (fieldname.size() > suffix.size())
        {
            if (!fieldname.compare(fieldname.size() - suffix.size(), suffix.size(), suffix))
            {
                fieldname.erase(fieldname.size() - suffix.size(), fieldname.size());
            }
        }
        return fieldname;
    }

    std::vector<std::string> removeSuffix(const std::vector<std::string> & fieldnamesIn,
                                          const std::string & suffix)
    {
        std::vector<std::string> fieldnames;
        fieldnames.reserve(fieldnamesIn.size());
        std::transform(fieldnamesIn.begin(),
                       fieldnamesIn.end(),
                       std::back_inserter(fieldnames),
                       [&suffix](const std::string & name) -> std::string
                       { return removeSuffix(name, suffix); });
        return fieldnames;
    }

    Eigen::VectorXd getLogVariable(const LogData & logData, const std::string & fieldname)
    {
        if (fieldname == GLOBAL_TIME)
        {
            return logData.times.cast<double>() * logData.timeUnit;
        }
        const auto & firstFieldnameIt = logData.variableNames.begin() + 1;  // Skip GLOBAL_TIME
        auto fieldnameIt = std::find(firstFieldnameIt, logData.variableNames.end(), fieldname);
        if (fieldnameIt == logData.variableNames.end())
        {
            THROW_ERROR(lookup_error, "Variable '", fieldname, "' does not exist.");
        }
        const int64_t varIndex = std::distance(firstFieldnameIt, fieldnameIt);
        const Eigen::Index numInt = logData.integerValues.rows();
        if (varIndex < numInt)
        {
            return logData.integerValues.row(varIndex).cast<double>();
        }
        return logData.floatValues.row(varIndex - numInt);
    }
}
