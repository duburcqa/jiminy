#ifndef _WIN32
#    include <pwd.h>
#    include <unistd.h>
#    include <getopt.h>
#else
#    include <stdlib.h>
#    include <stdio.h>
#endif

#include "jiminy/core/telemetry/telemetry_recorder.h"  // `LogData`
#include "jiminy/core/utilities/helpers.h"


namespace jiminy
{
    // *************** Local Mutex/Lock mechanism ******************

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

    // ************************* Timer **************************

    Timer::Timer() noexcept
    {
        tic();
    }

    void Timer::tic() noexcept
    {
        t0 = Time::now();
        dt = 0.0;
    }

    void Timer::toc() noexcept
    {
        tf = Time::now();
        std::chrono::duration<double> timeDiff = tf - t0;
        dt = timeDiff.count();
    }

    // ************ IO file and Directory utilities **************

#ifndef _WIN32
    std::string getUserDirectory()
    {
        struct passwd * pw = getpwuid(getuid());
        return pw->pw_dir;
    }
#else
    std::string getUserDirectory()
    {
        return {getenv("USERPROFILE")};
    }
#endif

    // ******************* Telemetry utilities **********************

    bool endsWith(const std::string & text, const std::string & ending)
    {
        if (text.length() >= ending.length())
        {
            return text.compare(text.length() - ending.length(), ending.length(), ending) == 0;
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
            PRINT_ERROR("Variable '", fieldname, "' does not exist.");
            return {};
        }
        const int64_t varIdx = std::distance(firstFieldnameIt, fieldnameIt);
        const Eigen::Index numInt = logData.integerValues.rows();
        if (varIdx < numInt)
        {
            return logData.integerValues.row(varIdx).cast<double>();
        }
        return logData.floatValues.row(varIdx - numInt);
    }
}
