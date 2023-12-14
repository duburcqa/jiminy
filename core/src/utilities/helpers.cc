#ifndef _WIN32
#    include <pwd.h>
#    include <unistd.h>
#    include <getopt.h>
#else
#    include <stdlib.h>
#    include <stdio.h>
#endif

#include "jiminy/core/constants.h"
#include "jiminy/core/exceptions.h"
#include "jiminy/core/telemetry/telemetry_data.h"      // `Global.Time`
#include "jiminy/core/telemetry/telemetry_recorder.h"  // `LogData`
#include "jiminy/core/utilities/helpers.h"


namespace jiminy
{
    // *************** Local Mutex/Lock mechanism ******************

    MutexLocal::MutexLocal() :
    isLocked_(new bool_t{false})
    {
    }

    MutexLocal::~MutexLocal()
    {
        *isLocked_ = false;
    }

    bool_t MutexLocal::isLocked() const
    {
        return *isLocked_;
    }

    LockGuardLocal::LockGuardLocal(MutexLocal & mutexLocal) :
    mutexFlag_(mutexLocal.isLocked_)
    {
        *mutexFlag_ = true;
    }

    LockGuardLocal::~LockGuardLocal()
    {
        *mutexFlag_ = false;
    }

    // ************************* Timer **************************

    Timer::Timer() :
    t0(),
    tf(),
    dt(0.0)
    {
        tic();
    }

    void Timer::tic()
    {
        t0 = Time::now();
        dt = 0.0;
    }

    void Timer::toc()
    {
        tf = Time::now();
        std::chrono::duration<float64_t> timeDiff = tf - t0;
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

    bool_t endsWith(const std::string & text, const std::string & ending)
    {
        if (text.length() >= ending.length())
        {
            return text.compare(text.length() - ending.length(), ending.length(), ending) == 0;
        }
        return false;
    }

    std::string addCircumfix(std::string fieldname,
                             const std::string & prefix,
                             const std::string & suffix,
                             const std::string & delimiter)
    {
        if (!prefix.empty())
        {
            fieldname = prefix + delimiter + fieldname;
        }
        if (!suffix.empty())
        {
            fieldname = fieldname + delimiter + suffix;
        }
        return fieldname;
    }

    std::vector<std::string> addCircumfix(const std::vector<std::string> & fieldnamesIn,
                                          const std::string & prefix,
                                          const std::string & suffix,
                                          const std::string & delimiter)
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
            return logData.times.cast<float64_t>() * logData.timeUnit;
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
            return logData.integerValues.row(varIdx).cast<float64_t>();
        }
        return logData.floatValues.row(varIdx - numInt);
    }
}
