#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#include <getopt.h>
#else
#include <stdlib.h>
#include <stdio.h>
#endif

#include "jiminy/core/Constants.h"
#include "jiminy/core/telemetry/TelemetryData.h"
#include "jiminy/core/utilities/Helpers.h"


namespace jiminy
{
    // *************** Local Mutex/Lock mechanism ******************

    MutexLocal::MutexLocal(void) :
    isLocked_(new bool_t{false})
    {
        // Empty on purpose
    }

    MutexLocal::~MutexLocal(void)
    {
        *isLocked_ = false;
    }

    bool_t const & MutexLocal::isLocked(void) const
    {
        return *isLocked_;
    }

    LockGuardLocal::LockGuardLocal(MutexLocal & mutexLocal) :
    mutexFlag_(mutexLocal.isLocked_)
    {
        *mutexFlag_ = true;
    }

    LockGuardLocal::~LockGuardLocal(void)
    {
        *mutexFlag_ = false;
    }

    // ************************* Timer **************************

    Timer::Timer(void) :
    t0(),
    tf(),
    dt(0.0)
    {
        tic();
    }

    void Timer::tic(void)
    {
        t0 = Time::now();
        dt = 0.0;
    }

    void Timer::toc(void)
    {
        tf = Time::now();
        std::chrono::duration<float64_t> timeDiff = tf - t0;
        dt = timeDiff.count();
    }

    // ************ IO file and Directory utilities **************

    #ifndef _WIN32
    std::string getUserDirectory(void)
    {
        struct passwd *pw = getpwuid(getuid());
        return pw->pw_dir;
    }
    #else
    std::string getUserDirectory(void)
    {
        return {getenv("USERPROFILE")};
    }
    #endif

    // ******************* Telemetry utilities **********************

    bool_t endsWith(std::string const & fullString, std::string const & ending)
    {
        if (fullString.length() >= ending.length())
        {
            return fullString.compare(fullString.length() - ending.length(), ending.length(), ending) == 0;
        }
        return false;
    }

    std::vector<std::string> defaultVectorFieldnames(std::string const & baseName,
                                                     uint32_t    const & size)
    {
        std::vector<std::string> fieldnames;
        fieldnames.reserve(size);
        for (uint32_t i = 0; i < size; ++i)
        {
            fieldnames.push_back(baseName + TELEMETRY_FIELDNAME_DELIMITER + std::to_string(i));
        }
        return fieldnames;
    }

    std::string addCircumfix(std::string         fieldname,
                             std::string const & prefix,
                             std::string const & suffix,
                             std::string const & delimiter)
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

    std::vector<std::string> addCircumfix(std::vector<std::string> const & fieldnamesIn,
                                          std::string              const & prefix,
                                          std::string              const & suffix,
                                          std::string              const & delimiter)
    {
        std::vector<std::string> fieldnames;
        fieldnames.reserve(fieldnamesIn.size());
        std::transform(fieldnamesIn.begin(), fieldnamesIn.end(),
                       std::back_inserter(fieldnames),
                       [&prefix, &suffix, &delimiter](std::string const & name) -> std::string
                       {
                           return addCircumfix(name, prefix, suffix, delimiter);
                       });
        return fieldnames;
    }

    std::string removeSuffix(std::string         fieldname,
                             std::string const & suffix)
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

    std::vector<std::string> removeSuffix(std::vector<std::string> const & fieldnamesIn,
                                          std::string              const & suffix)
    {
        std::vector<std::string> fieldnames;
        fieldnames.reserve(fieldnamesIn.size());
        std::transform(fieldnamesIn.begin(), fieldnamesIn.end(),
                       std::back_inserter(fieldnames),
                       [&suffix](std::string const & name) -> std::string
                       {
                           return removeSuffix(name, suffix);
                       });
        return fieldnames;
    }

    vectorN_t getLogVariable(logData_t   const & logData,
                             std::string const & fieldname)
    {
        if (fieldname == GLOBAL_TIME)
        {
            return logData.timestamps.cast<float64_t>() * logData.timeUnit;
        }
        auto iterator = std::find(
            logData.fieldnames.begin() + 1, logData.fieldnames.end(), fieldname);
        if (iterator == logData.fieldnames.end())
        {
            PRINT_ERROR("Variable '", fieldname, "' does not exist.");
            return {};
        }
        int64_t const varIdx = std::distance(
            logData.fieldnames.begin() + 1, iterator);  // Skip GLOBAL_TIME
        Eigen::Index const numInt = logData.intData.rows();
        if (varIdx < numInt)
        {
            return logData.intData.row(varIdx).cast<float64_t>();
        }
        return logData.floatData.row(varIdx - numInt);
    }
}
