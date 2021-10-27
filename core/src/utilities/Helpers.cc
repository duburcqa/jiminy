#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#include <getopt.h>
#else
#include <stdlib.h>
#include <stdio.h>
#endif

#include "jiminy/core/Constants.h"
#include "jiminy/core/utilities/Helpers.h"


namespace jiminy
{
    // *************** Local Mutex/Lock mechanism ******************

    MutexLocal::MutexLocal(void) :
    isLocked_(new bool_t{false})
    {
        // Empty
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

    Eigen::Ref<vectorN_t const> getLogFieldValue(std::string              const & fieldName,
                                                 std::vector<std::string> const & header,
                                                 matrixN_t                const & logData)
    {
        static vectorN_t fieldDataEmpty;

        auto iterator = std::find(header.begin(), header.end(), fieldName);
        if (iterator == header.end())
        {
            PRINT_ERROR("Field does not exist.");
            return fieldDataEmpty;
        }

        auto start = std::find(header.begin(), header.end(), "StartColumns");
        return logData.col(std::distance(start, iterator) - 1);
    }
}
