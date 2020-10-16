#include <math.h>
#include <climits>
#include <numeric>     /* iota */
#include <stdlib.h>     /* srand, rand */
#include <random>

#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#include <getopt.h>
#else
#include <stdlib.h>
#include <stdio.h>
#endif

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "jiminy/core/io/MemoryDevice.h"
#include "jiminy/core/io/JsonWriter.h"
#include "jiminy/core/io/JsonLoader.h"
#include "jiminy/core/Constants.h"

#include "jiminy/core/Utilities.h"


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

    MutexLocal::LockGuardLocal::LockGuardLocal(MutexLocal & mutexLocal) :
    mutexFlag_(mutexLocal.isLocked_)
    {
        *mutexFlag_ = true;
    }

    MutexLocal::LockGuardLocal::~LockGuardLocal(void)
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

    // *************** Convertion to JSON utilities *****************

    template<>
    Json::Value convertToJson<vectorN_t>(vectorN_t const & value)
    {
        Json::Value row(Json::arrayValue);
        for (int32_t i=0; i<value.size(); ++i)
        {
            row.append(value[i]);
        }
        return row;
    }

    template<>
    Json::Value convertToJson<matrixN_t>(matrixN_t const & value)
    {
        Json::Value mat(Json::arrayValue);
        if (value.rows() > 0)
        {
            for (int32_t i=0; i<value.rows(); ++i)
            {
                Json::Value row(Json::arrayValue);
                for (int32_t j=0; j<value.cols(); ++j)
                {
                    row.append(value(i,j));
                }
                mat.append(row);
            }
        }
        else
        {
            mat.append(Json::Value(Json::arrayValue));
        }
        return mat;
    }

    template<>
    Json::Value convertToJson<flexibleJointData_t>(flexibleJointData_t const & value)
    {
        Json::Value flex;
        flex["jointName"] = convertToJson(value.jointName);
        flex["stiffness"] = convertToJson(value.stiffness);
        flex["damping"] = convertToJson(value.damping);
        return flex;
    }

    template<>
    Json::Value convertToJson<heatMapFunctor_t>(heatMapFunctor_t const & value)
    {
        return {"not supported"};
    }

    class AppendBoostVariantToJson : public boost::static_visitor<>
    {
    public:
        explicit AppendBoostVariantToJson(Json::Value & root) :
        root_(root),
        field_()
        {
            // Empty on purpose
        }

        ~AppendBoostVariantToJson(void) = default;

        template <typename T>
        void operator()(T const & value)
        {
            root_[field_] = convertToJson(value);
        }

    public:
        Json::Value & root_;
        std::string field_;
    };

    template<>
    Json::Value convertToJson<configHolder_t>(configHolder_t const & value)
    {
        Json::Value root;
        AppendBoostVariantToJson visitor(root);
        for (auto const & option : value)
        {
            visitor.field_ = option.first;
            boost::apply_visitor(visitor, option.second);
        }
        return root;
    }

    hresult_t jsonDump(configHolder_t                    const & config,
                       std::shared_ptr<AbstractIODevice>       & device)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Create the memory device if necessary (the device is nullptr)
        if (!device)
        {
            device = std::make_shared<MemoryDevice>(0U);
        }

        // Wrapper the memory device in a JsonWriter
        JsonWriter ioWrite(device);

        // Convert the configuration in Json and write it in the device
        returnCode = ioWrite.dump(convertToJson(config));

        return returnCode;
    }

    // ************* Convertion from JSON utilities *****************

    template<>
    std::string convertFromJson<std::string>(Json::Value const & value)
    {
        return value.asString();
    }

    template<>
    bool_t convertFromJson<bool_t>(Json::Value const & value)
    {
        return value.asBool();
    }

    template<>
    int32_t convertFromJson<int32_t>(Json::Value const & value)
    {
        return value.asInt();
    }

    template<>
    uint32_t convertFromJson<uint32_t>(Json::Value const & value)
    {
        return value.asUInt();
    }

    template<>
    float64_t convertFromJson<float64_t>(Json::Value const & value)
    {
        return value.asDouble();
    }

    template<>
    vectorN_t convertFromJson<vectorN_t>(Json::Value const & value)
    {
        vectorN_t vec;
        if (value.size() > 0)
        {
            vec.resize(value.size());
            for (auto it = value.begin() ; it != value.end() ; ++it)
            {
                vec[it.index()] = convertFromJson<float64_t>(*it);
            }
        }
        return vec;
    }

    template<>
    matrixN_t convertFromJson<matrixN_t>(Json::Value const & value)
    {
        matrixN_t mat;
        if (value.size() > 0)
        {
            auto it = value.begin() ;
            if (it->size() > 0)
            {
                mat.resize(value.size(), it->size());
                for (; it != value.end() ; ++it)
                {
                    mat.row(it.index()) = convertFromJson<vectorN_t>(*it);
                }
            }
        }
        return mat;
    }

    template<>
    flexibleJointData_t convertFromJson<flexibleJointData_t>(Json::Value const & value)
    {
        return {
            flexibleJointData_t{
                convertFromJson<std::string>(value["jointName"]),
                convertFromJson<vectorN_t>(value["stiffness"]),
                convertFromJson<vectorN_t>(value["damping"])}
        };
    }

    template<>
    heatMapFunctor_t convertFromJson<heatMapFunctor_t>(Json::Value const & value)
    {
        return {
            heatMapFunctor_t(
                [](vector3_t const & pos) -> std::pair <float64_t, vector3_t>
                {
                    return {0.0, (vector3_t() << 0.0, 0.0, 1.0).finished()};
                })
        };
    }

    template<>
    configHolder_t convertFromJson<configHolder_t>(Json::Value const & value)
    {
        configHolder_t config;
        for (auto root = value.begin() ; root != value.end() ; ++root)
        {
            configField_t field;

            if (root->type() == Json::objectValue)
            {
                std::vector<std::string> keys = root->getMemberNames();
                std::vector<std::string> const stdVectorAttrib{
                    "type",
                    "value"
                };
                if (keys == stdVectorAttrib)
                {
                    std::string type = (*root)["type"].asString();
                    Json::Value data = (*root)["value"];
                    if (type == "list(string)")
                    {
                        field = convertFromJson<std::vector<std::string> >(data);
                    }
                    else if (type == "list(array)")
                    {
                        if (data.begin()->size() == 0
                         || data.begin()->begin()->type() == Json::realValue)
                        {
                            field = convertFromJson<std::vector<vectorN_t> >(data);
                        }
                        else
                        {
                            field = convertFromJson<std::vector<matrixN_t> >(data);
                        }
                    }
                    else if (type == "list(flexibility)")
                    {
                        field = convertFromJson<flexibilityConfig_t>(data);
                    }
                    else
                    {
                        std::cout << "Error - Utilities::convertFromJson - Unknown data type: "\
                                     "std::vector<" << type << ">"  << std::endl;
                        field = std::string{"ValueError"};
                    }
                }
                else
                {
                    field = convertFromJson<configHolder_t>(*root);
                }
            }
            else if (root->type() == Json::stringValue)
            {
                field = convertFromJson<std::string>(*root);
            }
            else if (root->type() == Json::booleanValue)
            {
                field = convertFromJson<bool_t>(*root);
            }
            else if (root->type() == Json::realValue)
            {
                field = convertFromJson<float64_t>(*root);
            }
            else if (root->type() == Json::uintValue)
            {
                field = convertFromJson<uint32_t>(*root);
            }
            else if (root->type() == Json::intValue)
            {
                field = convertFromJson<int32_t>(*root);
            }
            else if (root->type() == Json::arrayValue)
            {
                if (root->size() > 0)
                {
                    auto it = root->begin();
                    if (it->type() == Json::realValue)
                    {
                        field = convertFromJson<vectorN_t>(*root);
                    }
                    else if (it->type() == Json::arrayValue)
                    {
                        field = convertFromJson<matrixN_t>(*root);
                    }
                    else
                    {
                        std::cout << "Error - Utilities::convertFromJson - Unknown data type: "\
                                     "std::vector<" << it->type() << ">" << std::endl;
                        field = std::string{"ValueError"};
                    }
                }
                else
                {
                    field = vectorN_t();
                }
            }
            else
            {
                std::cout << "Error - Utilities::convertFromJson - Unknown data type: " << root->type() << std::endl;
                field = std::string{"ValueError"};
            }

            config[root.key().asString()] = field;
        }
        return config;
    }

    hresult_t jsonLoad(configHolder_t                    & config,
                       std::shared_ptr<AbstractIODevice> & device)
    {

        hresult_t returnCode = hresult_t::SUCCESS;

        JsonLoader ioRead(device);
        returnCode = ioRead.load();

        if (returnCode == hresult_t::SUCCESS)
        {
            config = convertFromJson<configHolder_t>(ioRead.getRoot());
        }

        return returnCode;
    }

    // ***************** Random number generator *****************
    // Based on Ziggurat generator by Marsaglia and Tsang (JSS, 2000)

    std::mt19937 generator_;
    std::uniform_real_distribution<float32_t> distUniform_(0.0,1.0);

    uint32_t kn[128];
    float32_t fn[128];
    float32_t wn[128];

    void r4_nor_setup(void)
    {
        float64_t const m1 = 2147483648.0;
        float64_t const vn = 9.91256303526217e-03;
        float64_t dn = 3.442619855899;
        float64_t tn = 3.442619855899;

        float64_t q = vn / exp (-0.5 * dn * dn);

        kn[0] = static_cast<uint32_t>((dn / q) * m1);
        kn[1] = 0;

        wn[0] = static_cast<float32_t>(q / m1);
        wn[127] = static_cast<float32_t>(dn / m1);

        fn[0] = 1.0f;
        fn[127] = static_cast<float32_t>(exp(-0.5 * dn * dn));

        for (uint8_t i=126; 1 <= i; i--)
        {
            dn = sqrt (-2.0 * log(vn / dn + exp(-0.5 * dn * dn)));
            kn[i+1] = static_cast<uint32_t>((dn / tn) * m1);
            tn = dn;
            fn[i] = static_cast<float32_t>(exp(-0.5 * dn * dn));
            wn[i] = static_cast<float32_t>(dn / m1);
        }
    }

    float32_t r4_uni(void)
    {
        return distUniform_(generator_);
    }

    float32_t r4_nor(void)
    {
        float32_t const r = 3.442620f;
        int32_t hz;
        uint32_t iz;
        float32_t x;
        float32_t y;

        hz = static_cast<int32_t>(generator_());
        iz = (hz & 127U);

        if (fabs(hz) < kn[iz])
        {
            return static_cast<float32_t>(hz) * wn[iz];
        }
        else
        {
            while (true)
            {
                if (iz == 0)
                {
                    while (true)
                    {
                        x = - 0.2904764f * log(r4_uni());
                        y = - log(r4_uni());
                        if (x * x <= y + y)
                        {
                            break;
                        }
                    }

                    if (hz <= 0)
                    {
                        return - r - x;
                    }
                    else
                    {
                        return + r + x;
                    }
                }

                x = static_cast<float32_t>(hz) * wn[iz];

                if (fn[iz] + r4_uni() * (fn[iz-1] - fn[iz]) < exp (-0.5f * x * x))
                {
                    return x;
                }

                hz = static_cast<int32_t>(generator_());
                iz = (hz & 127);

                if (fabs(hz) < kn[iz])
                {
                    return static_cast<float32_t>(hz) * wn[iz];
                }
            }
        }
    }

    // ************** Random number generator utilities ****************

	void resetRandGenerators(uint32_t const & seed)
	{
		srand(seed); // Eigen relies on srand for genering random matrix
        generator_.seed(seed);
        r4_nor_setup();
	}

	float64_t randUniform(float64_t const & lo,
	                      float64_t const & hi)
    {
        return lo + r4_uni() * (hi - lo);
    }

	float64_t randNormal(float64_t const & mean,
	                     float64_t const & std)
    {
        return mean + r4_nor() * std;
    }

    vectorN_t randVectorNormal(uint32_t  const & size,
                               float64_t const & mean,
                               float64_t const & std)
    {
        if (std > 0.0)
        {
            return vectorN_t::NullaryExpr(size,
            [&mean, &std] (vectorN_t::Index const &) -> float64_t
            {
                return randNormal(mean, std);
            });
        }
        else
        {
            return vectorN_t::Constant(size, mean);
        }
    }

    vectorN_t randVectorNormal(uint32_t  const & size,
                               float64_t const & std)
    {
        return randVectorNormal(size, 0, std);
    }

    vectorN_t randVectorNormal(vectorN_t const & mean,
                               vectorN_t const & std)
    {
        return vectorN_t::NullaryExpr(std.size(),
        [&mean, &std] (vectorN_t::Index const & i) -> float64_t
        {
            return randNormal(mean[i], std[i]);
        });
    }

    vectorN_t randVectorNormal(vectorN_t const & std)
    {
        return vectorN_t::NullaryExpr(std.size(),
        [&std] (vectorN_t::Index const & i) -> float64_t
        {
            return randNormal(0, std[i]);
        });
    }

    // ******************* Telemetry utilities **********************

    std::vector<std::string> defaultVectorFieldnames(std::string const & baseName,
                                                     uint32_t    const & size)
    {
        std::vector<std::string> fieldnames;
        fieldnames.reserve(size);
        for (uint32_t i=0; i<size; ++i)
        {
            fieldnames.emplace_back(std::move(baseName + TELEMETRY_DELIMITER + std::to_string(i)));
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
            std::cout << "Error - Utilities::getLogFieldValue - Field does not exist." << std::endl;
            return fieldDataEmpty;
        }

        auto start = std::find(header.begin(), header.end(), "StartColumns");
        return logData.col(std::distance(start, iterator) - 1);
    }

    // ********************** Pinocchio utilities **********************

    hresult_t getJointNameFromPositionIdx(pinocchio::Model const & model,
                                          int32_t          const & idIn,
                                          std::string            & jointNameOut)
    {
        // Iterate over all joints.
        for (int32_t i = 0; i < model.njoints; ++i)
        {
            // Get joint starting and ending index in position vector.
            int32_t startIndex = model.joints[i].idx_q();
            int32_t endIndex = startIndex + model.joints[i].nq();

            // If inIn is between start and end, we found the joint we were looking for.
            if (startIndex <= idIn && endIndex > idIn)
            {
                jointNameOut = model.names[i];
                return hresult_t::SUCCESS;
            }
        }

        std::cout << "Error - Utilities::getJointNameFromVelocityIdx - Position index out of range." << std::endl;
        return hresult_t::ERROR_BAD_INPUT;
    }

    hresult_t getJointNameFromVelocityIdx(pinocchio::Model const & model,
                                          int32_t          const & idIn,
                                          std::string            & jointNameOut)
    {
        // Iterate over all joints.
        for (int32_t i = 0; i < model.njoints; ++i)
        {
            // Get joint starting and ending index in velocity vector.
            int32_t startIndex = model.joints[i].idx_v();
            int32_t endIndex = startIndex + model.joints[i].nv();

            // If inIn is between start and end, we found the joint we were looking for.
            if (startIndex <= idIn && endIndex > idIn)
            {
                jointNameOut = model.names[i];
                return hresult_t::SUCCESS;
            }
        }

        std::cout << "Error - Utilities::getJointNameFromVelocityIdx - Velocity index out of range." << std::endl;
        return hresult_t::ERROR_BAD_INPUT;
    }

    hresult_t getJointTypeFromIdx(pinocchio::Model const & model,
                                  int32_t          const & idIn,
                                  joint_t                & jointTypeOut)
    {
        if (model.njoints < idIn - 1)
        {
            std::cout << "Error - Utilities::getJointTypeFromIdx - Joint index out of range." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        auto const & joint = model.joints[idIn];
        std::string const & jointTypeStr = joint.shortname();

        if (jointTypeStr == "JointModelFreeFlyer")
        {
            jointTypeOut = joint_t::FREE;
        }
        else if (jointTypeStr == "JointModelSpherical")
        {
            jointTypeOut = joint_t::SPHERICAL;
        }
        else if (jointTypeStr == "JointModelPlanar")
        {
            jointTypeOut = joint_t::PLANAR;
        }
        else if (jointTypeStr == "JointModelPX" ||
                 jointTypeStr == "JointModelPY" ||
                 jointTypeStr == "JointModelPZ")
        {
            jointTypeOut = joint_t::LINEAR;
        }
        else if (jointTypeStr == "JointModelRX" ||
                 jointTypeStr == "JointModelRY" ||
                 jointTypeStr == "JointModelRZ" ||
                 jointTypeStr == "JointModelRevoluteUnaligned")
        {
            jointTypeOut = joint_t::ROTARY;
        }
        else if (jointTypeStr == "JointModelRUBX" ||
                 jointTypeStr == "JointModelRUBY" ||
                 jointTypeStr == "JointModelRUBZ" ||
                 jointTypeStr == "JointModelRevoluteUnboundedUnaligned")
        {
            jointTypeOut = joint_t::ROTARY_UNBOUNDED;
        }
        else
        {
            // Unknown joint, throw an error to avoid any wrong manipulation.
            jointTypeOut = joint_t::NONE;
            std::cout << "Error - Utilities::getJointTypeFromIdx - Unknown joint type '"
                      <<  jointTypeStr << "'." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t getJointTypePositionSuffixes(joint_t                  const & jointTypeIn,
                                           std::vector<std::string>       & jointTypeSuffixesOut)
    {
        jointTypeSuffixesOut = std::vector<std::string>({std::string("")}); // If no extra discrimination is needed
        switch (jointTypeIn)
        {
        case joint_t::LINEAR:
            break;
        case joint_t::ROTARY:
            break;
        case joint_t::ROTARY_UNBOUNDED:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("Cos"),
                                                             std::string("Sin")});
            break;
        case joint_t::PLANAR:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("TransX"),
                                                             std::string("TransY"),
                                                             std::string("TransZ")});
            break;
        case joint_t::SPHERICAL:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("QuatX"),
                                                             std::string("QuatY"),
                                                             std::string("QuatZ"),
                                                             std::string("QuatW")});
            break;
        case joint_t::FREE:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("TransX"),
                                                             std::string("TransY"),
                                                             std::string("TransZ"),
                                                             std::string("QuatX"),
                                                             std::string("QuatY"),
                                                             std::string("QuatZ"),
                                                             std::string("QuatW")});
            break;
        case joint_t::NONE:
        default:
            std::cout << "Error - Utilities::getJointFieldnamesFromType - Joints of type 'NONE' do not have fieldnames." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t getJointTypeVelocitySuffixes(joint_t                  const & jointTypeIn,
                                           std::vector<std::string>       & jointTypeSuffixesOut)
    {
        jointTypeSuffixesOut = std::vector<std::string>({std::string("")}); // If no extra discrimination is needed
        switch (jointTypeIn)
        {
        case joint_t::LINEAR:
            break;
        case joint_t::ROTARY:
            break;
        case joint_t::ROTARY_UNBOUNDED:
            break;
        case joint_t::PLANAR:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("LinX"),
                                                             std::string("LinY"),
                                                             std::string("LinZ")});
            break;
        case joint_t::SPHERICAL:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("AngX"),
                                                             std::string("AngY"),
                                                             std::string("AngZ")});
            break;
        case joint_t::FREE:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("LinX"),
                                                             std::string("LinY"),
                                                             std::string("LinZ"),
                                                             std::string("AngX"),
                                                             std::string("AngY"),
                                                             std::string("AngZ")});
            break;
        case joint_t::NONE:
        default:
            std::cout << "Error - Utilities::getJointFieldnamesFromType - Joints of type 'NONE' do not have fieldnames." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t getFrameIdx(pinocchio::Model const & model,
                          std::string      const & frameName,
                          int32_t                & frameIdx)
    {
        if (!model.existFrame(frameName))
        {
            std::cout << "Error - Utilities::getFrameIdx - Frame '"
                      << frameName << "' not found in robot model." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        frameIdx = model.getFrameId(frameName);

        return hresult_t::SUCCESS;
    }

    hresult_t getFramesIdx(pinocchio::Model         const & model,
                           std::vector<std::string> const & framesNames,
                           std::vector<int32_t>           & framesIdx)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        framesIdx.resize(0);
        for (std::string const & name : framesNames)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                int32_t idx;
                returnCode = getFrameIdx(model, name, idx);
                framesIdx.push_back(std::move(idx));
            }
        }

        return returnCode;
    }

    hresult_t getBodyIdx(pinocchio::Model const & model,
                          std::string     const & bodyName,
                          int32_t               & bodyIdx)
    {
        if (!model.existBodyName(bodyName))
        {
            std::cout << "Error - Utilities::getBodyIdx - Body '"
                      << bodyName << "' not found in robot model." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        bodyIdx = model.getBodyId(bodyName);

        return hresult_t::SUCCESS;
    }

    hresult_t getBodiesIdx(pinocchio::Model         const & model,
                           std::vector<std::string> const & bodiesNames,
                           std::vector<int32_t>           & bodiesIdx)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        bodiesIdx.resize(0);
        for (std::string const & name : bodiesNames)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                int32_t idx;
                returnCode = getFrameIdx(model, name, idx);
                bodiesIdx.push_back(std::move(idx));
            }
        }

        return returnCode;
    }

    hresult_t getJointPositionIdx(pinocchio::Model     const & model,
                                  std::string          const & jointName,
                                  std::vector<int32_t>       & jointPositionIdx)
    {
        // It returns all the indices if the joint has multiple degrees of freedom

        if (!model.existJointName(jointName))
        {
            std::cout << "Error - Utilities::getJointPositionIdx - Joint '"
                      << jointName << "' not found in robot model." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        int32_t const & jointModelIdx = model.getJointId(jointName);
        int32_t const & jointPositionFirstIdx = model.joints[jointModelIdx].idx_q();
        int32_t const & jointNq = model.joints[jointModelIdx].nq();
        jointPositionIdx.resize(jointNq);
        std::iota(jointPositionIdx.begin(), jointPositionIdx.end(), jointPositionFirstIdx);

        return hresult_t::SUCCESS;
    }

    hresult_t getJointPositionIdx(pinocchio::Model const & model,
                                  std::string      const & jointName,
                                  int32_t                & jointPositionFirstIdx)
    {
        // It returns the first index even if the joint has multiple degrees of freedom

        if (!model.existJointName(jointName))
        {
            std::cout << "Error - Utilities::getJointPositionIdx - Joint '"
                      << jointName << "' not found in robot model." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        int32_t const & jointModelIdx = model.getJointId(jointName);
        jointPositionFirstIdx = model.joints[jointModelIdx].idx_q();

        return hresult_t::SUCCESS;
    }

    hresult_t getJointsPositionIdx(pinocchio::Model         const & model,
                                   std::vector<std::string> const & jointsNames,
                                   std::vector<int32_t>           & jointsPositionIdx,
                                   bool_t                   const & firstJointIdxOnly)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        jointsPositionIdx.clear();
        if (!firstJointIdxOnly)
        {
            std::vector<int32_t> jointPositionIdx;
            for (std::string const & jointName : jointsNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = getJointPositionIdx(model, jointName, jointPositionIdx);
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    jointsPositionIdx.insert(jointsPositionIdx.end(), jointPositionIdx.begin(), jointPositionIdx.end());
                }
            }
        }
        else
        {
            int32_t jointPositionIdx;
            for (std::string const & jointName : jointsNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = getJointPositionIdx(model, jointName, jointPositionIdx);
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    jointsPositionIdx.push_back(jointPositionIdx);
                }
            }
        }

        return returnCode;
    }

    hresult_t getJointModelIdx(pinocchio::Model const & model,
                               std::string      const & jointName,
                               int32_t                & jointModelIdx)
    {
        // It returns the first index even if the joint has multiple degrees of freedom

        if (!model.existJointName(jointName))
        {
            std::cout << "Error - Utilities::getJointModelIdx - Joint '"
                      << jointName << "' not found in robot model." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        jointModelIdx = model.getJointId(jointName);

        return hresult_t::SUCCESS;
    }

    hresult_t getJointsModelIdx(pinocchio::Model         const & model,
                                std::vector<std::string> const & jointsNames,
                                std::vector<int32_t>           & jointsModelIdx)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        jointsModelIdx.clear();
        int32_t jointModelIdx;
        for (std::string const & jointName : jointsNames)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = getJointModelIdx(model, jointName, jointModelIdx);
            }
            if (returnCode == hresult_t::SUCCESS)
            {
                jointsModelIdx.push_back(jointModelIdx);
            }
        }

        return returnCode;
    }

    hresult_t getJointVelocityIdx(pinocchio::Model     const & model,
                                  std::string          const & jointName,
                                  std::vector<int32_t>       & jointVelocityIdx)
    {
        // It returns all the indices if the joint has multiple degrees of freedom

        if (!model.existJointName(jointName))
        {
            std::cout << "Error - getJointVelocityIdx - Joint '"
                      << jointName << "' not found in robot model." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        int32_t const & jointModelIdx = model.getJointId(jointName);
        int32_t const & jointVelocityFirstIdx = model.joints[jointModelIdx].idx_v();
        int32_t const & jointNv = model.joints[jointModelIdx].nv();
        jointVelocityIdx.resize(jointNv);
        std::iota(jointVelocityIdx.begin(), jointVelocityIdx.end(), jointVelocityFirstIdx);

        return hresult_t::SUCCESS;
    }

    hresult_t getJointVelocityIdx(pinocchio::Model const & model,
                                  std::string      const & jointName,
                                  int32_t                & jointVelocityFirstIdx)
    {
        // It returns the first index even if the joint has multiple degrees of freedom

        if (!model.existJointName(jointName))
        {
            std::cout << "Error - getJointVelocityIdx - Joint '"
                      << jointName << "' not found in robot model." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        int32_t const & jointModelIdx = model.getJointId(jointName);
        jointVelocityFirstIdx = model.joints[jointModelIdx].idx_v();

        return hresult_t::SUCCESS;
    }

    hresult_t getJointsVelocityIdx(pinocchio::Model         const & model,
                                   std::vector<std::string> const & jointsNames,
                                   std::vector<int32_t>           & jointsVelocityIdx,
                                   bool_t                   const & firstJointIdxOnly)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        jointsVelocityIdx.clear();
        if (!firstJointIdxOnly)
        {
            std::vector<int32_t> jointVelocityIdx;
            for (std::string const & jointName : jointsNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = getJointVelocityIdx(model, jointName, jointVelocityIdx);
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    jointsVelocityIdx.insert(jointsVelocityIdx.end(), jointVelocityIdx.begin(), jointVelocityIdx.end());
                }
            }
        }
        else
        {
            int32_t jointVelocityIdx;
            for (std::string const & jointName : jointsNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = getJointVelocityIdx(model, jointName, jointVelocityIdx);
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    jointsVelocityIdx.push_back(jointVelocityIdx);
                }
            }
        }

        return returnCode;
    }

    void switchJoints(pinocchio::Model       & modelInOut,
                      uint32_t         const & firstJointIdx,
                      uint32_t         const & secondJointIdx)
    {
        // Only perform swap if firstJointIdx is less that secondJointId
        if (firstJointIdx < secondJointIdx)
        {
            // Update parents for other joints.
            for (uint32_t i = 0; i < modelInOut.parents.size(); ++i)
            {
                if (firstJointIdx == modelInOut.parents[i])
                {
                    modelInOut.parents[i] = secondJointIdx;
                }
                else if (secondJointIdx == modelInOut.parents[i])
                {
                    modelInOut.parents[i] = firstJointIdx;
                }
            }
            // Update frame parents.
            for (uint32_t i = 0; i < modelInOut.frames.size(); ++i)
            {
                if (firstJointIdx == modelInOut.frames[i].parent)
                {
                    modelInOut.frames[i].parent = secondJointIdx;
                }
                else if (secondJointIdx == modelInOut.frames[i].parent)
                {
                    modelInOut.frames[i].parent = firstJointIdx;
                }
            }
            // Update values in subtrees.
            for (uint32_t i = 0; i < modelInOut.subtrees.size(); ++i)
            {
                for (uint32_t j = 0; j < modelInOut.subtrees[i].size(); ++j)
                {
                    if (firstJointIdx == modelInOut.subtrees[i][j])
                    {
                        modelInOut.subtrees[i][j] = secondJointIdx;
                    }
                    else if (secondJointIdx == modelInOut.subtrees[i][j])
                    {
                        modelInOut.subtrees[i][j] = firstJointIdx;
                    }
                }
            }

            // Update vectors based on joint index: effortLimit, velocityLimit,
            // lowerPositionLimit and upperPositionLimit.
            swapVectorBlocks(modelInOut.effortLimit,
                             modelInOut.joints[firstJointIdx].idx_v(),
                             modelInOut.joints[firstJointIdx].nv(),
                             modelInOut.joints[secondJointIdx].idx_v(),
                             modelInOut.joints[secondJointIdx].nv());
            swapVectorBlocks(modelInOut.velocityLimit,
                             modelInOut.joints[firstJointIdx].idx_v(),
                             modelInOut.joints[firstJointIdx].nv(),
                             modelInOut.joints[secondJointIdx].idx_v(),
                             modelInOut.joints[secondJointIdx].nv());

            swapVectorBlocks(modelInOut.lowerPositionLimit,
                             modelInOut.joints[firstJointIdx].idx_q(),
                             modelInOut.joints[firstJointIdx].nq(),
                             modelInOut.joints[secondJointIdx].idx_q(),
                             modelInOut.joints[secondJointIdx].nq());
            swapVectorBlocks(modelInOut.upperPositionLimit,
                             modelInOut.joints[firstJointIdx].idx_q(),
                             modelInOut.joints[firstJointIdx].nq(),
                             modelInOut.joints[secondJointIdx].idx_q(),
                             modelInOut.joints[secondJointIdx].nq());

            // Switch elements in joint-indexed vectors:
            // parents, names, subtrees, joints, jointPlacements, inertias.
            uint32_t tempParent = modelInOut.parents[firstJointIdx];
            modelInOut.parents[firstJointIdx] = modelInOut.parents[secondJointIdx];
            modelInOut.parents[secondJointIdx] = tempParent;

            std::string tempName = modelInOut.names[firstJointIdx];
            modelInOut.names[firstJointIdx] = modelInOut.names[secondJointIdx];
            modelInOut.names[secondJointIdx] = tempName;

            std::vector<pinocchio::Index> tempSubtree = modelInOut.subtrees[firstJointIdx];
            modelInOut.subtrees[firstJointIdx] = modelInOut.subtrees[secondJointIdx];
            modelInOut.subtrees[secondJointIdx] = tempSubtree;

            pinocchio::JointModel jointTemp = modelInOut.joints[firstJointIdx];
            modelInOut.joints[firstJointIdx] = modelInOut.joints[secondJointIdx];
            modelInOut.joints[secondJointIdx] = jointTemp;

            pinocchio::SE3 tempPlacement = modelInOut.jointPlacements[firstJointIdx];
            modelInOut.jointPlacements[firstJointIdx] = modelInOut.jointPlacements[secondJointIdx];
            modelInOut.jointPlacements[secondJointIdx] = tempPlacement;

            pinocchio::Inertia tempInertia = modelInOut.inertias[firstJointIdx];
            modelInOut.inertias[firstJointIdx] = modelInOut.inertias[secondJointIdx];
            modelInOut.inertias[secondJointIdx] = tempInertia;

            /* Recompute all position and velocity indexes, as we may have
               switched joints that didn't have the same size.
               Skip 'universe' joint since it is not an actual joint. */
            uint32_t incrementalNq = 0;
            uint32_t incrementalNv = 0;
            for (uint32_t i = 1; i < modelInOut.joints.size(); ++i)
            {
                modelInOut.joints[i].setIndexes(i, incrementalNq, incrementalNv);
                incrementalNq += modelInOut.joints[i].nq();
                incrementalNv += modelInOut.joints[i].nv();
            }
        }
    }

    hresult_t insertFlexibilityInModel(pinocchio::Model       & modelInOut,
                                       std::string      const & childJointNameIn,
                                       std::string      const & newJointNameIn)
    {
        using namespace pinocchio;

        if (!modelInOut.existJointName(childJointNameIn))
        {
            std::cout << "Error - insertFlexibilityInModel - Child joint does not exist." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        int32_t const & childIdx = modelInOut.getJointId(childJointNameIn);

        // Flexible joint is placed at the same position as the child joint, in its parent frame.
        SE3 const jointPosition = modelInOut.jointPlacements[childIdx];

        // Create joint.
        int32_t const newIdx = modelInOut.addJoint(modelInOut.parents[childIdx],
                                                  JointModelSpherical(),
                                                  jointPosition,
                                                  newJointNameIn);

        // Set child joint to be a child of the new joint, at the origin.
        modelInOut.parents[childIdx] = newIdx;
        modelInOut.jointPlacements[childIdx] = SE3::Identity();

        // Add new joint to frame list.
        int32_t const & childFrameIdx = modelInOut.getFrameId(childJointNameIn);
        int32_t const & newFrameIdx = modelInOut.addJointFrame(
            newIdx, modelInOut.frames[childFrameIdx].previousFrame);

        // Update child joint previousFrame index.
        modelInOut.frames[childFrameIdx].previousFrame = newFrameIdx;

        // Update new joint subtree to include all the joints below it.
        for (uint32_t i = 0; i < modelInOut.subtrees[childIdx].size(); ++i)
        {
            modelInOut.subtrees[newIdx].push_back(modelInOut.subtrees[childIdx][i]);
        }

        /* Add weightless body.
           In practice having a zero inertia makes some of pinocchio algorithm crash,
           so we set a very small value instead: 1.0g. Anything below that creates
           numerical instability. */
        float64_t const mass = 1.0e-3;
        float64_t const length_semiaxis = 1.0;
        pinocchio::Inertia inertia = pinocchio::Inertia::FromEllipsoid(
            mass, length_semiaxis, length_semiaxis, length_semiaxis);

        modelInOut.appendBodyToJoint(newIdx, inertia, SE3::Identity());

        /* Pinocchio requires that joints are in increasing order as we move to the
            leaves of the kinematic tree. Here this is no longer the case, as an
            intermediate joint was appended at the end. We put back this joint at the
            correct position, by doing successive permutations. */
        for (int32_t i = childIdx; i < newIdx; ++i)
        {
            switchJoints(modelInOut, i, newIdx);
        }

        return hresult_t::SUCCESS;
    }

    pinocchio::Force convertForceGlobalFrameToJoint(pinocchio::Model const & model,
                                                    pinocchio::Data  const & data,
                                                    int32_t          const & frameIdx,
                                                    pinocchio::Force const & fextInGlobal)
    {
        // Compute transform from global frame to local joint frame.
        // Translation: joint_p_frame.
        // Rotation: joint_R_world
        pinocchio::SE3 joint_M_global(
            data.oMi[model.frames[frameIdx].parent].rotation().transpose(),
            model.frames[frameIdx].placement.translation());

        return joint_M_global.act(fextInGlobal);
    }

    // ********************** Math utilities *************************

    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief      Continuously differentiable piecewise-defined saturation function. More
    ///             precisely, it consists in adding fillets at the two discontinuous points:
    ///             - It is perfectly linear for `uc` between `-bevelStart` and `bevelStart`.
    ///             - It is  perfectly constant equal to mi (resp. ma) for `uc` lower than
    ///               `-bevelStop` (resp. higher than `-bevelStop`).
    ///             - Then, two arcs of a circle connect those two modes continuously between
    ///             `bevelStart` and `bevelStop` (resp. `-bevelStop` and `-bevelStart`).
    ///             See the implementation for details about how `uc`, `bevelStart` and `bevelStop`
    ///             are computed.
    ///
    ///////////////////////////////////////////////////////////////////////////////////////////////
    float64_t saturateSoft(float64_t const & in,
                           float64_t const & mi,
                           float64_t const & ma,
                           float64_t const & r)
    {
        float64_t uc, range, middle, bevelL, bevelXc, bevelYc, bevelStart, bevelStop, out;
        float64_t const alpha = M_PI/8;
        float64_t const beta = M_PI/4;

        range = ma - mi;
        middle = (ma + mi)/2;
        uc = 2*(in - middle)/range;

        bevelL = r * tan(alpha);
        bevelStart = 1 - cos(beta)*bevelL;
        bevelStop = 1 + bevelL;
        bevelXc = bevelStop;
        bevelYc = 1 - r;

        if (uc >= bevelStop)
        {
            out = ma;
        }
        else if (uc <= -bevelStop)
        {
            out = mi;
        }
        else if (uc <= bevelStart && uc >= -bevelStart)
        {
            out = in;
        }
        else if (uc > bevelStart)
        {
            out = sqrt(r * r - (uc - bevelXc) * (uc - bevelXc)) + bevelYc;
            out = 0.5 * out * range + middle;
        }
        else if (uc < -bevelStart)
        {
            out = -sqrt(r * r - (uc + bevelXc) * (uc + bevelXc)) - bevelYc;
            out = 0.5 * out * range + middle;
        }
        else
        {
            out = in;
        }
        return out;
    }

    vectorN_t clamp(Eigen::Ref<vectorN_t const> const & data,
                    float64_t                   const & minThr,
                    float64_t                   const & maxThr)
    {
        return data.unaryExpr(
        [&minThr, &maxThr](float64_t const & x) -> float64_t
        {
            return clamp(x, minThr, maxThr);
        });
    }

    float64_t clamp(float64_t const & data,
                    float64_t const & minThr,
                    float64_t const & maxThr)
    {
        if (!isnan(data))
        {
            return std::min(std::max(data, minThr), maxThr);
        }
        else
        {
            return 0.0;
        }
    }
    vectorN_t cat(std::vector<vectorN_t> const & xList)
    {
        vectorN_t xCat;

        // Initialize the vector the cumilative size
        uint32_t xCumSize = 0U;
        for (vectorN_t const & x : xList)
        {
            xCumSize += x.size();
        }
        xCat.resize(xCumSize);

        catInPlace(xList, xCat);

        return xCat;
    }

    void catInPlace(std::vector<vectorN_t> const & xList,
                    vectorN_t                    & xCat)
    {
        uint32_t xIdx = 0U;
        for (vectorN_t const & x : xList)
        {
            uint32_t const xSize = x.size();
            xCat.segment(xIdx, xSize) = x;
            xIdx += xSize;
        }
    }
}
