#ifndef FUNCTORS_WRAPPERS_PYTHON_H
#define FUNCTORS_WRAPPERS_PYTHON_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/control/controller_functor.h"

#define NO_IMPORT_ARRAY
#include "jiminy/python/fwd.h"
#include "jiminy/python/utilities.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ************************** FunPyWrapper ******************************

    template<typename T>
    struct InternalStorageType
    {
        using type = typename std::add_lvalue_reference_t<T>;
    };

    template<>
    struct InternalStorageType<pinocchio::Force>
    {
        using type = typename Eigen::Ref<Vector6d>;
    };

    template<typename T>
    typename InternalStorageType<T>::type setDataInternalBuffer(T * arg)
    {
        return *arg;
    }

    template<>
    typename InternalStorageType<pinocchio::Force>::type
    setDataInternalBuffer<pinocchio::Force>(pinocchio::Force * arg);

    template<typename T>
    T * createInternalBuffer()
    {
        return (new T());
    }

    template<>
    pinocchio::Force * createInternalBuffer<pinocchio::Force>();

    template<typename T>
    std::enable_if_t<std::is_arithmetic_v<T>, T> FunPyWrapperArgToPython(const T & arg)
    {
        return arg;
    }

    template<typename T>
    std::enable_if_t<is_eigen_object_v<T>, bp::handle<>> FunPyWrapperArgToPython(T & arg)
    {
        return bp::handle<>(getNumpyReference(arg));
    }

    template<typename T>
    std::enable_if_t<is_eigen_object_v<T>, bp::handle<>> FunPyWrapperArgToPython(const T & arg)
    {
        return bp::handle<>(getNumpyReference(arg));
    }

    template<typename T>
    std::enable_if_t<std::is_same_v<T, SensorMeasurementTree>,
                     boost::reference_wrapper<const SensorMeasurementTree>>
    FunPyWrapperArgToPython(const T & arg)
    {
        return boost::ref(arg);
    }

    template<typename OutputArg, typename... InputArgs>
    struct FunPyWrapper
    {
    public:
        using OutputStorageType = typename InternalStorageType<OutputArg>::type;

    public:
        // Disable copy-assignment
        FunPyWrapper & operator=(const FunPyWrapper & other) = delete;

    public:
        FunPyWrapper(const bp::object & funcPy) :
        funcPy_{funcPy},
        outPtr_(createInternalBuffer<OutputArg>()),
        outBuffer_(setDataInternalBuffer(outPtr_)),
        outPyPtr_{nullptr}
        {
            outPyPtr_ = getNumpyReference(outBuffer_);
        }

        // Copy constructor, same as the normal constructor
        FunPyWrapper(const FunPyWrapper & other) :
        funcPy_(other.funcPy_),
        outPtr_(createInternalBuffer<OutputArg>()),
        outBuffer_(setDataInternalBuffer(outPtr_)),
        outPyPtr_{nullptr}
        {
            *outPtr_ = *(other.outPtr_);
            outPyPtr_ = getNumpyReference(outBuffer_);
        }

        // Move constructor, takes a rvalue reference &&
        FunPyWrapper(FunPyWrapper && other) :
        funcPy_(other.funcPy_),
        outPtr_{nullptr},
        outBuffer_(other.outBuffer_),
        outPyPtr_{nullptr}
        {
            // Steal the resource from "other"
            outPtr_ = other.outPtr_;
            outPyPtr_ = other.outPyPtr_;

            /* "other" will soon be destroyed and its destructor will do nothing because we null
               out its resource here. */
            other.outPtr_ = nullptr;
            other.outPyPtr_ = nullptr;
        }

        // Destructor
        ~FunPyWrapper()
        {
            Py_XDECREF(outPyPtr_);
            delete outPtr_;
        }

        // Move assignment, takes a rvalue reference &&
        FunPyWrapper & operator=(FunPyWrapper && other)
        {
            /* "other" is soon going to be destroyed, so we let it destroy our current resource
               instead and we take "other"'s current resource via swapping. */
            std::swap(funcPy_, other.funcPy_);
            std::swap(outPtr_, other.outPtr_);
            std::swap(outBuffer_, other.outBuffer_);
            std::swap(outPyPtr_, other.outPyPtr_);
            return *this;
        }

        const OutputArg & operator()(const InputArgs &... args)
        {
            // Reset to 0 systematically
            PyArray_FILLWBYTE(reinterpret_cast<PyArrayObject *>(outPyPtr_), 0);
            bp::handle<> outPy(bp::borrowed(outPyPtr_));
            funcPy_(FunPyWrapperArgToPython(args)..., outPy);
            return *outPtr_;
        }

    private:
        bp::object funcPy_;
        OutputArg * outPtr_;
        OutputStorageType outBuffer_;
        PyObject * outPyPtr_;
    };

    template<typename T>
    using TimeStateFunPyWrapper = FunPyWrapper<T /* OutputType */,
                                               double /* t */,
                                               Eigen::VectorXd /* q */,
                                               Eigen::VectorXd /* v */>;

    template<typename T>
    using TimeBistateFunPyWrapper = FunPyWrapper<T /* OutputType */,
                                                 double /* t */,
                                                 Eigen::VectorXd /* q1 */,
                                                 Eigen::VectorXd /* v1 */,
                                                 Eigen::VectorXd /* q2 */,
                                                 Eigen::VectorXd /* v2 */>;

    // **************************** FunInOutPyWrapper *******************************

    template<typename Signature, typename = void>
    struct FunInOutPyWrapper;

    template<typename... Args>
    struct FunInOutPyWrapper<
        void(Args...),
        std::enable_if_t<
            std::is_same_v<select_last_t<Args...>, std::decay_t<select_last_t<Args...>> &>>>
    {
    public:
        FunInOutPyWrapper(const bp::object & funcPy) :
        funcPy_{funcPy}
        {
        }

        void operator()(Args... args)
        {
            if (!isNone_)
            {
                funcPy_(FunPyWrapperArgToPython(args)...);
            }
        }

    private:
        bp::object funcPy_;
        const bool isNone_{funcPy_.is_none()};
    };

    using ControllerFunPyWrapper = FunInOutPyWrapper<FunctionalControllerSignature>;

    // ************************** HeightmapFunPyWrapper ******************************

    enum class HeightmapType : uint8_t
    {
        CONSTANT = 0x01,
        STAIRS = 0x02,
        GENERIC = 0x03,
    };

    struct HeightmapFunPyWrapper
    {
        HeightmapFunPyWrapper(const bp::object & funcPy, HeightmapType objType) :
        heightmapType_{objType},
        handlePyPtr_{funcPy}
        {
        }

        void operator()(const Eigen::Vector2d & posFrame,
                        double & height,
                        std::optional<Eigen::Ref<Eigen::Vector3d>> normal)
        {
            switch (heightmapType_)
            {
            case HeightmapType::CONSTANT:
                height = bp::extract<double>(handlePyPtr_);
                if (normal.has_value())
                {
                    normal.value() = Eigen::Vector3d::UnitZ();
                }
                break;
            case HeightmapType::STAIRS:
                handlePyPtr_(posFrame, convertToPython(height, false));
                if (normal.has_value())
                {
                    normal.value() = Eigen::Vector3d::UnitZ();
                }
                break;
            case HeightmapType::GENERIC:
            default:
                handlePyPtr_(
                    posFrame, convertToPython(height, false), convertToPython(normal, false));
            }
        }

        HeightmapType heightmapType_;
        bp::object handlePyPtr_;
    };

    // **************************** HeightmapFunVisitor *****************************

    void exposeHeightmapFunction();
}

#endif  // FUNCTORS_PYTHON_H
