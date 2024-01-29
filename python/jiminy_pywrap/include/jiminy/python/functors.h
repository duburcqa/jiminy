#ifndef FUNCTORS_WRAPPERS_PYTHON_H
#define FUNCTORS_WRAPPERS_PYTHON_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/control/controller_functor.h"

#include "pinocchio/bindings/python/fwd.hpp"

#include "jiminy/python/utilities.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ************************** FctPyWrapper ******************************

    template<typename T>
    struct DataInternalBufferType
    {
        using type = typename std::add_lvalue_reference_t<T>;
    };

    template<>
    struct DataInternalBufferType<pinocchio::Force>
    {
        using type = typename Eigen::Ref<Vector6d>;
    };

    template<typename T>
    typename DataInternalBufferType<T>::type setDataInternalBuffer(T * arg)
    {
        return *arg;
    }

    template<>
    typename DataInternalBufferType<pinocchio::Force>::type
    setDataInternalBuffer<pinocchio::Force>(pinocchio::Force * arg);

    template<typename T>
    T * createInternalBuffer()
    {
        return (new T());
    }

    template<>
    pinocchio::Force * createInternalBuffer<pinocchio::Force>();

    template<typename T>
    std::enable_if_t<std::is_arithmetic_v<T>, T> FctPyWrapperArgToPython(const T & arg)
    {
        return arg;
    }

    template<typename T>
    std::enable_if_t<is_eigen_object_v<T>, bp::handle<>> FctPyWrapperArgToPython(T & arg)
    {
        return bp::handle<>(getNumpyReference(arg));
    }

    template<typename T>
    std::enable_if_t<is_eigen_object_v<T>, bp::handle<>> FctPyWrapperArgToPython(const T & arg)
    {
        return bp::handle<>(getNumpyReference(arg));
    }

    template<typename T>
    std::enable_if_t<std::is_same_v<T, SensorsDataMap>,
                     boost::reference_wrapper<const SensorsDataMap>>
    FctPyWrapperArgToPython(const T & arg)
    {
        return boost::ref(arg);
    }

    template<typename OutputArg, typename... InputArgs>
    struct FctPyWrapper
    {
    public:
        using OutputBufferType = typename DataInternalBufferType<OutputArg>::type;

    public:
        // Disable copy-assignment
        FctPyWrapper & operator=(const FctPyWrapper & other) = delete;

    public:
        FctPyWrapper(const bp::object & objPy) :
        funcPyPtr_{objPy},
        outPtr_(createInternalBuffer<OutputArg>()),
        outData_(setDataInternalBuffer(outPtr_)),
        outPyPtr_{nullptr}
        {
            outPyPtr_ = getNumpyReference(outData_);
        }

        // Copy constructor, same as the normal constructor
        FctPyWrapper(const FctPyWrapper & other) :
        funcPyPtr_(other.funcPyPtr_),
        outPtr_(createInternalBuffer<OutputArg>()),
        outData_(setDataInternalBuffer(outPtr_)),
        outPyPtr_{nullptr}
        {
            *outPtr_ = *(other.outPtr_);
            outPyPtr_ = getNumpyReference(outData_);
        }

        // Move constructor, takes a rvalue reference &&
        FctPyWrapper(FctPyWrapper && other) :
        funcPyPtr_(other.funcPyPtr_),
        outPtr_{nullptr},
        outData_(other.outData_),
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
        ~FctPyWrapper()
        {
            Py_XDECREF(outPyPtr_);
            delete outPtr_;
        }

        // Move assignment, takes a rvalue reference &&
        FctPyWrapper & operator=(FctPyWrapper && other)
        {
            /* "other" is soon going to be destroyed, so we let it destroy our current resource
               instead and we take "other"'s current resource via swapping. */
            std::swap(funcPyPtr_, other.funcPyPtr_);
            std::swap(outPtr_, other.outPtr_);
            std::swap(outData_, other.outData_);
            std::swap(outPyPtr_, other.outPyPtr_);
            return *this;
        }

        const OutputArg & operator()(const InputArgs &... args)
        {
            // Reset to 0 systematically
            PyArray_FILLWBYTE(reinterpret_cast<PyArrayObject *>(outPyPtr_), 0);
            bp::handle<> outPy(bp::borrowed(outPyPtr_));
            funcPyPtr_(FctPyWrapperArgToPython(args)..., outPy);
            return *outPtr_;
        }

    private:
        bp::object funcPyPtr_;
        OutputArg * outPtr_;
        OutputBufferType outData_;
        PyObject * outPyPtr_;
    };

    template<typename T>
    using TimeStateFctPyWrapper = FctPyWrapper<T /* OutputType */,
                                               double /* t */,
                                               Eigen::VectorXd /* q */,
                                               Eigen::VectorXd /* v */>;

    template<typename T>
    using TimeBistateFctPyWrapper = FctPyWrapper<T /* OutputType */,
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
        FunInOutPyWrapper(const bp::object & funPy) :
        funPy_{funPy}
        {
        }

        void operator()(Args... args)
        {
            if (!isNone_)
            {
                funPy_(FctPyWrapperArgToPython(args)...);
            }
        }

    private:
        bp::object funPy_;
        const bool isNone_{funPy_.is_none()};
    };

    using ControllerFunPyWrapper = FunInOutPyWrapper<ControllerFunctorSignature>;

    // ************************** HeightmapFunctorPyWrapper ******************************

    enum class heightmapType_t : uint8_t
    {
        CONSTANT = 0x01,
        STAIRS = 0x02,
        GENERIC = 0x03,
    };

    struct HeightmapFunctorPyWrapper
    {
        HeightmapFunctorPyWrapper(const bp::object & objPy, heightmapType_t objType) :
        heightmapType_{objType},
        handlePyPtr_{objPy}
        {
        }

        void operator()(
            const Eigen::Vector2d & posFrame, double & height, Eigen::Vector3d & normal)
        {
            switch (heightmapType_)
            {
            case heightmapType_t::CONSTANT:
                height = bp::extract<double>(handlePyPtr_);
                normal = Eigen::Vector3d::UnitZ();
                break;
            case heightmapType_t::STAIRS:
                handlePyPtr_(posFrame, convertToPython(height, false));
                normal = Eigen::Vector3d::UnitZ();
                break;
            case heightmapType_t::GENERIC:
            default:
                handlePyPtr_(
                    posFrame, convertToPython(height, false), convertToPython(normal, false));
            }
        }

        heightmapType_t heightmapType_;
        bp::object handlePyPtr_;
    };

    // **************************** HeightmapFunctorVisitor *****************************

    void exposeHeightmapFunctor();
}

#endif  // FUNCTORS_PYTHON_H
