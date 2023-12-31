#ifndef FUNCTORS_WRAPPERS_PYTHON_H
#define FUNCTORS_WRAPPERS_PYTHON_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/hardware/abstract_sensor.h"

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
    std::enable_if_t<is_eigen_v<T>, bp::handle<>> FctPyWrapperArgToPython(T & arg)
    {
        return bp::handle<>(getNumpyReference(arg));
    }

    template<typename T>
    std::enable_if_t<is_eigen_v<T>, bp::handle<>> FctPyWrapperArgToPython(const T & arg)
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

    // **************************** FctInOutPyWrapper *******************************

    template<typename OutputArg, typename... InputArgs>
    struct FctInOutPyWrapper
    {
    public:
        FctInOutPyWrapper(const bp::object & objPy) :
        funcPyPtr_{objPy}
        {
        }
        void operator()(const InputArgs &... argsIn, Eigen::VectorXd & argOut)
        {
            funcPyPtr_(FctPyWrapperArgToPython(argsIn)..., FctPyWrapperArgToPython(argOut));
        }

    private:
        bp::object funcPyPtr_;
    };

    using ControllerFctWrapper = FctInOutPyWrapper<Eigen::VectorXd /* OutputType */,
                                                   double /* t */,
                                                   Eigen::VectorXd /* q */,
                                                   Eigen::VectorXd /* v */,
                                                   SensorsDataMap /* sensorsData*/>;

    using ControllerFct = std::function<void(double /* t */,
                                             const Eigen::VectorXd & /* q */,
                                             const Eigen::VectorXd & /* v */,
                                             const SensorsDataMap & /* sensorsData */,
                                             Eigen::VectorXd & /* command */)>;

    // ************************** HeightmapFunctorPyWrapper ******************************

    enum class heightmapType_t : uint8_t
    {
        CONSTANT = 0x01,
        STAIRS = 0x02,
        GENERIC = 0x03,
    };

    struct HeightmapFunctorPyWrapper
    {
    public:
        // Disable copy-assignment
        HeightmapFunctorPyWrapper & operator=(const HeightmapFunctorPyWrapper & other) = delete;

    public:
        HeightmapFunctorPyWrapper(const bp::object & objPy, heightmapType_t objType) :
        heightmapType_{objType},
        handlePyPtr_{objPy},
        out1Ptr_(new double),
        out2Ptr_(new Eigen::Vector3d),
        out1PyPtr_{},
        out2PyPtr_{}
        {
            if (heightmapType_ == heightmapType_t::CONSTANT)
            {
                *out1Ptr_ = bp::extract<double>(handlePyPtr_);
                *out2Ptr_ = Eigen::Vector3d::UnitZ();
            }
            else if (heightmapType_ == heightmapType_t::STAIRS)
            {
                out1PyPtr_ = getNumpyReference(*out1Ptr_);
                *out2Ptr_ = Eigen::Vector3d::UnitZ();
            }
            else if (heightmapType_ == heightmapType_t::GENERIC)
            {
                out1PyPtr_ = getNumpyReference(*out1Ptr_);
                out2PyPtr_ = getNumpyReference(*out2Ptr_);
            }
        }

        // Copy constructor, same as the normal constructor
        HeightmapFunctorPyWrapper(const HeightmapFunctorPyWrapper & other) :
        heightmapType_(other.heightmapType_),
        handlePyPtr_(other.handlePyPtr_),
        out1Ptr_(new double),
        out2Ptr_(new Eigen::Vector3d),
        out1PyPtr_{},
        out2PyPtr_{}
        {
            *out1Ptr_ = *(other.out1Ptr_);
            *out2Ptr_ = *(other.out2Ptr_);
            out1PyPtr_ = getNumpyReference(*out1Ptr_);
            out2PyPtr_ = getNumpyReference(*out2Ptr_);
        }

        // Move constructor, takes a rvalue reference &&
        HeightmapFunctorPyWrapper(HeightmapFunctorPyWrapper && other) :
        heightmapType_(other.heightmapType_),
        handlePyPtr_(other.handlePyPtr_),
        out1Ptr_{nullptr},
        out2Ptr_{nullptr},
        out1PyPtr_{nullptr},
        out2PyPtr_{nullptr}
        {
            // Steal the resource from "other"
            out1Ptr_ = other.out1Ptr_;
            out2Ptr_ = other.out2Ptr_;
            out1PyPtr_ = other.out1PyPtr_;
            out2PyPtr_ = other.out2PyPtr_;

            /* "other" will soon be destroyed and its destructor will do nothing because we null
               out its resource here. */
            other.out1Ptr_ = nullptr;
            other.out2Ptr_ = nullptr;
            other.out1PyPtr_ = nullptr;
            other.out2PyPtr_ = nullptr;
        }

        // Destructor
        ~HeightmapFunctorPyWrapper()
        {
            Py_XDECREF(out1PyPtr_);
            Py_XDECREF(out2PyPtr_);
            delete out1Ptr_;
            delete out2Ptr_;
        }

        // Move assignment, takes a rvalue reference &&
        HeightmapFunctorPyWrapper & operator=(HeightmapFunctorPyWrapper && other)
        {
            /* "other" is soon going to be destroyed, so we let it destroy our current resource
               instead and we take "other"'s current resource via swapping. */
            std::swap(heightmapType_, other.heightmapType_);
            std::swap(handlePyPtr_, other.handlePyPtr_);
            std::swap(out1Ptr_, other.out1Ptr_);
            std::swap(out2Ptr_, other.out2Ptr_);
            std::swap(out1PyPtr_, other.out1PyPtr_);
            std::swap(out2PyPtr_, other.out2PyPtr_);
            return *this;
        }

        std::pair<double, Eigen::Vector3d> operator()(const Eigen::Vector3d & posFrame)
        {
            if (heightmapType_ == heightmapType_t::STAIRS)
            {
                *out1Ptr_ = qNAN;
                bp::handle<> out1Py(bp::borrowed(out1PyPtr_));
                handlePyPtr_(posFrame[0], posFrame[1], out1Py);
            }
            else if (heightmapType_ == heightmapType_t::GENERIC)
            {
                *out1Ptr_ = qNAN;
                out2Ptr_->setConstant(qNAN);
                bp::handle<> out1Py(bp::borrowed(out1PyPtr_));
                bp::handle<> out2Py(bp::borrowed(out2PyPtr_));
                handlePyPtr_(posFrame[0], posFrame[1], out1Py, out2Py);
            }
            if (std::isnan(*out1Ptr_))
            {
                throw std::runtime_error("Heightmap height output not set.");
            }
            if ((out2Ptr_->array() != out2Ptr_->array()).any())
            {
                throw std::runtime_error("Heightmap normal output not set.");
            }
            return {*out1Ptr_, *out2Ptr_};
        }

    public:
        heightmapType_t heightmapType_;
        bp::object handlePyPtr_;

    private:
        double * out1Ptr_;
        Eigen::Vector3d * out2Ptr_;
        PyObject * out1PyPtr_;
        PyObject * out2PyPtr_;
    };

    // **************************** HeightmapFunctorVisitor *****************************

    void exposeHeightmapFunctor();
}

#endif  // FUNCTORS_PYTHON_H
