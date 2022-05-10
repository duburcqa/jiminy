#ifndef FUNCTORS_WRAPPERS_PYTHON_H
#define FUNCTORS_WRAPPERS_PYTHON_H

#include "jiminy/core/Types.h"

#include <boost/python.hpp>

#include "jiminy/python/Utilities.h"


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    // ************************** FctPyWrapper ******************************

    template<typename T>
    struct DataInternalBufferType
    {
        using type = typename std::add_lvalue_reference_t<T>;
    };

    template<>
    struct DataInternalBufferType<pinocchio::Force> {
        using type = typename Eigen::Ref<vector6_t>;
    };

    template<typename T>
    typename DataInternalBufferType<T>::type
    setDataInternalBuffer(T * arg)
    {
        return *arg;
    }

    template<>
    typename DataInternalBufferType<pinocchio::Force>::type
    setDataInternalBuffer<pinocchio::Force>(pinocchio::Force * arg);

    template<typename T>
    T * createInternalBuffer(void)
    {
        return (new T());
    }

    template<>
    pinocchio::Force * createInternalBuffer<pinocchio::Force>(void);

    template<typename T>
    std::enable_if_t<std::is_arithmetic<T>::value, T>
    FctPyWrapperArgToPython(T const & arg)
    {
        return arg;
    }

    template<typename T>
    std::enable_if_t<is_eigen_v<T>, bp::handle<> >
    FctPyWrapperArgToPython(T & arg)
    {
        return bp::handle<>(getNumpyReference(arg));
    }

    template<typename T>
    std::enable_if_t<is_eigen_v<T>, bp::handle<> >
    FctPyWrapperArgToPython(T const & arg)
    {
        return bp::handle<>(getNumpyReference(arg));
    }

    template<typename T>
    std::enable_if_t<std::is_same<T, sensorsDataMap_t>::value, boost::reference_wrapper<sensorsDataMap_t const> >
    FctPyWrapperArgToPython(T const & arg)
    {
        return boost::ref(arg);
    }

    template<typename OutputArg, typename ... InputArgs>
    struct FctPyWrapper
    {
    public:
        using OutputBufferType = typename DataInternalBufferType<OutputArg>::type;
    public:
        // Disable the copy of the class
        FctPyWrapper & operator = (FctPyWrapper const & other) = delete;

    public:
        FctPyWrapper(bp::object const & objPy) :
        funcPyPtr_(objPy),
        outPtr_(createInternalBuffer<OutputArg>()),
        outData_(setDataInternalBuffer(outPtr_)),
        outPyPtr_(nullptr)
        {
            outPyPtr_ = getNumpyReference(outData_);
        }

        // Copy constructor, same as the normal constructor
        FctPyWrapper(FctPyWrapper const & other) :
        funcPyPtr_(other.funcPyPtr_),
        outPtr_(createInternalBuffer<OutputArg>()),
        outData_(setDataInternalBuffer(outPtr_)),
        outPyPtr_(nullptr)
        {
            *outPtr_ = *(other.outPtr_);
            outPyPtr_ = getNumpyReference(outData_);
        }

        // Move constructor, takes a rvalue reference &&
        FctPyWrapper(FctPyWrapper&& other) :
        funcPyPtr_(other.funcPyPtr_),
        outPtr_(nullptr),
        outData_(other.outData_),
        outPyPtr_(nullptr)
        {
            // Steal the resource from "other"
            outPtr_ = other.outPtr_;
            outPyPtr_ = other.outPyPtr_;

            /* "other" will soon be destroyed and its destructor will
               do nothing because we null out its resource here */
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
        FctPyWrapper& operator = (FctPyWrapper&& other)
        {
            /* "other" is soon going to be destroyed, so we let it destroy our current resource
               instead and we take "other"'s current resource via swapping */
            std::swap(funcPyPtr_, other.funcPyPtr_);
            std::swap(outPtr_, other.outPtr_);
            std::swap(outData_, other.outData_);
            std::swap(outPyPtr_, other.outPyPtr_);
            return *this;
        }

        OutputArg const & operator() (InputArgs const & ... args)
        {
            PyArray_FILLWBYTE(reinterpret_cast<PyArrayObject *>(outPyPtr_), 0);  // Reset to 0 systematically
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
                                               float64_t /* t */,
                                               vectorN_t /* q */,
                                               vectorN_t /* v */>;

    template<typename T>
    using TimeBistateFctPyWrapper = FctPyWrapper<T /* OutputType */,
                                                 float64_t /* t */,
                                                 vectorN_t /* q1 */,
                                                 vectorN_t /* v1 */,
                                                 vectorN_t /* q2 */,
                                                 vectorN_t /* v2 */ >;

    // **************************** FctInOutPyWrapper *******************************

    template<typename OutputArg, typename ... InputArgs>
    struct FctInOutPyWrapper
    {
    public:
        FctInOutPyWrapper(bp::object const & objPy) : funcPyPtr_(objPy) {}
        void operator() (InputArgs const & ... argsIn,
                         vectorN_t       &     argOut)
        {
            funcPyPtr_(FctPyWrapperArgToPython(argsIn)...,
                       FctPyWrapperArgToPython(argOut));
        }
    private:
        bp::object funcPyPtr_;
    };

    using ControllerFctWrapper = FctInOutPyWrapper<vectorN_t /* OutputType */,
                                                   float64_t /* t */,
                                                   vectorN_t /* q */,
                                                   vectorN_t /* v */,
                                                   sensorsDataMap_t /* sensorsData*/>;

    using ControllerFct = std::function<void(float64_t        const & /* t */,
                                             vectorN_t        const & /* q */,
                                             vectorN_t        const & /* v */,
                                             sensorsDataMap_t const & /* sensorsData */,
                                             vectorN_t              & /* command */)>;

    // ************************** HeightmapFunctorPyWrapper ******************************

    enum class heightmapType_t : uint8_t
    {
        CONSTANT = 0x01,
        STAIRS   = 0x02,
        GENERIC  = 0x03,
    };

    struct HeightmapFunctorPyWrapper
    {
    public:
        // Disable the copy of the class
        HeightmapFunctorPyWrapper & operator = (HeightmapFunctorPyWrapper const & other) = delete;

    public:
        HeightmapFunctorPyWrapper(bp::object      const & objPy,
                                  heightmapType_t const & objType) :
        heightmapType_(objType),
        handlePyPtr_(objPy),
        out1Ptr_(new float64_t),
        out2Ptr_(new vector3_t),
        out1PyPtr_(),
        out2PyPtr_()
        {
            if (heightmapType_ == heightmapType_t::CONSTANT)
            {
                *out1Ptr_ = bp::extract<float64_t>(handlePyPtr_);
                *out2Ptr_ = vector3_t::UnitZ();
            }
            else if (heightmapType_ == heightmapType_t::STAIRS)
            {
                out1PyPtr_ = getNumpyReference(*out1Ptr_);
                *out2Ptr_ = vector3_t::UnitZ();
            }
            else if (heightmapType_ == heightmapType_t::GENERIC)
            {
                out1PyPtr_ = getNumpyReference(*out1Ptr_);
                out2PyPtr_ = getNumpyReference(*out2Ptr_);
            }
        }

        // Copy constructor, same as the normal constructor
        HeightmapFunctorPyWrapper(HeightmapFunctorPyWrapper const & other) :
        heightmapType_(other.heightmapType_),
        handlePyPtr_(other.handlePyPtr_),
        out1Ptr_(new float64_t),
        out2Ptr_(new vector3_t),
        out1PyPtr_(),
        out2PyPtr_()
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
        out1Ptr_(nullptr),
        out2Ptr_(nullptr),
        out1PyPtr_(nullptr),
        out2PyPtr_(nullptr)
        {
            // Steal the resource from "other"
            out1Ptr_ = other.out1Ptr_;
            out2Ptr_ = other.out2Ptr_;
            out1PyPtr_ = other.out1PyPtr_;
            out2PyPtr_ = other.out2PyPtr_;

            /* "other" will soon be destroyed and its destructor will
               do nothing because we null out its resource here */
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
        HeightmapFunctorPyWrapper& operator = (HeightmapFunctorPyWrapper&& other)
        {
            /* "other" is soon going to be destroyed, so we let it destroy our current resource
               instead and we take "other"'s current resource via swapping */
            std::swap(heightmapType_, other.heightmapType_);
            std::swap(handlePyPtr_, other.handlePyPtr_);
            std::swap(out1Ptr_, other.out1Ptr_);
            std::swap(out2Ptr_, other.out2Ptr_);
            std::swap(out1PyPtr_, other.out1PyPtr_);
            std::swap(out2PyPtr_, other.out2PyPtr_);
            return *this;
        }

        std::pair<float64_t, vector3_t> operator() (vector3_t const & posFrame)
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
        float64_t * out1Ptr_;
        vector3_t * out2Ptr_;
        PyObject * out1PyPtr_;
        PyObject * out2PyPtr_;
    };

    // **************************** HeightmapFunctorVisitor *****************************

    void exposeHeightmapFunctor(void);
}  // End of namespace python.
}  // End of namespace jiminy.

#endif  // FUNCTORS_PYTHON_H
