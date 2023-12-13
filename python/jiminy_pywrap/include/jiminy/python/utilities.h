#ifndef UTILITIES_PYTHON_H
#define UTILITIES_PYTHON_H

#include <variant>
#include <optional>
#include <functional>

#include "jiminy/core/fwd.h"
#include "jiminy/core/traits.h"
#include "jiminy/core/exceptions.h"
#include "jiminy/core/hardware/abstract_sensor.h"

#include <boost/mpl/vector.hpp>

#include "pinocchio/bindings/python/fwd.hpp"
#include <boost/python/numpy.hpp>
#include <boost/python/signature.hpp>
#include <boost/python/object/function_doc_signature.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>


namespace boost::python::converter
{
#define EXPECTED_PYTYPE_FOR_ARG_IS_ARRAY(type)                             \
    template<>                                                             \
    struct expected_pytype_for_arg<type>                                   \
    {                                                                      \
        static const PyTypeObject * get_pytype() { return &PyArray_Type; } \
    };

    EXPECTED_PYTYPE_FOR_ARG_IS_ARRAY(numpy::ndarray)
    EXPECTED_PYTYPE_FOR_ARG_IS_ARRAY(numpy::ndarray const)
    EXPECTED_PYTYPE_FOR_ARG_IS_ARRAY(numpy::ndarray &)
    EXPECTED_PYTYPE_FOR_ARG_IS_ARRAY(const numpy::ndarray &)
}


namespace jiminy::python
{
    namespace bp = boost::python;
    namespace np = boost::python::numpy;

    // ****************************************************************************
    // ************************ BOOST PYTHON HELPERS ******************************
    // ****************************************************************************

    template<class E>
    PyObject * createExceptionClass(const char * name, PyObject * baseTypeObj = PyExc_Exception)
    {
        const std::string scopeName = bp::extract<std::string>(bp::scope().attr("__name__"));
        std::size_t moduleNameEnd = scopeName.find('.');
        const std::string qualifiedName0 = scopeName.substr(0, moduleNameEnd) + "." + name;
        PyObject * pythonExceptionTypeObj =
            PyErr_NewException(qualifiedName0.c_str(), baseTypeObj, 0);
        if (!pythonExceptionTypeObj)
            bp::throw_error_already_set();
        bp::scope().attr(name) = bp::handle<>(bp::borrowed(pythonExceptionTypeObj));

        bp::register_exception_translator<E>([ptr = pythonExceptionTypeObj](E const & e)
                                             { PyErr_SetString(ptr, e.what()); });

        return pythonExceptionTypeObj;
    }

#define BOOST_PYTHON_VISITOR_EXPOSE(class) \
    void expose##class()                   \
    {                                      \
        Py##class##Visitor::expose();      \
    }

    template<typename R, typename... Args>
    boost::mpl::vector<R, Args...> functionToMLP(std::function<R(Args...)> /* func */)
    {
        return {};
    }

    namespace detail
    {
        static char constexpr py_signature_tag[] = "PY signature :";
        static char constexpr cpp_signature_tag[] = "C++ signature :";
    }

    template<typename WrappedClassT>
    void setFunctionWrapperModule(bp::object & func)
    {
        /* Register it to the class to fix Ipython attribute lookup, which is looking for
          '__module__' attribute, and enable Python/C++ signatures in docstring.

           The intended way to do so is to call `add_to_namespace` function. However, the previous
           registration must be deleted first to avoid being detected as an overload and
           accumulating docstrings. To avoid such hassle, a hack is used instead by overwriting the
           internal attribute of the function directly. Beware it relies on `const_cast` to getter
           returning by reference, which may break in the future. Moreover, a hack is used to get
           the docstring, which consists in adding the expected tags as function doc. It works for
           now but it is not really reliable and may break in the future too. */
        const bp::converter::registration * r =
            bp::converter::registry::query(typeid(WrappedClassT));
        assert((std::string("Class ") + typeid(WrappedClassT).name() +
                    " not registered to Boost Python.",
                r != nullptr));
        PyTypeObject * nsPtr = r->get_class_object();
        bp::object nsName(
            bp::handle<>(PyObject_GetAttrString(reinterpret_cast<PyObject *>(nsPtr), "__name__")));
        bp::objects::function * funcPtr = bp::downcast<bp::objects::function>(func.ptr());
        bp::object & nsFunc = const_cast<bp::object &>(funcPtr->get_namespace());
        nsFunc = bp::object(nsName);
        bp::object & nameFunc = const_cast<bp::object &>(funcPtr->name());
        nameFunc = bp::str("function");
        // Add actual doc after those tags, if any
        funcPtr->doc(bp::str(detail::py_signature_tag) + bp::str(detail::cpp_signature_tag));
        // auto dict = bp::handle<>(bp::borrowed(nsPtr->tp_dict));
        // bp::str funcName("force_func");
        // if (PyObject_GetItem(dict.get(), funcName.ptr()))
        // {
        //     PyObject_DelItem(dict.get(), funcName.ptr());
        // }
        // bp::object ns(bp::handle<>(bp::borrowed(nsPtr)));
        // bp::objects::add_to_namespace(ns, "force_func", func);
    }

    inline const char * py_type_str(const bp::detail::signature_element & s)
    {
        if (strncmp(s.basename, "void", 4) == 0)
        {
            static const char * none = "None";
            return none;
        }
        const PyTypeObject * py_type = s.pytype_f ? s.pytype_f() : nullptr;
        if (py_type)
        {
            return py_type->tp_name;
        }
        else
        {
            static const char * object = "object";
            return object;
        }
    }

    template<typename ReturnT, typename... Args>
    std::string getPythonSignature()
    {
        std::ostringstream stringStream;
        stringStream << "(";
        constexpr std::size_t NArgs = sizeof...(Args);
        const bp::detail::signature_element * const signature = bp::detail::signature<
            boost::mpl::vector<std::add_lvalue_reference_t<ReturnT>,
                               std::add_lvalue_reference_t<Args>...>>::elements();
        if constexpr (NArgs > 0)
        {
            stringStream << " (" << py_type_str(signature[1]) << ")self";
            for (std::size_t i = 2; i < NArgs; ++i)
            {
                stringStream << ", (" << py_type_str(signature[i]) << ")arg" << i;
            }
        }
        stringStream << ") -> ";
        /* Special handling of the return type to rely primarily on `to_python_target_type` for
           type inference instead of `expected_pytype_for_arg` as `signature_element`. */
        const PyTypeObject * py_type = bp::converter::to_python_target_type<ReturnT>::get_pytype();
        if (py_type)
        {
            stringStream << py_type->tp_name;
        }
        else
        {
            stringStream << py_type_str(signature[0]);
        }
        return stringStream.str();
    }

    template<typename D, typename... Args>
    std::string getPythonSignature(D (* /* pm */)(Args...))
    {
        return getPythonSignature<D, Args...>();
    }

    template<typename C, typename D, typename... Args>
    std::enable_if_t<std::is_member_function_pointer_v<D (C::*)(Args...)>, std::string>
    getPythonSignature(D (C::* /* pm */)(Args...))
    {
        return getPythonSignature<D, C, Args...>();
    }

    template<typename C, typename D, typename... Args>
    std::enable_if_t<std::is_member_function_pointer_v<D (C::*)(Args...) const>, std::string>
    getPythonSignature(D (C::* /* pm */)(Args...) const)
    {
        return getPythonSignature<D, C, Args...>();
    }

    template<typename C, typename D>
    std::enable_if_t<std::is_member_object_pointer_v<D C::*>, std::string>
    getPythonSignature(D C::* /* pm */)
    {
        return getPythonSignature<D, C>();
    }

    template<typename... Args>
    std::string getPythonSignaturesWithDoc(const char * const doc,
                                           std::pair<const char *, Args>... sig)
    {
        std::ostringstream stringStream;
        ((stringStream << "\n" << sig.first << getPythonSignature(sig.second)), ...);
        if (doc)
        {
            stringStream << ":\n\n" << doc;
        }
        return stringStream.str().substr(
            std::min(static_cast<size_t>(stringStream.tellp()), size_t(1)));
    }

    template<typename Get>
    std::string getPropertySignaturesWithDoc(const char * const doc, Get getMemberFuncPtr)
    {
        return getPythonSignaturesWithDoc(doc, std::pair{"fget", getMemberFuncPtr});
    }

    template<typename Get, typename Set>
    std::string getPropertySignaturesWithDoc(
        const char * const doc, Get getMemberFuncPtr, Set setMemberFuncPtr)
    {
        return getPythonSignaturesWithDoc(
            doc, std::pair{"fget", getMemberFuncPtr}, std::pair{"fset", setMemberFuncPtr});
    }

// clang-format off
    #define DEF_READONLY3(namePy, memberFuncPtr, doc) \
        def_readonly(namePy, \
                     memberFuncPtr, \
                     getPropertySignaturesWithDoc(doc, memberFuncPtr).c_str())

    #define DEF_READONLY2(namePy, memberFuncPtr) \
        DEF_READONLY3(namePy, memberFuncPtr, nullptr)

    #define ADD_PROPERTY_GET3(namePy, memberFuncPtr, doc) \
        add_property(namePy, \
                     memberFuncPtr, \
                     getPropertySignaturesWithDoc(doc, memberFuncPtr).c_str())

    #define ADD_PROPERTY_GET2(namePy, memberFuncPtr) \
        ADD_PROPERTY_GET3(namePy, memberFuncPtr, nullptr)

    #define ADD_PROPERTY_GET_WITH_POLICY4(namePy, memberFuncPtr, policy, doc) \
        add_property(namePy, \
                     bp::make_function(memberFuncPtr, policy), \
                     getPropertySignaturesWithDoc(doc, memberFuncPtr).c_str())

    #define ADD_PROPERTY_GET_WITH_POLICY3(namePy, memberFuncPtr, policy) \
        ADD_PROPERTY_GET_WITH_POLICY4(namePy, memberFuncPtr, policy, nullptr)

    #define ADD_PROPERTY_GET_SET4(namePy, getMemberFuncPtr, setMemberFuncPtr, doc) \
        add_property(namePy, \
                     getMemberFuncPtr, \
                     setMemberFuncPtr, \
                     getPropertySignaturesWithDoc(doc, getMemberFuncPtr, setMemberFuncPtr).c_str())

    #define ADD_PROPERTY_GET_SET3(namePy, getMemberFuncPtr, setMemberFuncPtr) \
        ADD_PROPERTY_GET_SET4(namePy, getMemberFuncPtr, setMemberFuncPtr, nullptr)

    #define ADD_PROPERTY_GET_SET_WITH_POLICY5(namePy, getMemberFuncPtr, getPolicy, setMemberFuncPtr, doc) \
        add_property(namePy, \
                     bp::make_function(getMemberFuncPtr, getPolicy), \
                     setMemberFuncPtr, \
                     getPropertySignaturesWithDoc(doc, getMemberFuncPtr, setMemberFuncPtr).c_str())

    #define ADD_PROPERTY_GET_SET_WITH_POLICY4(namePy, getMemberFuncPtr, getPolicy, setMemberFuncPtr) \
        ADD_PROPERTY_GET_SET_WITH_POLICY5(namePy, getMemberFuncPtr, getPolicy, setMemberFuncPtr, nullptr)

    // Get number of arguments with __NARG__
    #define __ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N
    #define __NARG_I_(...) __ARG_N(__VA_ARGS__)
    #define __RSEQ_N() 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
    #define __NARG__(...)  __NARG_I_(__VA_ARGS__, __RSEQ_N())

    // General definition for any function name to resolve overloading based on number of arguments
    #define _VFUNC_(name, n) name ## n
    #define _VFUNC(name, n) _VFUNC_(name, n)
    #define VFUNC(func, ...) _VFUNC(func, __NARG__(__VA_ARGS__)) (__VA_ARGS__)

    // Handle overloading
    #define DEF_READONLY(...) VFUNC(DEF_READONLY, __VA_ARGS__)
    #define ADD_PROPERTY_GET(...) VFUNC(ADD_PROPERTY_GET, __VA_ARGS__)
    #define ADD_PROPERTY_GET_WITH_POLICY(...) VFUNC(ADD_PROPERTY_GET_WITH_POLICY, __VA_ARGS__)
    #define ADD_PROPERTY_GET_SET(...) VFUNC(ADD_PROPERTY_GET_SET, __VA_ARGS__)
    #define ADD_PROPERTY_GET_SET_WITH_POLICY(...) VFUNC(ADD_PROPERTY_GET_SET_WITH_POLICY, __VA_ARGS__)
    // clang-format on

    // Forward declaration
    template<class Container, bool NoProxy, class DerivedPolicies>
    class vector_indexing_suite_no_contains;

    namespace detail
    {
        template<class Container, bool NoProxy>
        class final_vector_derived_policies :
        public vector_indexing_suite_no_contains<
            Container,
            NoProxy,
            bp::detail::final_vector_derived_policies<Container, NoProxy>>
        {
        };
    }

    template<class Container,
             bool NoProxy = false,
             class DerivedPolicies = detail::final_vector_derived_policies<Container, NoProxy>>
    class vector_indexing_suite_no_contains :
    public bp::vector_indexing_suite<Container, NoProxy, DerivedPolicies>
    {
    public:
        static bool contains(Container & /* container */,
                             const typename Container::value_type & /* key */)
        {
            throw std::runtime_error("Contains method not supported.");
            return false;
        }
    };

    // ****************************************************************************
    // **************************** C++ TO PYTHON *********************************
    // ****************************************************************************

    /// C++ to Python type mapping

    template<typename T>
    int getPyType()
    {
        return NPY_OBJECT;
    }
    template<>
    inline int getPyType<bool_t>()
    {
        return NPY_BOOL;
    }
    template<>
    inline int getPyType<float32_t>()
    {
        return NPY_FLOAT32;
    }
    template<>
    inline int getPyType<float64_t>()
    {
        return NPY_FLOAT64;
    }
    template<>
    inline int getPyType<int32_t>()
    {
        return NPY_INT32;
    }
    template<>
    inline int getPyType<uint32_t>()
    {
        return NPY_UINT32;
    }
    template<>
    inline int getPyType<long>()
    {
        return NPY_LONG;
    }
    template<>
    inline int getPyType<unsigned long>()
    {
        return NPY_ULONG;
    }
    template<>
    inline int getPyType<long long>()
    {
        return NPY_LONGLONG;
    }
    template<>
    inline int getPyType<unsigned long long>()
    {
        return NPY_ULONGLONG;
    }
    template<>
    inline int getPyType<std::string>()
    {
        return NPY_UNICODE;
    }

    /// Convert Eigen scalar/vector/matrix to Numpy array by reference.

    template<typename T>
    inline PyObject * getNumpyReferenceFromScalar(T & value)
    {
        return PyArray_New(
            &PyArray_Type, 0, {}, getPyType<T>(), NULL, &value, 0, NPY_ARRAY_OUT_FARRAY, NULL);
    }

    template<typename T>
    PyObject * getNumpyReferenceFromScalar(const T & value)
    {
        return PyArray_New(&PyArray_Type,
                           0,
                           {},
                           getPyType<T>(),
                           NULL,
                           const_cast<T *>(&value),
                           0,
                           NPY_ARRAY_IN_FARRAY,
                           NULL);
    }

    template<typename T, int RowsAtCompileTime>
    PyObject * getNumpyReferenceFromEigenVector(Eigen::Matrix<T, RowsAtCompileTime, 1> & value)
    {
        npy_intp dims[1] = {{value.size()}};
        return PyArray_New(&PyArray_Type,
                           1,
                           dims,
                           getPyType<T>(),
                           NULL,
                           value.data(),
                           0,
                           NPY_ARRAY_OUT_FARRAY,
                           NULL);
    }

    template<typename T, int RowsAtCompileTime>
    PyObject *
    getNumpyReferenceFromEigenVector(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, 1>> & value)
    {
        npy_intp dims[1] = {{value.size()}};
        return PyArray_New(&PyArray_Type,
                           1,
                           dims,
                           getPyType<T>(),
                           NULL,
                           value.data(),
                           0,
                           NPY_ARRAY_OUT_FARRAY,
                           NULL);
    }

    template<typename T, int RowsAtCompileTime>
    PyObject *
    getNumpyReferenceFromEigenVector(const Eigen::Matrix<T, RowsAtCompileTime, 1> & value)
    {
        npy_intp dims[1] = {{value.size()}};
        return PyArray_New(&PyArray_Type,
                           1,
                           dims,
                           getPyType<T>(),
                           NULL,
                           const_cast<T *>(value.data()),
                           0,
                           NPY_ARRAY_IN_FARRAY,
                           NULL);
    }

    template<typename T, int RowsAtCompileTime>
    PyObject * getNumpyReferenceFromEigenVector(
        const Eigen::Ref<const Eigen::Matrix<T, RowsAtCompileTime, 1>> & value)
    {
        npy_intp dims[1] = {{value.size()}};
        return PyArray_New(&PyArray_Type,
                           1,
                           dims,
                           getPyType<T>(),
                           NULL,
                           const_cast<T *>(value.data()),
                           0,
                           NPY_ARRAY_IN_FARRAY,
                           NULL);
    }

    template<typename T, int RowsAtCompileTime, int ColsAtCompileTime>
    PyObject * getNumpyReferenceFromEigenMatrix(
        Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> & value)
    {
        npy_intp dims[2] = {{value.rows()}, {value.cols()}};
        return PyArray_New(&PyArray_Type,
                           2,
                           dims,
                           getPyType<T>(),
                           NULL,
                           const_cast<T *>(value.data()),
                           0,
                           NPY_ARRAY_OUT_FARRAY,
                           NULL);
    }

    template<typename T, int RowsAtCompileTime, int ColsAtCompileTime>
    PyObject * getNumpyReferenceFromEigenMatrix(
        Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime>> & value)
    {
        npy_intp dims[2] = {{value.rows()}, {value.cols()}};
        return PyArray_New(&PyArray_Type,
                           2,
                           dims,
                           getPyType<T>(),
                           NULL,
                           value.data(),
                           0,
                           NPY_ARRAY_OUT_FARRAY,
                           NULL);
    }

    template<typename T, int RowsAtCompileTime, int ColsAtCompileTime>
    PyObject * getNumpyReferenceFromEigenMatrix(
        const Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> & value)
    {
        npy_intp dims[2] = {{value.rows()}, {value.cols()}};
        return PyArray_New(&PyArray_Type,
                           2,
                           dims,
                           getPyType<T>(),
                           NULL,
                           const_cast<T *>(value.data()),
                           0,
                           NPY_ARRAY_IN_FARRAY,
                           NULL);
    }

    /// \brief Generic converter from Eigen Matrix to Numpy array by reference.
    template<typename T>
    std::enable_if_t<!is_eigen_v<T>, PyObject *> getNumpyReference(T & value)
    {
        return getNumpyReferenceFromScalar(value);
    }

    template<typename T>
    std::enable_if_t<is_eigen_vector_v<T>, PyObject *> getNumpyReference(T & value)
    {
        return getNumpyReferenceFromEigenVector(value);
    }

    template<typename T>
    std::enable_if_t<is_eigen_v<T> && !is_eigen_vector_v<T>, PyObject *>
    getNumpyReference(T & value)
    {
        return getNumpyReferenceFromEigenMatrix(value);
    }

    template<typename T>
    std::optional<
        Eigen::Map<Eigen::Matrix<T, -1, -1>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>>
    getEigenReferenceImpl(PyArrayObject * dataPyArray)
    {
        // Check array dtype
        if (PyArray_EquivTypenums(PyArray_TYPE(dataPyArray), getPyType<T>()) == NPY_FALSE)
        {
            PRINT_ERROR("'values' input array has dtype '",
                        PyArray_TYPE(dataPyArray),
                        "' but '",
                        getPyType<T>(),
                        "' was expected.");
            return {};
        }

        // Check array number of dimensions
        switch (PyArray_NDIM(dataPyArray))
        {
        case 0:
            return {
                {static_cast<T *>(PyArray_DATA(dataPyArray)), 1, 1, {1, 1}}
            };
        case 1:
            return {
                {static_cast<T *>(PyArray_DATA(dataPyArray)),
                 PyArray_SIZE(dataPyArray),
                 1, {PyArray_SIZE(dataPyArray), 1}}
            };
        case 2:
        {
            int32_t flags = PyArray_FLAGS(dataPyArray);
            npy_intp * dataPyArrayShape = PyArray_SHAPE(dataPyArray);
            if (flags & NPY_ARRAY_C_CONTIGUOUS)
            {
                return {
                    {static_cast<T *>(PyArray_DATA(dataPyArray)),
                     dataPyArrayShape[0],
                     dataPyArrayShape[1],
                     {1, dataPyArrayShape[1]}}
                };
            }
            if (flags & NPY_ARRAY_F_CONTIGUOUS)
            {
                return {
                    {static_cast<T *>(PyArray_DATA(dataPyArray)),
                     dataPyArrayShape[0],
                     dataPyArrayShape[1],
                     {dataPyArrayShape[0], 1}}
                };
            }
            PRINT_ERROR("Numpy arrays must be either row or column contiguous.");
            return {};
        }
        default:
            PRINT_ERROR("Only 1D and 2D 'np.ndarray' are supported.");
            return {};
        }
    }

    /// \brief Generic converter from Numpy array to Eigen Matrix by reference.
    inline std::optional<std::variant<
        Eigen::
            Map<Eigen::Matrix<float64_t, -1, -1>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>,
        Eigen::
            Map<Eigen::Matrix<int64_t, -1, -1>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>>>
    getEigenReference(PyObject * dataPy)
    {
        // Check if raw Python object pointer is actually a numpy array
        if (!PyArray_Check(dataPy))
        {
            PRINT_ERROR("'values' must have type 'np.ndarray'.");
            return {};
        }

        /* Cast raw Python object pointer to numpy array.
           Note that const qualifier is not supported by PyArray_DATA. */
        PyArrayObject * dataPyArray = reinterpret_cast<PyArrayObject *>(dataPy);

        // Check array dtype
        if (PyArray_EquivTypenums(PyArray_TYPE(dataPyArray), NPY_FLOAT64) == NPY_TRUE)
        {
            return {getEigenReferenceImpl<float64_t>(dataPyArray)};
        }
        if (PyArray_EquivTypenums(PyArray_TYPE(dataPyArray), NPY_INT64) == NPY_TRUE)
        {
            return {getEigenReferenceImpl<int64_t>(dataPyArray)};
        }
        else
        {
            PRINT_ERROR("'values' input array must have dtype 'np.float64' or 'np.int64'.");
            return {};
        }
    }

    /// Convert most C++ objects into Python objects by value.

    template<typename T>
    std::enable_if_t<!is_vector_v<T> && !is_eigen_v<T> && !std::is_arithmetic_v<T> &&
                         !std::is_integral_v<T>,
                     bp::object>
    convertToPython(const T & data, const bool & copy = true)
    {
        if (copy)
        {
            return bp::object(data);
        }
        bp::to_python_indirect<T, bp::detail::make_reference_holder> converter;
        return bp::object(bp::handle<>(converter(data)));
    }

    template<typename T>
    std::enable_if_t<std::is_arithmetic_v<T> || std::is_integral_v<T>, bp::object>
    convertToPython(T & data, const bool & copy = true)
    {
        if (copy)
        {
            return bp::object(data);
        }
        return bp::object(bp::handle<>(getNumpyReference(data)));
    }

    template<typename T>
    std::enable_if_t<is_eigen_v<T>, bp::object> convertToPython(T & data, const bool & copy = true)
    {
        PyObject * vecPyPtr = getNumpyReference(data);
        if (copy)
        {
            PyObject * copyVecPyPtr = PyArray_FROM_OF(vecPyPtr, NPY_ARRAY_ENSURECOPY);
            bp::decref(vecPyPtr);
            vecPyPtr = copyVecPyPtr;
        }
        return bp::object(bp::handle<>(vecPyPtr));
    }

    template<typename T>
    std::enable_if_t<is_vector_v<T>, bp::object> convertToPython(T & data,
                                                                 const bool & copy = true)
    {
        bp::list dataPy;
        for (auto & val : data)
        {
            dataPy.append(convertToPython(val, copy));
        }
        return std::move(dataPy);
    }

    template<typename T>
    std::enable_if_t<is_vector_v<T>, bp::object> convertToPython(const T & data,
                                                                 const bool & copy = true)
    {
        bp::list dataPy;
        for (const auto & val : data)
        {
            dataPy.append(convertToPython(val, copy));
        }
        return std::move(dataPy);
    }

    template<>
    inline bp::object convertToPython(const std::string & data, const bool & copy)
    {
        if (copy)
        {
            return bp::object(data);
        }
        return bp::object(bp::handle<>(PyUnicode_FromStringAndSize(data.c_str(), data.size())));
    }

    template<>
    inline bp::object convertToPython<FlexibleJointData>(
        const FlexibleJointData & flexibleJointData, const bool & /* copy */)
    {
        bp::dict flexibilityJointDataPy;
        flexibilityJointDataPy["frameName"] = flexibleJointData.frameName;
        flexibilityJointDataPy["stiffness"] = flexibleJointData.stiffness;
        flexibilityJointDataPy["damping"] = flexibleJointData.damping;
        flexibilityJointDataPy["inertia"] = flexibleJointData.inertia;
        return std::move(flexibilityJointDataPy);
    }

    template<>
    inline bp::object convertToPython<std::pair<const std::string, SensorDataTypeMap>>(
        const std::pair<const std::string, SensorDataTypeMap> & sensorDataTypeItem,
        const bool & copy)
    {
        const auto & [sensorGroupName, sensorDataType] = sensorDataTypeItem;
        return bp::make_tuple(sensorGroupName, convertToPython(sensorDataType.getAll(), copy));
    }

    class AppendBoostVariantToPython : public boost::static_visitor<bp::object>
    {
    public:
        AppendBoostVariantToPython(const bool & copy) :
        copy_(copy)
        {
        }

        template<typename T>
        auto operator()(const T & value) const
        {
            return convertToPython(value, copy_);
        }

    public:
        bool copy_;
    };

    template<>
    inline bp::object convertToPython<GenericConfig>(const GenericConfig & config,
                                                     const bool & copy)
    {
        bp::dict configPyDict;
        AppendBoostVariantToPython visitor(copy);
        for (const auto & [key, value] : config)
        {
            configPyDict[key] = boost::apply_visitor(visitor, value);
        }
        return std::move(configPyDict);
    }

    template<typename T, bool copy = true>
    struct converterToPython
    {
        static PyObject * convert(const T & data)
        {
            return bp::incref(convertToPython<T>(data, copy).ptr());
        }

        static const PyTypeObject * get_pytype()
        {
            if constexpr (is_vector_v<T>)
            {
                return &PyList_Type;
            }
            else if constexpr (std::is_same_v<T, GenericConfig> ||
                               std::is_same_v<T, FlexibleJointData>)
            {
                return &PyDict_Type;
            }
            return bp::converter::to_python_target_type<T>::get_pytype();
        }
    };

    template<bool copy>
    struct result_converter
    {
        template<typename T,
                 typename = typename std::enable_if_t<copy || std::is_reference_v<T> ||
                                                      is_eigen_ref_v<T>>>
        struct apply
        {
            struct type
            {
                typedef typename std::remove_reference_t<T> value_type;

                PyObject * operator()(T x) const
                {
                    return bp::incref(convertToPython<value_type>(x, copy).ptr());
                }

                const PyTypeObject * get_pytype() const
                {
                    return converterToPython<value_type, copy>::get_pytype();
                }
            };
        };
    };

    // ****************************************************************************
    // **************************** PYTHON TO C++ *********************************
    // ****************************************************************************

    /// \brief Convert a 1D python list into an Eigen vector by value.
    inline Eigen::VectorXd listPyToEigenVector(const bp::list & listPy)
    {
        Eigen::VectorXd x(len(listPy));
        for (bp::ssize_t i = 0; i < len(listPy); ++i)
        {
            x[i] = bp::extract<float64_t>(listPy[i]);
        }

        return x;
    }

    /// \brief Convert a 2D python list into an Eigen matrix.
    inline Eigen::MatrixXd listPyToEigenMatrix(const bp::list & listPy)
    {
        const bp::ssize_t nRows = len(listPy);
        assert(nRows > 0 && "empty list");

        const bp::ssize_t nCols = len(bp::extract<bp::list>(listPy[0]));
        assert(nCols > 0 && "empty row");

        Eigen::MatrixXd M(nRows, nCols);
        for (bp::ssize_t i = 0; i < nRows; ++i)
        {
            bp::list row = bp::extract<bp::list>(listPy[i]);  // Beware elements are not copied.
            assert(len(row) == nCols && "wrong number of columns");
            M.row(i) = listPyToEigenVector(row);
        }

        return M;
    }

    // Convert most Python objects in C++ objects by value.

    template<typename T>
    std::enable_if_t<!is_vector_v<T> && !is_map_v<T> && !is_eigen_v<T> &&
                         !std::is_same_v<T, SensorsDataMap>,
                     T>
    convertFromPython(const bp::object & dataPy)
    {
        try
        {
            if constexpr (std::is_same_v<std::string, T>)
            {
                /* Call Python string constructor explicitly to support conversion to string for
                   objects implementing `__str__` dunder method, typically `pathlib.Path objects`.
                */
                return bp::extract<T>(bp::str(dataPy));
            }
            return bp::extract<T>(dataPy);
        }
        catch (const bp::error_already_set &)
        {
            // Must clear the error indicator, otherwise 'PyArray_Check' will fail
            PyObject *e, *v, *t;
            PyErr_Fetch(&e, &v, &t);
            PyErr_Clear();

            // The input argument may be a 0D numpy array by any chance
            if (PyArray_Check(dataPy.ptr()))
            {
                PyArrayObject * dataPyArray = reinterpret_cast<PyArrayObject *>(dataPy.ptr());
                if (PyArray_NDIM(dataPyArray) == 0)
                {
                    if (PyArray_EquivTypenums(PyArray_TYPE(dataPyArray), getPyType<T>()) ==
                        NPY_TRUE)
                    {
                        return *static_cast<T *>(PyArray_DATA(dataPyArray));
                    }
                }
            }

            // Try dealing with unsigned/signed inconsistency in last resort
            if constexpr (std::is_integral_v<T> && !std::is_same_v<bool_t, T>)
            {
                try
                {
                    if constexpr (std::is_unsigned_v<T>)
                    {
                        return bp::extract<typename std::make_signed_t<T>>(dataPy);
                    }
                    return bp::extract<typename std::make_unsigned_t<T>>(dataPy);
                }
                catch (const bp::error_already_set &)
                {
                    PyErr_Clear();
                }
            }

            // Re-throw the exception if it was impossible to handle it
            PyErr_Restore(e, v, t);
            throw;
        }
    }

    template<typename T>
    std::enable_if_t<is_eigen_v<T>, T> convertFromPython(const bp::object & dataPy)
    {
        using Scalar = typename T::Scalar;

        try
        {
            np::ndarray dataNumpy = bp::extract<np::ndarray>(dataPy);
            if (dataNumpy.get_dtype() != np::dtype::get_builtin<Scalar>())
            {
                throw std::runtime_error(
                    "Scalar type of eigen object does not match dtype of numpy object.");
            }
            Scalar * dataPtr = reinterpret_cast<Scalar *>(dataNumpy.get_data());
            const Py_intptr_t * dataShape = dataNumpy.get_shape();
            if (is_eigen_vector_v<T>)
            {
                return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(dataPtr, dataShape[0]);
            }
            return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>(
                dataPtr, dataShape[0], dataShape[1]);
        }
        catch (const bp::error_already_set &)
        {
            PyErr_Clear();
            if (is_eigen_vector_v<T>)
            {
                return listPyToEigenVector(bp::extract<bp::list>(dataPy));
            }
            return listPyToEigenMatrix(bp::extract<bp::list>(dataPy));
        }
    }

    template<>
    inline FlexibleJointData convertFromPython<FlexibleJointData>(const bp::object & dataPy)
    {
        FlexibleJointData flexData;
        const bp::dict flexDataPy = bp::extract<bp::dict>(dataPy);
        flexData.frameName = convertFromPython<std::string>(flexDataPy["frameName"]);
        flexData.stiffness = convertFromPython<Eigen::VectorXd>(flexDataPy["stiffness"]);
        flexData.damping = convertFromPython<Eigen::VectorXd>(flexDataPy["damping"]);
        flexData.inertia = convertFromPython<Eigen::VectorXd>(flexDataPy["inertia"]);
        return flexData;
    }

    template<typename T>
    std::enable_if_t<is_vector_v<T>, T> convertFromPython(const bp::object & dataPy)
    {
        T vec;
        const bp::list listPy = bp::extract<bp::list>(dataPy);
        vec.reserve(bp::len(listPy));
        for (bp::ssize_t i = 0; i < bp::len(listPy); ++i)
        {
            const bp::object itemPy = listPy[i];
            vec.push_back(std::move(convertFromPython<typename T::value_type>(itemPy)));
        }
        return vec;
    }

    template<typename T>
    std::enable_if_t<std::is_same_v<T, SensorsDataMap>, T>
    convertFromPython(const bp::object & dataPy)
    {
        SensorsDataMap data;
        bp::dict sensorsGroupsPy = bp::extract<bp::dict>(dataPy);
        bp::list sensorsGroupsNamesPy = sensorsGroupsPy.keys();
        bp::list sensorsGroupsValuesPy = sensorsGroupsPy.values();
        for (bp::ssize_t i = 0; i < bp::len(sensorsGroupsNamesPy); ++i)
        {
            SensorDataTypeMap sensorGroupData{};
            std::string sensorGroupName = bp::extract<std::string>(sensorsGroupsNamesPy[i]);
            bp::dict sensorsDataPy = bp::extract<bp::dict>(sensorsGroupsValuesPy[i]);
            bp::list sensorsNamesPy = sensorsDataPy.keys();
            bp::list sensorsValuesPy = sensorsDataPy.values();
            for (bp::ssize_t j = 0; j < bp::len(sensorsNamesPy); ++j)
            {
                std::string sensorName = bp::extract<std::string>(sensorsNamesPy[j]);
                np::ndarray sensorDataNumpy = bp::extract<np::ndarray>(sensorsValuesPy[j]);
                auto sensorData =
                    convertFromPython<Eigen::Ref<const Eigen::VectorXd>>(sensorDataNumpy);
                sensorGroupData.insert({sensorName, static_cast<std::size_t>(j), sensorData});
            }
            data.emplace(sensorGroupName, std::move(sensorGroupData));
        }
        return data;
    }

    template<typename T>
    std::enable_if_t<is_map_v<T> && !std::is_same_v<T, SensorsDataMap>, T>
    convertFromPython(const bp::object & dataPy)
    {
        using K = typename T::key_type;
        using V = typename T::mapped_type;

        T map;
        const bp::dict dictPy = bp::extract<bp::dict>(dataPy);
        bp::list keysPy = dictPy.keys();
        bp::list valuesPy = dictPy.values();
        for (bp::ssize_t i = 0; i < bp::len(keysPy); ++i)
        {
            const K key = bp::extract<K>(keysPy[i]);
            map[key] = convertFromPython<V>(valuesPy[i]);
        }
        return map;
    }

    // Forward declaration
    inline void convertFromPython(const bp::object & configPy, GenericConfig & config);

    class AppendPythonToBoostVariant : public boost::static_visitor<>
    {
    public:
        AppendPythonToBoostVariant() = default;
        ~AppendPythonToBoostVariant() = default;

        template<typename T>
        std::enable_if_t<!std::is_same_v<T, GenericConfig>, void> operator()(T & value)
        {
            value = convertFromPython<T>(*objPy_);
        }

        template<typename T>
        std::enable_if_t<std::is_same_v<T, GenericConfig>, void> operator()(T & value)
        {
            convertFromPython(*objPy_, value);
        }

    public:
        bp::object * objPy_;
    };

    void convertFromPython(const bp::object & configPy, GenericConfig & config)
    {
        bp::dict configPyDict = bp::extract<bp::dict>(configPy);
        AppendPythonToBoostVariant visitor;
        for (auto & configField : config)
        {
            const std::string & name = configField.first;
            bp::object value = configPyDict[name];
            visitor.objPy_ = &value;
            boost::apply_visitor(visitor, configField.second);
        }
    }
}

#endif  // UTILITIES_PYTHON_H
