#ifndef UTILITIES_PYTHON_H
#define UTILITIES_PYTHON_H

#include <functional>

#include "jiminy/core/Types.h"
#include "jiminy/core/Macros.h"

#include <boost/mpl/vector.hpp>

/* Define Python C API, consistent with eigenpy, but do NOT import
   it to avoid "multiple definitions" error. Note that eigenpy must
   be imported before boost python to avoid compilation failure. */
#define BOOST_PYTHON_NUMPY_INTERNAL
#include "eigenpy/fwd.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/object/function_doc_signature.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;
    namespace np = boost::python::numpy;

    // ****************************************************************************
    // ************************ BOOST PYTHON HELPERS ******************************
    // ****************************************************************************

    #define BOOST_PYTHON_VISITOR_EXPOSE(class) \
    void expose ## class (void) \
    { \
        Py ## class ## Visitor::expose(); \
    }

    template<typename R, typename ...Args>
    boost::mpl::vector<R, Args...> functionToMLP(std::function<R(Args...)> /* func */)
    {
        return {};
    }

    namespace detail {
        static char constexpr py_signature_tag[] = "PY signature :";
        static char constexpr cpp_signature_tag[] = "C++ signature :";
    }

    template<typename WrappedClassT>
    void setFunctionWrapperModule(bp::object & func)
    {
        /* Register it to the class to fix Ipython attribute lookup, which is looking
           for '__module__' attribute, and enable Python/C++ signatures in docstring.

           The intended way to do so is to call `add_to_namespace` function. However,
           the previous registration must be deleted first to avoid being detected as
           an overload and accumulating docstrings. To avoid such hassle, a hack is
           used instead by overwritting the internal attribute of the function directly.
           Beware it relies on `const_cast` to getter returning by reference, which may
           break in the future. Moreover, a hack is used to get the docstring, which
           consists in adding the expected tags as function doc. It works for now but
           it is not really reliable and may break in the future too. */
        bp::converter::registration const * r = bp::converter::registry::query(typeid(WrappedClassT));
        assert((std::string("Class ") + typeid(WrappedClassT).name() + " not registered to Boost Python.", r != nullptr));
        PyTypeObject * nsPtr = r->get_class_object();
        bp::object nsName(bp::handle<>(PyObject_GetAttrString(reinterpret_cast<PyObject *>(nsPtr), "__name__")));
        bp::objects::function * funcPtr = bp::downcast<bp::objects::function>(func.ptr());
        bp::object & nsFunc = const_cast<bp::object &>(funcPtr->get_namespace());
        nsFunc = bp::object(nsName);
        bp::object & nameFunc = const_cast<bp::object &>(funcPtr->name());
        nameFunc = bp::str("function");
        funcPtr->doc(bp::str(detail::py_signature_tag) + bp::str(detail::cpp_signature_tag)); // Add actual doc after those tags, if any
        // auto dict = bp::handle<>(bp::borrowed(nsPtr->tp_dict));
        // bp::str funcName("force_func");
        // if (PyObject_GetItem(dict.get(), funcName.ptr()))
        // {
        //     PyObject_DelItem(dict.get(), funcName.ptr());
        // }
        // bp::object ns(bp::handle<>(bp::borrowed(nsPtr)));
        // bp::objects::add_to_namespace(ns, "force_func", func);
    }

    // Forward declaration
    template<class Container, bool NoProxy, class DerivedPolicies>
    class vector_indexing_suite_no_contains;

    namespace detail
    {
        template<class Container, bool NoProxy>
        class final_vector_derived_policies
            : public vector_indexing_suite_no_contains<Container,
                NoProxy, bp::detail::final_vector_derived_policies<Container, NoProxy> > {};
    }

    template<class Container,
              bool NoProxy = false,
              class DerivedPolicies = detail::final_vector_derived_policies<Container, NoProxy> >
    class vector_indexing_suite_no_contains : public bp::vector_indexing_suite<Container, NoProxy, DerivedPolicies>
    {
    public:
        static bool contains(Container & /* container */,
                             typename Container::value_type const & /* key */)
        {
            throw std::runtime_error("Contains method not supported.");
            return false;
        }
    };

    // ****************************************************************************
    // **************************** C++ TO PYTHON *********************************
    // ****************************************************************************

    /// C++ to Python type mapping

    inline int getPyType(bool_t const & /* data */) { return NPY_BOOL; }
    inline int getPyType(float32_t const & /* data */) { return NPY_FLOAT32; }
    inline int getPyType(float64_t const & /* data */) { return NPY_FLOAT64; }
    inline int getPyType(int32_t const & /* data */) { return NPY_INT32; }
    inline int getPyType(uint32_t const & /* data */) { return NPY_UINT32; }
    inline int getPyType(long const & /* data */) { return NPY_LONG; }
    inline int getPyType(unsigned long const & /* data */) { return NPY_ULONG; }
    inline int getPyType(long long const & /* data */) { return NPY_LONGLONG; }
    inline int getPyType(unsigned long long const & /* data */) { return NPY_ULONGLONG; }

    /// Convert Eigen scalar/vector/matrix to Numpy array by reference.

    template<typename T>
    inline PyObject * getNumpyReferenceFromScalar(T & value)
    {
        return PyArray_SimpleNewFromData(0, {}, getPyType(value), &value);
    }

    template<typename T>
    PyObject * getNumpyReferenceFromScalar(T const & value)
    {
        PyObject * array = getNumpyReferenceFromScalar(const_cast<T &>(value));
        PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(array), NPY_ARRAY_WRITEABLE);
        return array;
    }

    template<typename T, int RowsAtCompileTime>
    PyObject * getNumpyReferenceFromEigenVector(Eigen::Matrix<T, RowsAtCompileTime, 1> & value)
    {
        npy_intp dims[1] = {npy_intp(value.size())};
        return PyArray_SimpleNewFromData(1, dims, getPyType(*value.data()), value.data());
    }

    template<typename T, int RowsAtCompileTime>
    PyObject * getNumpyReferenceFromEigenVector(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, 1> > & value)
    {
        npy_intp dims[1] = {npy_intp(value.size())};
        return PyArray_SimpleNewFromData(1, dims, getPyType(*value.data()), value.data());
    }

    template<typename T, int RowsAtCompileTime>
    PyObject * getNumpyReferenceFromEigenVector(Eigen::Matrix<T, RowsAtCompileTime, 1> const & value)
    {
        PyObject * array = getNumpyReferenceFromEigenVector(
            const_cast<Eigen::Matrix<T, RowsAtCompileTime, 1> &>(value));
        PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(array), NPY_ARRAY_WRITEABLE);
        return array;
    }

    template<typename T, int RowsAtCompileTime>
    PyObject * getNumpyReferenceFromEigenVector(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, 1> const> const & value)
    {
        npy_intp dims[1] = {npy_intp(value.size())};
        PyObject * array = PyArray_SimpleNewFromData(1, dims, getPyType(*value.data()), const_cast<T*>(value.data()));
        PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(array), NPY_ARRAY_WRITEABLE);
        return array;
    }

    template<typename T, int RowsAtCompileTime, int ColsAtCompileTime>
    PyObject * getNumpyReferenceFromEigenMatrix(Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> & value)
    {
        npy_intp dims[2] = {npy_intp(value.cols()), npy_intp(value.rows())};
        PyObject * array = PyArray_SimpleNewFromData(2, dims, getPyType(*value.data()), const_cast<T*>(value.data()));
        PyObject * arrayT = PyArray_Transpose(reinterpret_cast<PyArrayObject *>(array), NULL);
        bp::decref(array);
        return arrayT;
    }

    template<typename T, int RowsAtCompileTime, int ColsAtCompileTime>
    PyObject * getNumpyReferenceFromEigenMatrix(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> > & value)
    {
        npy_intp dims[2] = {npy_intp(value.cols()), npy_intp(value.rows())};
        PyObject * array = PyArray_SimpleNewFromData(2, dims, getPyType(*value.data()), value.data());
        PyObject * arrayT = PyArray_Transpose(reinterpret_cast<PyArrayObject *>(array), NULL);
        bp::decref(array);
        return arrayT;
    }

    template<typename T, int RowsAtCompileTime, int ColsAtCompileTime>
    PyObject * getNumpyReferenceFromEigenMatrix(Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> const & value)
    {
        PyObject * array = getNumpyReferenceFromEigenMatrix(
            const_cast<Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> &>(value));
        PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(array), NPY_ARRAY_WRITEABLE);
        return array;
    }

    /// Generic converter from Eigen Matrix to Numpy array by reference

    template<typename T>
    std::enable_if_t<!is_eigen_v<T>, PyObject *>
    getNumpyReference(T & value)
    {
        return getNumpyReferenceFromScalar(value);
    }

    template<typename T>
    std::enable_if_t<is_eigen_vector_v<T>, PyObject *>
    getNumpyReference(T & value)
    {
        return getNumpyReferenceFromEigenVector(value);
    }

    template<typename T>
    std::enable_if_t<is_eigen_v<T> && !is_eigen_vector_v<T>, PyObject *>
    getNumpyReference(T & value)
    {
        return getNumpyReferenceFromEigenMatrix(value);
    }

    // Generic convert from Numpy array to Eigen Matrix by reference
    inline std::tuple<hresult_t, Eigen::Map<matrixN_t, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > >
    getEigenReference(PyObject * dataPy)
    {
        // Define dummy reference in case of error
        static matrixN_t dummyMat;
        static Eigen::Map<matrixN_t, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > dummyRef(
            dummyMat.data(), 0, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(0, 0));

        // Check if raw Python object pointer is actually a numpy array
        if (!PyArray_Check(dataPy))
        {
            PRINT_ERROR("'values' input array must have dtype 'np.float64'.");
            return {hresult_t::ERROR_BAD_INPUT, dummyRef};
        }

        // Cast raw Python object pointer to numpy array.
        // Note that const qualifier is not supported by PyArray_DATA.
        PyArrayObject * dataPyArray = reinterpret_cast<PyArrayObject *>(dataPy);

        // Check array dtype
        if (PyArray_TYPE(dataPyArray) != NPY_FLOAT64)
        {
            PRINT_ERROR("'values' input array must have dtype 'np.float64'.");
            return {hresult_t::ERROR_BAD_INPUT, dummyRef};
        }

        // Check array number of dimensions
        int dataPyArrayNdims = PyArray_NDIM(dataPyArray);
        if (dataPyArrayNdims == 0)
        {
            Eigen::Map<matrixN_t, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > data(
                static_cast<float64_t *>(PyArray_DATA(dataPyArray)),
                1, 1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, 1));
            return {hresult_t::SUCCESS, data};
        }
        else if (dataPyArrayNdims == 1)
        {
            Eigen::Map<matrixN_t, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > data(
                static_cast<float64_t *>(PyArray_DATA(dataPyArray)),
                PyArray_SIZE(dataPyArray), 1,
                Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(PyArray_SIZE(dataPyArray), 1));
            return {hresult_t::SUCCESS, data};
        }
        else if (dataPyArrayNdims == 2)
        {
            int32_t flags = PyArray_FLAGS(dataPyArray);
            npy_intp * dataPyArrayShape = PyArray_SHAPE(dataPyArray);
            if (flags & NPY_ARRAY_C_CONTIGUOUS)
            {
                Eigen::Map<matrixN_t, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > data(
                    static_cast<float64_t *>(PyArray_DATA(dataPyArray)),
                    dataPyArrayShape[0], dataPyArrayShape[1],
                    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, dataPyArrayShape[1]));
                return {hresult_t::SUCCESS, data};
            }
            else if (flags & NPY_ARRAY_F_CONTIGUOUS)
            {
                Eigen::Map<matrixN_t, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > data(
                    static_cast<float64_t *>(PyArray_DATA(dataPyArray)),
                    dataPyArrayShape[0], dataPyArrayShape[1],
                    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(dataPyArrayShape[0], 1));
                return {hresult_t::SUCCESS, data};
            }
            else
            {
                PRINT_ERROR("Numpy arrays must be either row or column contiguous.");
                return {hresult_t::ERROR_BAD_INPUT, dummyRef};
            }
        }
        else
        {
            PRINT_ERROR("Only 1D and 2D 'np.ndarray' are supported.");
            return {hresult_t::ERROR_BAD_INPUT, dummyRef};
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// Convert most C++ objects into Python objects by value.
    ///////////////////////////////////////////////////////////////////////////////

    template<typename T>
    std::enable_if_t<!is_vector_v<T> && !is_eigen_v<T> && !std::is_arithmetic_v<T> && !std::is_integral_v<T>, bp::object>
    convertToPython(T const & data, bool const & copy = true)
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
    convertToPython(T & data, bool const & copy = true)
    {
        if (copy)
        {
            return bp::object(data);
        }
        return bp::object(bp::handle<>(getNumpyReference(data)));
    }

    template<typename T>
    std::enable_if_t<is_eigen_v<T>, bp::object>
    convertToPython(T & data, bool const & copy = true)
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
    std::enable_if_t<is_eigen_v<T>, bp::object>
    convertToPython(T const & data, bool const & copy = true)
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
    std::enable_if_t<is_vector_v<T>, bp::object>
    convertToPython(T & data, bool const & copy = true)
    {
        bp::list dataPy;
        for (auto & val : data)
        {
            dataPy.append(convertToPython(val, copy));
        }
        return std::move(dataPy);
    }

    template<typename T>
    std::enable_if_t<is_vector_v<T>, bp::object>
    convertToPython(T const & data, bool const & copy = true)
    {
        bp::list dataPy;
        for (auto const & val : data)
        {
            dataPy.append(convertToPython(val, copy));
        }
        return std::move(dataPy);
    }

    template<>
    inline bp::object convertToPython(std::string const & data, bool const & copy)
    {
        if (copy)
        {
            return bp::object(data);
        }
        return bp::object(bp::handle<>(
            PyUnicode_FromStringAndSize(data.c_str(), data.size())));
    }

    template<>
    inline bp::object convertToPython<flexibleJointData_t>(
        flexibleJointData_t const & flexibleJointData,
        bool const & /* copy */)
    {
        bp::dict flexibilityJointDataPy;
        flexibilityJointDataPy["frameName"] = flexibleJointData.frameName;
        flexibilityJointDataPy["stiffness"] = flexibleJointData.stiffness;
        flexibilityJointDataPy["damping"] = flexibleJointData.damping;
        flexibilityJointDataPy["inertia"] = flexibleJointData.inertia;
        return std::move(flexibilityJointDataPy);
    }

    class AppendBoostVariantToPython : public boost::static_visitor<bp::object>
    {
    public:
        AppendBoostVariantToPython(bool const & copy) :
        copy_(copy)
        {
            // Empty on purpose
        }

        template<typename T>
        auto operator()(T const & value) const
        {
            return convertToPython(value, copy_);
        }

    public:
        bool copy_;
    };

    template<>
    inline bp::object convertToPython<configHolder_t>(
        configHolder_t const & config,
        bool const & copy)
    {
        bp::dict configPyDict;
        AppendBoostVariantToPython visitor(copy);
        for (auto const & [key, value] : config)
        {
            configPyDict[key] = boost::apply_visitor(visitor, value);
        }
        return std::move(configPyDict);
    }

    template<typename T, bool copy = true>
    struct converterToPython
    {
        static PyObject * convert(T const & data)
        {
            return bp::incref(convertToPython<T>(data, copy).ptr());
        }

        static PyTypeObject const * get_pytype()
        {
            if constexpr (is_vector_v<T>)
            {
                return &PyList_Type;
            }
            else if constexpr (std::is_same_v<T, configHolder_t>
                            || std::is_same_v<T, flexibleJointData_t>)
            {
                return &PyDict_Type;
            }
            std::type_info const * typeId(&typeid(bp::object));
            bp::converter::registration const * r = bp::converter::registry::query(*typeId);
            return r ? r->to_python_target_type(): 0;
        }
    };

    template<bool copy>
    struct result_converter
    {
        template<typename T, typename = typename std::enable_if_t<
            copy || std::is_reference_v<T> || is_eigen_ref_v<T> > >
        struct apply
        {
            struct type
            {
                typedef typename std::remove_reference_t<T> value_type;

                PyObject * operator()(T x) const
                {
                    return bp::incref(convertToPython<value_type>(x, copy).ptr());
                }

                PyTypeObject const * get_pytype(void) const
                {
                    return converterToPython<value_type, copy>::get_pytype();
                }
            };
        };
    };

    // ****************************************************************************
    // **************************** PYTHON TO C++ *********************************
    // ****************************************************************************

    /// \brief  Convert a 1D python list into an Eigen vector by value.
    inline vectorN_t listPyToEigenVector(bp::list const & listPy)
    {
        vectorN_t x(len(listPy));
        for (bp::ssize_t i = 0; i < len(listPy); ++i)
        {
            x[i] = bp::extract<float64_t>(listPy[i]);
        }

        return x;
    }

    /// \brief  Convert a 2D python list into an Eigen matrix.
    inline matrixN_t listPyToEigenMatrix(bp::list const & listPy)
    {
        bp::ssize_t const nRows = len(listPy);
        assert(nRows > 0 && "empty list");

        bp::ssize_t const nCols = len(bp::extract<bp::list>(listPy[0]));
        assert(nCols > 0 && "empty row");

        matrixN_t M(nRows, nCols);
        for (bp::ssize_t i = 0; i < nRows; ++i)
        {
            bp::list row = bp::extract<bp::list>(listPy[i]);  // Beware it is not an actual copy
            assert(len(row) == nCols && "wrong number of columns");
            M.row(i) = listPyToEigenVector(row);
        }

        return M;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// Convert most Python objects in C++ objects by value.
    ///////////////////////////////////////////////////////////////////////////////

    template<typename T>
    std::enable_if_t<!is_vector_v<T>
                  && !is_map_v<T>
                  && !is_eigen_v<T>
                  && !(std::is_integral_v<T> && !std::is_same_v<T, bool_t>)
                  && !std::is_same_v<T, sensorsDataMap_t>, T>
    convertFromPython(bp::object const & dataPy)
    {
        return bp::extract<T>(dataPy);
    }

    template<typename T>
    std::enable_if_t<std::is_integral_v<T>
                 && !std::is_same_v<T, bool_t>, T>
    convertFromPython(bp::object const & dataPy)
    {
        std::string const optionTypePyStr =
            bp::extract<std::string>(dataPy.attr("__class__").attr("__name__"));
        if (optionTypePyStr == "ndarray")
        {
            np::ndarray dataNumpy = bp::extract<np::ndarray>(dataPy);
            return *reinterpret_cast<T const *>(dataNumpy.get_data());
        }
        else if (optionTypePyStr == "matrix")
        {
            np::matrix dataMatrix = bp::extract<np::matrix>(dataPy);
            return *reinterpret_cast<T *>(dataMatrix.get_data());
        }
        else
        {
            bp::extract<T> getIntegral(dataPy);
            if (getIntegral.check())
            {
                return getIntegral();
            }
            if (std::is_unsigned_v<T>)
            {
                return bp::extract<typename std::make_signed_t<T> >(dataPy);
            }
            return bp::extract<typename std::make_unsigned_t<T> >(dataPy);
        }
    }

    template<typename T>
    std::enable_if_t<is_eigen_v<T>, T>
    convertFromPython(bp::object const & dataPy)
    {
        using Scalar = typename T::Scalar;

        std::string const optionTypePyStr =
            bp::extract<std::string>(dataPy.attr("__class__").attr("__name__"));
        if (optionTypePyStr == "ndarray")
        {
            np::ndarray dataNumpy = bp::extract<np::ndarray>(dataPy);
            if (dataNumpy.get_dtype() != np::dtype::get_builtin<Scalar>())
            {
                throw std::string("Scalar type of eigen object does not match dtype of numpy object.");
            }
            Scalar * dataPtr = reinterpret_cast<Scalar *>(dataNumpy.get_data());
            Py_intptr_t const * dataShape = dataNumpy.get_shape();
            if (is_eigen_vector_v<T>)
            {
                return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1> >(
                    dataPtr, dataShape[0]);
            }
            else
            {
                return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >(
                    dataPtr, dataShape[0], dataShape[1]);
            }
        }
        else if (optionTypePyStr == "matrix")
        {
            np::matrix dataMatrix = bp::extract<np::matrix>(dataPy);
            if (dataMatrix.get_dtype() != np::dtype::get_builtin<Scalar>())
            {
                throw std::string("Scalar type of eigen object does not match dtype of numpy object.");
            }
            Scalar * dataPtr = reinterpret_cast<Scalar *>(dataMatrix.get_data());
            Py_intptr_t const * dataShape = dataMatrix.get_shape();
            if (is_eigen_vector_v<T>)
            {
                return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1> >(
                    dataPtr, dataShape[0]);
            }
            else
            {
                return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >(
                    dataPtr, dataShape[0], dataShape[1]);
            }
        }
        else
        {
            if (is_eigen_vector_v<T>)
            {
                return listPyToEigenVector(bp::extract<bp::list>(dataPy));
            }
            else
            {
                return listPyToEigenMatrix(bp::extract<bp::list>(dataPy));
            }
        }
    }

    template<>
    inline flexibleJointData_t convertFromPython<flexibleJointData_t>(bp::object const & dataPy)
    {
        flexibleJointData_t flexData;
        bp::dict const flexDataPy = bp::extract<bp::dict>(dataPy);
        flexData.frameName = convertFromPython<std::string>(flexDataPy["frameName"]);
        flexData.stiffness = convertFromPython<vectorN_t>(flexDataPy["stiffness"]);
        flexData.damping = convertFromPython<vectorN_t>(flexDataPy["damping"]);
        flexData.inertia = convertFromPython<vectorN_t>(flexDataPy["inertia"]);
        return flexData;
    }

    template<typename T>
    std::enable_if_t<is_vector_v<T>, T>
    convertFromPython(bp::object const & dataPy)
    {
        T vec;
        bp::list const listPy = bp::extract<bp::list>(dataPy);
        vec.reserve(bp::len(listPy));
        for (bp::ssize_t i=0; i < bp::len(listPy); ++i)
        {
            bp::object const itemPy = listPy[i];
            vec.push_back(std::move(convertFromPython<typename T::value_type>(itemPy)));
        }
        return vec;
    }

    template<typename T>
    std::enable_if_t<std::is_same_v<T, sensorsDataMap_t>, T>
    convertFromPython(bp::object const & dataPy)
    {
        sensorsDataMap_t data;
        bp::dict sensorsGroupsPy = bp::extract<bp::dict>(dataPy);
        bp::list sensorsGroupsNamesPy = sensorsGroupsPy.keys();
        bp::list sensorsGroupsValuesPy = sensorsGroupsPy.values();
        for (bp::ssize_t i=0; i < bp::len(sensorsGroupsNamesPy); ++i)
        {
            sensorDataTypeMap_t sensorGroupData;
            std::string sensorGroupName = bp::extract<std::string>(sensorsGroupsNamesPy[i]);
            bp::dict sensorsDataPy = bp::extract<bp::dict>(sensorsGroupsValuesPy[i]);
            bp::list sensorsNamesPy = sensorsDataPy.keys();
            bp::list sensorsValuesPy = sensorsDataPy.values();
            for (bp::ssize_t j=0; j < bp::len(sensorsNamesPy); ++j)
            {
                std::string sensorName = bp::extract<std::string>(sensorsNamesPy[j]);
                np::ndarray sensorDataNumpy = bp::extract<np::ndarray>(sensorsValuesPy[j]);
                auto sensorData = convertFromPython<Eigen::Ref<vectorN_t const> >(sensorDataNumpy);
                sensorGroupData.emplace(sensorName, j, sensorData);
            }
            data.emplace(sensorGroupName, std::move(sensorGroupData));
        }
        return data;
    }

    template<typename T>
    std::enable_if_t<is_map_v<T>
                  && !std::is_same_v<T, sensorsDataMap_t>, T>
    convertFromPython(bp::object const & dataPy)
    {
        using K = typename T::key_type;
        using V = typename T::mapped_type;

        T map;
        bp::dict const dictPy = bp::extract<bp::dict>(dataPy);
        bp::list keysPy = dictPy.keys();
        bp::list valuesPy = dictPy.values();
        for (bp::ssize_t i=0; i < bp::len(keysPy); ++i)
        {
            K const key = bp::extract<K>(keysPy[i]);
            map[key] = convertFromPython<V>(valuesPy[i]);
        }
        return map;
    }

    inline void convertFromPython(bp::object const & configPy, configHolder_t & config);  // Forward declaration

    class AppendPythonToBoostVariant : public boost::static_visitor<>
    {
    public:
        AppendPythonToBoostVariant(void) = default;
        ~AppendPythonToBoostVariant(void) = default;

        template<typename T>
        std::enable_if_t<!std::is_same_v<T, configHolder_t>, void>
        operator()(T & value)
        {
            value = convertFromPython<T>(*objPy_);
        }

        template<typename T>
        std::enable_if_t<std::is_same_v<T, configHolder_t>, void>
        operator()(T & value)
        {
            convertFromPython(*objPy_, value);
        }

    public:
        bp::object * objPy_;
    };

    void convertFromPython(bp::object const & configPy, configHolder_t & config)
    {
        bp::dict configPyDict = bp::extract<bp::dict>(configPy);
        AppendPythonToBoostVariant visitor;
        for (auto & configField : config)
        {
            std::string const & name = configField.first;
            bp::object value = configPyDict[name];
            visitor.objPy_ = &value;
            boost::apply_visitor(visitor, configField.second);
        }
    }
}  // end of namespace python.
}  // end of namespace jiminy.

#endif  // UTILITIES_PYTHON_H
