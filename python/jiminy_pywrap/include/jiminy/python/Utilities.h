#ifndef JIMINY_PYTHON_UTILITIES_H
#define JIMINY_PYTHON_UTILITIES_H

// Make sure that the Python C API does not get redefined separately
#define PY_ARRAY_UNIQUE_SYMBOL BOOST_NUMPY_ARRAY_API
#define NO_IMPORT_ARRAY

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/numpy/ndarray.hpp>


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;
    namespace np = boost::python::numpy;

    inline int getPyType(bool_t const & data)
    {
        return NPY_BOOL;
    }

    inline int getPyType(float64_t const & data)
    {
        return NPY_FLOAT64;
    }

    inline int getPyType(float32_t const & data)
    {
        return NPY_FLOAT32;
    }

    inline int getPyType(int32_t const & data)
    {
        return NPY_INT32;
    }

    inline int getPyType(int64_t const & data)
    {
        return NPY_INT64;
    }

    // ****************************************************************************
    // **************************** C++ TO PYTHON *********************************
    // ****************************************************************************

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert scalar to Numpy array by reference.
    ///////////////////////////////////////////////////////////////////////////////
    template<typename T>
    PyObject * getNumpyReferenceFromScalar(T & value)
    {
        npy_intp dims[1] = {npy_intp(1)};
        return PyArray_SimpleNewFromData(1, dims, getPyType(value), &value);
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert Eigen vector to Numpy array by reference.
    ///////////////////////////////////////////////////////////////////////////////
    #define MAKE_FUNC(T) \
    PyObject * getNumpyReferenceFromEigenVector( \
        Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1> const> value /* Must use Ref to support fixed size array without copy */ ) \
    { \
        npy_intp dims[1] = {npy_intp(value.size())}; \
        return PyArray_SimpleNewFromData(1, dims, getPyType(*value.data()), const_cast<T*>(value.data())); \
    }

    MAKE_FUNC(int32_t)
    MAKE_FUNC(float32_t)
    MAKE_FUNC(float64_t)

    #undef MAKE_FUNC

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert Eigen matrix to Numpy array by reference.
    ///////////////////////////////////////////////////////////////////////////////
    PyObject * getNumpyReferenceFromEigenMatrix(Eigen::Ref<matrixN_t const> value)
    {
        npy_intp dims[2] = {npy_intp(value.cols()), npy_intp(value.rows())};
        return PyArray_Transpose(reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, const_cast<float64_t *>(value.data()))), NULL);
    }


    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Template specializations
    ///////////////////////////////////////////////////////////////////////////////
    template<typename T>
    PyObject * getNumpyReference(T & data)
    {
        return getNumpyReferenceFromScalar(data);
    }

    template<>
    PyObject * getNumpyReference<vector3_t>(vector3_t & data)
    {
        return getNumpyReferenceFromEigenVector(data);
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert an Eigen vector into a 1D python list by value.
    ///////////////////////////////////////////////////////////////////////////////
    bp::list eigenVectorTolistPy(vectorN_t const & v)
    {
        bp::list l;
        for (int32_t j = 0; j < v.rows(); j++)
        {
            l.append(v[j]);
        }
        return l;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert most C++ objects into Python objects by value.
    ///////////////////////////////////////////////////////////////////////////////

    template<typename CType>
    enable_if_t<!is_vector<CType>::value, bp::object>
    convertToPython(CType const & data)
    {
        return bp::object(data);
    }

    template<>
    bp::object convertToPython<flexibleJointData_t>(flexibleJointData_t const & flexibleJointData)
    {
        bp::dict flexibilityJointDataPy;
        flexibilityJointDataPy["jointName"] = flexibleJointData.jointName;
        flexibilityJointDataPy["stiffness"] = flexibleJointData.stiffness;
        flexibilityJointDataPy["damping"] = flexibleJointData.damping;
        return flexibilityJointDataPy;
    }

    template<>
    bp::object convertToPython<vectorN_t>(vectorN_t const & data)
    {
        PyObject * vecPyPtr = getNumpyReferenceFromEigenVector(data);
        return bp::object(bp::handle<>(PyArray_FROM_OF(vecPyPtr, NPY_ARRAY_ENSURECOPY)));
    }

    template<>
    bp::object convertToPython<matrixN_t>(matrixN_t const & data)
    {
        PyObject * matPyPtr = getNumpyReferenceFromEigenMatrix(data);
        return bp::object(bp::handle<>(PyArray_FROM_OF(matPyPtr, NPY_ARRAY_ENSURECOPY)));
    }

    template<typename CType>
    enable_if_t<is_vector<CType>::value, bp::object>
    convertToPython(CType const & data)
    {
        bp::list dataPy;
        for (auto const & val : data)
        {
            dataPy.append(convertToPython(val));
        }
        return dataPy;
    }

    class AppendBoostVariantToPython : public boost::static_visitor<bp::object>
    {
    public:
        template <typename T>
        bp::object operator()(T const & value) const
        {
            return convertToPython<T>(value);
        }
    };

    template<>
    bp::object convertToPython(configHolder_t const & config)
    {
        bp::dict configPyDict;
        AppendBoostVariantToPython visitor;
        for (auto const & configField : config)
        {
            std::string const & name = configField.first;
            configPyDict[name] = boost::apply_visitor(visitor, configField.second);
        }
        return configPyDict;
    }

    // ****************************************************************************
    // **************************** PYTHON TO C++ *********************************
    // ****************************************************************************

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert a 1D python list into an Eigen vector by value.
    ///////////////////////////////////////////////////////////////////////////////
    vectorN_t listPyToEigenVector(bp::list const & listPy)
    {
        vectorN_t x(len(listPy));
        for (int32_t i = 0; i < len(listPy); i++)
        {
            x(i) = bp::extract<float64_t>(listPy[i]);
        }

        return x;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert a 2D python list into an Eigen matrix.
    ///////////////////////////////////////////////////////////////////////////////
    matrixN_t listPyToEigenMatrix(bp::list const & listPy)
    {
        int32_t const nRows = len(listPy);
        assert(nRows > 0 && "empty list");

        int32_t const nCols = len(bp::extract<bp::list>(listPy[0]));
        assert(nCols > 0 && "empty row");

        matrixN_t M(nRows, nCols);
        for (int32_t i = 0; i < nRows; i++)
        {
            bp::list const row = bp::extract<bp::list>(listPy[i]);
            assert(len(row) == nCols && "wrong number of columns");
            M.row(i) = listPyToEigenVector(row).transpose();
        }

        return M;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert most Python objects in C++ objects by value.
    ///////////////////////////////////////////////////////////////////////////////

    template<typename CType>
    enable_if_t<!is_vector<CType>::value
             && !std::is_same<CType, int32_t>::value
             && !std::is_same<CType, uint32_t>::value
             && !std::is_same<CType, vectorN_t>::value
             && !std::is_same<CType, matrixN_t>::value, CType>
    convertFromPython(bp::object const & dataPy)
    {
        return bp::extract<CType>(dataPy);
    }

    template<typename CType>
    enable_if_t<std::is_same<CType, int32_t>::value
             || std::is_same<CType, uint32_t>::value, CType>
    convertFromPython(bp::object const & dataPy)
    {
        std::string const optionTypePyStr =
            bp::extract<std::string>(dataPy.attr("__class__").attr("__name__"));
        if (optionTypePyStr == "ndarray")
        {
            np::ndarray dataNumpy = bp::extract<np::ndarray>(dataPy);
            return *reinterpret_cast<CType const *>(dataNumpy.get_data());
        }
        else if (optionTypePyStr == "matrix")
        {
            np::matrix dataMatrix = bp::extract<np::matrix>(dataPy);
            return *reinterpret_cast<CType *>(dataMatrix.get_data());
        }
        else
        {
            return bp::extract<CType>(dataPy);
        }
    }

    template<typename CType>
    enable_if_t<std::is_same<CType, vectorN_t>::value
             || std::is_same<CType, matrixN_t>::value, CType>
    convertFromPython(bp::object const & dataPy)
    {
        std::string const optionTypePyStr =
            bp::extract<std::string>(dataPy.attr("__class__").attr("__name__"));
        if (optionTypePyStr == "ndarray")
        {
            np::ndarray dataNumpy = bp::extract<np::ndarray>(dataPy);
            dataNumpy = dataNumpy.astype(np::dtype::get_builtin<float64_t>());
            float64_t * dataPtr = reinterpret_cast<float64_t *>(dataNumpy.get_data());
            Py_intptr_t const * dataShape = dataNumpy.get_shape();
            if (std::is_same<CType, vectorN_t>::value)
            {
                return Eigen::Map<vectorN_t>(dataPtr, dataShape[0]);
            }
            else
            {
                return Eigen::Map<matrixN_t>(dataPtr, dataShape[0], dataShape[1]);
            }
        }
        else if (optionTypePyStr == "matrix")
        {
            np::matrix dataMatrix = bp::extract<np::matrix>(dataPy);
            np::ndarray dataNumpy = dataMatrix.astype(np::dtype::get_builtin<float64_t>());
            float64_t * dataPtr = reinterpret_cast<float64_t *>(dataNumpy.get_data());
            Py_intptr_t const * dataShape = dataNumpy.get_shape();
            if (std::is_same<CType, vectorN_t>::value)
            {
                return Eigen::Map<vectorN_t>(dataPtr, dataShape[0]);
            }
            else
            {
                return Eigen::Map<matrixN_t>(dataPtr, dataShape[0], dataShape[1]);
            }
        }
        else
        {
            if (std::is_same<CType, vectorN_t>::value)
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
    flexibleJointData_t convertFromPython<flexibleJointData_t>(bp::object const & dataPy)
    {
        flexibleJointData_t flexData;
        bp::dict const flexDataPy = bp::extract<bp::dict>(dataPy);
        flexData.jointName = convertFromPython<std::string>(flexDataPy["jointName"]);
        flexData.stiffness = convertFromPython<vectorN_t>(flexDataPy["stiffness"]);
        flexData.damping = convertFromPython<vectorN_t>(flexDataPy["damping"]);
        return flexData;
    }

    template<typename CType>
    enable_if_t<is_vector<CType>::value, CType>
    convertFromPython(bp::object const & dataPy)
    {
        CType vec;
        bp::list const listPy = bp::extract<bp::list>(dataPy);
        vec.reserve(bp::len(listPy));
        for (bp::ssize_t i=0; i < bp::len(listPy); i++)
        {
            bp::object const itemPy = listPy[i];
            vec.push_back(std::move(
                convertFromPython<typename CType::value_type>(itemPy)
            ));
        }
        return vec;
    }

    void convertFromPython(bp::object const & configPy, configHolder_t & config); // Forward declaration

    class AppendPythonToBoostVariant : public boost::static_visitor<>
    {
    public:
        AppendPythonToBoostVariant(void) :
        objPy_(nullptr)
        {
            // Empty on purpose
        }

        ~AppendPythonToBoostVariant(void) = default;

        template <typename T>
        enable_if_t<!std::is_same<T, configHolder_t>::value, void>
        operator()(T & value)
        {
            value = convertFromPython<T>(*objPy_);
        }

        template <typename T>
        enable_if_t<std::is_same<T, configHolder_t>::value, void>
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

#endif  // JIMINY_PYTHON_UTILITIES_H
