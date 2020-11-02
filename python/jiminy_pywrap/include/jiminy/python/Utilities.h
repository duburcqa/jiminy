#ifndef JIMINY_PYTHON_UTILITIES_H
#define JIMINY_PYTHON_UTILITIES_H

// Make sure that the Python C API does not get redefined separately
#define PY_ARRAY_UNIQUE_SYMBOL JIMINY_ARRAY_API
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

    // ****************************************************************************
    // **************************** C++ TO PYTHON *********************************
    // ****************************************************************************

    /// C++ to Python type mapping

    inline int getPyType(bool_t const & data) { return NPY_BOOL; }
    inline int getPyType(float64_t const & data) { return NPY_FLOAT64; }
    inline int getPyType(float32_t const & data) { return NPY_FLOAT32; }
    inline int getPyType(int32_t const & data) { return NPY_INT32; }
    inline int getPyType(int64_t const & data) { return NPY_INT64; }

    /// Convert Eigen scalar/vector/matrix to Numpy array by reference.

    template<typename T>
    PyObject * getNumpyReferenceFromScalar(T & value)
    {
        npy_intp dims[1] = {npy_intp(1)};
        return PyArray_SimpleNewFromData(1, dims, getPyType(value), &value);
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
    PyObject * getNumpyReferenceFromEigenVector(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, 1> const> const & value)
    {
        npy_intp dims[1] = {npy_intp(value.size())};
        return PyArray_SimpleNewFromData(1, dims, getPyType(*value.data()), const_cast<T*>(value.data()));
    }

    template<typename T>
    PyObject * getNumpyReferenceFromEigenMatrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & value)
    {
        npy_intp dims[2] = {npy_intp(value.cols()), npy_intp(value.rows())};
        return PyArray_Transpose(reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNewFromData(2, dims, getPyType(*value.data()), value.data())), NULL);
    }

    template<typename T>
    PyObject * getNumpyReferenceFromEigenMatrix(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > & value)
    {
        npy_intp dims[2] = {npy_intp(value.cols()), npy_intp(value.rows())};
        return PyArray_Transpose(reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNewFromData(2, dims, getPyType(*value.data()), value.data())), NULL);
    }

    /// Generic converter to Numpy array by reference

    template<typename T>
    std::enable_if_t<!is_eigen<T>::value, PyObject *>
    getNumpyReference(T & value)
    {
        return getNumpyReferenceFromScalar(value);
    }

    template<typename T>
    std::enable_if_t<is_eigen_vector<T>::value, PyObject *>
    getNumpyReference(T & value)
    {
        return getNumpyReferenceFromEigenVector(value);
    }

    template<typename T>
    std::enable_if_t<is_eigen<T>::value
                 && !is_eigen_vector<T>::value, PyObject *>
    getNumpyReference(T & value)
    {
        return getNumpyReferenceFromEigenMatrix(value);
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// Convert most C++ objects into Python objects by value.
    ///////////////////////////////////////////////////////////////////////////////

    template<typename T>
    std::enable_if_t<!is_vector<T>::value
                  && !is_eigen<T>::value, bp::object>
    convertToPython(T const & data)
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

    template<typename T>
    std::enable_if_t<is_eigen<T>::value, bp::object>
    convertToPython(T const & data)
    {
        PyObject * vecPyPtr = getNumpyReference(const_cast<T &>(data));
        return bp::object(bp::handle<>(PyArray_FROM_OF(vecPyPtr, NPY_ARRAY_ENSURECOPY)));
    }

    template<typename T>
    std::enable_if_t<is_vector<T>::value, bp::object>
    convertToPython(T const & data)
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

    /// \brief  Convert a 1D python list into an Eigen vector by value.
    vectorN_t listPyToEigenVector(bp::list const & listPy)
    {
        vectorN_t x(len(listPy));
        for (int32_t i = 0; i < len(listPy); ++i)
        {
            x(i) = bp::extract<float64_t>(listPy[i]);
        }

        return x;
    }

    /// \brief  Convert a 2D python list into an Eigen matrix.
    matrixN_t listPyToEigenMatrix(bp::list const & listPy)
    {
        int32_t const nRows = len(listPy);
        assert(nRows > 0 && "empty list");

        int32_t const nCols = len(bp::extract<bp::list>(listPy[0]));
        assert(nCols > 0 && "empty row");

        matrixN_t M(nRows, nCols);
        for (int32_t i = 0; i < nRows; ++i)
        {
            bp::list const row = bp::extract<bp::list>(listPy[i]);
            assert(len(row) == nCols && "wrong number of columns");
            M.row(i) = listPyToEigenVector(row).transpose();
        }

        return M;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// Convert most Python objects in C++ objects by value.
    ///////////////////////////////////////////////////////////////////////////////

    template<typename T>
    std::enable_if_t<!is_vector<T>::value
                  && !is_map<T>::value
                  && !is_eigen<T>::value
                  && !std::is_same<T, int32_t>::value
                  && !std::is_same<T, uint32_t>::value
                  && !std::is_same<T, sensorsDataMap_t>::value, T>
    convertFromPython(bp::object const & dataPy)
    {
        return bp::extract<T>(dataPy);
    }

    template<typename T>
    std::enable_if_t<std::is_same<T, int32_t>::value
                  || std::is_same<T, uint32_t>::value, T>
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
            return bp::extract<T>(dataPy);
        }
    }

    template<typename T>
    std::enable_if_t<is_eigen<T>::value, T>
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
            if (is_eigen_vector<T>::value)
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
            if (is_eigen_vector<T>::value)
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
            if (is_eigen_vector<T>::value)
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

    template<typename T>
    std::enable_if_t<is_vector<T>::value, T>
    convertFromPython(bp::object const & dataPy)
    {
        using V = typename T::value_type;

        T vec;
        bp::list const listPy = bp::extract<bp::list>(dataPy);
        vec.reserve(bp::len(listPy));
        for (bp::ssize_t i=0; i < bp::len(listPy); ++i)
        {
            bp::object const itemPy = listPy[i];
            vec.push_back(std::move(convertFromPython<V>(itemPy)));
        }
        return vec;
    }

    template<typename T>
    std::enable_if_t<std::is_same<T, sensorsDataMap_t>::value, T>
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
    std::enable_if_t<is_map<T>::value
                  && !std::is_same<T, sensorsDataMap_t>::value, T>
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
        std::enable_if_t<!std::is_same<T, configHolder_t>::value, void>
        operator()(T & value)
        {
            value = convertFromPython<T>(*objPy_);
        }

        template <typename T>
        std::enable_if_t<std::is_same<T, configHolder_t>::value, void>
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
