#ifndef SIMU_PYTHON_UTILITIES_H
#define SIMU_PYTHON_UTILITIES_H

#include <eigenpy/eigenpy.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/numpy/ndarray.hpp>


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    enum pyVector_t
    {
        vector,
        matrixCol,
        matrixRow
    };

    inline int getPyType(bool & data)
    {
        return NPY_BOOL;
    }

    inline int getPyType(float64_t & data)
    {
        return NPY_DOUBLE;
    }

    // ****************************************************************************
    // **************************** C++ TO PYTHON *********************************
    // ****************************************************************************

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert std::vector to Python list by value.
    ///////////////////////////////////////////////////////////////////////////////
    template<class T>
    struct stdVectorToListPyConverter
    {
        static PyObject* convert(std::vector<T> const & vec)
        {
            boost::python::list * l = new boost::python::list();
            for(size_t i = 0; i < vec.size(); i++)
            {
                l->append(vec[i]);
            }

            return l->ptr();
        }
    };

    template<class T>
    bp::list stdVectorToListPy(std::vector<T> const & v) {
        bp::list listPy;
        for (auto iter = v.begin(); iter != v.end(); ++iter)
        {
            listPy.append(*iter);
        }
        return listPy;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert scalar to Numpy array by reference.
    ///////////////////////////////////////////////////////////////////////////////
    template<typename T>
    PyObject * getNumpyReferenceFromScalar(T & data)
    {
        npy_intp dims[1] = {npy_intp(1)};
        return PyArray_SimpleNewFromData(1, dims, getPyType(data), &data);
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert Eigen vector to Numpy array by reference.
    ///////////////////////////////////////////////////////////////////////////////
    PyObject * getNumpyReferenceFromEigenVector(Eigen::Ref<vectorN_t const> data, // Must use Ref to support fixed size array without copy
                                                pyVector_t                  type = pyVector_t::vector)
    {
        if (type == pyVector_t::vector)
        {
            npy_intp dims[1] = {npy_intp(data.size())};
            return PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, const_cast<float64_t *>(data.data()));
        }
        else
        {
            npy_intp dims[2] = {npy_intp(1), npy_intp(data.size())};
            PyObject * pyData = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, const_cast<float64_t *>(data.data()));

            if (type == pyVector_t::matrixCol)
            {
                return PyArray_Transpose(reinterpret_cast<PyArrayObject *>(pyData), NULL);
            }
            else
            {
                return pyData;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert Eigen matrix to Numpy array by reference.
    ///////////////////////////////////////////////////////////////////////////////
    PyObject * getNumpyReferenceFromEigenMatrix(Eigen::Ref<matrixN_t const> data)
    {
        npy_intp dims[2] = {npy_intp(data.cols()), npy_intp(data.rows())};
        return  PyArray_Transpose(reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, const_cast<float64_t *>(data.data()))),NULL);
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert an Eigen vector into a 1D python list by value.
    ///////////////////////////////////////////////////////////////////////////////
    bp::list eigenVectorTolistPy(vectorN_t const & v)
    {
        bp::list l;
        for (int32_t j = 0; j < v.rows(); j++)
        {
            l.append(v(j));
        }
        return l;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert an Eigen matrix into a 2D python list by value.
    ///////////////////////////////////////////////////////////////////////////////
    bp::list eigenMatrixTolistPy(matrixN_t const & M)
    {
        bp::list l;
        for (int32_t j = 0; j < M.rows(); j++)
        {
            bp::list row;
            for (int32_t k = 0; k < M.cols(); k++)
            {
                row.append(M(j, k));
            }
            l.append(row);
        }
        return l;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert most C++ objects into Python objects by value.
    ///////////////////////////////////////////////////////////////////////////////
    template<typename CType>
    void convertToPy(CType const & data, bp::object & dataPy)
    {
        dataPy = bp::object(data);
    }

    template<>
    void convertToPy(flexibleJointData_t const & flexibleJointData, bp::object & dataPy)
    {
        bp::dict flexibilityJointDataPy;
        flexibilityJointDataPy["jointName"] = flexibleJointData.jointName;
        flexibilityJointDataPy["stiffness"] = flexibleJointData.stiffness;
        flexibilityJointDataPy["damping"] = flexibleJointData.damping;
        dataPy = flexibilityJointDataPy;
    }

    template<>
    void convertToPy(flexibilityConfig_t const & flexibilityConfig, bp::object & dataPy)
    {
        bp::list flexibilityConfigPy;
        for (auto const & flexibleJoint : flexibilityConfig)
        {
            bp::dict flexibilityJointDataPy;
            convertToPy(flexibleJoint, flexibilityJointDataPy);
            flexibilityConfigPy.append(flexibilityJointDataPy);
        }
        dataPy = flexibilityConfigPy;
    }

    template<>
    void convertToPy(vectorN_t const & data, bp::object & dataPy)
    {
        dataPy = eigenVectorTolistPy(data);
    }

    template<>
    void convertToPy(matrixN_t const & data, bp::object & dataPy)
    {
        dataPy = eigenMatrixTolistPy(data);
    }

    template<>
    void convertToPy(std::vector<std::string> const & data, bp::object & dataPy)
    {
        dataPy = stdVectorToListPy(data);
    }

    template<>
    void convertToPy(configHolder_t const & config, bp::object & configPy)
    {
        bp::dict configPyDict;
        for (auto const & configField : config)
        {
            std::string const & name = configField.first;
            const std::type_info & optionType = configField.second.type();

            if (optionType == typeid(bool_t))
            {
                bp::object dataPy;
                convertToPy(boost::get<bool_t>(configField.second), dataPy);
                configPyDict[name] = dataPy;
            }
            else if (optionType == typeid(int32_t))
            {
                bp::object dataPy;
                convertToPy(boost::get<int32_t>(configField.second), dataPy);
                configPyDict[name] = dataPy;
            }
            else if (optionType == typeid(uint32_t))
            {
                bp::object dataPy;
                convertToPy(boost::get<uint32_t>(configField.second), dataPy);
                configPyDict[name] = dataPy;
            }
            else if (optionType == typeid(float64_t))
            {
                bp::object dataPy;
                convertToPy(boost::get<float64_t>(configField.second), dataPy);
                configPyDict[name] = dataPy;
            }
            else if (optionType == typeid(std::string))
            {
                bp::object dataPy;
                convertToPy(boost::get<std::string>(configField.second), dataPy);
                configPyDict[name] = dataPy;
            }
            else if (optionType == typeid(heatMapFunctor_t))
            {
                bp::object dataPy;
                convertToPy(boost::get<heatMapFunctor_t>(configField.second), dataPy);
                configPyDict[name] = dataPy;
            }
            else if (optionType == typeid(flexibilityConfig_t))
            {
                bp::object dataPy;
                convertToPy(boost::get<flexibilityConfig_t>(configField.second), dataPy);
                configPyDict[name] = dataPy;
            }
            else if (optionType == typeid(vectorN_t))
            {
                bp::object dataPy;
                convertToPy(boost::get<vectorN_t>(configField.second), dataPy);
                configPyDict[name] = dataPy;
            }
            else if (optionType == typeid(matrixN_t))
            {
                bp::object dataPy;
                convertToPy(boost::get<matrixN_t>(configField.second), dataPy);
                configPyDict[name] = dataPy;
            }
            else if (optionType == typeid(std::vector<std::string>))
            {
                bp::object dataPy;
                convertToPy(boost::get<std::vector<std::string> >(configField.second), dataPy);
                configPyDict[name] = dataPy;
            }
            else if (optionType == typeid(std::vector<vectorN_t>))
            {
                bp::object dataPy;
                convertToPy(boost::get<std::vector<vectorN_t> >(configField.second), dataPy);
                configPyDict[name] = dataPy;
            }
            else if (optionType == typeid(std::vector<matrixN_t>))
            {
                bp::object dataPy;
                convertToPy(boost::get<std::vector<matrixN_t> >(configField.second), dataPy);
                configPyDict[name] = dataPy;
            }
            else if (optionType == typeid(configHolder_t))
            {
                bp::object dataPy;
                convertToPy(boost::get<configHolder_t>(configField.second), dataPy);
                configPyDict[name] = dataPy;
            }
            else
            {
                assert(false && "Unsupported type");
            }
        }
        configPy = configPyDict;
    }

    // ****************************************************************************
    // **************************** PYTHON TO C++ *********************************
    // ****************************************************************************

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert Python list/tuple to std::vector by value.
    ///////////////////////////////////////////////////////////////////////////////
    template<typename T>
    inline std::vector<T> listPyToStdVector(bp::list const & listPy)
    {
        std::vector<T> v;
        v.reserve(len(listPy));
        for (int32_t i = 0; i < len(listPy); i++)
        {
            v.push_back(bp::extract<T>(listPy[i]));
        }

        return v;
    }

    template<typename T>
    inline std::vector<T> listPyToStdVector(bp::tuple const & listPy)
    {
        std::vector<T> v;
        v.reserve(len(listPy));
        for (int32_t i = 0; i < len(listPy); i++)
        {
            v.push_back(bp::extract<T>(listPy[i]));
        }

        return v;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert Python list to std::vector of std::vector by value.
    ///////////////////////////////////////////////////////////////////////////////
    template<typename T>
    inline std::vector<std::vector<T> > listPyToStdVectorVector(bp::list const & listPy)
    {
        std::vector<std::vector<T>> v;
        v.reserve(len(listPy));
        for (int32_t i = 0; i < len(listPy); i++)
        {
            v.push_back(listPyToStdVector<T>(bp::extract<bp::list>(listPy[i])));
        }

        return v;
    }

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
            bp::list const & row = bp::extract<bp::list>(listPy[i]);
            assert(len(row) == nCols && "wrong number of columns");
            M.row(i) = listPyToEigenVector(row).transpose();
        }

        return M;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert most Python objects in C++ objects by value.
    ///////////////////////////////////////////////////////////////////////////////
    template<typename CType>
    typename std::enable_if<!std::is_same<CType, int32_t>::value
                         && !std::is_same<CType, uint32_t>::value
                         && !std::is_same<CType, vectorN_t>::value
                         && !std::is_same<CType, matrixN_t>::value
                         && !std::is_same<CType, std::vector<std::string> >::value
                         && !std::is_same<CType, std::vector<vectorN_t> >::value
                         && !std::is_same<CType, std::vector<matrixN_t> >::value, void>::type
    convertToC(bp::object const & dataPy, CType & data)
    {
        data = bp::extract<CType>(dataPy);
    }

    template<typename CType>
    typename std::enable_if<std::is_same<CType, int32_t>::value
                         || std::is_same<CType, uint32_t>::value, void>::type
    convertToC(bp::object const & dataPy, CType & data)
    {
        std::string const & optionTypePyStr =
            bp::extract<std::string>(dataPy.attr("__class__").attr("__name__"));
        if (optionTypePyStr == "ndarray")
        {
            bp::numpy::ndarray const & dataNumpy = bp::extract<bp::numpy::ndarray>(dataPy);
            data = *reinterpret_cast<CType *>(dataNumpy.get_data());
        }
        else if (optionTypePyStr == "matrix")
        {
            bp::numpy::matrix const & dataMatrix = bp::extract<bp::numpy::matrix>(dataPy);
            data = *reinterpret_cast<CType *>(dataMatrix.get_data());
        }
        else
        {
            data = bp::extract<CType>(dataPy);
        }
    }

    template<typename CType>
    typename std::enable_if<std::is_same<CType, vectorN_t>::value
                         || std::is_same<CType, matrixN_t>::value, void>::type
    convertToC(bp::object const & dataPy, CType & data)
    {
        std::string const & optionTypePyStr =
            bp::extract<std::string>(dataPy.attr("__class__").attr("__name__"));
        if (optionTypePyStr == "ndarray")
        {
            bp::numpy::ndarray dataNumpy = bp::extract<bp::numpy::ndarray>(dataPy);
            dataNumpy = dataNumpy.astype(bp::numpy::dtype::get_builtin<float64_t>());
            float64_t * dataPtr = reinterpret_cast<float64_t *>(dataNumpy.get_data());
            long int const * dataShape = dataNumpy.get_shape();
            if(std::is_same<CType, vectorN_t>::value)
            {
                data = Eigen::Map<vectorN_t>(dataPtr, dataShape[0]);
            }
            else
            {
                data = Eigen::Map<matrixN_t>(dataPtr, dataShape[0], dataShape[1]);
            }
        }
        else if (optionTypePyStr == "matrix")
        {
            bp::numpy::matrix dataMatrix = bp::extract<bp::numpy::matrix>(dataPy);
            bp::numpy::ndarray dataNumpy = dataMatrix.astype(bp::numpy::dtype::get_builtin<float64_t>());
            float64_t * dataPtr = reinterpret_cast<float64_t *>(dataNumpy.get_data());
            long int const * dataShape = dataNumpy.get_shape();
            if(std::is_same<CType, vectorN_t>::value)
            {
                data = Eigen::Map<vectorN_t>(dataPtr, dataShape[0]);
            }
            else
            {
                data = Eigen::Map<matrixN_t>(dataPtr, dataShape[0], dataShape[1]);
            }
        }
        else
        {
            if(std::is_same<CType, vectorN_t>::value)
            {
                data = listPyToEigenVector(bp::extract<bp::list>(dataPy));
            }
            else
            {
                data = listPyToEigenMatrix(bp::extract<bp::list>(dataPy));
            }
        }
    }

    template<typename CType>
    typename std::enable_if<std::is_same<CType, std::vector<std::string> >::value
                         || std::is_same<CType, std::vector<vectorN_t> >::value
                         || std::is_same<CType, std::vector<matrixN_t> >::value, void>::type
    convertToC(bp::object const & dataPy, CType & data)
    {
        data = listPyToStdVector<typename CType::value_type>(bp::extract<bp::list>(dataPy));
    }

    template<>
    void convertToC(bp::object const & dataPy, flexibleJointData_t & flexibleJointData)
    {
        bp::dict flexibilityJointDataPy = bp::extract<bp::dict>(dataPy);
        convertToC(flexibilityJointDataPy["jointName"], flexibleJointData.jointName);
        convertToC(flexibilityJointDataPy["stiffness"], flexibleJointData.stiffness);
        convertToC(flexibilityJointDataPy["damping"], flexibleJointData.damping);
    }

    template<>
    void convertToC(bp::object const & dataPy, flexibilityConfig_t & flexibilityConfig)
    {
        bp::list flexibilityConfigPy = bp::extract<bp::list>(dataPy);
        flexibilityConfig.resize(bp::len(flexibilityConfigPy));
        for (bp::ssize_t i=0; i < bp::len(flexibilityConfigPy); i++)
        {
            convertToC(flexibilityConfigPy[i], flexibilityConfig[i]);
        }
    }

    template<>
    void convertToC(bp::object const & configPy, configHolder_t & config)
    {
        for (auto const & configField : config)
        {
            std::string const & name = configField.first;
            const std::type_info & optionType = configField.second.type();
            if (optionType == typeid(bool_t))
            {
                convertToC(configPy[name], boost::get<bool_t>(config.at(name)));
            }
            else if (optionType == typeid(int32_t))
            {
                convertToC(configPy[name], boost::get<int32_t>(config.at(name)));
            }
            else if (optionType == typeid(uint32_t))
            {
                convertToC(configPy[name], boost::get<uint32_t>(config.at(name)));
            }
            else if (optionType == typeid(float64_t))
            {
                convertToC(configPy[name], boost::get<float64_t>(config.at(name)));
            }
            else if (optionType == typeid(std::string))
            {
                convertToC(configPy[name], boost::get<std::string>(config.at(name)));
            }
            else if (optionType == typeid(heatMapFunctor_t))
            {
                convertToC(configPy[name], boost::get<heatMapFunctor_t>(config.at(name)));
            }
            else if (optionType == typeid(flexibilityConfig_t))
            {
                convertToC(configPy[name], boost::get<flexibilityConfig_t>(config.at(name)));
            }
            else if (optionType == typeid(vectorN_t))
            {
                convertToC(configPy[name], boost::get<vectorN_t>(config.at(name)));
            }
            else if (optionType == typeid(matrixN_t))
            {
                convertToC(configPy[name], boost::get<matrixN_t>(config.at(name)));
            }
            else if (optionType == typeid(std::vector<std::string>))
            {
                convertToC(configPy[name], boost::get<std::vector<std::string> >(config.at(name)));
            }
            else if (optionType == typeid(std::vector<vectorN_t>))
            {
                convertToC(configPy[name], boost::get<std::vector<vectorN_t> >(config.at(name)));

            }
            else if (optionType == typeid(std::vector<matrixN_t>))
            {
                convertToC(configPy[name], boost::get<std::vector<matrixN_t> >(config.at(name)));
            }
            else if (optionType == typeid(configHolder_t))
            {
                convertToC(configPy[name], boost::get<configHolder_t>(config.at(name)));
            }
            else
            {
                assert(false && "Unsupported type");
            }
        }
    }
}  // end of namespace python.
}  // end of namespace jiminy.

#endif  // SIMU_PYTHON_UTILITIES_H
