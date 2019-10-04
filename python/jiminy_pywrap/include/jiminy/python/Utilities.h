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

    int getPyType(bool & data)
    {
        return NPY_BOOL;
    }

    int getPyType(float64_t & data)
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
    bp::list stdVectorToListPy(std::vector<T> const & v)
    {
        bp::object get_iter = bp::iterator<std::vector<T> >();
        bp::object iter = get_iter(v);
        bp::list l(iter);
        return l;
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
    /// \brief  Convert a config holder into a Python dictionary by value.
    ///////////////////////////////////////////////////////////////////////////////
    void convertConfigHolderPy(configHolder_t const & config,
                               bp::dict             & configPy)
    {
        std::vector<std::string> options;
        for(auto const& it : config)
        {
            options.push_back(it.first);
        }

        for (uint32_t i = 0; i < options.size(); i++)
        {
            std::string const name = options[i];
            const std::type_info & optionType = config.at(name).type();

            if (optionType == typeid(bool_t))
            {
                configPy[name] = boost::get<bool_t>(config.at(name));
            }
            else if (optionType == typeid(int32_t))
            {
                configPy[name] = boost::get<int32_t>(config.at(name));
            }
            else if (optionType == typeid(uint32_t))
            {
                configPy[name] = boost::get<uint32_t>(config.at(name));
            }
            else if (optionType == typeid(float64_t))
            {
                configPy[name] = boost::get<float64_t>(config.at(name));
            }
            else if (optionType == typeid(std::string))
            {
                configPy[name] = boost::get<std::string>(config.at(name));
            }
            else if (optionType == typeid(vectorN_t))
            {
                configPy[name] = eigenVectorTolistPy(boost::get<vectorN_t>(config.at(name)));
            }
            else if (optionType == typeid(matrixN_t))
            {
                configPy[name] = eigenMatrixTolistPy(boost::get<matrixN_t>(config.at(name)));
            }
            else if (optionType == typeid(std::vector<std::string>))
            {
                configPy[name] = boost::get<std::vector<std::string> >(config.at(name));
            }
            else if (optionType == typeid(std::vector<vectorN_t>))
            {
                configPy[name] = boost::get<std::vector<vectorN_t> >(config.at(name));
            }
            else if (optionType == typeid(std::vector<matrixN_t>))
            {
                configPy[name] = boost::get<std::vector<matrixN_t> >(config.at(name));
            }
            else if (optionType == typeid(configHolder_t))
            {
                bp::dict configPyTmp;
                convertConfigHolderPy(boost::get<configHolder_t>(config.at(name)), configPyTmp);
                configPy[name] = configPyTmp;
            }
            else
            {
                assert(false && "Unsupported type");
            }
        }
    }

    // ****************************************************************************
    // **************************** PYTHON TO C++ *********************************
    // **********µ*****************************************************************

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
            bp::list const row = bp::extract<bp::list>(listPy[i]);
            assert(len(row) == nCols && "wrong number of columns");
            M.row(i) = listPyToEigenVector(row).transpose();
        }

        return M;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// \brief  Convert a Python dictionary into a config holder.
    ///////////////////////////////////////////////////////////////////////////////
    void loadConfigHolder(bp::dict       const & configPy,
                          configHolder_t       & config)
    {
        std::vector<std::string> options;
        for(auto const& it : config)
        {
            options.push_back(it.first);
        }

        for (uint32_t i = 0; i < options.size(); i++)
        {
            std::string const name = options[i];
            std::type_info const & optionType = config[name].type();
            std::string const & optionTypePyStr =
                bp::extract<std::string>(configPy[name].attr("__class__").attr("__name__"));

            if (optionType == typeid(bool_t))
            {
                boost::get<bool_t>(config.at(name)) = bp::extract<bool_t>(configPy[name]);
            }
            else if (optionType == typeid(int32_t))
            {
                if (optionTypePyStr != "ndarray")
                {
                    boost::get<int32_t>(config.at(name)) = bp::extract<int32_t>(configPy[name]);
                }
                else
                {
                    bp::numpy::ndarray dataPy = bp::extract<bp::numpy::ndarray>(configPy[name]);
                    int32_t const & data = *reinterpret_cast<int32_t *>(dataPy.get_data());
                    boost::get<int32_t>(config.at(name)) = data;
                }
            }
            else if (optionType == typeid(uint32_t))
            {
                if (optionTypePyStr != "ndarray")
                {
                    boost::get<uint32_t>(config.at(name)) = bp::extract<uint32_t>(configPy[name]);
                }
                else
                {
                    bp::numpy::ndarray dataPy = bp::extract<bp::numpy::ndarray>(configPy[name]);
                    uint32_t const & data = *reinterpret_cast<uint32_t *>(dataPy.get_data());
                    boost::get<uint32_t>(config.at(name)) = data;
                }
            }
            else if (optionType == typeid(float64_t))
            {
                boost::get<float64_t>(config.at(name)) = bp::extract<float64_t>(configPy[name]);
            }
            else if (optionType == typeid(std::string))
            {
                 boost::get<std::string>(config.at(name)) = bp::extract<std::string>(configPy[name]);
            }
            else if (optionType == typeid(vectorN_t))
            {
                if (optionTypePyStr != "ndarray")
                {
                    boost::get<vectorN_t>(config.at(name)) =
                        listPyToEigenVector(bp::extract<bp::list>(configPy[name]));
                }
                else
                {
                    bp::numpy::ndarray dataPy = bp::extract<bp::numpy::ndarray>(configPy[name]);
                    dataPy = dataPy.astype(bp::numpy::dtype::get_builtin<float64_t>());
                    float64_t * dataPtr = reinterpret_cast<float64_t *>(dataPy.get_data());
                    long int const * dataShape = dataPy.get_shape();
                    Eigen::Map<vectorN_t> data(dataPtr, dataShape[0]);
                    boost::get<vectorN_t>(config.at(name)) = data;
                }
            }
            else if (optionType == typeid(matrixN_t))
            {
                boost::get<matrixN_t>(config.at(name)) =
                    listPyToEigenMatrix(bp::extract<bp::list>(configPy[name]));
            }
            else if (optionType == typeid(std::vector<std::string>))
            {
                boost::get<std::vector<std::string> >(config.at(name)) =
                    listPyToStdVector<std::string>(bp::extract<bp::list>(configPy[name]));
            }
            else if (optionType == typeid(std::vector<vectorN_t>))
            {
                boost::get<std::vector<vectorN_t> >(config.at(name)) =
                    listPyToStdVector<vectorN_t>(bp::extract<bp::list>(configPy[name]));

            }
            else if (optionType == typeid(std::vector<matrixN_t>))
            {
                boost::get<std::vector<matrixN_t> >(config.at(name)) =
                    listPyToStdVector<matrixN_t>(bp::extract<bp::list>(configPy[name]));
            }
            else if (optionType == typeid(configHolder_t))
            {
                loadConfigHolder(bp::extract<bp::dict>(configPy[name]),
                                 boost::get<configHolder_t>(config.at(name)));
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
