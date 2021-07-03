#define BOOST_PYTHON_NUMPY_INTERNAL
#include "eigenpy/fwd.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "jiminy/python/Compatibility.h"


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;
    namespace np = boost::python::numpy;

    template<typename T>
    class arrayScalarFromPython
    {
    public:
        static void * convertible(PyObject * obj)
        {
            PyTypeObject const * pytype = reinterpret_cast<PyArray_Descr*>(
                np::dtype::get_builtin<T>().ptr())->typeobj;
            if (obj->ob_type == pytype)
            {
                return obj;
            }
            else
            {
                np::dtype dt(bp::detail::borrowed_reference(obj->ob_type));
                if (equivalent(dt, np::dtype::get_builtin<T>()))
                {
                    return obj;
                }
            }
            return 0;
        }

        static void convert(PyObject * obj, bp::converter::rvalue_from_python_stage1_data * data)
        {
            void * storage = reinterpret_cast<bp::converter::rvalue_from_python_storage<T>*>(data)->storage.bytes;
            PyArray_ScalarAsCtype(obj, reinterpret_cast<T*>(storage));
            data->convertible = storage;
        }

        static void declare()
        {
            // Note that no `get_pytype` is provided, so that the already existing one will be used
            bp::converter::registry::push_back(&convertible, &convert, bp::type_id<T>());
        }
    };

    extern "C" PyObject * identity_unaryfunc(PyObject* x)
    {
        Py_INCREF(x);
        return x;
    }
    unaryfunc py_object_identity = identity_unaryfunc;
    unaryfunc py_unicode_as_string_unaryfunc = PyUnicode_AsUTF8String;

    // A SlotPolicy for extracting C++ strings from Python objects.
    struct stringFromPython
    {
        // If the underlying object is "string-able" this will succeed
        static unaryfunc * get_slot(PyObject * obj)
        {
            return (PyUnicode_Check(obj)) ? &py_unicode_as_string_unaryfunc :
                    PyBytes_Check(obj) ? &py_object_identity : 0;
        };

        static std::string extract(PyObject * intermediate)
        {
            return std::string(PyBytes_AsString(intermediate),
                               PyBytes_Size(intermediate));
        }

        static PyTypeObject const* get_pytype() { return &PyUnicode_Type; }
    };

    template<class T, class SlotPolicy>
    struct nativeFromPython
    {
    public:
        nativeFromPython()
        {
            bp::converter::registry::insert(
                &nativeFromPython<T, SlotPolicy>::convertible,
                &nativeFromPython<T, SlotPolicy>::construct,
                bp::type_id<T>(),
                &SlotPolicy::get_pytype);
        }

    private:
        static void * convertible(PyObject* obj)
        {
            unaryfunc* slot = SlotPolicy::get_slot(obj);
            return slot && *slot ? slot : 0;
        }

        static void construct(PyObject * obj, bp::converter::rvalue_from_python_stage1_data * data)
        {
            unaryfunc creator = *static_cast<unaryfunc*>(data->convertible);
            bp::handle<> intermediate(creator(obj));
            void* storage = reinterpret_cast<bp::converter::rvalue_from_python_storage<T> *>(data)->storage.bytes;
            new (storage) T(SlotPolicy::extract(intermediate.get()));
            data->convertible = storage;
        }
    };

    void exposeCompatibility(void)
    {
        // Add some automatic C++ to Python converters for numpy array of scalars,
        // which is different from a 0-dimensional numpy array.
        arrayScalarFromPython<bool>::declare();
        arrayScalarFromPython<npy_uint8>::declare();
        arrayScalarFromPython<npy_uint32>::declare();
    #ifdef _MSC_VER
        arrayScalarFromPython<boost::uint32_t>::declare();
    #endif
        arrayScalarFromPython<npy_float32>::declare();

        /* Add native string from python converter to resolve old VS new cxx11 string
           ABI conflict. Since it is impossible to guarantee other Boost.Python
           extension modules will be compiled with the same ABI, it is necessary to
           register manually the missing Native Python types converters, since they
           are normally only defined once by the first module to be loaded. Old and new
           ABI are different objects in practice, so there will be no conflicts.
           For references about this issue:
           - https://stackoverflow.com/a/33395489/4820605
           - https://groups.google.com/g/opengm/c/qv5q70YR8QQ */
        nativeFromPython<std::string, stringFromPython>();
    }

}  // End of namespace python.
}  // End of namespace jiminy.
