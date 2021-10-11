#include "pinocchio/spatial/force.hpp"  // `Pinocchio::Force`

#include "jiminy/core/Types.h"

#include "jiminy/python/Functors.h"


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    // ************************** FctPyWrapper ******************************

    template<>
    typename DataInternalBufferType<pinocchio::Force>::type
    setDataInternalBuffer<pinocchio::Force>(pinocchio::Force * arg)
    {
        return arg->toVector();
    }

    template<>
    pinocchio::Force * createInternalBuffer<pinocchio::Force>(void)
    {
        return (new pinocchio::Force(vector6_t::Zero()));
    }

    // **************************** PyHeightMapFunctorVisitor *****************************

    struct PyHeightMapFunctorVisitor
        : public bp::def_visitor<PyHeightMapFunctorVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .def("__init__", bp::make_constructor(&PyHeightMapFunctorVisitor::factory,
                                 bp::default_call_policies(),
                                (bp::arg("heightmap_function"),
                                 bp::arg("heightmap_type")=heightMapType_t::GENERIC)))
                .def("__call__", &PyHeightMapFunctorVisitor::eval,
                                 (bp::args("self", "position")))
                ;
        }

        static bp::tuple eval(heightMapFunctor_t       & self,
                              vector3_t          const & posFrame)
        {
            std::pair<float64_t, vector3_t> ground = self(posFrame);
            return bp::make_tuple(std::get<0>(ground), std::get<1>(ground));
        }

        static std::shared_ptr<heightMapFunctor_t> factory(bp::object            & objPy,
                                                           heightMapType_t const & objType)
        {
            return std::make_shared<heightMapFunctor_t>(HeightMapFunctorPyWrapper(objPy, objType));
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<heightMapFunctor_t,
                       std::shared_ptr<heightMapFunctor_t> >("HeightMapFunctor", bp::no_init)
                .def(PyHeightMapFunctorVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(HeightMapFunctor)
}  // End of namespace python.
}  // End of namespace jiminy.
