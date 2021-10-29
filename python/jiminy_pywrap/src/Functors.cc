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

    // **************************** PyHeightmapFunctorVisitor *****************************

    struct PyHeightmapFunctorVisitor
        : public bp::def_visitor<PyHeightmapFunctorVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .def("__init__", bp::make_constructor(&PyHeightmapFunctorVisitor::factory,
                                 bp::default_call_policies(),
                                (bp::arg("heightmap_function"),
                                 bp::arg("heightmap_type")=heightmapType_t::GENERIC)))
                .def("__call__", &PyHeightmapFunctorVisitor::eval,
                                 (bp::args("self", "position")))
                .add_property("py_function", bp::make_function(&PyHeightmapFunctorVisitor::getPyFun,
                                             bp::return_value_policy<bp::return_by_value>()));
                ;
        }

        static bp::tuple eval(heightmapFunctor_t       & self,
                              vector3_t          const & posFrame)
        {
            std::pair<float64_t, vector3_t> const ground = self(posFrame);
            return bp::make_tuple(std::get<float64_t>(ground), std::get<vector3_t>(ground));
        }

        static bp::object getPyFun(heightmapFunctor_t & self)
        {
            HeightmapFunctorPyWrapper * pyWrapper(self.target<HeightmapFunctorPyWrapper>());
            if (!pyWrapper || pyWrapper->heightmapType_ != heightmapType_t::GENERIC)
            {
                return {};
            }
            return pyWrapper->handlePyPtr_;
        }

        static std::shared_ptr<heightmapFunctor_t> factory(bp::object            & objPy,
                                                           heightmapType_t const & objType)
        {
            return std::make_shared<heightmapFunctor_t>(HeightmapFunctorPyWrapper(objPy, objType));
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<heightmapFunctor_t,
                       std::shared_ptr<heightmapFunctor_t> >("HeightmapFunctor", bp::no_init)
                .def(PyHeightmapFunctorVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(HeightmapFunctor)
}  // End of namespace python.
}  // End of namespace jiminy.
