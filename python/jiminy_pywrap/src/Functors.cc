#include "jiminy/core/Types.h"

#include "jiminy/python/Functors.h"

#include <boost/python.hpp>


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    // **************************** PyHeatMapFunctorVisitor *****************************

    struct PyHeatMapFunctorVisitor
        : public bp::def_visitor<PyHeatMapFunctorVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .def("__init__", bp::make_constructor(&PyHeatMapFunctorVisitor::factory,
                                 bp::default_call_policies(),
                                (bp::args("heatmap_function", "heatmap_type"))))
                .def("__call__", &PyHeatMapFunctorVisitor::eval,
                                 (bp::arg("self"), bp::arg("position")))
                ;
        }

        static bp::tuple eval(heatMapFunctor_t       & self,
                              vector3_t        const & posFrame)
        {
            std::pair<float64_t, vector3_t> ground = self(posFrame);
            return bp::make_tuple(std::move(std::get<0>(ground)), std::move(std::get<1>(ground)));
        }

        static std::shared_ptr<heatMapFunctor_t> factory(bp::object          & objPy,
                                                         heatMapType_t const & objType)
        {
            return std::make_shared<heatMapFunctor_t>(HeatMapFunctorPyWrapper(std::move(objPy), objType));
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<heatMapFunctor_t,
                       std::shared_ptr<heatMapFunctor_t> >("HeatMapFunctor", bp::no_init)
                .def(PyHeatMapFunctorVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(HeatMapFunctor)
}  // End of namespace python.
}  // End of namespace jiminy.
