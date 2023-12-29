#include "pinocchio/spatial/force.hpp"  // `Pinocchio::Force`

#include "jiminy/python/functors.h"


namespace jiminy::python
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
    pinocchio::Force * createInternalBuffer<pinocchio::Force>()
    {
        return (new pinocchio::Force(Vector6d::Zero()));
    }

    // **************************** PyHeightmapFunctorVisitor *****************************

    struct PyHeightmapFunctorVisitor : public bp::def_visitor<PyHeightmapFunctorVisitor>
    {
    public:
        /// \brief Expose C++ API through the visitor.
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .def("__init__", bp::make_constructor(&PyHeightmapFunctorVisitor::factory,
                                 bp::default_call_policies(),
                                (bp::arg("heightmap_function"),
                                 bp::arg("heightmap_type")=heightmapType_t::GENERIC)))
                .def("__call__", &PyHeightmapFunctorVisitor::eval,
                                 (bp::arg("self"), "position"))
                .ADD_PROPERTY_GET_WITH_POLICY("py_function",
                                              &PyHeightmapFunctorVisitor::getPyFun,
                                              bp::return_value_policy<bp::return_by_value>());
                ;
            // clang-format on
        }

        static bp::tuple eval(HeightmapFunctor & self, const Eigen::Vector3d & posFrame)
        {
            const std::pair<double, Eigen::Vector3d> ground = self(posFrame);
            return bp::make_tuple(std::get<double>(ground), std::get<Eigen::Vector3d>(ground));
        }

        static bp::object getPyFun(HeightmapFunctor & self)
        {
            HeightmapFunctorPyWrapper * pyWrapper(self.target<HeightmapFunctorPyWrapper>());
            if (!pyWrapper || pyWrapper->heightmapType_ != heightmapType_t::GENERIC)
            {
                return {};
            }
            return pyWrapper->handlePyPtr_;
        }

        static std::shared_ptr<HeightmapFunctor> factory(bp::object & objPy,
                                                         heightmapType_t objType)
        {
            return std::make_shared<HeightmapFunctor>(HeightmapFunctorPyWrapper(objPy, objType));
        }

        static void expose()
        {
            // clang-format off
            bp::class_<HeightmapFunctor,
                       std::shared_ptr<HeightmapFunctor>>("HeightmapFunctor", bp::no_init)
                .def(PyHeightmapFunctorVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(HeightmapFunctor)
}
