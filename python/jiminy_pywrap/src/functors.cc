#include "pinocchio/spatial/force.hpp"  // `Pinocchio::Force`

#include "jiminy/python/functors.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ************************** FunPyWrapper ******************************

    template<>
    typename InternalStorageType<pinocchio::Force>::type
    setDataInternalBuffer<pinocchio::Force>(pinocchio::Force * arg)
    {
        return arg->toVector();
    }

    template<>
    pinocchio::Force * createInternalBuffer<pinocchio::Force>()
    {
        return (new pinocchio::Force(Vector6d::Zero()));
    }

    // **************************** PyHeightmapFunctionVisitor *****************************

    struct PyHeightmapFunctionVisitor : public bp::def_visitor<PyHeightmapFunctionVisitor>
    {
    public:
        /// \brief Expose C++ API through the visitor.
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .def("__init__", bp::make_constructor(&PyHeightmapFunctionVisitor::factory,
                                 bp::default_call_policies(),
                                (bp::arg("heightmap_function"),
                                 bp::arg("heightmap_type")=heightmapType_t::GENERIC)))
                .def("__call__", &PyHeightmapFunctionVisitor::eval,
                                 (bp::arg("self"), "position"))
                .ADD_PROPERTY_GET_WITH_POLICY("py_function",
                                              &PyHeightmapFunctionVisitor::getPyFun,
                                              bp::return_value_policy<bp::return_by_value>());
                ;
            // clang-format on
        }

        static bp::tuple eval(HeightmapFunction & self, const Eigen::Vector2d & position)
        {
            double height;
            Eigen::Vector3d normal;
            self(position, height, normal);
            return bp::make_tuple(height, normal);
        }

        static bp::object getPyFun(HeightmapFunction & self)
        {
            HeightmapFunPyWrapper * pyWrapper(self.target<HeightmapFunPyWrapper>());
            if (!pyWrapper || pyWrapper->heightmapType_ != heightmapType_t::GENERIC)
            {
                return {};
            }
            return pyWrapper->handlePyPtr_;
        }

        static std::shared_ptr<HeightmapFunction> factory(bp::object & objPy,
                                                          heightmapType_t objType)
        {
            return std::make_shared<HeightmapFunction>(HeightmapFunPyWrapper(objPy, objType));
        }

        static void expose()
        {
            // clang-format off
            bp::class_<HeightmapFunction,
                       std::shared_ptr<HeightmapFunction>>("HeightmapFunction", bp::no_init)
                .def(PyHeightmapFunctionVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(HeightmapFunction)
}
