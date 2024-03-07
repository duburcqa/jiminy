#include "pinocchio/spatial/force.hpp"  // `Pinocchio::Force`

#include "jiminy/python/functors.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ************************************** FunPyWrapper ************************************* //

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

    // *********************************** HeightmapFunction *********************************** //

    namespace internal::heightmap_function
    {
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
            if (!pyWrapper || pyWrapper->heightmapType_ != HeightmapType::GENERIC)
            {
                return {};
            }
            return pyWrapper->handlePyPtr_;
        }

        static std::shared_ptr<HeightmapFunction> factory(bp::object & objPy,
                                                          HeightmapType objType)
        {
            return std::make_shared<HeightmapFunction>(HeightmapFunPyWrapper(objPy, objType));
        }
    }

    void exposeHeightmapFunction()
    {
        bp::class_<HeightmapFunction, std::shared_ptr<HeightmapFunction>>("HeightmapFunction",
                                                                          bp::no_init)
            .def("__init__",
                 bp::make_constructor(&internal::heightmap_function::factory,
                                      bp::default_call_policies(),
                                      (bp::arg("heightmap_function"),
                                       bp::arg("heightmap_type") = HeightmapType::GENERIC)))
            .def("__call__", &internal::heightmap_function::eval, (bp::arg("self"), "position"))
            .ADD_PROPERTY_GET_WITH_POLICY("py_function",
                                          &internal::heightmap_function::getPyFun,
                                          bp::return_value_policy<bp::return_by_value>());
    }
}
