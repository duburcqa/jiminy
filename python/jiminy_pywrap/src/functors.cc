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

    void queryHeightMap(HeightmapFunction & heightmap,
                        const Eigen::Matrix2Xd & positions,
                        Eigen::Ref<Eigen::VectorXd> heights)
    {
        // Make sure that the number of query points is consistent between all arguments
        if (heights.size() != positions.cols())
        {
            JIMINY_THROW(
                std::invalid_argument,
                "'positions' and/or 'heights' are inconsistent with each other. 'position' must "
                "be a 2D array whose first dimension gathers the 2 position coordinates in world "
                "plane (X, Y) while the second dimension corresponds to individual query points.");
        }

        // Loop over all query points sequentially
        for (Eigen::Index i = 0; i < positions.cols(); ++i)
        {
            heightmap(positions.col(i), heights[i], std::nullopt);
        }
    }

    void queryHeightMapWithNormals(HeightmapFunction & heightmap,
                                   const Eigen::Matrix2Xd & positions,
                                   Eigen::Ref<Eigen::VectorXd> heights,
                                   Eigen::Ref<Eigen::Matrix3Xd> normals)
    {
        // Make sure that the number of query points is consistent between all arguments
        if (heights.size() != positions.cols() || normals.cols() != positions.cols())
        {
            JIMINY_THROW(std::invalid_argument,
                         "'positions', 'heights' and/or 'normals' are inconsistent with each "
                         "other. 'normals' must be a 2D array whose first dimension gathers the 3 "
                         "position coordinates (X, Y, Z) while the second dimension corresponds "
                         "to individual query points.");
        }

        // Loop over all query points sequentially
        for (Eigen::Index i = 0; i < positions.cols(); ++i)
        {
            heightmap(positions.col(i), heights[i], normals.col(i));
        }
    }

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

        bp::def(
            "query_heightmap", &queryHeightMap, (bp::args("heightmap"), "positions", "heights"));
        bp::def("query_heightmap",
                &queryHeightMapWithNormals,
                (bp::args("heightmap"), "positions", "heights", "normals"));
    }
}
