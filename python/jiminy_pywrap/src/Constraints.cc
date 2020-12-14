#include "jiminy/core/robot/AbstractConstraint.h"
#include "jiminy/core/robot/FixedFrameConstraint.h"
#include "jiminy/core/robot/WheelConstraint.h"

#include "jiminy/python/Constraints.h"

#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/dict.hpp>


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    // ***************************** PyConstraintVisitor ***********************************

    struct PyConstraintVisitor
        : public bp::def_visitor<PyConstraintVisitor>
    {
    public:
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .add_property("jacobian", bp::make_function(&AbstractConstraint::getJacobian,
                                          bp::return_internal_reference<>()))
                .add_property("drift", bp::make_function(&AbstractConstraint::getDrift,
                                       bp::return_internal_reference<>()))
                ;
        }

    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<AbstractConstraint,
                       std::shared_ptr<AbstractConstraint>,
                       boost::noncopyable>("AbstractConstraint", bp::no_init)
                .def(PyConstraintVisitor());

            bp::class_<FixedFrameConstraint, bp::bases<AbstractConstraint>,
                       std::shared_ptr<FixedFrameConstraint>,
                       boost::noncopyable>("FixedFrameConstraint", bp::init<std::string>());

            bp::class_<WheelConstraint, bp::bases<AbstractConstraint>,
                       std::shared_ptr<WheelConstraint>,
                       boost::noncopyable>("WheelConstraint", bp::init<std::string, float64_t, vector3_t, vector3_t>());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(Constraint)
}  // End of namespace python.
}  // End of namespace jiminy.
