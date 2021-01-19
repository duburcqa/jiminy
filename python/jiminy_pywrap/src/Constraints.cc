#include "jiminy/core/robot/AbstractConstraint.h"
#include "jiminy/core/robot/FixedFrameConstraint.h"
#include "jiminy/core/robot/WheelConstraint.h"

#include "jiminy/python/Functors.h"
#include "jiminy/python/Constraints.h"

#include <boost/python.hpp>


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    // ***************************** PyConstraintVisitor ***********************************

    // Using an intermediary class is a trick to enable defining bp::base<...> in conjunction with bp::wrapper<...>
    class AbstractConstraintImpl: public AbstractConstraint {};

    class AbstractConstraintWrapper: public AbstractConstraintImpl, public bp::wrapper<AbstractConstraintImpl>
    {
    public:
        hresult_t reset(void)
        {
            bp::override func = this->get_override("reset");
            if (func)
            {
                func();
            }
            return hresult_t::SUCCESS;
        }

        hresult_t computeJacobianAndDrift(vectorN_t const & q,
                                          vectorN_t const & v)
        {
            bp::override func = this->get_override("compute_jacobian_and_drift");
            if (func)
            {
                func(FctPyWrapperArgToPython(q),
                     FctPyWrapperArgToPython(v));
            }
            return hresult_t::SUCCESS;
        }
    };

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

            bp::class_<AbstractConstraintWrapper, bp::bases<AbstractConstraint>,
                       std::shared_ptr<AbstractConstraintWrapper>,
                       boost::noncopyable>("BaseConstraint")
                .def("reset", bp::pure_virtual(&AbstractConstraint::reset))
                .def("compute_jacobian_and_drift", bp::pure_virtual(&AbstractConstraint::computeJacobianAndDrift))
                ;

            bp::class_<FixedFrameConstraint, bp::bases<AbstractConstraint>,
                       std::shared_ptr<FixedFrameConstraint>,
                       boost::noncopyable>("FixedFrameConstraint", bp::init<std::string>())
                .def("reset", &FixedFrameConstraint::reset)
                .def("compute_jacobian_and_drift", &FixedFrameConstraint::computeJacobianAndDrift,
                                                   (bp::arg("self"), "q", "v"))
                ;

            bp::class_<WheelConstraint, bp::bases<AbstractConstraint>,
                       std::shared_ptr<WheelConstraint>,
                       boost::noncopyable>("WheelConstraint", bp::init<std::string, float64_t, vector3_t, vector3_t>())
                .def("reset", &WheelConstraint::reset)
                .def("compute_jacobian_and_drift", &WheelConstraint::computeJacobianAndDrift,
                                                   (bp::arg("self"), "q", "v"))
                ;
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(Constraint)
}  // End of namespace python.
}  // End of namespace jiminy.
