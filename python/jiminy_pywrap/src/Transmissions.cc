#include "jiminy/core/robot/BasicTransmissions.h"

#include <boost/python.hpp>

#include "jiminy/python/Utilities.h"
#include "jiminy/python/Transmissions.h"


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    // ***************************** PyAbstractTransmissionVisitor ***********************************

    struct PyAbstractTransmissionVisitor
        : public bp::def_visitor<PyAbstractTransmissionVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////

        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .add_property("is_initialized", bp::make_function(&AbstractTransmissionBase::getIsInitialized,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("name", bp::make_function(&AbstractTransmissionBase::getName,
                                        bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("idx", bp::make_function(&AbstractTransmissionBase::getIdx,
                                        bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("joint_names", bp::make_function(&AbstractTransmissionBase::getJointNames,
                                             bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("joint_indices", bp::make_function(&AbstractTransmissionBase::getJointModelIndices,
                                               bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("joint_types", bp::make_function(&AbstractTransmissionBase::getJointTypes,
                                             bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("joint_position_indices", bp::make_function(&AbstractTransmissionBase::getJointPositionIndices,
                                                        bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("joint_velocity_indices", bp::make_function(&AbstractTransmissionBase::getJointVelocityIndices,
                                                        bp::return_value_policy<bp::copy_const_reference>()))

                .def("set_options", &PyAbstractTransmissionVisitor::setOptions)
                .def("get_options", &AbstractTransmissionBase::getOptions)
                .def("compute_transform", &AbstractTransmissionBase::computeTransform)
                .def("compute_inverse_transform", &AbstractTransmissionBase::computeInverseTransform)
                ;
        }

    public:
        static void setOptions(AbstractTransmissionBase       & self,
                               bp::dict          const & configPy)
        {
            configHolder_t config = self.getOptions();
            convertFromPython(configPy, config);
            self.setOptions(config);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<AbstractTransmissionBase,
                       std::shared_ptr<AbstractTransmissionBase>,
                       boost::noncopyable>("AbstractTransmission", bp::no_init)
                .def(PyAbstractTransmissionVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(AbstractTransmission)

    // ***************************** PySimpleTransmissionVisitor ***********************************

    struct PySimpleTransmissionVisitor
        : public bp::def_visitor<PySimpleTransmissionVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////

        template<class PyClass>
        class PyTransmissionVisitorImpl
        {
        public:
            using TTransmission = typename PyClass::wrapped_type;

            static void visit(PyClass & cl)
            {
                cl
                    .def("initialize", &TTransmission::initialize)
                    ;
            }
        };

    public:
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            PyTransmissionVisitorImpl<PyClass>::visit(cl);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<SimpleTransmission, bp::bases<AbstractTransmissionBase>,
                       std::shared_ptr<SimpleTransmission>,
                       boost::noncopyable>("SimpleTransmission",
                       bp::init<std::string const &>(
                       bp::args("self", "transmission_name")))
                .def(PySimpleTransmissionVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(SimpleTransmission)
}  // End of namespace python.
}  // End of namespace jiminy.
