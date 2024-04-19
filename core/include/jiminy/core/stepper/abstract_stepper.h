#ifndef JIMINY_ABSTRACT_STEPPER_H
#define JIMINY_ABSTRACT_STEPPER_H

#include <optional>
#include <functional>
#include <exception>

#include "jiminy/core/fwd.h"
#include "jiminy/core/stepper/lie_group.h"

namespace jiminy
{
    namespace stepper
    {
        enum class ReturnCode : uint8_t
        {
            IS_SUCCESS = 0,
            IS_FAILURE = 1,
            IS_ERROR = 2
        };

        struct JIMINY_DLLAPI StatusInfo
        {
            ReturnCode returnCode;
            std::exception_ptr exception;
        };
    }

    using systemDynamics = std::function<void(double /*t*/,
                                              const std::vector<Eigen::VectorXd> & /*qSplit*/,
                                              const std::vector<Eigen::VectorXd> & /*vSplit*/,
                                              std::vector<Eigen::VectorXd> & /*aSplit*/)>;

    /// \brief Generic interface for steppers used to integrate the dynamic equations of motion.
    ///
    /// \details The generalized position of a dynamical system evolves not in a vector space, but
    ///          in a Lie group: this is visible for instance in the fact that quaternions have 4
    ///          components, but only 3 are degrees of freedom, as quaternion must remain unitary
    ///          to represent a rotation. As such, the velocity vector v is not the term-by-term
    ///          derivative of the configuration q. This means that classical stepper
    ///          implementations cannot be used as is: this class thus defines an interface for
    ///          implementing variable-step numerical solvers on this type of system.
    class JIMINY_DLLAPI AbstractStepper
    {
    public:
        /// \param[in] f Dynamics function, with signature `a = f(t, q, v)`.
        /// \param[in] robots Robots whose dynamics the stepper will work on.
        explicit AbstractStepper(const systemDynamics & f,
                                 const std::vector<const Robot *> & robots) noexcept;
        virtual ~AbstractStepper() = default;

        /// \brief Attempt to integrate the system from time t to time t + dt.
        ///
        /// \details Each stepper is responsible for implementing its own error checking routine to
        ///          assess whether an integration step was successful. On success, the value of
        ///          the state is updated in place. Regardless of success or failure, the parameter
        ///          dt an be updated by variable step schemes to indicate the next advised dt.
        ///
        /// \param[in, out] q System starting position.
        /// \param[in, out] v System starting velocity.
        /// \param[in, out] a System starting acceleration.
        /// \param[in, out] t Integration start time.
        /// \param[in, out] dt Input: desired integration duration. Output: recommended step size
        ///                    for variable-step schemes. Constant-step schemes leave this value
        ///                    unmodified.
        ///
        /// \return Whether integration was successful. If not, (q, v, a) are not updated.
        stepper::StatusInfo tryStep(std::vector<Eigen::VectorXd> & q,
                                    std::vector<Eigen::VectorXd> & v,
                                    std::vector<Eigen::VectorXd> & a,
                                    double & t,
                                    double & dt);

    protected:
        /// \brief Internal tryStep method wrapping the arguments as State and
        /// StateDerivative.
        virtual bool tryStepImpl(
            State & state, StateDerivative & stateDerivative, double t, double & dt) = 0;

        /// \brief Wrapper around the system dynamics: `stateDerivative = f(t, state)`.
        const StateDerivative & f(double t, const State & state);

    private:
        /// \brief Dynamics to integrate.
        systemDynamics f_;
        /// \brief Robots on which to perform integration.
        std::vector<const Robot *> robots_;
        /// \brief State derivative computation buffer.
        State state_;
        /// \brief State derivative computation buffer.
        StateDerivative stateDerivative_;
        /// \brief State derivative computation buffer.
        StateDerivative fOutput_;
    };
}

#endif  // JIMINY_ABSTRACT_STEPPER_H
