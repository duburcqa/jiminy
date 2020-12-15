///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Generic interface for a stepper to integrate the equation of dynamics.
///
/// \details    The generalized position of a mechanical system evolves not in a vector space,
///             but in a Lie group: this is visible for instance in the fact that quaternions
///             have four components, but only three are meaningful, as quaternion must remain
///             unitary to represent a rotation. As such, the velocity vector v is not the
///             term-by-term derivative of the configuration q.
///             This means that classical stepper implementations cannot be used as is: this
///             class thus defines an interface for implementing variable-step numerical
///             solvers on this type of system.
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_ABSTRACT_STEPPER_H
#define JIMINY_ABSTRACT_STEPPER_H

#include <functional>

#include "jiminy/core/Types.h"
#include "jiminy/core/stepper/LieGroup.h"

namespace jiminy
{
    using systemDynamics = std::function<void(float64_t const & /*t*/,
                                              std::vector<vectorN_t> const & /*qSplit*/,
                                              std::vector<vectorN_t> const & /*vSplit*/,
                                              std::vector<vectorN_t> & /*aSplit*/)>;

    class AbstractStepper
    {
    public:
        /// \brief Constructor
        /// \param[in] f       Dynamics function, with signature a = f(t, q, v)
        /// \param[in] robots  Robots whose dynamics the stepper will work on.
        AbstractStepper(systemDynamics f, /* Copy on purpose */
                        std::vector<Robot const *> const & robots);

        /// \brief Attempt to integrate the system from time t to time t + dt.
        /// \details Each stepper is responsible for implementing its own error checking routine to determine
        ///          if an integration step was successful or not. On success, the value of the state is
        ///          updated in place. Regardless of success or failure, the parameter dt can be updated
        ///          by variable step schemes to indicate the next advised dt.
        ///
        /// \param[in, out] q   System starting position.
        /// \param[in, out] v   System starting velocity.
        /// \param[in, out] a   System starting acceleration.
        /// \param[in, out] t   Integration start time.
        /// \param[in, out] dt  Input: desired integration duration. Output: recommended step size for
        ///                     variable-step schemes. Constant-step schemes leave this value unmodified.
        /// \return True if integration was successful, false otherwise. In that case, (q, v, a) are not updated.
        bool_t tryStep(std::vector<vectorN_t> & q,
                       std::vector<vectorN_t> & v,
                       std::vector<vectorN_t> & a,
                       float64_t              & t,
                       float64_t              & dt);

    protected:
        /// \brief Internal tryStep method wrapping the arguments as state_t and stateDerivative_t.
        virtual bool_t tryStepImpl(state_t                 & state,
                                   stateDerivative_t       & stateDerivative,
                                   float64_t         const & t,
                                   float64_t               & dt) = 0;

        /// \brief Wrapper around the system dynamics: stateDerivative = f(t, state)
        stateDerivative_t const & f(float64_t const & t,
                                    state_t   const & state);

    private:
        systemDynamics f_;                   ///< Dynamics to integrate.
        std::vector<Robot const *> robots_;  ///< Robots on which to perform integration.
        state_t state_;                      ///< State derivative computation buffer.
        stateDerivative_t stateDerivative_;  ///< State derivative computation buffer.
        stateDerivative_t fOutput_;          ///< State derivative computation buffer.
    };
}

#endif //end of JIMINY_ENGINE_MULTIROBOT_H
