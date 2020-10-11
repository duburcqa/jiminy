///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Implements genering Runge-Kutta methods with step adjustement.
/// \details    This class constructs a Runge-Kutta stepper from the supplied weights forming the
///             Butcher table (a_ij, c_i, b_i) ; the vector e is used in error evaluation for
///             step adjustement.
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_RUNGE_KUTTA_STEPPER_H
#define JIMINY_RUNGE_KUTTA_STEPPER_H

#include "jiminy/core/stepper/AbstractStepper.h"

namespace jiminy
{
    class AbstractRungeKuttaStepper: public AbstractStepper
    {
        public:
            /// \brief Constructor
            /// \param[in] f      Dynamics function, with signature a = f(t, q, v)
            /// \param[in] robots Robots whose dynamics the stepper will work on.
            AbstractRungeKuttaStepper(systemDynamics f, /* Copy on purpose */
                                      std::vector<Robot const *> const & robots,
                                      matrixN_t const & RungeKuttaMatrix,
                                      vectorN_t const & bWeights,
                                      vectorN_t const & cNodes,
                                      bool_t    const & isFSAL);

        protected:
            /// \brief Internal tryStep method wrapping the arguments as state_t and stateDerivative_t.
            virtual bool_t tryStepImpl(state_t                 & state,
                                       stateDerivative_t       & stateDerivative,
                                       float64_t         const & t,
                                       float64_t               & dt) final override;

            /// \brief Determine if step has succeeded or failed, and adjust dt.
            /// \param[in] intialState Starting state, used to compute alternative estimates of the solution.
            /// \param[in] solution Current solution computed by the main Runge-Kutta step.
            /// \param[in, out] dt  Timestep to be scaled.
            /// \return True on step success, false otherwise. dt is updated in place.
            virtual bool_t adjustStep(state_t   const & initialState,
                                      state_t   const & solution,
                                      float64_t       & dt);

        private:
            matrixN_t A_;    ///< Weight matrix.
            vectorN_t b_;    ///< Solution coefficients.
            vectorN_t c_;    ///< Nodes
            bool_t isFSAL_;  ///< Does scheme support first-same-as-last.

        protected:
            std::vector<stateDerivative_t> ki_;  ///< Internal computation steps.
            stateDerivative_t stateIncrement_;   ///< Internal buffer storing intermediary computation of state increment.
            state_t stateBuffer_;                ///< Internal buffer storing intermediary state during knots computations.
            state_t candidateSolution_;          ///< Internal buffer storing the candidate solution (i.e. before knowing if the step is successful).

    };
}

#endif //end of JIMINY_RUNGE_KUTTA_STEPPER_H
