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
    // Constants describing the behavior of step adjustement.
    namespace RK
    {
        float64_t const ERROR_EXPONENT = -1.0 / 3.0; ///< Exponent used to scale the error.
        float64_t const SAFETY = 0.9; ///< Safety factor when updating the error, should be less than 1.
        float64_t const MIN_FACTOR = 0.2; ///< Miminum allowed relative step decrease.
        float64_t const MAX_FACTOR = 10.0; ///< Maximum allowed relative step increase.
    }

    class RungeKuttaStepper: public AbstractStepper
    {
        public:
            /// \brief Constructor
            /// \param[in] f      Dynamics function, with signature a = f(t, q, v)
            /// \param[in] robots Robots whose dynamics the stepper will work on.
            /// \param[in] tolRel Relative tolerance, used to determine step success and timestep update.
            /// \param[in] tolAbs Relative tolerance, used to determine step success and timestep update.
            RungeKuttaStepper(systemDynamics f, /* Copy on purpose */
                              std::vector<Robot const *> const & robots,
                              float64_t const & tolRel,
                              float64_t const & tolAbs,
                              matrixN_t const & RungeKuttaMatrix,
                              vectorN_t const & cNodes,
                              vectorN_t const & bWeights,
                              vectorN_t const & eWeights);

        protected:
            /// \brief Internal tryStep method wrapping the arguments as state_t and stateDerivative_t.
            bool tryStepImpl(state_t           & state,
                             stateDerivative_t & stateDerivative,
                             float64_t   const & t,
                             float64_t         & dt) final override;

        private:
            float64_t tolRel_; ///< Relative tolerance
            float64_t tolAbs_; ///< Absolute tolerance
            matrixN_t A_;      ///< A weight matrix.
            vectorN_t c_;      ///< Nodes
            vectorN_t b_;      ///< Solution eights
            vectorN_t e_;      ///< Error evaluation weights
    };
}

#endif //end of JIMINY_RUNGE_KUTTA_STEPPER_H
