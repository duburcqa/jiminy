///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Implements the Dormand-Prince Runge-Kutta algorithm.
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_RUNGE_KUTTA_DOPRI_STEPPER_H
#define JIMINY_RUNGE_KUTTA_DOPRI_STEPPER_H

#include "jiminy/core/stepper/AbstractRungeKuttaStepper.h"

namespace jiminy
{
    namespace DOPRI
    {
        matrixN_t const A((matrixN_t(7, 7) << 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                                              1.0 / 5.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                                              3.0 / 40.0, 9.0 / 40.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                                              44.0 / 45.0, -56.0 / 15.0,  32.0 / 9.0,  0.0,  0.0,  0.0,  0.0,
                                              19372.0 /6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0,  0.0,  0.0,  0.0,
                                              9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0, 0.0,
                                              35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0).finished());
        vectorN_t const c((vectorN_t(7) << 0.0,
                                           2.0 / 10.0,
                                           3.0 / 10.0,
                                           4.0 / 5.0,
                                           8.0 / 9.0,
                                           1.0,
                                           1.0).finished());
        vectorN_t const b((vectorN_t(7) << 35.0/ 384.0,
                                           0.0,
                                           500.0 / 1113.0,
                                           125.0 / 192.0,
                                           -2187.0 / 6784.0,
                                           11.0 / 84.0,
                                           0.0).finished());
        vectorN_t const e((vectorN_t(7) << 5179.0 / 57600.0,
                                           0.0,
                                           7571.0 / 16695.0,
                                           393.0 / 640.0,
                                           -92097.0 / 339200.0,
                                           187.0 / 2100.0,
                                           1.0 / 40.0).finished());

        // These parameters are from boost's stepper implementation.
        float64_t const STEPPER_ORDER = 5.0;  ///< Stepper order, used to scale the error.
        float64_t const SAFETY = 0.9;         ///< Safety factor when updating the error, should be less than 1.
        float64_t const MIN_FACTOR = 0.2;     ///< Miminum allowed relative step decrease.
        float64_t const MAX_FACTOR = 5.0;     ///< Maximum allowed relative step increase.
    }

    class RungeKuttaDOPRIStepper: public AbstractRungeKuttaStepper
    {
        public:
            /// \brief Constructor
            /// \param[in] f      Dynamics function, with signature a = f(t, q, v)
            /// \param[in] robots Robots whose dynamics the stepper will work on.
            /// \param[in] tolRel Relative tolerance, used to determine step success and timestep update.
            /// \param[in] tolAbs Relative tolerance, used to determine step success and timestep update.
            RungeKuttaDOPRIStepper(systemDynamics f, /* Copy on purpose */
                                   std::vector<Robot const *> const & robots,
                                   float64_t const & tolRel,
                                   float64_t const & tolAbs);

        protected:
            /// \brief Determine if step has succeeded or failed, and adjust dt.
            /// \param[in] intialState Starting state, used to compute alternative estimates of the solution.
            /// \param[in] solution Current solution computed by the main Runge-Kutta step.
            /// \param[in, out] dt  Timestep to be scaled.
            /// \return True on step success, false otherwise. dt is updated in place.
            virtual bool adjustStep(state_t   const & initialState,
                                    state_t   const & solution,
                                    float64_t       & dt) override final;

        private:
            /// \brief Run error computation algorithm to return normalized error.
            /// \param[in] intialState Starting state, used to compute alternative estimates of the solution.
            /// \param[in] solution Current solution computed by the main Runge-Kutta step.
            /// \return Normalized error, >1 indicates step failure.
            float64_t computeError(state_t   const & initialState,
                                   state_t   const & solution,
                                   float64_t const & dt);

            /// \brief Scale timestep based on normalized error value.
            bool_t adjustStepImpl(float64_t const & error,
                                  float64_t       & dt);

        private:
            float64_t tolRel_;                 ///< Relative tolerance
            float64_t tolAbs_;                 ///< Absolute tolerance
            state_t alternativeSolution_;      ///< Internal buffer for alternative solution during error computation
            stateDerivative_t errorSolution_;  ///< Internal buffer for difference between solutions during error computation
    };
}

#endif //end of JIMINY_RUNGE_KUTTA_DOPRI_STEPPER_H
