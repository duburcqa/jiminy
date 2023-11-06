#ifndef JIMINY_RUNGE_KUTTA_DOPRI_STEPPER_H
#define JIMINY_RUNGE_KUTTA_DOPRI_STEPPER_H

#include "jiminy/core/macros.h"
#include "jiminy/core/stepper/abstract_runge_kutta_stepper.h"

namespace jiminy
{
    namespace DOPRI
    {
        // clang-format off
        const Eigen::MatrixXd A((Eigen::MatrixXd(7, 7) <<
                        0.0,               0.0,              0.0,            0.0,               0.0,         0.0, 0.0,
                  1.0 / 5.0,               0.0,              0.0,            0.0,               0.0,         0.0, 0.0,
                 3.0 / 40.0,        9.0 / 40.0,              0.0,            0.0,               0.0,         0.0, 0.0,
                44.0 / 45.0,      -56.0 / 15.0,       32.0 / 9.0,            0.0,               0.0,         0.0, 0.0,
            19372.0 /6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0,               0.0,         0.0, 0.0,
            9017.0 / 3168.0,     -355.0 / 33.0, 46732.0 / 5247.0,   49.0 / 176.0, -5103.0 / 18656.0,         0.0, 0.0,
               35.0 / 384.0,               0.0,   500.0 / 1113.0,  125.0 / 192.0,  -2187.0 / 6784.0, 11.0 / 84.0, 0.0
        ).finished());
        const Eigen::VectorXd c((Eigen::VectorXd(7) <<
            0.0, 2.0 / 10.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0
        ).finished());
        const Eigen::VectorXd b((Eigen::VectorXd(7) <<
            35.0/ 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0
        ).finished());
        const Eigen::VectorXd e((Eigen::VectorXd(7) <<
            5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0
        ).finished());
        // clang-format on

        // These parameters are coming from boost's stepper implementation.

        /// \brief Stepper order, used to scale the error.
        const float64_t STEPPER_ORDER = 5.0;
        /// \brief Safety factor when updating the error, should be less than 1.
        const float64_t SAFETY = 0.8;
        /// \brief Miminum allowed relative step decrease.
        const float64_t MIN_FACTOR = 0.2;
        /// \brief Maximum allowed relative step increase.
        const float64_t MAX_FACTOR = 5.0;
    }

    /// \brief Implements the Dormand-Prince Runge-Kutta algorithm.
    class JIMINY_DLLAPI RungeKuttaDOPRIStepper : public AbstractRungeKuttaStepper
    {
    public:
        /// \param[in] f Dynamics function, with signature a = f(t, q, v)
        /// \param[in] robots Robots whose dynamics the stepper will work on.
        /// \param[in] tolRel Relative tolerance used to assess step success and timestep update.
        /// \param[in] tolAbs Absolute tolerance used to assess step success and timestep update.
        RungeKuttaDOPRIStepper(const systemDynamics & f,
                               const std::vector<const Robot *> & robots,
                               const float64_t & tolRel,
                               const float64_t & tolAbs);

    protected:
        /// \brief Determine if step has succeeded or failed, and adjust dt.
        ///
        /// \param[in] intialState Starting state used to compute alternative estimates of the
        ///                        solution.
        /// \param[in] solution Current solution computed by the main Runge-Kutta step.
        /// \param[in, out] dt Timestep to be scaled.
        ///
        /// \return True on step success, false otherwise. dt is updated in place.
        virtual bool_t adjustStep(
            const state_t & initialState, const state_t & solution, float64_t & dt) override final;

    private:
        /// \brief Run error computation algorithm to return normalized error.
        ///
        /// \param[in] intialState Starting state, used to compute alternative estimates of the
        ///                        solution.
        /// \param[in] solution Current solution computed by the main Runge-Kutta step.
        ///
        /// \returns Normalized error, >1 indicates step failure.
        float64_t computeError(
            const state_t & initialState, const state_t & solution, const float64_t & dt);

        /// \brief Scale timestep based on normalized error value.
        bool_t adjustStepImpl(const float64_t & error, float64_t & dt);

    private:
        /// \brief Relative tolerance.
        float64_t tolRel_;
        /// \brief Absolute tolerance.
        float64_t tolAbs_;
        /// \brief Internal buffer for error scale using during relative error computation.
        stateDerivative_t scale_;
        /// \brief Internal buffer for alternative solution during error computation.
        state_t otherSolution_;
        /// \brief Internal buffer for difference between solutions during error computation.
        stateDerivative_t error_;
    };
}

#endif  // end of JIMINY_RUNGE_KUTTA_DOPRI_STEPPER_H
