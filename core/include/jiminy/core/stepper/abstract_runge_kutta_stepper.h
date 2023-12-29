#ifndef JIMINY_RUNGE_KUTTA_STEPPER_H
#define JIMINY_RUNGE_KUTTA_STEPPER_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/stepper/abstract_stepper.h"

namespace jiminy
{
    /// \brief Implements genering Runge-Kutta methods with step adjustement.
    /// \details This class constructs a Runge-Kutta stepper from the supplied weights forming the
    ///          Butcher table (a_ij, c_i, b_i) ; the vector e is used in error evaluation for step
    ///          adjustment.
    class JIMINY_DLLAPI AbstractRungeKuttaStepper : public AbstractStepper
    {
    public:
        /// \param[in] f Dynamics function, with signature `a = f(t, q, v)`.
        /// \param[in] robots Robots whose dynamics the stepper will work on.
        explicit AbstractRungeKuttaStepper(const systemDynamics & f,
                                           const std::vector<const Robot *> & robots,
                                           const Eigen::MatrixXd & RungeKuttaMatrix,
                                           const Eigen::VectorXd & bWeights,
                                           const Eigen::VectorXd & cNodes,
                                           bool isFSAL) noexcept;
        virtual ~AbstractRungeKuttaStepper() = default;

    protected:
        /// \brief Internal tryStep method wrapping the arguments as state_t and stateDerivative_t.
        virtual bool tryStepImpl(state_t & state,
                                 stateDerivative_t & stateDerivative,
                                 double t,
                                 double & dt) final override;

        /// \brief Determine if step has succeeded or failed, and adjust dt.
        ///
        /// \param[in] intialState Starting state, used to compute alternative estimates of the
        ///                        solution.
        /// \param[in] solution Current solution computed by the main Runge-Kutta step.
        /// \param[in, out] dt Timestep to be scaled.
        ///
        /// \return Whether the step is successful. The timestep dt is updated in place.
        virtual bool adjustStep(
            const state_t & initialState, const state_t & solution, double & dt);

    private:
        /// \brief Weight matrix.
        Eigen::MatrixXd A_;
        /// \brief Solution coefficients.
        Eigen::VectorXd b_;
        /// \brief Nodes
        Eigen::VectorXd c_;
        /// \brief Does scheme support first-same-as-last.
        bool isFSAL_;

    protected:
        /// \brief Derivatives at knots.
        std::vector<stateDerivative_t> ki_;
        /// \brief Intermediary computation of state increment.
        stateDerivative_t stateIncrement_;
        /// \brief Intermediary state during knots computations.
        state_t stateBuffer_;
        /// \brief Candidate solution before knowing if the step is successful.
        state_t candidateSolution_;
    };
}

#endif  // end of JIMINY_RUNGE_KUTTA_STEPPER_H
