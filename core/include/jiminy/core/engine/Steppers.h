
#ifndef JIMINY_STEPPERS_H
#define JIMINY_STEPPERS_H

#include <set>

#include "jiminy/core/Types.h"

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>


namespace jiminy
{
    using namespace boost::numeric::odeint;

    namespace stepper
    {
        using state_type = vectorN_t;  ///< x
        using deriv_type = vectorN_t;  ///< dxdt
        using time_type = float64_t;   ///< t
        using value_type = float64_t;  ///< err

        // ************************************************************************
        // **************************** Euler explicit ****************************
        // ************************************************************************

        class EulerExplicit
        {
        public:
            using stepper_category = controlled_stepper_tag;

            static unsigned short order(void)
            {
                return 1;
            }

            template<class System>
            controlled_step_result try_step(System       system,
                                            state_type & x,
                                            deriv_type & dxdt,
                                            time_type  & t,
                                            time_type  & dt) const
            {
                t += dt;
                system(x, dxdt, t);
                x += dt * dxdt;
                return controlled_step_result::success;
            }
        };

        // ************************************************************************
        // **************************** Bulirsch-Stoer ****************************
        // ************************************************************************

        using BulirschStoer = bulirsch_stoer<
            state_type,
            value_type,
            deriv_type,
            time_type,
            vector_space_algebra
        >;

        // ************************************************************************
        // *************************** Runge-Kutta Dopri **************************
        // ************************************************************************

        namespace runge_kutta
        {
            using Stepper = runge_kutta_dopri5<
                state_type,
                value_type,
                deriv_type,
                time_type,
                vector_space_algebra
            >;

            using ErrorChecker = default_error_checker<
                value_type,
                vector_space_algebra,
                typename operations_dispatcher<state_type>::operations_type
            >;

            template<typename Value, typename Time>
            class StepAdjusterImpl
            {
            public:
                using time_type = Time;
                using value_type = Value;

                StepAdjusterImpl(void) = default;

                Time decrease_step(Time          dt,
                                Value const & error,
                                int   const & error_order) const
                {
                    // Returns the decreased time step
                    dt *= std::max(
                        static_cast<value_type>(static_cast<value_type>(9) / static_cast<value_type>(10) *
                                                std::pow(error, static_cast<value_type>(-1) / (error_order - 1))),
                        static_cast<value_type>(static_cast<value_type>(1) / static_cast<value_type> (5))
                    );
                    return dt;
                }

                Time increase_step(Time          dt,
                                Value         error,
                                int   const & stepper_order) const
                {
                    if(error < 0.5)
                    {
                        // Error should be > 0
                        error = std::max(
                            error,
                            static_cast<value_type>(std::pow(static_cast<value_type>(5.0),
                                                    -static_cast<value_type>(stepper_order)))
                        );

                        // Error too small - increase dt and keep the evolution and limit scaling factor to 5.0
                        dt *= static_cast<value_type>(9)/static_cast<value_type>(10)
                            * pow(error, static_cast<value_type>(-1) / stepper_order);
                    }
                    return dt;
                }

                bool check_step_size_limit(Time const & dt) { return true; }
                Time get_max_dt(void) { return {0.0}; }
            };

            using StepAdjuster = StepAdjusterImpl<value_type, time_type>;
        }

        using RungeKutta = controlled_runge_kutta<
            runge_kutta::Stepper,
            runge_kutta::ErrorChecker,
            runge_kutta::StepAdjuster
        >;
    }

    // ****************************************************************
    // *************************** Helpers ****************************
    // ****************************************************************

    using stepper_t = boost::variant<
        stepper::BulirschStoer,
        stepper::RungeKutta,
        stepper::EulerExplicit
    >;

    std::set<std::string> const STEPPERS{"runge_kutta_dopri5",
                                         "bulirsch_stoer",
                                         "euler_explicit"};

    template<typename system_t>
    bool_t try_step(stepper_t & stepper,
                    system_t  & rhs,
                    vectorN_t & x,
                    vectorN_t & dxdt,
                    float64_t & t,
                    float64_t & dt)
    {
        return (boost::apply_visitor(
        [&rhs, &x, &dxdt, &t, &dt](auto && one) -> bool_t
        {
            return one.try_step(rhs, x, dxdt, t, dt);
        }, stepper) == success);
    }
}

#endif //end of JIMINY_STEPPERS_H
