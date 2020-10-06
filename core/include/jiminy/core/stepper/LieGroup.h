///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Handling of the joint configuration q as part of a Lie Group, not a vector space.
///
/// \details    The state of the system, (q, v), lies in a Lie Group, while its derivative
///             (v, a) is a tangent vector to this group: the classes in this file are
///             meant to handle this.
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_LIE_GROUP_H
#define JIMINY_LIE_GROUP_H

#include "jiminy/core/Types.h"

namespace jiminy
{
    class Robot;
    struct stateDerivative_t;

    struct state_t
    {
        friend struct stateDerivative_t;

        public:
            /// \brief Constructor
            state_t(std::vector<Robot const *> const & robots);
            state_t(std::vector<Robot const *> const & robots,
                    std::vector<vectorN_t>     const & qIn,
                    std::vector<vectorN_t>     const & vIn);
            state_t(std::vector<Robot const *> const & robots,
                    std::vector<vectorN_t>          && qIn,
                    std::vector<vectorN_t>          && vIn);
            state_t(state_t && other);
            state_t(state_t const & other);

            state_t & operator=(state_t const & other);
            state_t & operator=(state_t && other);

            /// \brief Compute the state reached from the current state given the velocity.
            /// \details This function returns the state reached from this with the velocity
            ///          v (integrated over 1s): a vector-space equivalent would be the simple
            ///          sum stateOut = this + velocity. However, since the state evolves in a
            ///          Lie group, the actual operation is stateOut = exp(velocity) * this
            state_t & operator+=(stateDerivative_t const & velocity);
            friend state_t operator+(state_t                   position,  // copy on purpose
                                     stateDerivative_t const & velocity);

            /// \brief Compute the difference between two states, i.e. the velocity to go from other to this.
            stateDerivative_t difference(state_t const & other) const;

            /// \brief Compute the norm of the current state.
            ///
            /// \details This function returns the infinity-norm (max(abs(x))) of (q_, v_),
            ///          and is meant to be used for step adjustement by steppers.
            float64_t normInf(void)  const;

        public:
            uint32_t nrobots;
            std::vector<vectorN_t> q;
            std::vector<vectorN_t> v;

        protected:
            std::vector<Robot const *> robots_;
    };

    struct stateDerivative_t
    {
        friend struct state_t;

        public:
            /// \brief Constructor
            stateDerivative_t(std::vector<Robot const *> const & robots);
            stateDerivative_t(std::vector<Robot const *> const & robots,
                              std::vector<vectorN_t>     const & vIn,
                              std::vector<vectorN_t>     const & aIn);
            stateDerivative_t(std::vector<Robot const *> const & robots,
                              std::vector<vectorN_t>          && vIn,
                              std::vector<vectorN_t>          && aIn);
            stateDerivative_t(stateDerivative_t && other);
            stateDerivative_t(stateDerivative_t const & other);

            stateDerivative_t & operator=(stateDerivative_t const & other);
            stateDerivative_t & operator=(stateDerivative_t && other);

            /// \brief Sum operator.
            /// \details Given (v1, a1) and (v2, a2), this simply returns (v1 + v2, a1 + a2)
            /// \param[in] other Other state derivative to sum.
            stateDerivative_t & operator+=(stateDerivative_t const & other);
            friend stateDerivative_t operator+(stateDerivative_t         velocity,
                                               stateDerivative_t const & other);

            /// \brief Multiplication by a scalar.
            /// \details Given alpha and (v1, a1), this simply returns (alpha * v1, alpha * a1)
            /// \param[in] alpha Scalar coefficient.
            /// \param[in] state state_t to scale.
            friend stateDerivative_t operator*(float64_t         const & alpha,
                                               stateDerivative_t         velocity);  // copy on purpose.

            /// \brief Compute the norm-2 of the current stateDerivative_t.
            ///
            /// \details This function simply computes the induced norm
            ///          sum(norm2(x) for x in v, a)
            float64_t norm(void) const;

        public:
            uint32_t nrobots;
            std::vector<vectorN_t> v;
            std::vector<vectorN_t> a;

        protected:
            std::vector<Robot const *> robots_;
    };
}

#endif //end of JIMINY_LIE_GROUP_H
