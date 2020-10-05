#include "pinocchio/algorithm/joint-configuration.hpp"

#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/stepper/LieGroup.h"

namespace jiminy
{
    state_t::state_t(std::vector<Robot const *> robots,
                     std::vector<vectorN_t> const & qIn,
                     std::vector<vectorN_t> const & vIn):
    q(qIn),
    v(vIn),
    robots_(robots)
    {
        assert(q.size() == robots.size() && v.size() == robots.size()); // TODO : Check consistency between robot[i].pncModel_.nq/nv and q[i]/v[i]
    }

    state_t::state_t(state_t const& state):
    q(state.q),
    v(state.v),
    robots_(state.robots_)
    {
        // Empty on purpose
    }


    state_t state_t::operator+(stateDerivative_t const & velocity)
    {
        assert(v.size() == velocity.v.size());

        state_t s(*this);
        for (uint32_t i = 0; i < v.size(); ++i)
        {
            // 'Sum' q = q + v, remember q is part of a Lie group (dim(q) != dim(v))
            pinocchio::integrate(robots_[i]->pncModel_, q[i], velocity.v[i], s.q[i]);
            s.v[i] += velocity.a[i];
        }
        return s;
    }


    void state_t::operator+=(stateDerivative_t const & velocity)
    {
        assert(v.size() == velocity.v.size());

        for (uint32_t i = 0; i < v.size(); ++i)
        {
            // 'Sum' q = q + v, remember q is part of a Lie group (dim(q) != dim(v))
            pinocchio::integrate(robots_[i]->pncModel_, q[i], velocity.v[i], q[i]);
            v[i] += velocity.a[i];
        }
    }


    stateDerivative_t state_t::difference(state_t const & other)
    {
        assert(v.size() == other.v.size());

        stateDerivative_t s(v, v);

        for (uint32_t i = 0; i < v.size(); i++)
        {
            pinocchio::difference(robots_[i]->pncModel_, q[i], other.q[i], s.v[i]);
            s.a[i] -= other.v[i];
        }
        return s;
    }


    float64_t state_t::normInf(void)
    {
        float64_t norm = 0.0;
        for (uint32_t i = 0; i < v.size(); ++i)
        {
            float64_t n = q[i].lpNorm<Eigen::Infinity>();
            if (n > norm)
            {
                norm = n;
            }
            n = v[i].lpNorm<Eigen::Infinity>();
            if (n > norm)
            {
                norm = n;
            }
        }
        return norm;
    }


    stateDerivative_t::stateDerivative_t(std::vector<vectorN_t> const & vIn,
                                         std::vector<vectorN_t> const & aIn):
    v(vIn),
    a(aIn)
    {
        assert(v.size() == a.size()); // TODO : Check consistency between q[i].size() and v[i].size()
    }


    stateDerivative_t::stateDerivative_t(stateDerivative_t const & stateIn):
    v(stateIn.v),
    a(stateIn.a)
    {
        // Empty on purpose.
    }


    stateDerivative_t stateDerivative_t::operator+(stateDerivative_t const & other)
    {
        assert(v.size() == other.v.size());

        stateDerivative_t s(*this);

        for (uint32_t i = 0; i < v.size(); ++i)
        {
            s.v[i] += other.v[i];
            s.a[i] += other.a[i];
        }
        return s;
    }


    float64_t stateDerivative_t::norm(void)
    {
        float64_t norm = 0.0;
        for (uint32_t i = 0; i < v.size(); ++i)
        {
            norm += v[i].norm();
            norm += a[i].norm();
        }
        return norm;
    }


    stateDerivative_t operator*(float64_t         const & alpha,
                                stateDerivative_t const & state)
    {
        stateDerivative_t s(state);
        for (uint32_t i = 0; i < s.v.size(); ++i)
        {
            s.v[i] *= alpha;
            s.a[i] *= alpha;
        }
        return s;
    }
}
