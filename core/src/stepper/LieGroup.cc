#include "pinocchio/algorithm/joint-configuration.hpp"

#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/stepper/LieGroup.h"

namespace jiminy
{
    // ====================================================
    // ====================== state_t =====================
    // ====================================================

    state_t::state_t(std::vector<Robot const *> const & robots):
    nrobots(robots.size()),
    q(),
    v(),
    robots_(robots)
    {
        q.reserve(robots_.size());
        std::transform(robots_.begin(), robots_.end(),
                       std::back_inserter(q),
                       [](Robot const * robot) -> vectorN_t
                       {
                           return vectorN_t(robot->nq());
                       });
        v.reserve(robots_.size());
        std::transform(robots_.begin(), robots_.end(),
                       std::back_inserter(v),
                       [](Robot const * robot) -> vectorN_t
                       {
                           return vectorN_t(robot->nv());
                       });
    }

    state_t::state_t(std::vector<Robot const *> const & robots,
                     std::vector<vectorN_t>     const & qIn,
                     std::vector<vectorN_t>     const & vIn):
    nrobots(robots.size()),
    q(qIn),
    v(vIn),
    robots_(robots)
    {
        assert(q.size() == nrobots && v.size() == nrobots);
        for (uint32_t i = 0; i < nrobots; i++)
        {
            assert(q[i].size() == robots_[i]->nq());
            assert(v[i].size() == robots_[i]->nv());
        }
    }

    state_t::state_t(std::vector<Robot const *> const & robots,
                     std::vector<vectorN_t>          && qIn,
                     std::vector<vectorN_t>          && vIn):
    nrobots(robots.size()),
    q(std::move(qIn)),
    v(std::move(vIn)),
    robots_(robots)
    {
        assert(q.size() == nrobots && v.size() == nrobots);
        for (uint32_t i = 0; i < nrobots; i++)
        {
            assert(q[i].size() == robots_[i]->nq());
            assert(v[i].size() == robots_[i]->nv());
        }
    }

    state_t::state_t(state_t && other):
    state_t(other.robots_, std::move(other.q), std::move(other.v))
    {
        // Empty on purpose.
    }

    state_t::state_t(state_t const & other):
    nrobots(other.nrobots),
    q(other.q),
    v(other.v),
    robots_(other.robots_)
    {
        // Empty on purpose.
    }

    state_t & state_t::operator=(state_t const & other)
    {
        nrobots = other.nrobots;
        q = other.q;
        v = other.v;
        robots_ = other.robots_;
        return *this;
    }

    state_t & state_t::operator=(state_t && other)
    {
        nrobots = std::move(other.nrobots);
        q = std::move(other.q);
        v = std::move(other.v);
        robots_ = other.robots_;
        return *this;
    }

    state_t & state_t::operator+=(stateDerivative_t const & velocity)
    {
        assert(robots_ == velocity.robots_);

        for (uint32_t i = 0; i < nrobots; ++i)
        {
            // 'Sum' q = q + v, remember q is part of a Lie group (dim(q) != dim(v))
            pinocchio::integrate(robots_[i]->pncModel_, q[i], velocity.v[i], q[i]);
            v[i] += velocity.a[i];
        }
        return *this;
    }

    state_t operator+(state_t                   position,
                      stateDerivative_t const & velocity)
    {
        position += velocity;
        return position;
    }

    stateDerivative_t state_t::difference(state_t const & other)
    {
        assert(robots_ == other.robots_);

        stateDerivative_t s(robots_, v, v);
        for (uint32_t i = 0; i < nrobots; i++)
        {
            pinocchio::difference(robots_[i]->pncModel_, q[i], other.q[i], s.v[i]);
            s.a[i] -= other.v[i];
        }
        return s;
    }

    float64_t state_t::normInf(void)
    {
        float64_t norm = 0.0;
        for (uint32_t i = 0; i < nrobots; ++i)
        {
            float64_t const qnorm = q[i].lpNorm<Eigen::Infinity>();
            if (qnorm > norm)
            {
                norm = qnorm;
            }
            float64_t const vnorm = v[i].lpNorm<Eigen::Infinity>();
            if (vnorm > norm)
            {
                norm = vnorm;
            }
        }
        return norm;
    }

    // ====================================================
    // ================ stateDerivative_t =================
    // ====================================================

    stateDerivative_t::stateDerivative_t(std::vector<Robot const *> const & robots):
    nrobots(robots.size()),
    v(),
    a(),
    robots_(robots)
    {
        v.reserve(robots.size());
        std::transform(robots.begin(), robots.end(),
                       std::back_inserter(v),
                       [](Robot const * robot) -> vectorN_t
                       {
                           return vectorN_t(robot->nv());
                       });
        a = v;
    }

    stateDerivative_t::stateDerivative_t(std::vector<Robot const *> const & robots,
                                         std::vector<vectorN_t>     const & vIn,
                                         std::vector<vectorN_t>     const & aIn):
    nrobots(robots.size()),
    v(vIn),
    a(aIn),
    robots_(robots)
    {
        assert(v.size() == nrobots && a.size() == nrobots);
        for (uint32_t i = 0; i < nrobots; i++)
        {
            assert(v[i].size() == robots_[i]->nv());
            assert(a[i].size() == robots_[i]->nv());
        }
    }

    stateDerivative_t::stateDerivative_t(std::vector<Robot const *> const & robots,
                                         std::vector<vectorN_t>          && vIn,
                                         std::vector<vectorN_t>          && aIn):
    nrobots(robots.size()),
    v(std::move(vIn)),
    a(std::move(aIn)),
    robots_(robots)
    {
        assert(v.size() == nrobots && a.size() == nrobots);
        for (uint32_t i = 0; i < nrobots; i++)
        {
            assert(v[i].size() == robots_[i]->nv());
            assert(a[i].size() == robots_[i]->nv());
        }
    }

    stateDerivative_t::stateDerivative_t(stateDerivative_t && other):
    stateDerivative_t(other.robots_, std::move(other.v), std::move(other.a))
    {
        // Empty on purpose.
    }

    stateDerivative_t::stateDerivative_t(stateDerivative_t const & other):
    nrobots(other.nrobots),
    v(other.v),
    a(other.a),
    robots_(other.robots_)
    {
        // Empty on purpose.
    }

    stateDerivative_t & stateDerivative_t::operator=(stateDerivative_t const & other)
    {
        nrobots = other.nrobots;
        v = other.v;
        a = other.a;
        robots_ = other.robots_;
        return *this;
    }

    stateDerivative_t & stateDerivative_t::operator=(stateDerivative_t && other)
    {
        nrobots = std::move(other.nrobots);
        v = std::move(other.v);
        a = std::move(other.a);
        return *this;
    }

    stateDerivative_t & stateDerivative_t::operator+=(stateDerivative_t const & other)
    {
        assert(robots_ == other.robots_);

        for (uint32_t i = 0; i < nrobots; ++i)
        {
            v[i] += other.v[i];
            a[i] += other.a[i];
        }
        return *this;
    }

    stateDerivative_t operator+(stateDerivative_t         velocity,
                                stateDerivative_t const & other)
    {
        velocity += other;
        return velocity;
    }

    stateDerivative_t operator*(float64_t         const & alpha,
                                stateDerivative_t         velocity)
    {
        for (uint32_t i = 0; i < velocity.nrobots; ++i)
        {
            velocity.v[i] *= alpha;
            velocity.a[i] *= alpha;
        }
        return velocity;
    }

    float64_t stateDerivative_t::norm(void)
    {
        float64_t norm = 0.0;
        for (uint32_t i = 0; i < nrobots; ++i)
        {
            norm += v[i].norm();
            norm += a[i].norm();
        }
        return norm;
    }
}
