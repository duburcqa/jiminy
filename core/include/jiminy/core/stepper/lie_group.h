/// \brief Handling of the joint configuration q as part of a Lie Group, not a vector space.
///
/// \details The state of the system, (q, v), lies in a Lie Group, while its derivative (v, a) is a
///          tangent vector to this group: the classes in this file are meant to handle this.

#ifndef JIMINY_LIE_GROUP_H
#define JIMINY_LIE_GROUP_H

#include <numeric>

#include "pinocchio/algorithm/joint-configuration.hpp"  // `pinocchio::integrate`, `pinocchio::difference`

#include "jiminy/core/fwd.h"
#include "jiminy/core/robot/robot.h"


namespace Eigen
{
    // Forward declarations

    using namespace jiminy;

    template<typename Derived>
    class StateBase;
    template<typename _DataType>
    class State;
    template<typename _DataType>
    class StateWrapper;

    template<typename Derived>
    class StateDerivativeBase;
    template<typename _DataType>
    class StateDerivative;
    template<typename _DataType>
    class StateDerivativeWrapper;

    template<typename Derived>
    class VectorContainerBase;
    template<typename _ValueType>
    class VectorContainer;
    template<typename _ValueType>
    class VectorContainerWrapper;

    class StateShared;
    class StateDerivativeShared;

    // ====================================================
    // ============== Generic StateDerivative =============
    // ====================================================

    namespace internal
    {
        template<typename Derived>
        struct traits<StateDerivativeBase<Derived>>
        {
            typedef typename internal::traits<Derived>::DataType DataType;
        };
    }

    template<typename Derived>
    class StateDerivativeBase
    {
    public:
        typedef typename internal::traits<Derived>::DataType DataType;
        typedef typename DataType::Scalar Scalar;
        typedef typename NumTraits<Scalar>::Real RealScalar;

        const Derived & derived() const { return *static_cast<const Derived *>(this); }
        Derived & derived() { return *static_cast<Derived *>(this); }

        const Robot * const & robot() const { return derived().robot(); }

        const DataType & v() const { return derived().v(); }
        DataType & v() { return derived().v(); }

        const DataType & a() const { return derived().a(); }
        DataType & a() { return derived().a(); }

        template<int p>
        RealScalar lpNorm() const;
        RealScalar norm() const { return lpNorm<2>(); };
        RealScalar normInf() const { return lpNorm<Infinity>(); };

        void setZero()
        {
            derived().v().setZero();
            derived().a().setZero();
        }

#define GENERATE_OPERATOR_MULT(OP, NAME)                                              \
    template<typename OtherDerived>                                                   \
    StateDerivativeBase &(operator EIGEN_CAT(OP, =))(                                 \
        const StateDerivativeBase<OtherDerived> & other)                              \
    {                                                                                 \
        assert(robot() == other.robot());                                             \
        v().array() EIGEN_CAT(OP, =) other.v().array();                               \
        a().array() EIGEN_CAT(OP, =) other.a().array();                               \
        return *this;                                                                 \
    }                                                                                 \
                                                                                      \
    StateDerivativeBase &(operator EIGEN_CAT(OP, =))(const Scalar & scalar)           \
    {                                                                                 \
        v().array() EIGEN_CAT(OP, =) scalar;                                          \
        a().array() EIGEN_CAT(OP, =) scalar;                                          \
        return *this;                                                                 \
    }                                                                                 \
                                                                                      \
    const StateDerivativeWrapper<const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(        \
        DataType, Scalar, NAME)>(operator OP)(const Scalar & scalar) const            \
    {                                                                                 \
        return StateDerivativeWrapper<const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(   \
            DataType, Scalar, NAME)>(robot(), v() OP scalar, a() OP scalar);          \
    }                                                                                 \
                                                                                      \
    friend const StateDerivativeWrapper<const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE( \
        DataType, Scalar, NAME)>(operator OP)(const Scalar & scalar,                  \
                                              const StateDerivativeBase & other)      \
    {                                                                                 \
        return other OP scalar;                                                       \
    }

        GENERATE_OPERATOR_MULT(*, product)
        GENERATE_OPERATOR_MULT(/, quotient)

#undef GENERATE_OPERATOR_MULT

#define GENERATE_OPERATOR_ADD(OP, NAME)                                                   \
    template<typename OtherDerived>                                                       \
    StateDerivativeBase &(operator EIGEN_CAT(OP, =))(                                     \
        const StateDerivativeBase<OtherDerived> & other)                                  \
    {                                                                                     \
        assert(robot() == other.robot());                                                 \
        v().array() EIGEN_CAT(OP, =) other.v().array();                                   \
        a().array() EIGEN_CAT(OP, =) other.a().array();                                   \
        return *this;                                                                     \
    }                                                                                     \
                                                                                          \
    StateDerivativeBase &(operator EIGEN_CAT(OP, =))(const Scalar & scalar)               \
    {                                                                                     \
        v().array() EIGEN_CAT(OP, =) scalar;                                              \
        a().array() EIGEN_CAT(OP, =) scalar;                                              \
        return *this;                                                                     \
    }                                                                                     \
                                                                                          \
    const StateDerivativeWrapper<const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(            \
        DataType, Scalar, NAME)>(operator OP)(const Scalar & scalar) const                \
    {                                                                                     \
        const typename internal::plain_constant_type<DataType, Scalar>::type scalarConst( \
            v().size(), 1, scalar);                                                       \
        return StateDerivativeWrapper<const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(       \
            DataType, Scalar, NAME)>(robot(), v() OP scalarConst, a() OP scalarConst);    \
    }                                                                                     \
                                                                                          \
    friend const StateDerivativeWrapper<const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(     \
        DataType, Scalar, NAME)>(operator OP)(const Scalar & scalar,                      \
                                              const StateDerivativeBase & other)          \
    {                                                                                     \
        return other OP scalar;                                                           \
    }                                                                                     \
                                                                                          \
    template<typename OtherDerived>                                                       \
    const StateDerivativeWrapper<const EIGEN_CWISE_BINARY_RETURN_TYPE(                    \
        DataType, typename OtherDerived::DataType, NAME)>(operator OP)(                   \
        const StateDerivativeBase<OtherDerived> & other) const                            \
    {                                                                                     \
        assert(robot() == other.robot());                                                 \
        return StateDerivativeWrapper<const EIGEN_CWISE_BINARY_RETURN_TYPE(               \
            DataType, typename OtherDerived::DataType, NAME)>(                            \
            robot(), v() OP other.v(), a() OP other.a());                                 \
    }

        GENERATE_OPERATOR_ADD(+, sum)
        GENERATE_OPERATOR_ADD(-, difference)

#undef GENERATE_OPERATOR_ADD
    };

    namespace internal
    {
        template<typename Derived, int p>
        struct StateDerivativeLpNormImpl
        {
            typedef typename StateDerivativeBase<Derived>::RealScalar RealScalar;
            static inline RealScalar run(const StateDerivativeBase<Derived> & velocity)
            {
                return std::pow(velocity.v().cwiseAbs().array().pow(p).sum() +
                                    velocity.a().cwiseAbs().array().pow(p).sum(),
                                RealScalar(1) / p);
            }
        };

        template<typename Derived>
        struct StateDerivativeLpNormImpl<Derived, 1>
        {
            static inline typename StateDerivativeBase<Derived>::RealScalar run(
                const StateDerivativeBase<Derived> & velocity)
            {
                return velocity.v().cwiseAbs().sum() + velocity.a().cwiseAbs().sum();
            }
        };

        template<typename Derived>
        struct StateDerivativeLpNormImpl<Derived, Infinity>
        {
            static inline typename StateDerivativeBase<Derived>::RealScalar run(
                const StateDerivativeBase<Derived> & velocity)
            {
                return std::max(velocity.v().cwiseAbs().maxCoeff(),
                                velocity.a().cwiseAbs().maxCoeff());
            }
        };
    }

    template<typename Derived>
    template<int p>
    typename StateDerivativeBase<Derived>::RealScalar StateDerivativeBase<Derived>::lpNorm() const
    {
        return internal::StateDerivativeLpNormImpl<Derived, p>::run(*this);
    }

    namespace internal
    {
        template<typename _DataType>
        struct traits<StateDerivative<_DataType>>
        {
            typedef _DataType DataType;
        };
    }

    template<typename _DataType = Eigen::VectorXd>
    class StateDerivative : public StateDerivativeBase<StateDerivative<_DataType>>
    {
    public:
        typedef _DataType DataType;
        typedef typename DataType::Scalar Scalar;
        typedef const StateDerivative & Nested;

    public:
        explicit StateDerivative(const Robot * const & robotIn,
                                 const Eigen::VectorXd & vIn,
                                 const Eigen::VectorXd & aIn) :
        robot_(robotIn),
        v_(vIn),
        a_(aIn)
        {
        }

        explicit StateDerivative(
            const Robot * const & robotIn, Eigen::VectorXd && vIn, Eigen::VectorXd && aIn) :
        robot_(robotIn),
        v_(std::move(vIn)),
        a_(std::move(aIn))
        {
        }

        explicit StateDerivative(const Robot * const & robotIn, Eigen::VectorXd && vIn) :
        robot_(robotIn),
        v_(std::move(vIn)),
        a_(robot_->nv())
        {
        }

        explicit StateDerivative(const Robot * const & robotIn) :
        robot_(robotIn),
        v_(robot_->nv()),
        a_(robot_->nv())
        {
        }

        StateDerivative(const StateDerivative & other) :
        robot_(other.robot()),
        v_(other.v()),
        a_(other.a())
        {
        }

        StateDerivative(StateDerivative && other) :
        robot_(other.robot()),
        v_(std::move(other.v())),
        a_(std::move(other.a()))
        {
        }

        template<typename OtherDerived>
        StateDerivative(const StateDerivativeBase<OtherDerived> & other) :
        robot_(other.robot()),
        v_(other.v()),
        a_(other.a())
        {
        }

        const Robot * const & robot() const { return robot_; }
        const DataType & v() const { return v_; }
        DataType & v() { return v_; }
        const DataType & a() const { return a_; }
        DataType & a() { return a_; }

        static const StateDerivativeWrapper<const typename DataType::ConstantReturnType> Zero(
            const Robot * const & robotIn);
        static const StateDerivativeWrapper<const typename DataType::ConstantReturnType> Ones(
            const Robot * const & robotIn);

        // This method allows you to assign Eigen expressions to StateDerivative
        template<typename OtherDerived>
        StateDerivative & operator=(const StateDerivativeBase<OtherDerived> & other)
        {
            robot_ = other.robot();
            v_ = other.v();
            a_ = other.a();
            return *this;
        }

        StateDerivative & operator=(const StateDerivative & other)
        {
            robot_ = other.robot();
            v_ = other.v();
            a_ = other.a();
            return *this;
        }

        StateDerivative & operator=(StateDerivative && other)
        {
            robot_ = other.robot_;
            v_ = std::move(other.v_);
            a_ = std::move(other.a_);
            return *this;
        }

    protected:
        const Robot * robot_;
        DataType v_;
        DataType a_;
    };

    namespace internal
    {
        template<typename _DataType>
        struct traits<StateDerivativeWrapper<_DataType>>
        {
            typedef _DataType DataType;
            typedef Eigen::Dense StorageKind;
        };
    }

    template<typename _DataType>
    class StateDerivativeWrapper :
    public StateDerivativeBase<StateDerivativeWrapper<_DataType>>,
        internal::no_assignment_operator
    {
    public:
        typedef _DataType DataType;
        typedef StateDerivativeWrapper Nested;

        explicit StateDerivativeWrapper(
            const Robot * const & robotIn, DataType & vIn, DataType & aIn) :
        robot_(robotIn),
        vRef_(vIn),
        aRef_(aIn)
        {
        }

        explicit StateDerivativeWrapper(
            const Robot * const & robotIn, DataType && vIn, DataType && aIn) :
        robot_(robotIn),
        vRef_(std::move(vIn)),
        aRef_(std::move(aIn))
        {
        }

        const Robot * const & robot() const { return robot_; }
        const DataType & v() const { return vRef_; }
        const DataType & a() const { return aRef_; }

    protected:
        const Robot * robot_;
        typename DataType::Nested vRef_;
        typename DataType::Nested aRef_;
    };

    template<typename _DataType>
    const StateDerivativeWrapper<const typename _DataType::ConstantReturnType>
    StateDerivative<_DataType>::Zero(const Robot * const & robotIn)
    {
        return StateDerivativeWrapper<const typename DataType::ConstantReturnType>(
            robotIn, DataType::Zero(robotIn->nv()), DataType::Zero(robotIn->nv()));
    }

    template<typename _DataType>
    const StateDerivativeWrapper<const typename _DataType::ConstantReturnType>
    StateDerivative<_DataType>::Ones(const Robot * const & robotIn)
    {
        return StateDerivativeWrapper<const typename DataType::ConstantReturnType>(
            robotIn, DataType::Ones(robotIn->nv()), DataType::Ones(robotIn->nv()));
    }

    // ====================================================
    // =================== Generic State ==================
    // ====================================================

    namespace internal
    {
        template<typename Derived>
        struct traits<StateBase<Derived>>
        {
            typedef typename internal::traits<Derived>::DataType DataType;
        };
    }

    template<typename Derived>
    class StateBase
    {
    public:
        typedef typename internal::traits<Derived>::DataType DataType;
        typedef typename DataType::Scalar Scalar;
        typedef typename NumTraits<Scalar>::Real RealScalar;

        const Derived & derived() const { return *static_cast<const Derived *>(this); }
        Derived & derived() { return *static_cast<Derived *>(this); }

        const Robot * const & robot() const { return derived().robot(); }

        const DataType & q() const { return derived().q(); }
        DataType & q() { return derived().q(); }

        const DataType & v() const { return derived().v(); }
        DataType & v() { return derived().v(); }

        template<int p>
        RealScalar lpNorm() const;
        RealScalar norm() const { return lpNorm<2>(); };
        RealScalar normInf() const { return lpNorm<Infinity>(); };

        void setZero()
        {
            pinocchio::neutral(robot()->pncModel_, derived().q());
            derived().v().setZero();
        }

        template<typename OtherDerived, typename OutDerived>
        StateBase<OutDerived> & sum(const StateDerivativeBase<OtherDerived> & velocity,
                                    StateBase<OutDerived> & out) const
        {
            // 'Sum' q = q + v, remember q is part of a Lie group (dim(q) != dim(v))
            assert(robot() == velocity.robot() && robot() == out.robot());
            pinocchio::integrate(robot()->pncModel_, q(), velocity.v(), out.q());
            out.v() = v() + velocity.a();
            return out;
        }

        template<typename OtherDerived>
        StateBase & sumInPlace(const StateDerivativeBase<OtherDerived> & velocity)
        {
            return sum(velocity, *this);
        }

        template<typename OtherDerived, typename OutDerived>
        StateDerivativeBase<OutDerived> & difference(const StateBase<OtherDerived> & position,
                                                     StateDerivativeBase<OutDerived> & out) const
        {
            assert(robot() == position.robot() && robot() == out.robot());
            pinocchio::difference(robot()->pncModel_, q(), position.q(), out.v());
            out.a() = v() - position.v();
            return out;
        }
    };

    namespace internal
    {
        template<typename _DataType>
        struct traits<State<_DataType>>
        {
            typedef _DataType DataType;
        };
    }

    template<typename _DataType = Eigen::VectorXd>
    class State : public StateBase<State<_DataType>>
    {
    public:
        typedef _DataType DataType;
        typedef typename DataType::Scalar Scalar;
        typedef const State & Nested;

    public:
        explicit State(const Robot * const & robotIn,
                       const Eigen::VectorXd & qIn,
                       const Eigen::VectorXd & vIn) :
        robot_(robotIn),
        q_(qIn),
        v_(vIn)
        {
        }

        explicit State(
            const Robot * const & robotIn, Eigen::VectorXd && qIn, Eigen::VectorXd && vIn) :
        robot_(robotIn),
        q_(std::move(qIn)),
        v_(std::move(vIn))
        {
        }

        explicit State(const Robot * const & robotIn) :
        robot_(robotIn),
        q_(robot_->nq()),
        v_(robot_->nv())
        {
        }

        State(const State & other) :
        robot_(other.robot()),
        q_(other.q()),
        v_(other.v())
        {
        }

        State(State && other) :
        robot_(other.robot()),
        q_(std::move(other.q())),
        v_(std::move(other.v()))
        {
        }

        template<typename OtherDerived>
        State(const StateBase<OtherDerived> & other) :
        robot_(other.robot()),
        q_(other.q()),
        v_(other.v())
        {
        }

        const Robot * const & robot() const { return robot_; }
        const DataType & q() const { return q_; }
        DataType & q() { return q_; }
        const DataType & v() const { return v_; }
        DataType & v() { return v_; }

        static const StateWrapper<const typename DataType::ConstantReturnType> Zero(
            const Robot * const & robotIn);
        static const StateWrapper<const typename DataType::ConstantReturnType> Ones(
            const Robot * const & robotIn);

        // This method allows you to assign Eigen expressions to State
        template<typename OtherDerived>
        State & operator=(const StateBase<OtherDerived> & other)
        {
            robot_ = other.robot();
            q_ = other.q();
            v_ = other.v();
            return *this;
        }

        State & operator=(const State & other)
        {
            robot_ = other.robot();
            q_ = other.q();
            v_ = other.v();
            return *this;
        }

        State & operator=(State && other)
        {
            robot_ = other.robot();
            q_ = std::move(other.q_);
            v_ = std::move(other.v_);
            return *this;
        }

    protected:
        const Robot * robot_;
        DataType q_;
        DataType v_;
    };

    namespace internal
    {
        template<typename _DataType>
        struct traits<StateWrapper<_DataType>>
        {
            typedef _DataType DataType;
            typedef Eigen::Dense StorageKind;
        };
    }

    template<typename _DataType>
    class StateWrapper :
    public StateBase<StateWrapper<_DataType>>,
        internal::no_assignment_operator
    {
    public:
        typedef _DataType DataType;
        typedef StateWrapper Nested;

        explicit StateWrapper(const Robot * const & robotIn, DataType & qIn, DataType & vIn) :
        robot_(robotIn),
        qRef_(qIn),
        vRef_(vIn)
        {
        }

        explicit StateWrapper(const Robot * const & robotIn, DataType && qIn, DataType && vIn) :
        robot_(robotIn),
        qRef_(std::move(qIn)),
        vRef_(std::move(vIn))
        {
        }

        const Robot * const & robot() const { return robot_; }
        const DataType & q() const { return qRef_; }
        const DataType & v() const { return vRef_; }

    protected:
        const Robot * robot_;
        typename DataType::Nested qRef_;
        typename DataType::Nested vRef_;
    };

    template<typename _DataType>
    const StateWrapper<const typename _DataType::ConstantReturnType>
    State<_DataType>::Zero(const Robot * const & robotIn)
    {
        return StateWrapper<const typename DataType::ConstantReturnType>(
            robotIn, DataType::Zero(robotIn->nq()), DataType::Zero(robotIn->nv()));
    }

    template<typename _DataType>
    const StateWrapper<const typename _DataType::ConstantReturnType>
    State<_DataType>::Ones(const Robot * const & robotIn)
    {
        return StateWrapper<const typename DataType::ConstantReturnType>(
            robotIn, DataType::Ones(robotIn->nq()), DataType::Ones(robotIn->nv()));
    }

    // ====================================================
    // ============= Generic Vector container =============
    // ====================================================

    namespace internal
    {
        template<typename Derived>
        struct traits<VectorContainerBase<Derived>>
        {
            typedef typename internal::traits<Derived>::ValueType ValueType;
        };
    }

    template<typename Derived>
    class VectorContainerBase
    {
    public:
        typedef typename internal::traits<Derived>::ValueType ValueType;
        typedef typename ValueType::DataType DataType;
        typedef typename ValueType::Scalar Scalar;
        typedef typename NumTraits<Scalar>::Real RealScalar;

        const Derived & derived() const { return *static_cast<const Derived *>(this); }
        Derived & derived() { return *static_cast<Derived *>(this); }

        const std::vector<ValueType> & vector() const { return derived().vector(); }
        std::vector<ValueType> & vector() { return derived().vector(); };

        template<int p>
        RealScalar lpNorm() const;
        RealScalar norm() const { return lpNorm<2>(); };
        RealScalar normInf() const { return lpNorm<Infinity>(); };

        void setZero()
        {
            for (ValueType & element : vector())
            {
                element.setZero();
            }
        }

#define GENERATE_OPERATOR_ARITHMETIC(OP, NAME)                                                     \
    auto(operator OP)(const Scalar & scalar) const                                                 \
    {                                                                                              \
        typedef std::remove_const_t<decltype(std::declval<ValueType>() OP std::declval<Scalar>())> \
            wrappedType;                                                                           \
        std::vector<wrappedType> result;                                                           \
        const std::vector<ValueType> & vector_ = vector();                                         \
        result.reserve(vector_.size());                                                            \
        for (const ValueType & element : vector_)                                                  \
        {                                                                                          \
            result.emplace_back(element OP scalar);                                                \
        }                                                                                          \
        return VectorContainerWrapper<wrappedType>(std::move(result));                             \
    }                                                                                              \
                                                                                                   \
    friend auto(operator OP)(const Scalar & scalar, const VectorContainerBase & other)             \
    {                                                                                              \
        return other OP scalar;                                                                    \
    }                                                                                              \
                                                                                                   \
    template<typename OtherDerived>                                                                \
    auto(operator OP)(const VectorContainerBase<OtherDerived> & other) const                       \
    {                                                                                              \
        typedef std::remove_const_t<decltype(std::declval<ValueType>() OP std::declval<            \
                                             typename OtherDerived::ValueType>())>                 \
            wrappedType;                                                                           \
        const std::vector<ValueType> & vector_ = vector();                                         \
        const std::vector<typename OtherDerived::ValueType> & vectorIn = other.vector();           \
        assert(vector_.size() == vectorIn.size());                                                 \
        std::vector<wrappedType> result;                                                           \
        result.reserve(vector_.size());                                                            \
        for (std::size_t i = 0; i < vector_.size(); ++i)                                           \
        {                                                                                          \
            result.emplace_back(vector_[i] OP vectorIn[i]);                                        \
        }                                                                                          \
        return VectorContainerWrapper<wrappedType>(std::move(result));                             \
    }                                                                                              \
                                                                                                   \
    VectorContainerBase &(operator EIGEN_CAT(OP, =))(const Scalar & scalar)                        \
    {                                                                                              \
        for (ValueType & element : vector())                                                       \
        {                                                                                          \
            element EIGEN_CAT(OP, =) scalar;                                                       \
        }                                                                                          \
        return *this;                                                                              \
    }

        GENERATE_OPERATOR_ARITHMETIC(*, product)
        GENERATE_OPERATOR_ARITHMETIC(/, quotient)
        GENERATE_OPERATOR_ARITHMETIC(+, sum)
        GENERATE_OPERATOR_ARITHMETIC(-, difference)

#undef GENERATE_OPERATOR_ARITHMETIC

#define GENERATE_OPERATOR_COMPOUND(OP, NAME)                                               \
    template<typename OtherDerived>                                                        \
    VectorContainerBase &(operator EIGEN_CAT(OP, =))(                                      \
        const VectorContainerBase<OtherDerived> & other)                                   \
    {                                                                                      \
        std::vector<ValueType> & vector_ = vector();                                       \
        const std::vector<typename internal::traits<OtherDerived>::ValueType> & vectorIn = \
            other.vector();                                                                \
        assert(vector_.size() == vectorIn.size());                                         \
        for (std::size_t i = 0; i < vector_.size(); ++i)                                   \
        {                                                                                  \
            vector_[i] EIGEN_CAT(OP, =) vectorIn[i];                                       \
        }                                                                                  \
        return *this;                                                                      \
    }

        GENERATE_OPERATOR_COMPOUND(*, product)
        GENERATE_OPERATOR_COMPOUND(/, quotient)
        GENERATE_OPERATOR_COMPOUND(+, sum)
        GENERATE_OPERATOR_COMPOUND(-, difference)

#undef GENERATE_OPERATOR_COMPOUND
    };

    namespace internal
    {
        template<typename Derived, int p>
        struct VectorContainerLpNormImpl
        {
            typedef typename VectorContainerBase<Derived>::RealScalar RealScalar;
            static inline RealScalar run(const VectorContainerBase<Derived> & container)
            {
                RealScalar total = 0.0;
                for (const typename internal::traits<Derived>::ValueType & element :
                     container.vector())
                {
                    total += std::pow(element.template lpNorm<p>(), p);
                }
                return std::pow(total, RealScalar(1) / p);
            }
        };

        template<typename Derived>
        struct VectorContainerLpNormImpl<Derived, 1>
        {
            typedef typename VectorContainerBase<Derived>::RealScalar RealScalar;
            static inline RealScalar run(const VectorContainerBase<Derived> & container)
            {
                return std::accumulate(
                    container.vector().begin(),
                    container.vector().end(),
                    RealScalar(0.0),
                    [](RealScalar & /* cumsum */,
                       const typename internal::traits<Derived>::ValueType & element)
                    { return element.template lpNorm<1>(); });
            }
        };

        template<typename Derived>
        struct VectorContainerLpNormImpl<Derived, Infinity>
        {
            typedef typename VectorContainerBase<Derived>::RealScalar RealScalar;
            static inline RealScalar run(const VectorContainerBase<Derived> & container)
            {
                RealScalar maxValue = 0.0;
                for (const typename internal::traits<Derived>::ValueType & element :
                     container.vector())
                {
                    RealScalar value = element.template lpNorm<Infinity>();
                    if (value > maxValue)
                    {
                        maxValue = value;
                    }
                }
                return maxValue;
            }
        };
    }

    template<typename Derived>
    template<int p>
    typename VectorContainerBase<Derived>::RealScalar VectorContainerBase<Derived>::lpNorm() const
    {
        return internal::VectorContainerLpNormImpl<Derived, p>::run(*this);
    }

    namespace internal
    {
        template<typename _ValueType>
        struct traits<VectorContainer<_ValueType>>
        {
            typedef _ValueType ValueType;
        };
    }

    template<typename _ValueType>
    class VectorContainer : public VectorContainerBase<VectorContainer<_ValueType>>
    {
    public:
        typedef _ValueType ValueType;
        typedef typename ValueType::DataType DataType;
        typedef const VectorContainer & Nested;

        VectorContainer() :
        vector_()
        {
            // Empty on purpose
        }

        template<typename... Args>
        explicit VectorContainer(const std::vector<const Robot *> & robots,
                                 const std::vector<Args> &... args) :
        vector_()
        {
            do_for(
                [&robots](auto arg)
                {
                    assert(arg.size() == robots.size());
                    (void)(arg);  // Necessary to fix unused variable warning
                },
                args...);
            vector_.reserve(robots.size());
            for (std::size_t i = 0; i < robots.size(); ++i)
            {
                vector_.emplace_back(robots[i], args[i]...);
            }
        }

        template<typename... Args>
        explicit VectorContainer(const std::vector<const Robot *> & robots,
                                 std::vector<Args> &&... args) :
        vector_()
        {
            do_for(
                [&robots](auto arg)
                {
                    assert(arg.size() == robots.size());
                    (void)(arg);  // Necessary to fix unused variable warning
                },
                args...);
            vector_.reserve(robots.size());
            for (std::size_t i = 0; i < robots.size(); ++i)
            {
                vector_.emplace_back(robots[i], std::move(args[i])...);
            }
        }

        explicit VectorContainer(const std::vector<const Robot *> & robots) :
        vector_()
        {
            vector_.reserve(robots.size());
            for (std::size_t i = 0; i < robots.size(); ++i)
            {
                vector_.emplace_back(robots[i]);
            }
        }

        template<typename OtherDerived>
        VectorContainer(const VectorContainerBase<OtherDerived> & other) :
        VectorContainer(other.vector())
        {
        }

        VectorContainer(VectorContainer && other) :
        vector_(std::move(other.vector()))
        {
        }

        VectorContainer(std::vector<ValueType> && vectorIn) :
        vector_(std::move(vectorIn))
        {
        }

        template<typename OtherValueType>
        VectorContainer(const std::vector<OtherValueType> & vectorIn) :
        vector_()
        {
            vector_.reserve(vectorIn.size());
            std::copy(vectorIn.begin(), vectorIn.end(), std::back_inserter(vector_));
        }

        const std::vector<ValueType> & vector() const { return vector_; }
        std::vector<ValueType> & vector() { return vector_; }

        static auto Zero(const std::vector<const Robot *> & robots);
        static auto Ones(const std::vector<const Robot *> & robots);

        // This method allows you to assign Eigen expressions to VectorContainer
        template<typename OtherDerived>
        VectorContainer & operator=(const VectorContainerBase<OtherDerived> & other)
        {
            const std::vector<typename internal::traits<OtherDerived>::ValueType> & vectorIn =
                other.vector();
            vector_.resize(vectorIn.size());
            for (std::size_t i = 0; i < vector_.size(); ++i)
            {
                vector_[i] = vectorIn[i];
            }
            return *this;
        }

        VectorContainer & operator=(VectorContainer && other)
        {
            vector_ = std::move(other.vector());
            return *this;
        }

    protected:
        std::vector<ValueType> vector_;
    };

    namespace internal
    {
        template<typename _ValueType>
        struct traits<VectorContainerWrapper<_ValueType>>
        {
            typedef _ValueType ValueType;
            typedef Eigen::Dense StorageKind;
        };
    }

    template<typename _ValueType>
    class VectorContainerWrapper :
    public VectorContainerBase<VectorContainerWrapper<_ValueType>>,
        internal::no_assignment_operator
    {
    public:
        typedef _ValueType ValueType;
        typedef VectorContainerWrapper Nested;

        explicit VectorContainerWrapper(std::vector<ValueType> & vectorIn) :
        vector_(vectorIn)
        {
        }

        explicit VectorContainerWrapper(std::vector<ValueType> && vectorIn) :
        vector_(std::move(vectorIn))
        {
        }

        const std::vector<ValueType> & vector() const { return vector_; }

    protected:
        std::vector<ValueType> vector_;
    };

    template<typename _ValueType>
    auto VectorContainer<_ValueType>::Zero(const std::vector<const Robot *> & robots)
    {
        typedef std::remove_const_t<decltype(_ValueType::Zero(
            std::declval<const Robot * const &>()))>
            wrappedType;
        std::vector<wrappedType> vectorIn;
        vectorIn.reserve(robots.size());
        std::transform(robots.begin(),
                       robots.end(),
                       std::back_inserter(vectorIn),
                       [](const Robot * const & robot) -> wrappedType
                       { return _ValueType::Zero(robot); });
        return VectorContainerWrapper<wrappedType>(std::move(vectorIn));
    }

    template<typename _ValueType>
    auto VectorContainer<_ValueType>::Ones(const std::vector<const Robot *> & robots)
    {
        typedef std::remove_const_t<decltype(_ValueType::Ones(
            std::declval<const Robot * const &>()))>
            wrappedType;
        std::vector<wrappedType> vectorIn;
        vectorIn.reserve(robots.size());
        std::transform(robots.begin(),
                       robots.end(),
                       std::back_inserter(vectorIn),
                       [](const Robot * const & robot) -> wrappedType
                       { return _ValueType::Ones(robot); });
        return VectorContainerWrapper<wrappedType>(std::move(vectorIn));
    }

    // ====================================================
    // ================== Specializations =================
    // ====================================================

#define GENERATE_SHARED_IMPL(BASE, VAR1, SIZE1, VAR2, SIZE2)                                   \
    namespace internal                                                                         \
    {                                                                                          \
        template<>                                                                             \
        struct traits<EIGEN_CAT(BASE, Shared)>                                                 \
        {                                                                                      \
            typedef Eigen::Ref<Eigen::VectorXd> DataType;                                      \
        };                                                                                     \
    }                                                                                          \
                                                                                               \
    class EIGEN_CAT(BASE, Shared) :                                                            \
    public EIGEN_CAT(BASE, Base)<EIGEN_CAT(BASE, Shared)>                                      \
    {                                                                                          \
    public:                                                                                    \
        typedef typename internal::traits<EIGEN_CAT(BASE, Shared)>::DataType DataType;         \
        typedef EIGEN_CAT(BASE, Shared) Nested;                                                \
                                                                                               \
        explicit EIGEN_CAT(BASE, Shared)(const Robot * const & robot,                          \
                                         const Eigen::Ref<Eigen::VectorXd> & VAR1,             \
                                         const Eigen::Ref<Eigen::VectorXd> & VAR2) :           \
        robot_(robot),                                                                         \
        EIGEN_CAT(VAR1, Ref_)(VAR1),                                                           \
        EIGEN_CAT(VAR2, Ref_)(VAR2)                                                            \
        {                                                                                      \
        }                                                                                      \
                                                                                               \
        const Robot * const & robot() const                                                    \
        {                                                                                      \
            return robot_;                                                                     \
        }                                                                                      \
        Eigen::Ref<Eigen::VectorXd> & VAR1()                                                   \
        {                                                                                      \
            return EIGEN_CAT(VAR1, Ref_);                                                      \
        }                                                                                      \
        const Eigen::Ref<Eigen::VectorXd> & VAR1() const                                       \
        {                                                                                      \
            return EIGEN_CAT(VAR1, Ref_);                                                      \
        }                                                                                      \
        Eigen::Ref<Eigen::VectorXd> & VAR2()                                                   \
        {                                                                                      \
            return EIGEN_CAT(VAR2, Ref_);                                                      \
        }                                                                                      \
        const Eigen::Ref<Eigen::VectorXd> & VAR2() const                                       \
        {                                                                                      \
            return EIGEN_CAT(VAR2, Ref_);                                                      \
        }                                                                                      \
                                                                                               \
    protected:                                                                                 \
        const Robot * robot_;                                                                  \
        Eigen::Ref<Eigen::VectorXd> EIGEN_CAT(VAR1, Ref_);                                     \
        Eigen::Ref<Eigen::VectorXd> EIGEN_CAT(VAR2, Ref_);                                     \
    };                                                                                         \
                                                                                               \
    class EIGEN_CAT(BASE, Vector) :                                                            \
    public VectorContainer<EIGEN_CAT(BASE, Shared)>                                            \
    {                                                                                          \
    public:                                                                                    \
        EIGEN_CAT(BASE, Vector)                                                                \
        () :                                                                                   \
        VectorContainer<EIGEN_CAT(BASE, Shared)>(),                                            \
        VAR1(),                                                                                \
        VAR2()                                                                                 \
        {                                                                                      \
        }                                                                                      \
                                                                                               \
        explicit EIGEN_CAT(BASE,                                                               \
                           Vector)(const std::vector<const Robot *> & robots,                  \
                                   const std::vector<Eigen::VectorXd> & EIGEN_CAT(VAR1, In),   \
                                   const std::vector<Eigen::VectorXd> & EIGEN_CAT(VAR2, In)) : \
        VectorContainer<EIGEN_CAT(BASE, Shared)>(),                                            \
        VAR1(EIGEN_CAT(VAR1, In)),                                                             \
        VAR2(EIGEN_CAT(VAR2, In))                                                              \
        {                                                                                      \
            assert(VAR1.size() == robots.size() && VAR2.size() == robots.size());              \
            vector_.reserve(robots.size());                                                    \
            for (std::size_t i = 0; i < robots.size(); ++i)                                    \
            {                                                                                  \
                vector_.emplace_back(robots[i], VAR1[i], VAR2[i]);                             \
            }                                                                                  \
        }                                                                                      \
                                                                                               \
        explicit EIGEN_CAT(BASE, Vector)(const std::vector<const Robot *> & robots) :          \
        VectorContainer<EIGEN_CAT(BASE, Shared)>(),                                            \
        VAR1(),                                                                                \
        VAR2()                                                                                 \
        {                                                                                      \
            vector_.reserve(robots.size());                                                    \
            VAR1.reserve(robots.size());                                                       \
            VAR2.reserve(robots.size());                                                       \
            for (std::size_t i = 0; i < robots.size(); ++i)                                    \
            {                                                                                  \
                VAR1.emplace_back(robots[i]->SIZE1());                                         \
                VAR2.emplace_back(robots[i]->SIZE2());                                         \
                vector_.emplace_back(robots[i], VAR1[i], VAR2[i]);                             \
            }                                                                                  \
        }                                                                                      \
                                                                                               \
        EIGEN_CAT(BASE, Vector)                                                                \
        (EIGEN_CAT(BASE, Vector) const & other) :                                              \
        VectorContainer<EIGEN_CAT(BASE, Shared)>(),                                            \
        VAR1(other.VAR1),                                                                      \
        VAR2(other.VAR2)                                                                       \
        {                                                                                      \
            const std::vector<ValueType> & vectorIn = other.vector();                          \
            vector_.reserve(vectorIn.size());                                                  \
            for (std::size_t i = 0; i < vectorIn.size(); ++i)                                  \
            {                                                                                  \
                vector_.emplace_back(vectorIn[i].robot(), VAR1[i], VAR2[i]);                   \
            }                                                                                  \
        }                                                                                      \
                                                                                               \
        EIGEN_CAT(BASE, Vector)                                                                \
        (EIGEN_CAT(BASE, Vector) && other) :                                                   \
        VectorContainer<EIGEN_CAT(BASE, Shared)>(std::move(other.vector())),                   \
        VAR1(std::move(other.VAR1)),                                                           \
        VAR2(std::move(other.VAR2))                                                            \
        {                                                                                      \
        }                                                                                      \
                                                                                               \
        template<typename OtherDerived>                                                        \
        EIGEN_CAT(BASE, Vector)                                                                \
        (const VectorContainerBase<OtherDerived> & other) :                                    \
        VectorContainer<EIGEN_CAT(BASE, Shared)>(),                                            \
        VAR1(),                                                                                \
        VAR2()                                                                                 \
        {                                                                                      \
            const std::vector<typename internal::traits<OtherDerived>::ValueType> & vectorIn = \
                other.vector();                                                                \
            vector_.reserve(vectorIn.size());                                                  \
            VAR1.reserve(vectorIn.size());                                                     \
            VAR2.reserve(vectorIn.size());                                                     \
            for (std::size_t i = 0; i < vectorIn.size(); ++i)                                  \
            {                                                                                  \
                VAR1.push_back(vectorIn[i].VAR1());                                            \
                VAR2.push_back(vectorIn[i].VAR2());                                            \
                vector_.emplace_back(vectorIn[i].robot(), VAR1[i], VAR2[i]);                   \
            }                                                                                  \
        }                                                                                      \
                                                                                               \
        template<typename OtherValueType>                                                      \
        EIGEN_CAT(BASE, Vector)                                                                \
        (std::vector<OtherValueType> && vectorIn) :                                            \
        VectorContainer<EIGEN_CAT(BASE, Shared)>(),                                            \
        VAR1(),                                                                                \
        VAR2()                                                                                 \
        {                                                                                      \
            vector_.reserve(vectorIn.size());                                                  \
            VAR1.reserve(vectorIn.size());                                                     \
            VAR2.reserve(vectorIn.size());                                                     \
            for (std::size_t i = 0; i < vectorIn.size(); ++i)                                  \
            {                                                                                  \
                VAR1.push_back(std::move(vectorIn[i].VAR1()));                                 \
                VAR2.push_back(std::move(vectorIn[i].VAR2()));                                 \
                vector_.emplace_back(vectorIn[i].robot(), VAR1[i], VAR2[i]);                   \
            }                                                                                  \
        }                                                                                      \
                                                                                               \
        static EIGEN_CAT(BASE, Vector) Ones(const std::vector<const Robot *> & robots)         \
        {                                                                                      \
            EIGEN_CAT(BASE, Vector) container(robots);                                         \
            for (std::size_t i = 0; i < robots.size(); ++i)                                    \
            {                                                                                  \
                container.VAR1[i].setOnes();                                                   \
                container.VAR2[i].setOnes();                                                   \
            }                                                                                  \
            return container;                                                                  \
        }                                                                                      \
                                                                                               \
        static EIGEN_CAT(BASE, Vector) Zero(const std::vector<const Robot *> & robots)         \
        {                                                                                      \
            EIGEN_CAT(BASE, Vector) container(robots);                                         \
            for (std::size_t i = 0; i < robots.size(); ++i)                                    \
            {                                                                                  \
                container.VAR1[i].setZero();                                                   \
                container.VAR2[i].setZero();                                                   \
            }                                                                                  \
            return container;                                                                  \
        }                                                                                      \
                                                                                               \
        template<typename OtherDerived>                                                        \
        EIGEN_CAT(BASE, Vector) & operator=(const VectorContainerBase<OtherDerived> & other)   \
        {                                                                                      \
            const std::vector<typename internal::traits<OtherDerived>::ValueType> & vectorIn = \
                other.vector();                                                                \
            assert(vectorIn.size() == vector_.size());                                         \
            for (std::size_t i = 0; i < vector_.size(); ++i)                                   \
            {                                                                                  \
                assert(vectorIn[i].robot() == vector_[i].robot());                             \
                VAR1[i] = vectorIn[i].VAR1();                                                  \
                VAR2[i] = vectorIn[i].VAR2();                                                  \
            }                                                                                  \
            return *this;                                                                      \
        }                                                                                      \
                                                                                               \
        EIGEN_CAT(BASE, Vector) & operator=(EIGEN_CAT(BASE, Vector) const & other)             \
        {                                                                                      \
            const std::vector<ValueType> & vectorIn = other.vector();                          \
            assert(vectorIn.size() == vector_.size());                                         \
            for (std::size_t i = 0; i < vectorIn.size(); ++i)                                  \
            {                                                                                  \
                assert(vectorIn[i].robot() == vector_[i].robot());                             \
                VAR1[i] = other.VAR1[i];                                                       \
                VAR2[i] = other.VAR2[i];                                                       \
            }                                                                                  \
            return *this;                                                                      \
        }                                                                                      \
                                                                                               \
        EIGEN_CAT(BASE, Vector) & operator=(EIGEN_CAT(BASE, Vector) && other)                  \
        {                                                                                      \
            VAR1 = std::move(other.VAR1);                                                      \
            VAR2 = std::move(other.VAR2);                                                      \
            vector_ = std::move(other.vector());                                               \
            return *this;                                                                      \
        }                                                                                      \
                                                                                               \
        EIGEN_CAT(BASE, _SHARED_ADDON)                                                         \
                                                                                               \
    public:                                                                                    \
        std::vector<Eigen::VectorXd> VAR1;                                                     \
        std::vector<Eigen::VectorXd> VAR2;                                                     \
    };

#define StateDerivative_SHARED_ADDON                                                          \
    template<typename Derived,                                                                \
             typename = typename std::enable_if_t<                                            \
                 is_base_of_template_v<StateDerivativeBase,                                   \
                                       typename internal::traits<Derived>::ValueType>::value, \
                 void>>                                                                       \
    StateDerivativeVector & sumInPlace(const VectorContainerBase<Derived> & other,            \
                                       const Scalar & scale)                                  \
    {                                                                                         \
        const std::vector<typename internal::traits<Derived>::ValueType> & vectorIn =         \
            other.vector();                                                                   \
        assert(vector_.size() == vectorIn.size());                                            \
        for (std::size_t i = 0; i < vector_.size(); ++i)                                      \
        {                                                                                     \
            vector_[i] += scale * vectorIn[i];                                                \
        }                                                                                     \
        return *this;                                                                         \
    }

#define State_SHARED_ADDON                                                                        \
    template<                                                                                     \
        typename Derived,                                                                         \
        typename OtherDerived,                                                                    \
        typename = typename std::enable_if_t<                                                     \
            is_base_of_template_v<StateDerivativeBase,                                            \
                                  typename internal::traits<Derived>::ValueType>::value &&        \
                is_base_of_template_v<StateBase,                                                  \
                                      typename internal::traits<OtherDerived>::ValueType>::value, \
            void>>                                                                                \
    VectorContainerBase<OtherDerived> & sum(const VectorContainerBase<Derived> & other,           \
                                            VectorContainerBase<OtherDerived> & out) const        \
    {                                                                                             \
        const std::vector<typename internal::traits<Derived>::ValueType> & vectorIn =             \
            other.vector();                                                                       \
        std::vector<typename internal::traits<OtherDerived>::ValueType> & vectorOut =             \
            out.vector();                                                                         \
        assert(vectorIn.size() == vectorOut.size());                                              \
        for (std::size_t i = 0; i < vector_.size(); ++i)                                          \
        {                                                                                         \
            vector_[i].sum(vectorIn[i], vectorOut[i]);                                            \
        }                                                                                         \
        return out;                                                                               \
    }                                                                                             \
                                                                                                  \
    template<typename Derived,                                                                    \
             typename = typename std::enable_if_t<                                                \
                 is_base_of_template_v<StateDerivativeBase,                                       \
                                       typename internal::traits<Derived>::ValueType>::value,     \
                 void>>                                                                           \
    StateVector & sumInPlace(const VectorContainerBase<Derived> & other)                          \
    {                                                                                             \
        sum(other, *this);                                                                        \
        return *this;                                                                             \
    }                                                                                             \
                                                                                                  \
    template<typename Derived,                                                                    \
             typename = typename std::enable_if_t<                                                \
                 is_base_of_template_v<StateDerivativeBase,                                       \
                                       typename internal::traits<Derived>::ValueType>::value,     \
                 void>>                                                                           \
    StateVector & sumInPlace(const VectorContainerBase<Derived> & other, const Scalar & scale)    \
    {                                                                                             \
        const std::vector<typename internal::traits<Derived>::ValueType> & vectorIn =             \
            other.vector();                                                                       \
        assert(vector_.size() == vectorIn.size());                                                \
        for (std::size_t i = 0; i < vector_.size(); ++i)                                          \
        {                                                                                         \
            vector_[i].sumInPlace(scale * vectorIn[i]);                                           \
        }                                                                                         \
        return *this;                                                                             \
    }                                                                                             \
                                                                                                  \
    template<                                                                                     \
        typename Derived,                                                                         \
        typename OtherDerived,                                                                    \
        typename = typename std::enable_if_t<                                                     \
            is_base_of_template_v<StateBase,                                                      \
                                  typename internal::traits<Derived>::ValueType>::value &&        \
                is_base_of_template_v<StateDerivativeBase,                                        \
                                      typename internal::traits<OtherDerived>::ValueType>::value, \
            void>>                                                                                \
    VectorContainerBase<OtherDerived> & difference(const VectorContainerBase<Derived> & other,    \
                                                   VectorContainerBase<OtherDerived> & out) const \
    {                                                                                             \
        const std::vector<typename internal::traits<Derived>::ValueType> & vectorIn =             \
            other.vector();                                                                       \
        std::vector<typename internal::traits<OtherDerived>::ValueType> & vectorOut =             \
            out.vector();                                                                         \
        assert(vectorIn.size() == vectorOut.size());                                              \
        for (std::size_t i = 0; i < vector_.size(); ++i)                                          \
        {                                                                                         \
            vector_[i].difference(vectorIn[i], vectorOut[i]);                                     \
        }                                                                                         \
        return out;                                                                               \
    }

    GENERATE_SHARED_IMPL(StateDerivative, v, nv, a, nv)
    GENERATE_SHARED_IMPL(State, q, nq, v, nv)

#undef GENERATE_SHARED_IMPL
}

namespace jiminy
{
    using state_t = Eigen::StateVector;
    using stateDerivative_t = Eigen::StateDerivativeVector;
}

#endif  // JIMINY_LIE_GROUP_H
