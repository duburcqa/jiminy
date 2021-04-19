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

#include <numeric>

#include "pinocchio/algorithm/joint-configuration.hpp"  // `pinocchio::integrate`, `pinocchio::difference`

#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Macros.h"
#include "jiminy/core/Types.h"


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

    namespace internal {
        template<typename Derived>
        struct traits<StateDerivativeBase<Derived> >
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

        Derived const & derived(void) const { return *static_cast<Derived const *>(this); }
        Derived & derived(void) { return *static_cast<Derived *>(this); }

        Robot const * const & robot(void) const { return derived().robot(); }

        DataType const & v(void) const { return derived().v(); }
        DataType & v(void) { return derived().v(); }

        DataType const & a(void) const { return derived().a(); }
        DataType & a(void) { return derived().a(); }

        template<int p>
        RealScalar lpNorm(void) const;
        RealScalar norm(void) const { return lpNorm<2>(); };
        RealScalar normInf(void) const { return lpNorm<Infinity>(); };

        void setZero(void)
        {
            derived().v().setZero();
            derived().a().setZero();
        }

        #define GENERATE_OPERATOR_MULT(OP,NAME) \
        StateDerivativeBase & (operator EIGEN_CAT(OP,=))(Scalar const & scalar) \
        { \
            v().array() EIGEN_CAT(OP,=) scalar; \
            a().array() EIGEN_CAT(OP,=) scalar; \
            return *this; \
        } \
         \
        StateDerivativeWrapper<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(DataType,Scalar,NAME) const> const \
        (operator OP)(Scalar const & scalar) const \
        { \
            return StateDerivativeWrapper<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(DataType,Scalar,NAME) const>( \
                robot(), v() OP scalar, a() OP scalar); \
        } \
         \
        friend StateDerivativeWrapper<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(DataType,Scalar,NAME) const> const \
        (operator OP)(Scalar              const & scalar, \
                      StateDerivativeBase const & other) \
        { \
            return other OP scalar; \
        }

        GENERATE_OPERATOR_MULT(*,product)
        GENERATE_OPERATOR_MULT(/,quotient)

        #undef GENERATE_OPERATOR_MULT

        #define GENERATE_OPERATOR_ADD(OP,NAME) \
        template<typename OtherDerived> \
        StateDerivativeBase & (operator EIGEN_CAT(OP,=))(StateDerivativeBase<OtherDerived> const & other) \
        { \
            assert(robot() == other.robot()); \
            v() EIGEN_CAT(OP,=) other.v(); \
            a() EIGEN_CAT(OP,=) other.a(); \
            return *this; \
        } \
         \
        StateDerivativeBase & (operator EIGEN_CAT(OP,=))(Scalar const & scalar) \
        { \
            v().array() EIGEN_CAT(OP,=) scalar; \
            a().array() EIGEN_CAT(OP,=) scalar; \
            return *this; \
        } \
         \
        StateDerivativeWrapper<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(DataType,Scalar,NAME) const> const \
        (operator OP)(Scalar const & scalar) const \
        { \
            typename internal::plain_constant_type<DataType,Scalar>::type const scalarConst( \
                v().size(), 1, scalar); \
            return StateDerivativeWrapper<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(DataType,Scalar,NAME) const>( \
                robot(), v() OP scalarConst, a() OP scalarConst); \
        } \
         \
        friend StateDerivativeWrapper<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(DataType,Scalar,NAME) const> const \
        (operator OP)(Scalar              const & scalar, \
                      StateDerivativeBase const & other) \
        { \
            return other OP scalar; \
        } \
         \
        template<typename OtherDerived> \
        StateDerivativeWrapper<EIGEN_CWISE_BINARY_RETURN_TYPE( \
            DataType,typename OtherDerived::DataType,NAME) const> const \
        (operator OP)(StateDerivativeBase<OtherDerived> const & other) const \
        { \
            assert(robot() == other.robot()); \
            return StateDerivativeWrapper<EIGEN_CWISE_BINARY_RETURN_TYPE( \
                DataType,typename OtherDerived::DataType,NAME) const>( \
                    robot(), v() OP other.v(), a() OP other.a()); \
        }

        GENERATE_OPERATOR_ADD(+,sum)
        GENERATE_OPERATOR_ADD(-,difference)

        #undef GENERATE_OPERATOR_ADD
    };

    namespace internal {
        template<typename Derived, int p>
        struct StateDerivativeLpNormImpl
        {
            typedef typename StateDerivativeBase<Derived>::RealScalar RealScalar;
            static inline RealScalar run(StateDerivativeBase<Derived> const & velocity)
            {
                EIGEN_USING_STD_MATH(pow)
                return pow(velocity.v().cwiseAbs().array().pow(p).sum() +
                           velocity.a().cwiseAbs().array().pow(p).sum(),
                           RealScalar(1)/p);
            }
        };

        template<typename Derived>
        struct StateDerivativeLpNormImpl<Derived, 1>
        {
            static inline typename StateDerivativeBase<Derived>::RealScalar
            run(StateDerivativeBase<Derived> const & velocity)
            {
                return velocity.v().cwiseAbs().sum() + velocity.a().cwiseAbs().sum();
            }
        };

        template<typename Derived>
        struct StateDerivativeLpNormImpl<Derived, Infinity>
        {
            static inline typename StateDerivativeBase<Derived>::RealScalar
            run(StateDerivativeBase<Derived> const & velocity)
            {
                return std::max(velocity.v().cwiseAbs().maxCoeff(),
                                velocity.a().cwiseAbs().maxCoeff());
            }
        };
    }

    template<typename Derived>
    template<int p>
    typename StateDerivativeBase<Derived>::RealScalar StateDerivativeBase<Derived>::lpNorm(void) const
    {
        return internal::StateDerivativeLpNormImpl<Derived, p>::run(*this);
    }

    namespace internal {
        template<typename _DataType>
        struct traits<StateDerivative<_DataType> >
        {
            typedef _DataType DataType;
        };
    }

    template<typename _DataType = vectorN_t>
    class StateDerivative : public StateDerivativeBase<StateDerivative<_DataType> >
    {
    public:
        typedef _DataType DataType;
        typedef typename DataType::Scalar Scalar;
        typedef StateDerivative const & Nested;

    public:
        explicit StateDerivative(Robot const * const & robotIn,
                                 vectorN_t const & vIn,
                                 vectorN_t const & aIn) :
        robot_(robotIn),
        v_(vIn),
        a_(aIn)
        {
            // Empty on purpose
        }

        explicit StateDerivative(Robot const * const & robotIn,
                                 vectorN_t && vIn,
                                 vectorN_t && aIn) :
        robot_(robotIn),
        v_(std::move(vIn)),
        a_(std::move(aIn))
        {
            // Empty on purpose
        }

        explicit StateDerivative(Robot const * const & robotIn,
                                 vectorN_t && vIn) :
        robot_(robotIn),
        v_(std::move(vIn)),
        a_(robot_->nv())
        {
            // Empty on purpose
        }

        explicit StateDerivative(Robot const * const & robotIn) :
        robot_(robotIn),
        v_(robot_->nv()),
        a_(robot_->nv())
        {
            // Empty on purpose
        }

        StateDerivative(StateDerivative const & other) :
        robot_(other.robot()),
        v_(other.v()),
        a_(other.a())
        {
            // Empty on purpose
        }

        StateDerivative(StateDerivative && other) :
        robot_(other.robot()),
        v_(std::move(other.v())),
        a_(std::move(other.a()))
        {
            // Empty on purpose
        }

        template<typename OtherDerived>
        StateDerivative(StateDerivativeBase<OtherDerived> const & other) :
        robot_(other.robot()),
        v_(other.v()),
        a_(other.a())
        {
            // Empty on purpose
        }

        Robot const * const & robot(void) const { return robot_; }
        DataType const & v(void) const { return v_; }
        DataType & v(void) { return v_; }
        DataType const & a(void) const { return a_; }
        DataType & a(void) { return a_; }

        static StateDerivativeWrapper<typename DataType::ConstantReturnType const> const
        Zero(Robot const * const & robotIn);
        static StateDerivativeWrapper<typename  DataType::ConstantReturnType const> const
        Ones(Robot const * const & robotIn);

        // This method allows you to assign Eigen expressions to StateDerivative
        template<typename OtherDerived>
        StateDerivative & operator=(StateDerivativeBase<OtherDerived> const & other)
        {
            robot_ = other.robot();
            v_ = other.v();
            a_ = other.a();
            return *this;
        }

        StateDerivative & operator=(StateDerivative const & other)
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
        Robot const * robot_;
        DataType v_;
        DataType a_;
    };

    namespace internal {
        template<typename _DataType>
        struct traits<StateDerivativeWrapper<_DataType> >
        {
            typedef _DataType DataType;
            typedef Eigen::Dense StorageKind;
        };
    }

    template<typename _DataType>
    class StateDerivativeWrapper :
    public StateDerivativeBase<StateDerivativeWrapper<_DataType> >,
    internal::no_assignment_operator
    {
    public:
        typedef _DataType DataType;
        typedef StateDerivativeWrapper Nested;

        explicit StateDerivativeWrapper(Robot const * const & robotIn,
                                        DataType & vIn,
                                        DataType & aIn) :
        robot_(robotIn),
        vRef_(vIn),
        aRef_(aIn)
        {
            // Empty on purpose
        }

        explicit StateDerivativeWrapper(Robot const * const & robotIn,
                                        DataType && vIn,
                                        DataType && aIn) :
        robot_(robotIn),
        vRef_(std::move(vIn)),
        aRef_(std::move(aIn))
        {
            // Empty on purpose
        }

        Robot const * const & robot(void) const { return robot_; }
        DataType const & v(void) const { return vRef_; }
        DataType const & a(void) const { return aRef_; }

    protected:
        Robot const * robot_;
        typename DataType::Nested vRef_;
        typename DataType::Nested aRef_;
    };

    template<typename _DataType>
    StateDerivativeWrapper<typename _DataType::ConstantReturnType const> const
    StateDerivative<_DataType>::Zero(Robot const * const & robotIn)
    {
        return StateDerivativeWrapper<typename DataType::ConstantReturnType const>(
            robotIn, DataType::Zero(robotIn->nv()), DataType::Zero(robotIn->nv()));
    }

    template<typename _DataType>
    StateDerivativeWrapper<typename _DataType::ConstantReturnType const> const
    StateDerivative<_DataType>::Ones(Robot const * const & robotIn)
    {
        return StateDerivativeWrapper<typename DataType::ConstantReturnType const>(
            robotIn, DataType::Ones(robotIn->nv()), DataType::Ones(robotIn->nv()));
    }

    // ====================================================
    // =================== Generic State ==================
    // ====================================================

    namespace internal {
        template<typename Derived>
        struct traits<StateBase<Derived> >
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

        Derived const & derived(void) const { return *static_cast<Derived const *>(this); }
        Derived & derived(void) { return *static_cast<Derived *>(this); }

        Robot const * const & robot(void) const { return derived().robot(); }

        DataType const & q(void) const { return derived().q(); }
        DataType & q(void) { return derived().q(); }

        DataType const & v(void) const { return derived().v(); }
        DataType & v(void) { return derived().v(); }

        template<int p>
        RealScalar lpNorm(void) const;
        RealScalar norm(void) const { return lpNorm<2>(); };
        RealScalar normInf(void) const { return lpNorm<Infinity>(); };

        void setZero(void)
        {
            derived().q().setZero();
            derived().v().setZero();
        }

        template<typename OtherDerived, typename OutDerived>
        StateBase<OutDerived> & sum(StateDerivativeBase<OtherDerived> const & velocity,
                                    StateBase<OutDerived>                   & out) const
        {
            // 'Sum' q = q + v, remember q is part of a Lie group (dim(q) != dim(v))
            assert(robot() == velocity.robot() == out.robot());
            pinocchio::integrate(robot()->pncModel_, q(), velocity.v(), out.q());
            out.v() = v() + velocity.a();
            return out;
        }

        template<typename OtherDerived>
        StateBase & sumInPlace(StateDerivativeBase<OtherDerived> const & velocity)
        {
            return sum(velocity, *this);
        }

        template<typename OtherDerived, typename OutDerived>
        StateDerivativeBase<OutDerived> & difference(StateBase<OtherDerived>         const & position,
                                                     StateDerivativeBase<OutDerived>       & out) const
        {
            assert(robot() == position.robot() == out.robot());
            pinocchio::difference(robot()->pncModel_, q(), position.q(), out.v());
            out.a() = v() - position.v();
            return out;
        }
    };

    namespace internal {
        template<typename Derived, int p>
        struct StateLpNormImpl
        {
            typedef typename StateBase<Derived>::RealScalar RealScalar;
            static inline RealScalar run(StateBase<Derived> const & velocity)
            {
                EIGEN_USING_STD_MATH(pow)
                return pow(velocity.q().cwiseAbs().array().pow(p).sum() +
                           velocity.v().cwiseAbs().array().pow(p).sum(),
                           RealScalar(1)/p);
            }
        };

        template<typename Derived>
        struct StateLpNormImpl<Derived, 1>
        {
            static inline typename StateBase<Derived>::RealScalar
            run(StateBase<Derived> const & velocity)
            {
                return velocity.q().cwiseAbs().sum() + velocity.v().cwiseAbs().sum();
            }
        };

        template<typename Derived>
        struct StateLpNormImpl<Derived, Infinity>
        {
            static inline typename StateBase<Derived>::RealScalar
            run(StateBase<Derived> const & velocity)
            {
                return std::max(velocity.q().cwiseAbs().maxCoeff(),
                                velocity.v().cwiseAbs().maxCoeff());
            }
        };
    }

    template<typename Derived>
    template<int p>
    typename StateBase<Derived>::RealScalar StateBase<Derived>::lpNorm(void) const
    {
        return internal::StateLpNormImpl<Derived, p>::run(*this);
    }

    namespace internal {
        template<typename _DataType>
        struct traits<State<_DataType> >
        {
            typedef _DataType DataType;
        };
    }

    template<typename _DataType = vectorN_t>
    class State : public StateBase<State<_DataType> >
    {
    public:
        typedef _DataType DataType;
        typedef typename DataType::Scalar Scalar;
        typedef State const & Nested;

    public:
        explicit State(Robot const * const & robotIn,
                       vectorN_t const & qIn,
                       vectorN_t const & vIn) :
        robot_(robotIn),
        q_(qIn),
        v_(vIn)
        {
            // Empty on purpose
        }

        explicit State(Robot const * const & robotIn,
                       vectorN_t && qIn,
                       vectorN_t && vIn) :
        robot_(robotIn),
        q_(std::move(qIn)),
        v_(std::move(vIn))
        {
            // Empty on purpose
        }

        explicit State(Robot const * const & robotIn) :
        robot_(robotIn),
        q_(robot_->nq()),
        v_(robot_->nv())
        {
            // Empty on purpose
        }

        State(State const & other) :
        robot_(other.robot()),
        q_(other.q()),
        v_(other.v())
        {
            // Empty on purpose
        }

        State(State && other) :
        robot_(other.robot()),
        q_(std::move(other.q())),
        v_(std::move(other.v()))
        {
            // Empty on purpose
        }

        template<typename OtherDerived>
        State(StateBase<OtherDerived> const & other) :
        robot_(other.robot()),
        q_(other.q()),
        v_(other.v())
        {
            // Empty on purpose
        }

        Robot const * const & robot(void) const { return robot_; }
        DataType const & q(void) const { return q_; }
        DataType & q(void) { return q_; }
        DataType const & v(void) const { return v_; }
        DataType & v(void) { return v_; }

        static StateWrapper<typename DataType::ConstantReturnType const> const
        Zero(Robot const * const & robotIn);
        static StateWrapper<typename DataType::ConstantReturnType const> const
        Ones(Robot const * const & robotIn);

        // This method allows you to assign Eigen expressions to State
        template<typename OtherDerived>
        State & operator=(StateBase<OtherDerived> const & other)
        {
            robot_ = other.robot();
            q_ = other.q();
            v_ = other.v();
            return *this;
        }

        State & operator=(State const & other)
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
        Robot const * robot_;
        DataType q_;
        DataType v_;
    };

    namespace internal {
        template<typename _DataType>
        struct traits<StateWrapper<_DataType> >
        {
            typedef _DataType DataType;
            typedef Eigen::Dense StorageKind;
        };
    }

    template<typename _DataType>
    class StateWrapper :
    public StateBase<StateWrapper<_DataType> >,
    internal::no_assignment_operator
    {
    public:
        typedef _DataType DataType;
        typedef StateWrapper Nested;

        explicit StateWrapper(Robot const * const & robotIn,
                              DataType & qIn,
                              DataType & vIn) :
        robot_(robotIn),
        qRef_(qIn),
        vRef_(vIn)
        {
            // Empty on purpose
        }

        explicit StateWrapper(Robot const * const & robotIn,
                              DataType && qIn,
                              DataType && vIn) :
        robot_(robotIn),
        qRef_(std::move(qIn)),
        vRef_(std::move(vIn))
        {
            // Empty on purpose
        }

        Robot const * const & robot(void) const { return robot_; }
        DataType const & q(void) const { return qRef_; }
        DataType const & v(void) const { return vRef_; }

    protected:
        Robot const * robot_;
        typename DataType::Nested qRef_;
        typename DataType::Nested vRef_;
    };

    template<typename _DataType>
    StateWrapper<typename _DataType::ConstantReturnType const> const
    State<_DataType>::Zero(Robot const * const & robotIn)
    {
        return StateWrapper<typename DataType::ConstantReturnType const>(
            robotIn, DataType::Zero(robotIn->nq()), DataType::Zero(robotIn->nv()));
    }

    template<typename _DataType>
    StateWrapper<typename _DataType::ConstantReturnType const> const
    State<_DataType>::Ones(Robot const * const & robotIn)
    {
        return StateWrapper<typename DataType::ConstantReturnType const>(
            robotIn, DataType::Ones(robotIn->nq()), DataType::Ones(robotIn->nv()));
    }

    // ====================================================
    // ============= Generic Vector container =============
    // ====================================================

    namespace internal {
        template<typename Derived>
        struct traits<VectorContainerBase<Derived> >
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

        Derived const & derived(void) const { return *static_cast<Derived const *>(this); }
        Derived & derived(void) { return *static_cast<Derived *>(this); }

        std::vector<ValueType> const & vector(void) const { return derived().vector(); }
        std::vector<ValueType> & vector(void) { return derived().vector(); };

        template<int p>
        RealScalar lpNorm(void) const;
        RealScalar norm(void) const { return lpNorm<2>(); };
        RealScalar normInf(void) const { return lpNorm<Infinity>(); };

        void setZero(void)
        {
            for (ValueType & element : vector())
            {
                element.setZero();
            }
        }

        #define GENERATE_OPERATOR_ARITHEMTIC(OP,NAME) \
        auto const (operator OP)(Scalar const & scalar) const \
        { \
            typedef std::remove_const_t<decltype( \
                std::declval<ValueType>() OP std::declval<Scalar>())> wrappedType; \
            std::vector<wrappedType> result; \
            std::vector<ValueType> const & vector_ = vector(); \
            result.reserve(vector_.size()); \
            for (ValueType const & element : vector_) \
            { \
                result.emplace_back(element OP scalar); \
            } \
            return VectorContainerWrapper<wrappedType>(std::move(result)); \
        } \
         \
        friend auto const (operator OP)(Scalar              const & scalar, \
                                        VectorContainerBase const & other) \
        { \
            return other OP scalar; \
        } \
         \
        template<typename OtherDerived> \
        auto const (operator OP)(VectorContainerBase<OtherDerived> const & other) const \
        { \
            typedef std::remove_const_t<decltype( \
                std::declval<ValueType>() OP std::declval<typename OtherDerived::ValueType>())> wrappedType; \
            std::vector<ValueType> const & vector_ = vector(); \
            std::vector<typename OtherDerived::ValueType> const & vectorIn = other.vector(); \
            assert(vector_.size() == vectorIn.size()); \
            std::vector<wrappedType> result; \
            result.reserve(vector_.size()); \
            for (uint32_t i = 0; i < vector_.size(); ++i) \
            { \
                result.emplace_back(vector_[i] OP vectorIn[i]); \
            } \
            return VectorContainerWrapper<wrappedType>(std::move(result)); \
        } \
         \
        VectorContainerBase & (operator EIGEN_CAT(OP,=))(Scalar const & scalar) \
        { \
            for (ValueType & element : vector()) \
            { \
                element EIGEN_CAT(OP,=) scalar; \
            } \
            return *this; \
        }

        GENERATE_OPERATOR_ARITHEMTIC(*,product)
        GENERATE_OPERATOR_ARITHEMTIC(/,quotient)
        GENERATE_OPERATOR_ARITHEMTIC(+,sum)
        GENERATE_OPERATOR_ARITHEMTIC(-,difference)

        #undef GENERATE_OPERATOR_ARITHEMTIC

        #define GENERATE_OPERATOR_COMPOUND(OP,NAME) \
        template<typename OtherDerived> \
        VectorContainerBase & (operator EIGEN_CAT(OP,=))( \
            VectorContainerBase<OtherDerived> const & other) \
        { \
            std::vector<ValueType> & vector_ = vector(); \
            std::vector<typename internal::traits<OtherDerived>::ValueType> const & \
                vectorIn = other.vector(); \
            assert(vector_.size() == vectorIn.size()); \
            for (uint32_t i = 0; i < vector_.size(); ++i) \
            { \
                vector_[i] EIGEN_CAT(OP,=) vectorIn[i]; \
            } \
            return *this; \
        }

        GENERATE_OPERATOR_COMPOUND(+,sum)
        GENERATE_OPERATOR_COMPOUND(-,difference)

        #undef GENERATE_OPERATOR_COMPOUND
    };

    namespace internal {
        template<typename Derived, int p>
        struct VectorContainerLpNormImpl
        {
            typedef typename VectorContainerBase<Derived>::RealScalar RealScalar;
            static inline RealScalar run(VectorContainerBase<Derived> const & container)
            {
                EIGEN_USING_STD_MATH(pow)
                RealScalar total = 0.0;
                for (typename internal::traits<Derived>::ValueType const & element : container.vector())
                {
                    total += pow(element.template lpNorm<p>(), p);
                }
                return pow(total, RealScalar(1)/p);
            }
        };

        template<typename Derived>
        struct VectorContainerLpNormImpl<Derived, 1>
        {
            typedef typename VectorContainerBase<Derived>::RealScalar RealScalar;
            static inline RealScalar run(VectorContainerBase<Derived> const & container)
            {
                return std::accumulate(
                    container.vector().begin(), container.vector().end(), RealScalar(0.0),
                    [](RealScalar & cumsum, typename internal::traits<Derived>::ValueType const & element)
                    {
                        return element.template lpNorm<1>();
                    }
                );
            }
        };

        template<typename Derived>
        struct VectorContainerLpNormImpl<Derived, Infinity>
        {
            typedef typename VectorContainerBase<Derived>::RealScalar RealScalar;
            static inline RealScalar run(VectorContainerBase<Derived> const & container)
            {
                RealScalar maxValue = 0.0;
                for (typename internal::traits<Derived>::ValueType const & element : container.vector())
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
    typename VectorContainerBase<Derived>::RealScalar VectorContainerBase<Derived>::lpNorm(void) const
    {
        return internal::VectorContainerLpNormImpl<Derived, p>::run(*this);
    }

    namespace internal {
        template<typename _ValueType>
        struct traits<VectorContainer<_ValueType> >
        {
            typedef _ValueType ValueType;
        };
    }

    template<typename _ValueType>
    class VectorContainer : public VectorContainerBase<VectorContainer<_ValueType> >
    {
    public:
        typedef _ValueType ValueType;
        typedef typename ValueType::DataType DataType;
        typedef VectorContainer const & Nested;

        VectorContainer(void) :
        vector_()
        {
            // Empty on purpose
        }

        template <typename ... Args>
        explicit VectorContainer(std::vector<Robot const *> const & robots,
                                 std::vector<Args> const & ... args) :
        vector_()
        {
            do_for([&robots](auto arg) {
                assert(arg.size() == robots.size());
            }, args...);
            vector_.reserve(robots.size());
            for (uint32_t i = 0; i < robots.size(); ++i)
            {
                vector_.emplace_back(robots[i], args[i]...);
            }
        }

        template <typename ... Args>
        explicit VectorContainer(std::vector<Robot const *> const & robots,
                                 std::vector<Args> && ... args) :
        vector_()
        {
            do_for([&robots](auto arg) {
                assert(arg.size() == robots.size());
            }, args...);
            vector_.reserve(robots.size());
            for (uint32_t i = 0; i < robots.size(); ++i)
            {
                vector_.emplace_back(robots[i], std::move(args[i])...);
            }
        }

        explicit VectorContainer(std::vector<Robot const *> const & robots) :
        vector_()
        {
            vector_.reserve(robots.size());
            for (uint32_t i = 0; i < robots.size(); ++i)
            {
                vector_.emplace_back(robots[i]);
            }
        }

        template<typename OtherDerived>
        VectorContainer(VectorContainerBase<OtherDerived> const & other) :
        VectorContainer(other.vector())
        {
            // Empty on purpose
        }

        VectorContainer(VectorContainer && other) :
        vector_(std::move(other.vector()))
        {
            // Empty on purpose
        }

        VectorContainer(std::vector<ValueType> && vectorIn) :
        vector_(std::move(vectorIn))
        {
            // Empty on purpose
        }

        template<typename OtherValueType>
        VectorContainer(std::vector<OtherValueType> const & vectorIn) :
        vector_()
        {
            vector_.reserve(vectorIn.size());
            std::copy(vectorIn.begin(), vectorIn.end(), std::back_inserter(vector_));
        }

        std::vector<ValueType> const & vector(void) const { return vector_; }
        std::vector<ValueType> & vector(void) { return vector_; }

        static auto const Zero(std::vector<Robot const *> const & robots);
        static auto const Ones(std::vector<Robot const *> const & robots);

        // This method allows you to assign Eigen expressions to VectorContainer
        template<typename OtherDerived>
        VectorContainer & operator=(VectorContainerBase<OtherDerived> const & other)
        {
            std::vector<typename internal::traits<OtherDerived>::ValueType> const & vectorIn = other.vector();
            vector_.resize(vectorIn.size());
            for (uint32_t i = 0; i < vector_.size(); ++i)
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

    namespace internal {
        template<typename _ValueType>
        struct traits<VectorContainerWrapper<_ValueType> >
        {
            typedef _ValueType ValueType;
            typedef Eigen::Dense StorageKind;
        };
    }

    template<typename _ValueType>
    class VectorContainerWrapper :
    public VectorContainerBase<VectorContainerWrapper<_ValueType> >,
    internal::no_assignment_operator
    {
    public:
        typedef _ValueType ValueType;
        typedef VectorContainerWrapper Nested;

        explicit VectorContainerWrapper(std::vector<ValueType> & vectorIn) :
        vector_(vectorIn)
        {
            // Empty on purpose
        }

        explicit VectorContainerWrapper(std::vector<ValueType> && vectorIn) :
        vector_(std::move(vectorIn))
        {
            // Empty on purpose
        }

        std::vector<ValueType> const & vector(void) const { return vector_; }

    protected:
        std::vector<ValueType> vector_;
    };

    template<typename _ValueType>
    auto const VectorContainer<_ValueType>::Zero(std::vector<Robot const *> const & robots)
    {
        typedef std::remove_const_t<decltype( \
            _ValueType::Zero(std::declval<Robot const * const &>()))> wrappedType; \
        std::vector<wrappedType> vectorIn;
        vectorIn.reserve(robots.size());
        std::transform(robots.begin(), robots.end(),
                       std::back_inserter(vectorIn),
                       [](Robot const * const & robot) -> wrappedType
                       {
                           return _ValueType::Zero(robot);
                       });
        return VectorContainerWrapper<wrappedType>(std::move(vectorIn));
    }

    template<typename _ValueType>
    auto const VectorContainer<_ValueType>::Ones(std::vector<Robot const *> const & robots)
    {
        typedef std::remove_const_t<decltype( \
            _ValueType::Ones(std::declval<Robot const * const &>()))> wrappedType; \
        std::vector<wrappedType> vectorIn;
        vectorIn.reserve(robots.size());
        std::transform(robots.begin(), robots.end(),
                       std::back_inserter(vectorIn),
                       [](Robot const * const & robot) -> wrappedType
                       {
                           return _ValueType::Ones(robot);
                       });
        return VectorContainerWrapper<wrappedType>(std::move(vectorIn));
    }

    // ====================================================
    // ================== Specializations =================
    // ====================================================

    #define GENERATE_SHARED_IMPL(BASE,VAR1,SIZE1,VAR2,SIZE2) \
    namespace internal { \
        template<> \
        struct traits<EIGEN_CAT(BASE,Shared)> \
        { \
            typedef Eigen::Ref<vectorN_t> DataType; \
        }; \
    } \
     \
    class EIGEN_CAT(BASE,Shared) : public EIGEN_CAT(BASE,Base)<EIGEN_CAT(BASE,Shared)> \
    { \
    public: \
        typedef typename internal::traits<EIGEN_CAT(BASE,Shared)>::DataType DataType; \
        typedef EIGEN_CAT(BASE,Shared) Nested; \
         \
        explicit EIGEN_CAT(BASE,Shared)(Robot const * const & robot, \
                                        Eigen::Ref<vectorN_t> const & VAR1, \
                                        Eigen::Ref<vectorN_t> const & VAR2) : \
        robot_(robot), \
        EIGEN_CAT(VAR1,Ref_)(VAR1), \
        EIGEN_CAT(VAR2,Ref_)(VAR2) \
        { \
            /* Empty on purpose */ \
        } \
         \
        Robot const * const & robot(void) const { return robot_; } \
        Eigen::Ref<vectorN_t> & VAR1(void) { return EIGEN_CAT(VAR1,Ref_); } \
        Eigen::Ref<vectorN_t> const & VAR1(void) const { return EIGEN_CAT(VAR1,Ref_); } \
        Eigen::Ref<vectorN_t> & VAR2(void) { return EIGEN_CAT(VAR2,Ref_); } \
        Eigen::Ref<vectorN_t> const & VAR2(void) const { return EIGEN_CAT(VAR2,Ref_); } \
         \
    protected: \
        Robot const * robot_; \
        Eigen::Ref<vectorN_t> EIGEN_CAT(VAR1,Ref_); \
        Eigen::Ref<vectorN_t> EIGEN_CAT(VAR2,Ref_); \
    }; \
     \
    class EIGEN_CAT(BASE,Vector) : public VectorContainer<EIGEN_CAT(BASE,Shared)> \
    { \
    public: \
        EIGEN_CAT(BASE,Vector)(void) : \
        VectorContainer<EIGEN_CAT(BASE,Shared)>(), \
        VAR1(), \
        VAR2() \
        { \
            /* Empty on purpose */ \
        } \
         \
        explicit EIGEN_CAT(BASE,Vector)(std::vector<Robot const *> const & robots, \
                                        std::vector<vectorN_t> const & EIGEN_CAT(VAR1,In), \
                                        std::vector<vectorN_t> const & EIGEN_CAT(VAR2,In)) : \
        VectorContainer<EIGEN_CAT(BASE,Shared)>(), \
        VAR1(EIGEN_CAT(VAR1,In)), \
        VAR2(EIGEN_CAT(VAR2,In)) \
        { \
            assert(VAR1.size() == robots.size() && VAR2.size() == robots.size()); \
            vector_.reserve(robots.size()); \
            for (uint32_t i = 0; i < robots.size(); ++i) \
            { \
                vector_.emplace_back(robots[i], VAR1[i], VAR2[i]); \
            } \
        } \
         \
        explicit EIGEN_CAT(BASE,Vector)(std::vector<Robot const *> const & robots) : \
        VectorContainer<EIGEN_CAT(BASE,Shared)>(), \
        VAR1(), \
        VAR2() \
        { \
            vector_.reserve(robots.size()); \
            VAR1.reserve(robots.size()); \
            VAR2.reserve(robots.size()); \
            for (uint32_t i = 0; i < robots.size(); ++i) \
            { \
                VAR1.emplace_back(robots[i]->SIZE1()); \
                VAR2.emplace_back(robots[i]->SIZE2()); \
                vector_.emplace_back(robots[i], VAR1[i], VAR2[i]); \
            } \
        } \
         \
        EIGEN_CAT(BASE,Vector)(EIGEN_CAT(BASE,Vector) const & other) : \
        VectorContainer<EIGEN_CAT(BASE,Shared)>(), \
        VAR1(other.VAR1), \
        VAR2(other.VAR2) \
        { \
            std::vector<ValueType> const & vectorIn = other.vector(); \
            vector_.reserve(vectorIn.size()); \
            for (uint32_t i = 0; i < vectorIn.size(); ++i) \
            { \
                vector_.emplace_back(vectorIn[i].robot(), VAR1[i], VAR2[i]); \
            } \
        } \
         \
        EIGEN_CAT(BASE,Vector)(EIGEN_CAT(BASE,Vector) && other) : \
        VectorContainer<EIGEN_CAT(BASE,Shared)>(std::move(other.vector())), \
        VAR1(std::move(other.VAR1)), \
        VAR2(std::move(other.VAR2)) \
        { \
            /* Empty on purpose */ \
        } \
         \
        template<typename OtherDerived> \
        EIGEN_CAT(BASE,Vector)(VectorContainerBase<OtherDerived> const & other) : \
        VectorContainer<EIGEN_CAT(BASE,Shared)>(), \
        VAR1(), \
        VAR2() \
        { \
            std::vector<typename internal::traits<OtherDerived>::ValueType> const & vectorIn = other.vector(); \
            vector_.reserve(vectorIn.size()); \
            VAR1.reserve(vectorIn.size()); \
            VAR2.reserve(vectorIn.size()); \
            for (uint32_t i = 0; i < vectorIn.size(); ++i) \
            { \
                VAR1.push_back(vectorIn[i].VAR1()); \
                VAR2.push_back(vectorIn[i].VAR2()); \
                vector_.emplace_back(vectorIn[i].robot(), VAR1[i], VAR2[i]); \
            } \
        } \
         \
        template<typename OtherValueType> \
        EIGEN_CAT(BASE,Vector)(std::vector<OtherValueType> && vectorIn) : \
        VectorContainer<EIGEN_CAT(BASE,Shared)>(), \
        VAR1(), \
        VAR2() \
        { \
            vector_.reserve(vectorIn.size()); \
            VAR1.reserve(vectorIn.size()); \
            VAR2.reserve(vectorIn.size()); \
            for (uint32_t i = 0; i < vectorIn.size(); ++i) \
            { \
                VAR1.push_back(std::move(vectorIn[i].VAR1())); \
                VAR2.push_back(std::move(vectorIn[i].VAR2())); \
                vector_.emplace_back(vectorIn[i].robot(), VAR1[i], VAR2[i]); \
            } \
        } \
         \
        static EIGEN_CAT(BASE,Vector) Ones(std::vector<Robot const *> const & robots) \
        { \
            EIGEN_CAT(BASE,Vector) container(robots); \
            for (uint32_t i = 0; i < robots.size(); ++i) \
            { \
                container.VAR1[i].setOnes(); \
                container.VAR2[i].setOnes(); \
            } \
            return container; \
        } \
         \
        static EIGEN_CAT(BASE,Vector) Zero(std::vector<Robot const *> const & robots) \
        { \
            EIGEN_CAT(BASE,Vector) container(robots); \
            for (uint32_t i = 0; i < robots.size(); ++i) \
            { \
                container.VAR1[i].setZero(); \
                container.VAR2[i].setZero(); \
            } \
            return container; \
        } \
         \
        template<typename OtherDerived> \
        EIGEN_CAT(BASE,Vector) & operator=(VectorContainerBase<OtherDerived> const & other) \
        { \
            std::vector<typename internal::traits<OtherDerived>::ValueType> const & vectorIn = other.vector(); \
            assert(vectorIn.size() == vector_.size()); \
            for (uint32_t i = 0; i < vector_.size(); ++i) \
            { \
                assert(vectorIn[i].robot() == vector_[i].robot()); \
                VAR1[i] = vectorIn[i].VAR1(); \
                VAR2[i] = vectorIn[i].VAR2(); \
            } \
            return *this; \
        } \
         \
        EIGEN_CAT(BASE,Vector) & operator=(EIGEN_CAT(BASE,Vector) const & other) \
        { \
            std::vector<ValueType> const & vectorIn = other.vector(); \
            assert(vectorIn.size() == vector_.size()); \
            for (uint32_t i = 0; i < vectorIn.size(); ++i) \
            { \
                assert(vectorIn[i].robot() == vector_[i].robot()); \
                VAR1[i] = other.VAR1[i]; \
                VAR2[i] = other.VAR2[i]; \
            } \
            return *this; \
        } \
         \
        EIGEN_CAT(BASE,Vector) & operator=(EIGEN_CAT(BASE,Vector) && other) \
        { \
            VAR1 = std::move(other.VAR1); \
            VAR2 = std::move(other.VAR2); \
            vector_ = std::move(other.vector()); \
            return *this; \
        } \
         \
        EIGEN_CAT(BASE,_SHARED_ADDON) \
    public: \
        std::vector<vectorN_t> VAR1; \
        std::vector<vectorN_t> VAR2; \
    };

    #define StateDerivative_SHARED_ADDON \
    template<typename Derived, \
             typename = typename std::enable_if_t<is_base_of_template<StateDerivativeBase, \
                        typename internal::traits<Derived>::ValueType>::value, void> > \
    StateDerivativeVector & sumInPlace(VectorContainerBase<Derived> const & other, \
                                       Scalar const & scale) \
    { \
        std::vector<typename internal::traits<Derived>::ValueType> const & \
            vectorIn = other.vector(); \
        assert(vector_.size() == vectorOut.size()); \
        for (uint32_t i = 0; i < vector_.size(); ++i) \
        { \
            vector_[i] += scale * vectorIn[i]; \
        } \
        return *this; \
    }

    #define State_SHARED_ADDON \
    template<typename Derived, typename OtherDerived, \
             typename = typename std::enable_if_t< \
                is_base_of_template<StateDerivativeBase, \
                                    typename internal::traits<Derived>::ValueType>::value && \
                is_base_of_template<StateBase, \
                                    typename internal::traits<OtherDerived>::ValueType>::value, \
                void> > \
    VectorContainerBase<OtherDerived> & sum(VectorContainerBase<Derived> const & other, \
                                            VectorContainerBase<OtherDerived> & out) const \
    { \
        std::vector<typename internal::traits<Derived>::ValueType> const & \
            vectorIn = other.vector(); \
        std::vector<typename internal::traits<OtherDerived>::ValueType> & \
            vectorOut = out.vector(); \
        assert(vectorIn.size() == vectorOut.size()); \
        for (uint32_t i = 0; i < vector_.size(); ++i) \
        { \
            vector_[i].sum(vectorIn[i], vectorOut[i]); \
        } \
        return out; \
    } \
    \
    template<typename Derived, \
             typename = typename std::enable_if_t<is_base_of_template<StateDerivativeBase, \
                        typename internal::traits<Derived>::ValueType>::value, void> > \
    StateVector & sumInPlace(VectorContainerBase<Derived> const & other) \
    { \
        sum(other, *this); \
        return *this; \
    } \
    \
    template<typename Derived, \
             typename = typename std::enable_if_t<is_base_of_template<StateDerivativeBase, \
                        typename internal::traits<Derived>::ValueType>::value, void> > \
    StateVector & sumInPlace(VectorContainerBase<Derived> const & other, \
                             Scalar const & scale) \
    { \
        std::vector<typename internal::traits<Derived>::ValueType> const & \
            vectorIn = other.vector(); \
        assert(vector_.size() == vectorOut.size()); \
        for (uint32_t i = 0; i < vector_.size(); ++i) \
        { \
            vector_[i].sumInPlace(scale * vectorIn[i]); \
        } \
        return *this; \
    } \
    \
    template<typename Derived, typename OtherDerived, \
             typename = typename std::enable_if_t< \
                is_base_of_template<StateBase, \
                                    typename internal::traits<Derived>::ValueType>::value && \
                is_base_of_template<StateDerivativeBase, \
                                    typename internal::traits<OtherDerived>::ValueType>::value, \
                void> > \
    VectorContainerBase<OtherDerived> & difference(VectorContainerBase<Derived> const & other, \
                                                   VectorContainerBase<OtherDerived> & out) const \
    { \
        std::vector<typename internal::traits<Derived>::ValueType> const & \
            vectorIn = other.vector(); \
        std::vector<typename internal::traits<OtherDerived>::ValueType> & \
            vectorOut = out.vector(); \
        assert(vectorIn.size() == vectorOut.size()); \
        for (uint32_t i = 0; i < vector_.size(); ++i) \
        { \
            vector_[i].difference(vectorIn[i], vectorOut[i]); \
        } \
        return out; \
    }

    // Disable "-Weffc++" flag while generting this code because it is buggy...
    #pragma GCC diagnostic ignored "-Weffc++"
    GENERATE_SHARED_IMPL(StateDerivative,v,nv,a,nv)
    GENERATE_SHARED_IMPL(State,q,nq,v,nv)
    #pragma GCC diagnostic pop

    #undef GENERATE_SHARED_IMPL
}

namespace jiminy
{
    using state_t = Eigen::StateVector;
    using stateDerivative_t = Eigen::StateDerivativeVector;
}

#endif //end of JIMINY_LIE_GROUP_H
