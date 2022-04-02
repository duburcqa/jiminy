#ifndef JIMINY_LCP_SOLVERS_H
#define JIMINY_LCP_SOLVERS_H

#include "jiminy/core/Types.h"


namespace jiminy
{
    class constraintsHolder_t;

    class AbstractContactSolver
    {
    public:
        AbstractContactSolver(void) = default;
        virtual ~AbstractContactSolver(void) = default;

        /// \brief Compute the solution of the Nonlinear Complementary Problem:
        ///        A x + b = w,
        ///        s.t. (w[i] > 0 and x[i] = 0) or (w[i] = 0 and x[i] > 0
        ///
        ///        for non-linear boxed bounds lo(x) < x < hi(x):
        ///        s.t. if fIndices[i].size() == 0, lo[i] < x[i] < hi[i]
        ///             else, sqrt(x[i] ** 2 + sum_{j>=1}(x[fIndices[i][j]] ** 2)) < hi[i] * max(0.0, x[fIndices[i][0]])
        ///
        /// The result x will be stored in data.lambda_c.
        virtual bool_t BoxedForwardDynamics(pinocchio::Model const & model,
                                            pinocchio::Data & data,
                                            constraintsHolder_t & constraintsHolder,
                                            std::vector<int32_t> const & boundJointsActiveDir,
                                            float64_t const & friction,
                                            float64_t const & torsion,
                                            float64_t const & inv_damping) = 0;
    };

    class PGSSolver : public AbstractContactSolver
    {
    public:
        PGSSolver(uint32_t const & maxIter,
                  float64_t const & tolAbs,
                  float64_t const & tolRel);

        virtual bool_t BoxedForwardDynamics(pinocchio::Model const & model,
                                            pinocchio::Data & data,
                                            constraintsHolder_t & constraintsHolder,
                                            std::vector<int32_t> const & boundJointsActiveDir,
                                            float64_t const & friction,
                                            float64_t const & torsion,
                                            float64_t const & inv_damping) override final;

    private:
        void ProjectedGaussSeidelIter(matrixN_t const & A,
                                      vectorN_t const & b,
                                      vectorN_t & x);
        bool_t ProjectedGaussSeidelSolver(matrixN_t const & A,
                                          vectorN_t const & b,
                                          vectorN_t & x);

    private:
        uint32_t maxIter_;
        float64_t tolAbs_;
        float64_t tolRel_;

        matrixN_t J_;                 ///< Matrix holding the jacobian of the constraints
        vectorN_t gamma_;             ///< Vector holding the drift of the constraints
        std::vector<float64_t> lo_;
        std::vector<float64_t> hi_;
        std::vector<std::vector<int32_t> > fIndices_;

        vectorN_t b_;
        vectorN_t y_;
        vectorN_t dy_;
        vectorN_t yPrev_;
        vectorN_t xPrev_;
    };
}

#endif  // JIMINY_LCP_SOLVERS_H
