#ifndef JIMINY_LCP_SOLVERS_H
#define JIMINY_LCP_SOLVERS_H

#include "jiminy/core/Types.h"


namespace jiminy
{
    class AbstractLCPSolver
    {
    public:
        AbstractLCPSolver(void) = default;
        virtual ~AbstractLCPSolver(void) = default;

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
                                            vectorN_t const & tau,
                                            Eigen::Ref<matrixN_t const> const & J,
                                            Eigen::Ref<vectorN_t const> const & gamma,
                                            float64_t const & inv_damping,
                                            vectorN_t const & lo,
                                            vectorN_t const & hi,
                                            std::vector<std::vector<int32_t> > const & fIndices) = 0;
    };

    class PGSSolver : public AbstractLCPSolver
    {
    public:
        PGSSolver(uint32_t const & maxIter,
                  float64_t const & tolAbs,
                  float64_t const & tolRel);

        virtual bool_t BoxedForwardDynamics(pinocchio::Model const & model,
                                            pinocchio::Data & data,
                                            vectorN_t const & tau,
                                            Eigen::Ref<matrixN_t const> const & J,
                                            Eigen::Ref<vectorN_t const> const & gamma,
                                            float64_t const & inv_damping,
                                            vectorN_t const & lo,
                                            vectorN_t const & hi,
                                            std::vector<std::vector<int32_t> > const & fIndices) override final;

    private:
        void ProjectedGaussSeidelIter(matrixN_t const & A,
                                      vectorN_t const & b,
                                      vectorN_t const & lo,
                                      vectorN_t const & hi,
                                      std::vector<std::vector<int32_t> > const & fIndices,
                                      vectorN_t & x);
        bool_t ProjectedGaussSeidelSolver(matrixN_t & A,
                                          vectorN_t & b,
                                          vectorN_t const & lo,
                                          vectorN_t const & hi,
                                          std::vector<std::vector<int32_t> > const & fIndices,
                                          vectorN_t & x);

    private:
        uint32_t maxIter_;
        float64_t tolAbs_;
        float64_t tolRel_;
        vectorN_t b_;
        vectorN_t xPrev_;
        vectorN_t y_;
        vectorN_t yPrev_;
        vectorN_t dy_;
    };


}

#endif  // JIMINY_LCP_SOLVERS_H
