#ifndef JIMINY_LCP_SOLVERS_H
#define JIMINY_LCP_SOLVERS_H

#include "jiminy/core/Types.h"


namespace jiminy
{
    float64_t const EPS_DIVISION = 1.0e-9;

    class AbstractLCPSolver
    {
    public:
        /// \brief Compute the solution of the Linear Complementary Problem:
        ///        A x + b = w,
        ///        s.t. (w[i] > 0 and x[i] = 0) or (w[i] = 0 and x[i] > 0
        ///
        ///        using boxed bounds lo < x < hi instead of 0 < x:
        ///        s.t. if fIdx[i] < 0, lo[i] < x[i] < hi[i]
        ///             else, - hi[i] x[fIdx[i]] < x[i] < hi[i] x[fIdx[i]]
        ///
        /// The result x will be stored in data.lambda_c.
        virtual bool_t BoxedForwardDynamics(pinocchio::Model const & model,
                                            pinocchio::Data & data,
                                            vectorN_t const & tau,
                                            matrixN_t const & J,
                                            vectorN_t const & gamma,
                                            float64_t const & inv_damping,
                                            vectorN_t const & lo,
                                            vectorN_t const & hi,
                                            std::vector<int32_t> const & fIdx) = 0;
    };

    class PGSSolver : public AbstractLCPSolver
    {
    public:
        PGSSolver(uint32_t const & maxIter,
                  uint32_t const & randomPermutationPeriod,
                  float64_t const & tolAbs,
                  float64_t const & tolRel);

        virtual bool_t BoxedForwardDynamics(pinocchio::Model const & model,
                                            pinocchio::Data & data,
                                            vectorN_t const & tau,
                                            matrixN_t const & J,
                                            vectorN_t const & gamma,
                                            float64_t const & inv_damping,
                                            vectorN_t const & lo,
                                            vectorN_t const & hi,
                                            std::vector<int32_t> const & fIdx) override final;

    private:
        bool_t ProjectedGaussSeidelIter(matrixN_t const & A,
                                        vectorN_t const & b,
                                        vectorN_t const & lo,
                                        vectorN_t const & hi,
                                        std::vector<int32_t> const & fIdx,
                                        bool_t const & checkAbs,
                                        bool_t const & checkRel,
                                        vectorN_t & x);
        bool_t ProjectedGaussSeidelSolver(matrixN_t & A,
                                          vectorN_t & b,
                                          vectorN_t const & lo,
                                          vectorN_t const & hi,
                                          std::vector<int32_t> const & fIdx,
                                          vectorN_t & x);

    private:
        uint32_t maxIter_;
        uint32_t randomPermutationPeriod_;
        float64_t tolAbs_;
        float64_t tolRel_;
        std::vector<uint32_t> indices_;
        uint32_t lastShuffle_;
        vectorN_t b_;
    };


}

#endif  // JIMINY_LCP_SOLVERS_H
