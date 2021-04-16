#include <numeric>

#include "jiminy/core/utilities/Random.h"


namespace jiminy
{
    static float64_t const PERLIN_NOISE_PERSISTENCE = 1.50;
    static float64_t const PERLIN_NOISE_LACUNARITY = 1.15;

    // ***************** Random number generator_ *****************

    // Based on Ziggurat generator by Marsaglia and Tsang (JSS, 2000):
    // https://people.sc.fsu.edu/~jburkardt/cpp_src/ziggurat/ziggurat.html

    std::mt19937 generator_;
    std::uniform_real_distribution<float32_t> distUniform_(0.0, 1.0);
    bool_t isInitialized_ = false;
    uint32_t seed_ = 0U;

    uint32_t kn[128];
    float32_t fn[128];
    float32_t wn[128];

    void r4_nor_setup(void)
    {
        float64_t const m1 = 2147483648.0;
        float64_t const vn = 9.91256303526217e-03;
        float64_t dn = 3.442619855899;
        float64_t tn = dn;

        float64_t q = vn / exp(-0.5 * dn * dn);

        kn[0] = static_cast<uint32_t>((dn / q) * m1);
        kn[1] = 0;

        wn[0] = static_cast<float32_t>(q / m1);
        wn[127] = static_cast<float32_t>(dn / m1);

        fn[0] = 1.0f;
        fn[127] = static_cast<float32_t>(exp(-0.5 * dn * dn));

        for (uint8_t i=126; 1 <= i; i--)
        {
            dn = sqrt(-2.0 * log(vn / dn + exp(-0.5 * dn * dn)));
            kn[i+1] = static_cast<uint32_t>((dn / tn) * m1);
            tn = dn;
            fn[i] = static_cast<float32_t>(exp(-0.5 * dn * dn));
            wn[i] = static_cast<float32_t>(dn / m1);
        }
    }

    float32_t r4_uni(void)
    {
        return distUniform_(generator_);
    }

    float32_t r4_nor(void)
    {
        float32_t const r = 3.442620f;
        int32_t hz;
        uint32_t iz;
        float32_t x;
        float32_t y;

        hz = static_cast<int32_t>(generator_());
        iz = (hz & 127U);

        if (fabs(hz) < kn[iz])
        {
            return static_cast<float32_t>(hz) * wn[iz];
        }
        else
        {
            while (true)
            {
                if (iz == 0)
                {
                    while (true)
                    {
                        x = - 0.2904764f * log(r4_uni());
                        y = - log(r4_uni());
                        if (x * x <= y + y)
                        {
                            break;
                        }
                    }

                    if (hz <= 0)
                    {
                        return - r - x;
                    }
                    else
                    {
                        return + r + x;
                    }
                }

                x = static_cast<float32_t>(hz) * wn[iz];

                if (fn[iz] + r4_uni() * (fn[iz-1] - fn[iz]) < exp (-0.5f * x * x))
                {
                    return x;
                }

                hz = static_cast<int32_t>(generator_());
                iz = (hz & 127);

                if (fabs(hz) < kn[iz])
                {
                    return static_cast<float32_t>(hz) * wn[iz];
                }
            }
        }
    }

    void resetRandomGenerators(boost::optional<uint32_t> const & seed)
    {
        uint32_t newSeed = seed.value_or(seed_);
        srand(newSeed);  // Eigen relies on srand for genering random numbers
        generator_.seed(newSeed);
        r4_nor_setup();
        seed_ = newSeed;
        isInitialized_ = true;
    }

    hresult_t getRandomSeed(uint32_t & seed)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Random number generator not initialized.");
            return hresult_t::ERROR_GENERIC;
        }

        seed = seed_;

        return hresult_t::SUCCESS;
    }

    float64_t randUniform(float64_t const & lo,
                          float64_t const & hi)
    {
        assert(isInitialized_ && "Random number genetors not initialized. "
                                 "Please call `resetRandomGenerators` at least once.");
        return lo + r4_uni() * (hi - lo);
    }

    float64_t randNormal(float64_t const & mean,
                         float64_t const & std)
    {
        assert(isInitialized_ && "Random number genetors not initialized. "
                                 "Please call `resetRandomGenerators` at least once.");
        return mean + r4_nor() * std;
    }

    vectorN_t randVectorNormal(uint32_t  const & size,
                               float64_t const & mean,
                               float64_t const & std)
    {
        if (std > 0.0)
        {
            return vectorN_t::NullaryExpr(size,
            [&mean, &std] (vectorN_t::Index const &) -> float64_t
            {
                return randNormal(mean, std);
            });
        }
        else
        {
            return vectorN_t::Constant(size, mean);
        }
    }

    vectorN_t randVectorNormal(uint32_t  const & size,
                               float64_t const & std)
    {
        return randVectorNormal(size, 0, std);
    }

    vectorN_t randVectorNormal(vectorN_t const & mean,
                               vectorN_t const & std)
    {
        return vectorN_t::NullaryExpr(std.size(),
        [&mean, &std] (vectorN_t::Index const & i) -> float64_t
        {
            return randNormal(mean[i], std[i]);
        });
    }

    vectorN_t randVectorNormal(vectorN_t const & std)
    {
        return vectorN_t::NullaryExpr(std.size(),
        [&std] (vectorN_t::Index const & i) -> float64_t
        {
            return randNormal(0, std[i]);
        });
    }

    void shuffleIndices(std::vector<uint32_t> & vector)
    {
        std::shuffle(vector.begin(), vector.end(), generator_);
    }

    //-----------------------------------------------------------------------------
    // MurmurHash3 was written by Austin Appleby, and is placed in the public
    // domain. The author hereby disclaims copyright to this source code:
    // https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp

    inline uint32_t rotl32(uint32_t x, int8_t r)
    {
        return (x << r) | (x >> (32 - r));
    }

    uint32_t MurmurHash3(void const * key,
                        int32_t const & len,
                        uint32_t const & seed)
    {
        // Define some internal constants
        uint32_t const c1 = 0xcc9e2d51;
        uint32_t const c2 = 0x1b873593;
        uint32_t const c3 = 0xe6546b64;

        // Initialize has to seed value
        uint32_t h1 = seed;

        // Extract bytes from key
        uint8_t const * data = (uint8_t const *) key;
        int32_t const nblocks = len / 4;  // len in bytes, so 32-bits blocks

        // Body
        uint32_t const * blocks = (uint32_t const *)(data + nblocks * 4);
        for(int32_t i = -nblocks; i; ++i)
        {
            uint32_t k1 = blocks[i];
            k1 *= c1;
            k1 = rotl32(k1, 15);
            k1 *= c2;
            h1 ^= k1;
            h1 = rotl32(h1, 13);
            h1 = h1 * 5 + c3;
        }

        // Tail
        uint8_t const * tail = (uint8_t const *)(data + nblocks * 4);
        uint32_t k1 = 0U;
        switch(len & 3)
        {
        case 3:
            k1 ^= tail[2] << 16;
            /* Falls through. */  // [[fallthrough]] is not supported by gcc<7.3
        case 2:
            k1 ^= tail[1] << 8;
            /* Falls through. */  // [[fallthrough]] is not supported by gcc<7.3
        case 1:
            k1 ^= tail[0];
            k1 *= c1;
            k1 = rotl32(k1,15);
            k1 *= c2;
            h1 ^= k1;
            /* Falls through. */ // [[fallthrough]] is not supported by gcc<7.3
        case 0:
        default:
            break;
        };

        // Finalization mix - force all bits of a hash block to avalanche
        h1 ^= len;
        h1 ^= h1 >> 16;
        h1 *= 0x85ebca6b;
        h1 ^= h1 >> 13;
        h1 *= 0xc2b2ae35;
        h1 ^= h1 >> 16;

        return h1;
    }

    //-----------------------------------------------------------------------------

    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief  Lower Cholesky factor of a Toeplitz positive semi_definite matrix.
    ///
    /// \details  In practice, it is advisable to combine this algorithm with Tikhonov
    ///           regularization of relative magnitude 1e-9 to avoid numerical instabilities
    ///           because of float64_t machine precision.
    ///
    /// \sa  Michael Stewart,
    ///      Cholesky factorization of semi-definite Toeplitz matrices.
    ///      Linear Algebra and its Applications,
    ///      Volume 254, pages 497-525, 1997.
    ///
    /// \sa  https://people.sc.fsu.edu/~jburkardt/cpp_src/toeplitz_cholesky/toeplitz_cholesky.html
    ///
    /// \param[in]   a  Toeplitz matrix to decompose.
    /// \param[out]  l  Lower Cholesky factor.
    ///
    ///////////////////////////////////////////////////////////////////////////////////////////////
    template<typename DerivedType1, typename DerivedType2>
    void toeplitzCholeskyLower(Eigen::MatrixBase<DerivedType1> const & a,
                            Eigen::MatrixBase<DerivedType2> & l)
    {
        /* Initialize lower Cholesky factor.
        Resizing is enough, no need to initialize it. */
        uint32_t const n = a.rows();
        l.resize(n, n);

        /* Compute compressed representation of the matrix, which
        coincide with the Schur genetor for Toepliz matrices. */
        matrixN_t g = matrixN_t(2, n);
        g.row(0) = a.row(0);
        g.row(1).tail(n - 1) = a.col(0).tail(n - 1);
        g(1, 0) = 0.0;

        // Run progressive Schur algorithm, adapted to Toepliz matrices
        l.col(0) = g.row(0);
        g.row(0).tail(n - 1) = g.row(0).head(n - 1).eval();
        g(0, 0) = 0.0;
        matrix2_t H;
        for (uint32_t i = 1; i < n ; ++i)
        {
            float64_t const rho = - g(1, i) / g(0, i);
            H << 1.0, rho,
                 rho, 1.0;
            g.rightCols(n - i + 1) = (H * g.rightCols(n - i + 1) / std::sqrt((1 - rho) * (1 + rho))).eval();
            l.col(i).tail(n - i + 1) = g.row(0).tail(n - i + 1);
            g.row(0).tail(n - i) = g.row(0).segment(i - 1, n - i).eval();
            g(0, i - 1) = 0.0;
        }
    }

    PeriodicGaussianProcess::PeriodicGaussianProcess(float64_t const & wavelength,
                                                     float64_t const & period) :
    wavelength_(wavelength),
    period_(period),
    dt_(0.02 * wavelength_),
    numTimes_(std::ceil(period_ / dt_)),
    isInitialized_(false),
    values_(numTimes_),
    covSqrtRoot_(numTimes_, numTimes_)
    {
        // Empty on purpose
    }

    void PeriodicGaussianProcess::reset(void)
    {
        // Initialize the process if not already done
        if (!isInitialized_)
        {
            initialize();
        }

        // Sample normal vector
        vectorN_t const normalVec = vectorN_t(numTimes_).unaryExpr(
            [](float64_t const &) { return randNormal(); });

        // Compute discrete periodic gaussian process values
        values_.noalias() = covSqrtRoot_.triangularView<Eigen::Lower>() * normalVec;
    }

    float64_t PeriodicGaussianProcess::operator()(float const & t)
    {
        // Reset the process if not initialized
        if (!isInitialized_)
        {
            reset();
        }

        // Wrap requested time in guassian process period
        float64_t tWrap = std::fmod(t, period_);
        if (tWrap < 0)
        {
            tWrap += period_;
        }

        // Compute closest left and right indices
        int32_t const tLeftIdx = std::floor(tWrap / dt_);
        int32_t const tRightIdx = (tLeftIdx + 1) % numTimes_;

        // Perform First order interpolation
        float64_t const ratio = tWrap / dt_ - tLeftIdx;
        return (1.0 - ratio) * values_[tLeftIdx] + ratio * values_[tRightIdx];
    }

    float64_t const & PeriodicGaussianProcess::getWavelength(void) const
    {
        return wavelength_;
    }

    float64_t const & PeriodicGaussianProcess::getPeriod(void) const
    {
        return period_;
    }

    float64_t const & PeriodicGaussianProcess::getDt(void) const
    {
        return dt_;
    }

    void PeriodicGaussianProcess::initialize(void)
    {
        // Compute distance matrix
        matrixN_t distMat(numTimes_, numTimes_);
        for (uint32_t i=0; i < numTimes_; ++i)
        {
            distMat.diagonal(i).setConstant(dt_ * i);
        }
        distMat.triangularView<Eigen::Lower>() = distMat.transpose();

        // Compute covariance matrix
        matrixN_t cov(numTimes_, numTimes_);
        cov = distMat.array().abs().unaryExpr(
            [period = period_, wavelength2 = std::pow(wavelength_, 2)](float64_t const & dist)
            {
                return std::exp(-2.0 * std::pow(std::sin(M_PI / period * dist), 2) / wavelength2);
            });

        /* Perform Square-Root-Free Cholesky decomposition (LDLT) .
        In practice, since the covariance matrix is symmetric, Eigen Value, Singular Value,
        Square-Root-Free Cholesky or Schur decompositions are equivalent, but Cholesky is
        by far the most efficient one:
        - https://math.stackexchange.com/q/22825/375496
        Moreover, note that the covariance is positive semi-definite toepliz
        matrix, so computational complexity of the decomposition could be reduced
        even further using adapted Cholesky algorithm since speed is very critical. */
        //auto covLDLT = cov.ldlt();
        //covSqrtRoot_ = covLDLT.matrixL();
        //covSqrtRoot_ *= covLDLT.vectorD().array().cwiseMax(0.0).sqrt().matrix().asDiagonal();
        toeplitzCholeskyLower(cov  + 1.0e-9 * matrixN_t::Identity(numTimes_, numTimes_),
                                covSqrtRoot_);

        // At this point, it is fully initialized
        isInitialized_ = true;
    }

    PeriodicFourierProcess::PeriodicFourierProcess(float64_t const & wavelength,
                                                   float64_t const & period) :
    wavelength_(wavelength),
    period_(period),
    dt_(0.02 * wavelength_),
    numTimes_(std::ceil(period_ / dt_)),
    numHarmonics_(std::ceil(period_ / wavelength_)),
    isInitialized_(false),
    values_(numTimes_),
    cosMat_(numTimes_, numHarmonics_),
    sinMat_(numTimes_, numHarmonics_)
    {
        // Empty on purpose
    }

    void PeriodicFourierProcess::reset(void)
    {
        // Initialize the process if not already done
        if (!isInitialized_)
        {
            initialize();
        }

        // Sample normal vectors
        vectorN_t normalVec1 = vectorN_t(numHarmonics_).unaryExpr(
            [](float64_t const &) { return randNormal(); });
        vectorN_t normalVec2 = vectorN_t(numHarmonics_).unaryExpr(
            [](float64_t const &) { return randNormal(); });

        // Compute discrete periodic gaussian process values
        values_ = M_SQRT2 / std::sqrt(2 * numHarmonics_ + 1) * (
            cosMat_ * normalVec1 + sinMat_ * normalVec2);
    }

    float64_t PeriodicFourierProcess::operator()(float const & t)
    {
        // Reset the process if not initialized
        if (!isInitialized_)
        {
            reset();
        }

        // Wrap requested time in guassian process period
        float64_t tWrap = std::fmod(t, period_);
        if (tWrap < 0)
        {
            tWrap += period_;
        }

        // Compute closest left and right indices
        int32_t const tLeftIdx = std::floor(tWrap / dt_);
        int32_t const tRightIdx = (tLeftIdx + 1) % numTimes_;

        // Perform First order interpolation
        float64_t ratio = tWrap / dt_ - tLeftIdx;
        return (1.0 - ratio) * values_[tLeftIdx] + ratio * values_[tRightIdx];
    }

    float64_t const & PeriodicFourierProcess::getWavelength(void) const
    {
        return wavelength_;
    }

    float64_t const & PeriodicFourierProcess::getPeriod(void) const
    {
        return period_;
    }

    uint32_t const & PeriodicFourierProcess::getNumHarmonics(void) const
    {
        return numHarmonics_;
    }

    float64_t const & PeriodicFourierProcess::getDt(void) const
    {
        return dt_;
    }

    void PeriodicFourierProcess::initialize(void)
    {
        // Compute exponential base at given time
        for (uint32_t colIdx = 0; colIdx < numHarmonics_; ++colIdx)
        {
            for (uint32_t rowIdx = 0; rowIdx < numTimes_; ++rowIdx)
            {
                float64_t const freq = colIdx / period_;
                float64_t const t = dt_ * rowIdx;
                float64_t const phase = 2 * M_PI * freq * t;
                cosMat_(rowIdx, colIdx) = std::cos(phase);
                sinMat_(rowIdx, colIdx) = std::sin(phase);
            }
        }

        // At this point, it is fully initialized
        isInitialized_ = true;
    }

    AbstractPerlinNoiseOctave::AbstractPerlinNoiseOctave(float64_t const & wavelength,
                                                         float64_t const & scale) :
    wavelength_(wavelength),
    scale_(scale),
    shift_(0.0)
    {
        // Empty on purpose
    }

    void AbstractPerlinNoiseOctave::reset(void)
    {
        // Sample random phase shift
        shift_ = randUniform();
    }

    float64_t AbstractPerlinNoiseOctave::operator()(float64_t const & t) const
    {
        // Get current phase
        float64_t const phase = t / wavelength_ + shift_;

        // Compute closest right and left knots
        int32_t const phaseIdxLeft = phase;
        int32_t const phaseIdxRight = phaseIdxLeft + 1;

        // Compute smoothed ratio of current phase wrt to the closest knots
        float64_t const dtLeft = phase - phaseIdxLeft;
        float64_t const dtRight = dtLeft - 1.0;
        float64_t const ratio = fade(dtLeft);

        /* Compute gradients at knots, and perform linear interpolation
           between them to get value at current phase.*/
        float64_t const yLeft = grad(phaseIdxLeft, dtLeft);
        float64_t const yRight = grad(phaseIdxRight, dtRight);
        return scale_ * lerp(ratio, yLeft, yRight);
    }

    float64_t const & AbstractPerlinNoiseOctave::getWavelength(void) const
    {
        return wavelength_;
    }

    float64_t const & AbstractPerlinNoiseOctave::getScale(void) const
    {
        return scale_;
    }

    float64_t AbstractPerlinNoiseOctave::fade(float64_t const & delta) const
    {
        /* Improved Smoothstep function by Ken Perlin (aka Smootherstep).
           It has zero 1st and 2nd-order derivatives at dt = 0.0, and 1.0:
           https://en.wikipedia.org/wiki/Smoothstep#Variations */
        return std::pow(delta, 3) * (delta * (delta * 6.0 - 15.0) + 10.0);
    }

    float64_t AbstractPerlinNoiseOctave::lerp(float64_t const & ratio,
                                              float64_t const & yLeft,
                                              float64_t const & yRight) const
    {
        return yLeft + ratio * (yRight - yLeft);
    }

    RandomPerlinNoiseOctave::RandomPerlinNoiseOctave(float64_t const & wavelength,
                                                     float64_t const & scale) :
    AbstractPerlinNoiseOctave(wavelength, scale),
    seed_(0)
    {
        // Empty on purpose
    }

    void RandomPerlinNoiseOctave::reset(void)
    {
        // Call base implementation
        AbstractPerlinNoiseOctave::reset();

        // Sample new random seed for MurmurHash
        seed_ = generator_();
    }

    float64_t RandomPerlinNoiseOctave::grad(int32_t knot,
                                            float64_t const & delta) const
    {
        // Get hash of knot
        uint32_t const hash = MurmurHash3(&knot, sizeof(int32_t), seed_);

        // Convert to double in [0.0, 1.0)
        float64_t const s = static_cast<float64_t>(hash) /
            static_cast<float64_t>(std::numeric_limits<uint32_t>::max());

        // Compute rescaled gradient between [-1.0, 1.0)
        float64_t const grad = 2.0 * s - 1.0;

        // Return scalar product between distance and gradient
        return 2.0 * grad * delta;
    }

    PeriodicPerlinNoiseOctave::PeriodicPerlinNoiseOctave(float64_t const & wavelength,
                                                         float64_t const & period,
                                                         float64_t const & scale) :
    AbstractPerlinNoiseOctave(wavelength, scale),
    period_(period),
    perm_(256)
    {
        // Make sure the wavelength is multiple of the period
        assert (std::abs(period_ - period) < 1e-6);
    }

    void PeriodicPerlinNoiseOctave::reset(void)
    {
        // Call base implementation
        AbstractPerlinNoiseOctave::reset();

        // Initialize the permutation vector with values from 0 to 255
        std::iota(perm_.begin(), perm_.end(), 0);

        // Suffle the permutation vector
        std::shuffle(perm_.begin(), perm_.end(), generator_);
    }

    float64_t PeriodicPerlinNoiseOctave::grad(int32_t knot,
                                              float64_t const & delta) const
    {
        // Wrap knot is period interval
        knot %= static_cast<uint32_t>(period_ / wavelength_);

        // Convert to double in [0.0, 1.0)
        float64_t const s = perm_[knot] / 256.0;

        // Compute rescaled gradient between [-1.0, 1.0)
        float64_t const grad = 2.0 * s - 1.0;

        // Return scalar product between distance and gradient
        return 2.0 * grad * delta;
    }

    AbstractPerlinProcess::AbstractPerlinProcess(float64_t const & wavelength,
                                                 uint32_t const & numOctaves) :
    wavelength_(wavelength),
    numOctaves_(numOctaves),
    isInitialized_(false),
    octaves_(),
    amplitude_(0.0)
    {
        // Empty on purpose
    }

    void AbstractPerlinProcess::reset(void)
    {
        // Initialize the process if not already done
        if (!isInitialized_)
        {
            initialize();
        }

        // Reset every octave successively
        for (auto & octave: octaves_)
        {
            octave->reset();
        }

        // Compute scaling factor to get values in range [-1.0, 1.0]
        float64_t amplitudeSquared = 0.0;
        for (auto const & octave: octaves_)
        {
            amplitudeSquared += std::pow(octave->getScale(), 2);
        }
        amplitude_ = std::sqrt(amplitudeSquared);
    }

    float64_t AbstractPerlinProcess::operator()(float const & t)
    {
        // Reset the process if not initialized
        if (!isInitialized_)
        {
            reset();
        }

        // Compute sum of octaves' values
        float64_t value = 0.0;
        for (auto const & octave: octaves_)
        {
            value += (*octave)(t);
        }

        // Return scaled value by total amplitude
        return value / amplitude_;
    }

    float64_t const & AbstractPerlinProcess::getWavelength(void) const
    {
        return wavelength_;
    }

    uint32_t const & AbstractPerlinProcess::getNumOctaves(void) const
    {
        return numOctaves_;
    }

    RandomPerlinProcess::RandomPerlinProcess(float64_t const & wavelength,
                                             uint32_t const & numOctaves) :
    AbstractPerlinProcess(wavelength, numOctaves)
    {
        // Empty on purpose
    }

    void RandomPerlinProcess::initialize(void)
    {
        // Add desired perlin noise octaves
        octaves_.reserve(numOctaves_);
        float64_t octaveWavelength = wavelength_;
        float64_t octaveScale = 1.0;
        for (uint32_t i = 0; i < numOctaves_; ++i)
        {
            octaves_.emplace_back(std::make_unique<RandomPerlinNoiseOctave>(
                octaveWavelength, octaveScale));
            octaveScale *= PERLIN_NOISE_PERSISTENCE;
            octaveWavelength *= PERLIN_NOISE_LACUNARITY;
        }

        // At this point, it is fully initialized
        isInitialized_ = true;
    }

    PeriodicPerlinProcess::PeriodicPerlinProcess(float64_t const & wavelength,
                                                 float64_t const & period,
                                                 uint32_t const & numOctaves) :
    AbstractPerlinProcess(wavelength, numOctaves),
    period_(period)
    {
        // Make sure the period is larger than the wavelength
        assert(period_ >= wavelength && "Period must be larger than wavelength.");
    }

    void PeriodicPerlinProcess::initialize(void)
    {
        // Add desired perlin noise octaves
        octaves_.reserve(numOctaves_);
        float64_t octaveWavelength = wavelength_;
        float64_t octaveScale = 1.0;
        for (uint32_t i = 0; i < numOctaves_; ++i)
        {
            // Make sure the octave wavelength is divisor of the period
            if (octaveWavelength > period_)
            {
                // Do not add more octoves if current wavelength is larger than the period
                break;
            }
            octaveWavelength = period_ / std::floor(period_ / octaveWavelength);

            // Instantiate and add octave
            octaves_.emplace_back(std::make_unique<PeriodicPerlinNoiseOctave>(
                octaveWavelength, period_, octaveScale));

            // Update scale and wavelength for next octave
            octaveScale *= PERLIN_NOISE_PERSISTENCE;
            octaveWavelength *= PERLIN_NOISE_LACUNARITY;
        }

        // At this point, it is fully initialized
        isInitialized_ = true;
    }

    float64_t const & PeriodicPerlinProcess::getPeriod(void) const
    {
        return period_;
    }
}