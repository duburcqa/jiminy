#include <numeric>

#include "jiminy/core/utilities/random.h"


namespace jiminy
{
    static const float64_t PERLIN_NOISE_PERSISTENCE = 1.50;
    static const float64_t PERLIN_NOISE_LACUNARITY = 1.15;

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

    void r4_nor_setup()
    {
        const float64_t m1 = 2147483648.0;
        const float64_t vn = 9.91256303526217e-03;
        float64_t dn = 3.442619855899;
        float64_t tn = dn;

        float64_t q = vn / exp(-0.5 * dn * dn);

        kn[0] = static_cast<uint32_t>((dn / q) * m1);
        kn[1] = 0;

        wn[0] = static_cast<float32_t>(q / m1);
        wn[127] = static_cast<float32_t>(dn / m1);

        fn[0] = 1.0f;
        fn[127] = static_cast<float32_t>(exp(-0.5 * dn * dn));

        for (uint8_t i = 126; 1 <= i; i--)
        {
            dn = sqrt(-2.0 * log(vn / dn + exp(-0.5 * dn * dn)));
            kn[i + 1] = static_cast<uint32_t>((dn / tn) * m1);
            tn = dn;
            fn[i] = static_cast<float32_t>(exp(-0.5 * dn * dn));
            wn[i] = static_cast<float32_t>(dn / m1);
        }
    }

    float32_t r4_uni()
    {
        return distUniform_(generator_);
    }

    float32_t r4_nor()
    {
        const float32_t r = 3.442620f;
        int32_t hz;
        uint32_t iz;
        float32_t x;
        float32_t y;

        hz = static_cast<int32_t>(generator_());
        iz = (static_cast<uint32_t>(hz) & 127U);

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
                        x = -0.2904764f * log(r4_uni());
                        y = -log(r4_uni());
                        if (x * x <= y + y)
                        {
                            break;
                        }
                    }

                    if (hz <= 0)
                    {
                        return -r - x;
                    }
                    else
                    {
                        return +r + x;
                    }
                }

                x = static_cast<float32_t>(hz) * wn[iz];

                if (fn[iz] + r4_uni() * (fn[iz - 1] - fn[iz]) < exp(-0.5f * x * x))
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

    void resetRandomGenerators(const std::optional<uint32_t> & seed)
    {
        uint32_t newSeed = seed.value_or(seed_);
        srand(newSeed);  // Eigen relies on srand for generating random numbers
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

    float64_t randUniform(const float64_t & lo, const float64_t & hi)
    {
        assert(isInitialized_ && "Random number genetors not initialized. "
                                 "Please call `resetRandomGenerators` at least once.");
        return lo + r4_uni() * (hi - lo);
    }

    float64_t randNormal(const float64_t & mean, const float64_t & std)
    {
        assert(isInitialized_ && "Random number genetors not initialized. "
                                 "Please call `resetRandomGenerators` at least once.");
        return mean + r4_nor() * std;
    }

    vectorN_t randVectorNormal(
        const uint32_t & size, const float64_t & mean, const float64_t & std)
    {
        if (std > 0.0)
        {
            return vectorN_t::NullaryExpr(size,
                                          [&mean, &std](const vectorN_t::Index &) -> float64_t
                                          { return randNormal(mean, std); });
        }
        else
        {
            return vectorN_t::Constant(size, mean);
        }
    }

    vectorN_t randVectorNormal(const uint32_t & size, const float64_t & std)
    {
        return randVectorNormal(size, 0.0, std);
    }

    vectorN_t randVectorNormal(const vectorN_t & mean, const vectorN_t & std)
    {
        return vectorN_t::NullaryExpr(std.size(),
                                      [&mean, &std](const vectorN_t::Index & i) -> float64_t
                                      { return randNormal(mean[i], std[i]); });
    }

    vectorN_t randVectorNormal(const vectorN_t & std)
    {
        return vectorN_t::NullaryExpr(std.size(),
                                      [&std](const vectorN_t::Index & i) -> float64_t
                                      { return randNormal(0, std[i]); });
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

    uint32_t MurmurHash3(const void * key, const int32_t & len, const uint32_t & seed)
    {
        // Define some internal constants
        const uint32_t c1 = 0xcc9e2d51;
        const uint32_t c2 = 0x1b873593;
        const uint32_t c3 = 0xe6546b64;

        // Initialize has to seed value
        uint32_t h1 = seed;

        // Extract bytes from key
        const uint8_t * data = reinterpret_cast<const uint8_t *>(key);
        const int32_t nblocks = len / 4;  // len in bytes, so 32-bits blocks

        // Body
        const uint32_t * blocks = reinterpret_cast<const uint32_t *>(data + nblocks * 4);
        for (int32_t i = -nblocks; i; ++i)
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
        const uint8_t * tail = reinterpret_cast<const uint8_t *>(data + nblocks * 4);
        uint32_t k1 = 0U;
        switch (len & 3)
        {
        case 3:
            k1 ^= tail[2] << 16;
            [[fallthrough]];
        case 2:
            k1 ^= tail[1] << 8;
            [[fallthrough]];
        case 1:
            k1 ^= tail[0];
            k1 *= c1;
            k1 = rotl32(k1, 15);
            k1 *= c2;
            h1 ^= k1;
            [[fallthrough]];
        case 0:
        default:
            break;
        };

        // Finalization mix - force all bits of a hash block to avalanche
        h1 ^= static_cast<uint32_t>(len);
        h1 ^= h1 >> 16;
        h1 *= 0x85ebca6b;
        h1 ^= h1 >> 13;
        h1 *= 0xc2b2ae35;
        h1 ^= h1 >> 16;

        return h1;
    }

    //-----------------------------------------------------------------------------

    /// \brief Lower Cholesky factor of a Toeplitz positive semi-definite matrix.
    ///
    /// \details In practice, it is advisable to combine this algorithm with Tikhonov
    ///          regularization of relative magnitude 1e-9 to avoid numerical instabilities because
    ///          of float64_t machine precision.
    ///
    /// \see Michael Stewart, Cholesky factorization of semi-definite Toeplitz matrices. Linear
    ///      Algebra and its Applications, Volume 254, pages 497-525, 1997.
    ///
    /// \see https://people.sc.fsu.edu/~jburkardt/cpp_src/toeplitz_cholesky/toeplitz_cholesky.html
    ///
    /// \param[in] a Toeplitz matrix to decompose.
    /// \param[out] l Lower Cholesky factor.
    template<typename DerivedType1, typename DerivedType2>
    void toeplitzCholeskyLower(const Eigen::MatrixBase<DerivedType1> & a,
                               Eigen::MatrixBase<DerivedType2> & l)
    {
        // Initialize lower Cholesky factor. Resizing is enough, no need to initialize it.
        const uint64_t n = a.rows();
        l.resize(n, n);

        /* Compute compressed representation of the matrix, which coincide with the Schur generator
           for Toepliz matrices. */
        matrixN_t g(2, n);
        g.row(0) = a.row(0);
        g.row(1).tail(n - 1) = a.col(0).tail(n - 1);
        g(1, 0) = 0.0;

        // Run progressive Schur algorithm, adapted to Toepliz matrices
        l.col(0) = g.row(0);
        g.row(0).tail(n - 1) = g.row(0).head(n - 1).eval();
        g(0, 0) = 0.0;
        matrix2_t H = matrix2_t::Ones();
        for (uint32_t i = 1; i < n; ++i)
        {
            const float64_t rho = -g(1, i) / g(0, i);
            // H << 1.0, rho,
            //      rho, 1.0;
            Eigen::Map<vector4_t>(H.data()).segment<2>(1).fill(rho);
            g.rightCols(n - i + 1) =
                (H * g.rightCols(n - i + 1) / std::sqrt(1.0 - rho * rho)).eval();
            l.col(i).tail(n - i + 1) = g.row(0).tail(n - i + 1);
            g.row(0).tail(n - i) = g.row(0).segment(i - 1, n - i).eval();
            g(0, i - 1) = 0.0;
        }
    }

    PeriodicGaussianProcess::PeriodicGaussianProcess(
        const float64_t & wavelength, const float64_t & period, const float64_t & scale) :
    wavelength_(wavelength),
    period_(period),
    scale_(scale),
    dt_(0.02 * wavelength_),
    numTimes_(static_cast<int32_t>(std::ceil(period_ / dt_))),
    isInitialized_(false),
    values_(numTimes_),
    covSqrtRoot_(numTimes_, numTimes_)
    {
    }

    void PeriodicGaussianProcess::reset()
    {
        // Initialize the process if not already done
        if (!isInitialized_)
        {
            initialize();
        }

        // Sample normal vector
        const vectorN_t normalVec =
            vectorN_t::NullaryExpr(numTimes_, [](const float64_t &) { return randNormal(); });

        // Compute discrete periodic gaussian process values
        values_.noalias() = covSqrtRoot_.triangularView<Eigen::Lower>() * normalVec;
    }

    float64_t PeriodicGaussianProcess::operator()(const float & t)
    {
        // Reset the process if not initialized
        if (!isInitialized_)
        {
            reset();
        }

        // Wrap requested time in gaussian process period
        float64_t tWrap = std::fmod(t, period_);
        if (tWrap < 0)
        {
            tWrap += period_;
        }

        // Compute closest left and right indices
        const int32_t tLeftIdx = static_cast<int32_t>(std::floor(tWrap / dt_));
        const int32_t tRightIdx = (tLeftIdx + 1) % numTimes_;

        // Perform First order interpolation
        const float64_t ratio = tWrap / dt_ - tLeftIdx;
        return scale_ * (values_[tLeftIdx] + ratio * (values_[tRightIdx] - values_[tLeftIdx]));
    }

    const float64_t & PeriodicGaussianProcess::getWavelength() const
    {
        return wavelength_;
    }

    const float64_t & PeriodicGaussianProcess::getPeriod() const
    {
        return period_;
    }

    const float64_t & PeriodicGaussianProcess::getDt() const
    {
        return dt_;
    }

    void PeriodicGaussianProcess::initialize()
    {
        // Compute distance matrix
        matrixN_t distMat(numTimes_, numTimes_);
        for (int32_t i = 0; i < numTimes_; ++i)
        {
            distMat.diagonal(i).setConstant(dt_ * i);
        }
        distMat.triangularView<Eigen::StrictlyLower>() = distMat.transpose();

        // Compute covariance matrix
        matrixN_t cov(numTimes_, numTimes_);
        cov = distMat.array().abs().unaryExpr(
            [period = period_, wavelength = wavelength_](const float64_t & dist)
            { return std::exp(-2.0 * std::pow(std::sin(M_PI / period * dist) / wavelength, 2)); });

        /* Perform Square-Root-Free Cholesky decomposition (LDLT).
           All decompositions are equivalent as the covariance matrix is symmetric, namely Eigen
           Value, Singular Value, Square-Root-Free Cholesky and Schur decompositions. Cholesky
           is by far the most efficient one (https://math.stackexchange.com/q/22825/375496).
           Moreover, the covariance is positive semi-definite toepliz matrix, so computational
           complexity can be reduced even further using optimized Cholesky algorithm. */
        toeplitzCholeskyLower(cov + 1.0e-9 * matrixN_t::Identity(numTimes_, numTimes_),
                              covSqrtRoot_);

        // At this point, it is fully initialized
        isInitialized_ = true;
    }

    PeriodicFourierProcess::PeriodicFourierProcess(
        const float64_t & wavelength, const float64_t & period, const float64_t & scale) :
    wavelength_(wavelength),
    period_(period),
    scale_(scale),
    dt_(0.02 * wavelength_),
    numTimes_(static_cast<int32_t>(std::ceil(period_ / dt_))),
    numHarmonics_(static_cast<int32_t>(std::ceil(period_ / wavelength_))),
    isInitialized_(false),
    values_(numTimes_),
    cosMat_(numTimes_, numHarmonics_),
    sinMat_(numTimes_, numHarmonics_)
    {
    }

    void PeriodicFourierProcess::reset()
    {
        // Initialize the process if not already done
        if (!isInitialized_)
        {
            initialize();
        }

        // Sample normal vectors
        vectorN_t normalVec1 =
            vectorN_t::NullaryExpr(numHarmonics_, [](const float64_t &) { return randNormal(); });
        vectorN_t normalVec2 =
            vectorN_t::NullaryExpr(numHarmonics_, [](const float64_t &) { return randNormal(); });

        // Compute discrete periodic gaussian process values
        values_ = M_SQRT2 / std::sqrt(2 * numHarmonics_ + 1) *
                  (cosMat_ * normalVec1 + sinMat_ * normalVec2);
    }

    float64_t PeriodicFourierProcess::operator()(const float & t)
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
        const int32_t tLeftIdx = static_cast<int32_t>(std::floor(tWrap / dt_));
        const int32_t tRightIdx = (tLeftIdx + 1) % numTimes_;

        // Perform First order interpolation
        const float64_t ratio = tWrap / dt_ - tLeftIdx;
        return scale_ * (values_[tLeftIdx] + ratio * (values_[tRightIdx] - values_[tLeftIdx]));
    }

    const float64_t & PeriodicFourierProcess::getWavelength() const
    {
        return wavelength_;
    }

    const float64_t & PeriodicFourierProcess::getPeriod() const
    {
        return period_;
    }

    const int32_t & PeriodicFourierProcess::getNumHarmonics() const
    {
        return numHarmonics_;
    }

    const float64_t & PeriodicFourierProcess::getDt() const
    {
        return dt_;
    }

    void PeriodicFourierProcess::initialize()
    {
        // Compute exponential base at given time
        for (int32_t colIdx = 0; colIdx < numHarmonics_; ++colIdx)
        {
            for (int32_t rowIdx = 0; rowIdx < numTimes_; ++rowIdx)
            {
                const float64_t freq = colIdx / period_;
                const float64_t t = dt_ * rowIdx;
                const float64_t phase = 2 * M_PI * freq * t;
                cosMat_(rowIdx, colIdx) = std::cos(phase);
                sinMat_(rowIdx, colIdx) = std::sin(phase);
            }
        }

        // At this point, it is fully initialized
        isInitialized_ = true;
    }

    AbstractPerlinNoiseOctave::AbstractPerlinNoiseOctave(const float64_t & wavelength,
                                                         const float64_t & scale) :
    wavelength_(wavelength),
    scale_(scale),
    shift_(0.0)
    {
    }

    void AbstractPerlinNoiseOctave::reset()
    {
        // Sample random phase shift
        shift_ = randUniform();
    }

    float64_t AbstractPerlinNoiseOctave::operator()(const float64_t & t) const
    {
        // Get current phase
        const float64_t phase = t / wavelength_ + shift_;

        // Compute closest right and left knots
        const int32_t phaseIdxLeft = static_cast<int32_t>(phase);
        const int32_t phaseIdxRight = phaseIdxLeft + 1;

        // Compute smoothed ratio of current phase wrt to the closest knots
        const float64_t dtLeft = phase - phaseIdxLeft;
        const float64_t dtRight = dtLeft - 1.0;
        const float64_t ratio = fade(dtLeft);

        /* Compute gradients at knots, and perform linear interpolation between them to get value
           at current phase.*/
        const float64_t yLeft = grad(phaseIdxLeft, dtLeft);
        const float64_t yRight = grad(phaseIdxRight, dtRight);
        return scale_ * lerp(ratio, yLeft, yRight);
    }

    const float64_t & AbstractPerlinNoiseOctave::getWavelength() const
    {
        return wavelength_;
    }

    const float64_t & AbstractPerlinNoiseOctave::getScale() const
    {
        return scale_;
    }

    float64_t AbstractPerlinNoiseOctave::fade(const float64_t & delta) const
    {
        /* Improved Smoothstep function by Ken Perlin (aka Smootherstep).
           It has zero 1st and 2nd-order derivatives at dt = 0.0, and 1.0:
           https://en.wikipedia.org/wiki/Smoothstep#Variations */
        return std::pow(delta, 3) * (delta * (delta * 6.0 - 15.0) + 10.0);
    }

    float64_t AbstractPerlinNoiseOctave::lerp(
        const float64_t & ratio, const float64_t & yLeft, const float64_t & yRight) const
    {
        return yLeft + ratio * (yRight - yLeft);
    }

    RandomPerlinNoiseOctave::RandomPerlinNoiseOctave(const float64_t & wavelength,
                                                     const float64_t & scale) :
    AbstractPerlinNoiseOctave(wavelength, scale),
    seed_(0)
    {
    }

    void RandomPerlinNoiseOctave::reset()
    {
        // Call base implementation
        AbstractPerlinNoiseOctave::reset();

        // Sample new random seed for MurmurHash
        seed_ = static_cast<uint32_t>(generator_());
    }

    float64_t RandomPerlinNoiseOctave::grad(int32_t knot, const float64_t & delta) const
    {
        // Get hash of knot
        const uint32_t hash = MurmurHash3(&knot, sizeof(int32_t), seed_);

        // Convert to double in [0.0, 1.0)
        const float64_t s = static_cast<float64_t>(hash) /
                            static_cast<float64_t>(std::numeric_limits<uint32_t>::max());

        // Compute rescaled gradient between [-1.0, 1.0)
        const float64_t grad = 2.0 * s - 1.0;

        // Return scalar product between distance and gradient
        return 2.0 * grad * delta;
    }

    PeriodicPerlinNoiseOctave::PeriodicPerlinNoiseOctave(
        const float64_t & wavelength, const float64_t & period, const float64_t & scale) :
    AbstractPerlinNoiseOctave(wavelength, scale),
    period_(period),
    perm_(256)
    {
        // Make sure the wavelength is multiple of the period
        assert(std::abs(period_ - period) < 1e-6);
    }

    void PeriodicPerlinNoiseOctave::reset()
    {
        // Call base implementation
        AbstractPerlinNoiseOctave::reset();

        // Initialize the permutation vector with values from 0 to 255
        std::iota(perm_.begin(), perm_.end(), 0);

        // Shuffle the permutation vector
        std::shuffle(perm_.begin(), perm_.end(), generator_);
    }

    float64_t PeriodicPerlinNoiseOctave::grad(int32_t knot, const float64_t & delta) const
    {
        // Wrap knot is period interval
        knot %= static_cast<uint32_t>(period_ / wavelength_);

        // Convert to double in [0.0, 1.0)
        const float64_t s = perm_[knot] / 256.0;

        // Compute rescaled gradient between [-1.0, 1.0)
        const float64_t grad = 2.0 * s - 1.0;

        // Return scalar product between distance and gradient
        return 2.0 * grad * delta;
    }

    AbstractPerlinProcess::AbstractPerlinProcess(
        const float64_t & wavelength, const float64_t & scale, const uint32_t & numOctaves) :
    wavelength_(wavelength),
    numOctaves_(numOctaves),
    scale_(scale),
    isInitialized_(false),
    octaves_(),
    amplitude_(0.0)
    {
    }

    void AbstractPerlinProcess::reset()
    {
        // Initialize the process if not already done
        if (!isInitialized_)
        {
            initialize();
        }

        // Reset every octave successively
        for (auto & octave : octaves_)
        {
            octave->reset();
        }

        // Compute scaling factor to get values in range [-1.0, 1.0]
        float64_t amplitudeSquared = 0.0;
        for (const auto & octave : octaves_)
        {
            amplitudeSquared += std::pow(octave->getScale(), 2);
        }
        amplitude_ = scale_ * std::sqrt(amplitudeSquared);
    }

    float64_t AbstractPerlinProcess::operator()(const float & t)
    {
        // Reset the process if not initialized
        if (!isInitialized_)
        {
            reset();
        }

        // Compute sum of octaves' values
        float64_t value = 0.0;
        for (const auto & octave : octaves_)
        {
            value += (*octave)(t);
        }

        // Return scaled value by total amplitude
        return value / amplitude_;
    }

    const float64_t & AbstractPerlinProcess::getWavelength() const
    {
        return wavelength_;
    }

    const uint32_t & AbstractPerlinProcess::getNumOctaves() const
    {
        return numOctaves_;
    }

    const float64_t & AbstractPerlinProcess::getScale() const
    {
        return scale_;
    }

    RandomPerlinProcess::RandomPerlinProcess(
        const float64_t & wavelength, const float64_t & scale, const uint32_t & numOctaves) :
    AbstractPerlinProcess(wavelength, scale, numOctaves)
    {
    }

    void RandomPerlinProcess::initialize()
    {
        // Add desired perlin noise octaves
        octaves_.clear();
        octaves_.reserve(numOctaves_);
        float64_t octaveWavelength = wavelength_;
        float64_t octaveScale = 1.0;
        for (uint32_t i = 0; i < numOctaves_; ++i)
        {
            octaves_.emplace_back(
                std::make_unique<RandomPerlinNoiseOctave>(octaveWavelength, octaveScale));
            octaveScale *= PERLIN_NOISE_PERSISTENCE;
            octaveWavelength *= PERLIN_NOISE_LACUNARITY;
        }

        // At this point, it is fully initialized
        isInitialized_ = true;
    }

    PeriodicPerlinProcess::PeriodicPerlinProcess(const float64_t & wavelength,
                                                 const float64_t & period,
                                                 const float64_t & scale,
                                                 const uint32_t & numOctaves) :
    AbstractPerlinProcess(wavelength, scale, numOctaves),
    period_(period)
    {
        // Make sure the period is larger than the wavelength
        assert(period_ >= wavelength && "Period must be larger than wavelength.");
    }

    void PeriodicPerlinProcess::initialize()
    {
        // Add desired perlin noise octaves
        octaves_.clear();
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

    const float64_t & PeriodicPerlinProcess::getPeriod() const
    {
        return period_;
    }

    template<typename VectorLike>
    std::enable_if_t<is_eigen_vector_v<VectorLike>, float64_t>
    randomDouble(const Eigen::MatrixBase<VectorLike> & key,
                 const int64_t & sparsity,
                 const float64_t & scale,
                 const uint32_t & seed)
    {
        const int32_t keyLen = static_cast<int32_t>(sizeof(typename VectorLike::Scalar)) *
                               static_cast<int32_t>(key.size());
        const uint32_t hash = MurmurHash3(key.derived().data(), keyLen, seed);
        if (hash % sparsity == 0)
        {
            float64_t encoding(hash);
            encoding /= std::numeric_limits<uint32_t>::max();
            return scale * encoding;
        }
        return 0.0;
    }

    std::pair<float64_t, float64_t> tile2dInterp1d(Eigen::Matrix<int32_t, 2, 1> & posIdx,
                                                   const vector2_t & posRel,
                                                   const uint32_t & dim,
                                                   const vector2_t & size,
                                                   const int64_t & sparsity,
                                                   const float64_t & heightMax,
                                                   const vector2_t & interpThreshold,
                                                   const uint32_t & seed)
    {
        const float64_t z = randomDouble(posIdx, sparsity, heightMax, seed);
        float64_t height, dheight;
        if (posRel[dim] < interpThreshold[dim])
        {
            posIdx[dim] -= 1;
            const float64_t z_m = randomDouble(posIdx, sparsity, heightMax, seed);
            posIdx[dim] += 1;

            const float64_t ratio = (1.0 - posRel[dim] / interpThreshold[dim]) / 2.0;
            height = z + (z_m - z) * ratio;
            dheight = (z - z_m) / (2.0 * size[dim] * interpThreshold[dim]);
        }
        else if (1.0 - posRel[dim] < interpThreshold[dim])
        {
            posIdx[dim] += 1;
            const float64_t z_p = randomDouble(posIdx, sparsity, heightMax, seed);
            posIdx[dim] -= 1;

            const float64_t ratio = (1.0 + (posRel[dim] - 1.0) / interpThreshold[dim]) / 2.0;
            height = z + (z_p - z) * ratio;
            dheight = (z_p - z) / (2.0 * size[dim] * interpThreshold[dim]);
        }
        else
        {
            height = z;
            dheight = 0.0;
        }

        return {height, dheight};
    }

    heightmapFunctor_t randomTileGround(const vector2_t & size,
                                        const float64_t & heightMax,
                                        const vector2_t & interpDelta,
                                        const uint32_t & sparsity,
                                        const float64_t & orientation,
                                        const uint32_t & seed)
    {
        if ((0.01 <= interpDelta.array()).all() &&
            (interpDelta.array() <= size.array() / 2.0).all())
        {
            PRINT_WARNING("'interpDelta' must be in range [0.01, 'size'/2.0].");
        }

        vector2_t interpThreshold = interpDelta.cwiseMax(0.01).cwiseMin(size / 2.0);
        interpThreshold.array() /= size.array();

        const vector2_t offset = vector2_t::NullaryExpr(
            [&size, &seed](const vectorN_t::Index & i) -> float64_t
            {
                Eigen::Matrix<vectorN_t::Index, 1, 1> key;
                key[0] = i;
                return randomDouble(key, 1, size[i], seed);
            });

        Eigen::Rotation2D<float64_t> rotationMat(orientation);

        return
            [size, heightMax, interpDelta, rotationMat, sparsity, interpThreshold, offset, seed](
                const vector3_t & pos3) -> std::pair<float64_t, vector3_t>
        {
            // Compute the tile index and relative coordinate
            vector2_t pos = rotationMat * (pos3.head<2>() + offset);
            vector2_t posRel = pos.array() / size.array();
            Eigen::Matrix<int32_t, 2, 1> posIdx = posRel.array().floor().cast<int32_t>();
            posRel -= posIdx.cast<float64_t>();

            // Interpolate height based on nearby tiles if necessary
            float64_t height;
            vector3_t normal;
            Eigen::Matrix<bool_t, 2, 1> isEdge = (posRel.array() < interpThreshold.array()) ||
                                                 (1.0 - posRel.array() < interpThreshold.array());
            if (isEdge[0] && !isEdge[1])
            {
                float64_t dheight_x;
                std::tie(height, dheight_x) = tile2dInterp1d(
                    posIdx, posRel, 0, size, sparsity, heightMax, interpThreshold, seed);
                const float64_t normInv = std::sqrt(dheight_x * dheight_x + 1.0);
                normal << -dheight_x * normInv, 0.0, normInv;
            }
            else if (!isEdge[0] && isEdge[1])
            {
                float64_t dheight_y;
                std::tie(height, dheight_y) = tile2dInterp1d(
                    posIdx, posRel, 1, size, sparsity, heightMax, interpThreshold, seed);
                const float64_t normInv = std::sqrt(dheight_y * dheight_y + 1.0);
                normal << 0.0, -dheight_y * normInv, normInv;
            }
            else if (isEdge[0] && isEdge[1])
            {
                const auto [height_0, dheight_x_0] = tile2dInterp1d(
                    posIdx, posRel, 0, size, sparsity, heightMax, interpThreshold, seed);
                if (posRel[1] < interpThreshold[1])
                {
                    posIdx[1] -= 1;
                    const auto [height_m, dheight_x_m] = tile2dInterp1d(
                        posIdx, posRel, 0, size, sparsity, heightMax, interpThreshold, seed);

                    const float64_t ratio = (1.0 - posRel[1] / interpThreshold[1]) / 2.0;
                    height = height_0 + (height_m - height_0) * ratio;
                    const float64_t dheight_x = dheight_x_0 + (dheight_x_m - dheight_x_0) * ratio;
                    const float64_t dheight_y =
                        (height_0 - height_m) / (2.0 * size[1] * interpThreshold[1]);
                    normal << -dheight_x, -dheight_y, 1.0;
                    normal.normalize();
                }
                else
                {
                    posIdx[1] += 1;
                    const auto [height_p, dheight_x_p] = tile2dInterp1d(
                        posIdx, posRel, 0, size, sparsity, heightMax, interpThreshold, seed);

                    const float64_t ratio = (1.0 + (posRel[1] - 1.0) / interpThreshold[1]) / 2.0;
                    height = height_0 + (height_p - height_0) * ratio;
                    const float64_t dheight_x = dheight_x_0 + (dheight_x_p - dheight_x_0) * ratio;
                    const float64_t dheight_y =
                        (height_p - height_0) / (2.0 * size[1] * interpThreshold[1]);
                    normal << -dheight_x, -dheight_y, 1.0;
                    normal.normalize();
                }
            }
            else
            {
                height = randomDouble(posIdx, sparsity, heightMax, seed);
                normal = vector3_t::UnitZ();
            }

            return std::make_pair(height, std::move(normal));
        };
    }

    heightmapFunctor_t sumHeightmap(const std::vector<heightmapFunctor_t> & heightmaps)
    {
        if (heightmaps.size() == 1)
        {
            return heightmaps[0];
        }
        return [heightmaps](const vector3_t & pos3) -> std::pair<float64_t, vector3_t>
        {
            float64_t height = 0.0;
            vector3_t normal = vector3_t::Zero();
            for (const heightmapFunctor_t & heightmap : heightmaps)
            {
                const auto [height_i, normal_i] = heightmap(pos3);
                height += height_i;
                normal += normal_i;
            }
            normal.normalize();
            return std::make_pair(height, std::move(normal));
        };
    }

    heightmapFunctor_t mergeHeightmap(const std::vector<heightmapFunctor_t> & heightmaps)
    {
        if (heightmaps.size() == 1)
        {
            return heightmaps[0];
        }
        return [heightmaps](const vector3_t & pos3) -> std::pair<float64_t, vector3_t>
        {
            float64_t heightmax = -INF;
            vector3_t normal;  // It will be initialized to `normal_i`
            bool_t isDirty = false;
            for (const heightmapFunctor_t & heightmap : heightmaps)
            {
                const auto [height, normal_i] = heightmap(pos3);
                if (std::abs(height - heightmax) < EPS)
                {
                    normal += normal_i;
                    isDirty = true;
                }
                else if (height > heightmax)
                {
                    heightmax = height;
                    normal = normal_i;
                    isDirty = false;
                }
            }
            if (isDirty)
            {
                normal.normalize();
            }
            return std::make_pair(heightmax, std::move(normal));
        };
    }

    matrixN_t discretizeHeightmap(const heightmapFunctor_t & heightmap,
                                  const float64_t & gridSize,
                                  const float64_t & gridUnit)
    {
        // Allocate empty discrete grid
        uint32_t gridDim = static_cast<int32_t>(std::ceil(gridSize / gridUnit)) + 1U;
        matrixN_t heightGrid(gridDim * gridDim, 6);

        // Fill x and y discrete grid coordinates
        const vectorN_t values =
            (vectorN_t::LinSpaced(gridDim, 0, gridDim - 1) * gridUnit).array() -
            (gridDim - 1) * (gridUnit / 2.0);
        Eigen::Map<matrixN_t>(heightGrid.col(0).data(), gridDim, gridDim).colwise() = values;
        Eigen::Map<matrixN_t>(heightGrid.col(1).data(), gridDim, gridDim).rowwise() =
            values.transpose();

        // Fill discrete grid
        for (uint32_t i = 0; i < heightGrid.rows(); ++i)
        {
            auto result = heightmap(heightGrid.block<1, 3>(i, 0));
            heightGrid(i, 2) = std::get<float64_t>(result);
            heightGrid.block<1, 3>(i, 3) = std::get<vector3_t>(result);
        }

        return heightGrid;
    }
}
