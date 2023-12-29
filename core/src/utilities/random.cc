#include <numeric>

#include "jiminy/core/utilities/random.h"


namespace jiminy
{
    static inline constexpr double PERLIN_NOISE_PERSISTENCE{1.50};
    static inline constexpr double PERLIN_NOISE_LACUNARITY{1.15};

    // ***************** Random number generator_ *****************

    // Based on Ziggurat generator by Marsaglia and Tsang (JSS, 2000):
    // https://people.sc.fsu.edu/~jburkardt/cpp_src/ziggurat/ziggurat.html

    std::mt19937 generator_{};
    std::uniform_real_distribution<float> distUniform_(0.0, 1.0);
    bool isInitialized_{false};
    uint32_t seed_{0U};

    uint32_t kn[128];
    float fn[128];
    float wn[128];

    void r4_nor_setup() noexcept
    {
        const double m1 = 2147483648.0;
        const double vn = 9.91256303526217e-03;
        double dn = 3.442619855899;
        double tn = dn;

        double q = vn / exp(-0.5 * dn * dn);

        kn[0] = static_cast<uint32_t>((dn / q) * m1);
        kn[1] = 0;

        wn[0] = static_cast<float>(q / m1);
        wn[127] = static_cast<float>(dn / m1);

        fn[0] = 1.0f;
        fn[127] = static_cast<float>(exp(-0.5 * dn * dn));

        for (uint8_t i = 126; 1 <= i; i--)
        {
            dn = sqrt(-2.0 * log(vn / dn + exp(-0.5 * dn * dn)));
            kn[i + 1] = static_cast<uint32_t>((dn / tn) * m1);
            tn = dn;
            fn[i] = static_cast<float>(exp(-0.5 * dn * dn));
            wn[i] = static_cast<float>(dn / m1);
        }
    }

    float r4_uni()
    {
        return distUniform_(generator_);
    }

    float r4_nor()
    {
        const float r = 3.442620f;
        int32_t hz;
        uint32_t iz;
        float x;
        float y;

        hz = static_cast<int32_t>(generator_());
        iz = (static_cast<uint32_t>(hz) & 127U);

        if (fabs(hz) < kn[iz])
        {
            return static_cast<float>(hz) * wn[iz];
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

                x = static_cast<float>(hz) * wn[iz];

                if (fn[iz] + r4_uni() * (fn[iz - 1] - fn[iz]) < exp(-0.5f * x * x))
                {
                    return x;
                }

                hz = static_cast<int32_t>(generator_());
                iz = (hz & 127);

                if (fabs(hz) < kn[iz])
                {
                    return static_cast<float>(hz) * wn[iz];
                }
            }
        }
    }

    void resetRandomGenerators(const std::optional<uint32_t> & seed) noexcept
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

    double randUniform(double lo, double hi)
    {
        assert(isInitialized_ && "Random number genetors not initialized. "
                                 "Please call `resetRandomGenerators` at least once.");
        return lo + r4_uni() * (hi - lo);
    }

    double randNormal(double mean, double std)
    {
        assert(isInitialized_ && "Random number genetors not initialized. "
                                 "Please call `resetRandomGenerators` at least once.");
        return mean + r4_nor() * std;
    }

    Eigen::VectorXd randVectorNormal(uint32_t size, double mean, double std)
    {
        if (std > 0.0)
        {
            return Eigen::VectorXd::NullaryExpr(
                size, [mean, std](Eigen::Index) -> double { return randNormal(mean, std); });
        }
        return Eigen::VectorXd::Constant(size, mean);
    }

    Eigen::VectorXd randVectorNormal(uint32_t size, double std)
    {
        return randVectorNormal(size, 0.0, std);
    }

    Eigen::VectorXd randVectorNormal(const Eigen::VectorXd & mean, const Eigen::VectorXd & std)
    {
        return Eigen::VectorXd::NullaryExpr(std.size(),
                                            [&mean, &std](Eigen::Index i) -> double
                                            { return randNormal(mean[i], std[i]); });
    }

    Eigen::VectorXd randVectorNormal(const Eigen::VectorXd & std)
    {
        return Eigen::VectorXd::NullaryExpr(
            std.size(), [&std](Eigen::Index i) -> double { return randNormal(0, std[i]); });
    }

    void shuffleIndices(std::vector<uint32_t> & vector)
    {
        std::shuffle(vector.begin(), vector.end(), generator_);
    }

    //-----------------------------------------------------------------------------
    // MurmurHash3 was written by Austin Appleby, and is placed in the public
    // domain. The author hereby disclaims copyright to this source code:
    // https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp

    inline uint32_t rotl32(uint32_t x, int8_t r) noexcept
    {
        return (x << r) | (x >> (32 - r));
    }

    uint32_t MurmurHash3(const void * key, int32_t len, uint32_t seed) noexcept
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

    PeriodicGaussianProcess::PeriodicGaussianProcess(
        double wavelength, double period, double scale) noexcept :
    wavelength_{wavelength},
    period_{period},
    scale_{scale}
    {
    }

    void PeriodicGaussianProcess::reset()
    {
        // Sample normal vector
        const Eigen::VectorXd normalVec =
            Eigen::VectorXd::NullaryExpr(numTimes_, [](Eigen::Index) { return randNormal(); });

        // Compute discrete periodic gaussian process values
        values_.noalias() = covSqrtRoot_.triangularView<Eigen::Lower>() * normalVec;
    }

    double PeriodicGaussianProcess::operator()(const float & t)
    {
        // Wrap requested time in gaussian process period
        double tWrap = std::fmod(t, period_);
        if (tWrap < 0)
        {
            tWrap += period_;
        }

        // Compute closest left and right indices
        const int32_t tLeftIdx = static_cast<int32_t>(std::floor(tWrap / dt_));
        const int32_t tRightIdx = (tLeftIdx + 1) % numTimes_;

        // Perform First order interpolation
        const double ratio = tWrap / dt_ - tLeftIdx;
        return scale_ * (values_[tLeftIdx] + ratio * (values_[tRightIdx] - values_[tLeftIdx]));
    }

    double PeriodicGaussianProcess::getWavelength() const
    {
        return wavelength_;
    }

    double PeriodicGaussianProcess::getPeriod() const
    {
        return period_;
    }

    double PeriodicGaussianProcess::getDt() const
    {
        return dt_;
    }

    PeriodicFourierProcess::PeriodicFourierProcess(
        double wavelength, double period, double scale) noexcept :
    wavelength_{wavelength},
    period_{period},
    scale_{scale}
    {
    }

    void PeriodicFourierProcess::reset()
    {
        // Sample normal vectors
        Eigen::VectorXd normalVec1 =
            Eigen::VectorXd::NullaryExpr(numHarmonics_, [](Eigen::Index) { return randNormal(); });
        Eigen::VectorXd normalVec2 =
            Eigen::VectorXd::NullaryExpr(numHarmonics_, [](Eigen::Index) { return randNormal(); });

        // Compute discrete periodic gaussian process values
        values_ = M_SQRT2 / std::sqrt(2 * numHarmonics_ + 1) *
                  (cosMat_ * normalVec1 + sinMat_ * normalVec2);
    }

    double PeriodicFourierProcess::operator()(const float & t)
    {
        // Wrap requested time in guassian process period
        double tWrap = std::fmod(t, period_);
        if (tWrap < 0)
        {
            tWrap += period_;
        }

        // Compute closest left and right indices
        const int32_t tLeftIdx = static_cast<int32_t>(std::floor(tWrap / dt_));
        const int32_t tRightIdx = (tLeftIdx + 1) % numTimes_;

        // Perform First order interpolation
        const double ratio = tWrap / dt_ - tLeftIdx;
        return scale_ * (values_[tLeftIdx] + ratio * (values_[tRightIdx] - values_[tLeftIdx]));
    }

    double PeriodicFourierProcess::getWavelength() const
    {
        return wavelength_;
    }

    double PeriodicFourierProcess::getPeriod() const
    {
        return period_;
    }

    int32_t PeriodicFourierProcess::getNumHarmonics() const
    {
        return numHarmonics_;
    }

    double PeriodicFourierProcess::getDt() const
    {
        return dt_;
    }

    AbstractPerlinNoiseOctave::AbstractPerlinNoiseOctave(double wavelength, double scale) noexcept
    :
    wavelength_{wavelength},
    scale_{scale}
    {
    }

    void AbstractPerlinNoiseOctave::reset()
    {
        // Sample random phase shift
        shift_ = randUniform();
    }

    double AbstractPerlinNoiseOctave::operator()(double t) const
    {
        // Get current phase
        const double phase = t / wavelength_ + shift_;

        // Compute closest right and left knots
        const int32_t phaseIdxLeft = static_cast<int32_t>(phase);
        const int32_t phaseIdxRight = phaseIdxLeft + 1;

        // Compute smoothed ratio of current phase wrt to the closest knots
        const double dtLeft = phase - phaseIdxLeft;
        const double dtRight = dtLeft - 1.0;
        const double ratio = fade(dtLeft);

        /* Compute gradients at knots, and perform linear interpolation between them to get value
           at current phase.*/
        const double yLeft = grad(phaseIdxLeft, dtLeft);
        const double yRight = grad(phaseIdxRight, dtRight);
        return scale_ * lerp(ratio, yLeft, yRight);
    }

    double AbstractPerlinNoiseOctave::getWavelength() const
    {
        return wavelength_;
    }

    double AbstractPerlinNoiseOctave::getScale() const
    {
        return scale_;
    }

    double AbstractPerlinNoiseOctave::fade(double delta) const
    {
        /* Improved Smoothstep function by Ken Perlin (aka Smootherstep).
           It has zero 1st and 2nd-order derivatives at dt = 0.0, and 1.0:
           https://en.wikipedia.org/wiki/Smoothstep#Variations */
        return std::pow(delta, 3) * (delta * (delta * 6.0 - 15.0) + 10.0);
    }

    double AbstractPerlinNoiseOctave::lerp(double ratio, double yLeft, double yRight) const
    {
        return yLeft + ratio * (yRight - yLeft);
    }

    void RandomPerlinNoiseOctave::reset()
    {
        // Call base implementation
        AbstractPerlinNoiseOctave::reset();

        // Sample new random seed for MurmurHash
        seed_ = static_cast<uint32_t>(generator_());
    }

    double RandomPerlinNoiseOctave::grad(int32_t knot, double delta) const
    {
        // Get hash of knot
        const uint32_t hash = MurmurHash3(&knot, sizeof(int32_t), seed_);

        // Convert to double in [0.0, 1.0)
        const double s =
            static_cast<double>(hash) / static_cast<double>(std::numeric_limits<uint32_t>::max());

        // Compute rescaled gradient between [-1.0, 1.0)
        const double grad = 2.0 * s - 1.0;

        // Return scalar product between distance and gradient
        return 2.0 * grad * delta;
    }

    PeriodicPerlinNoiseOctave::PeriodicPerlinNoiseOctave(
        double wavelength, double period, double scale) :
    AbstractPerlinNoiseOctave(wavelength, scale),
    period_{period}
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

    double PeriodicPerlinNoiseOctave::grad(int32_t knot, double delta) const
    {
        // Wrap knot is period interval
        knot %= static_cast<uint32_t>(period_ / wavelength_);

        // Convert to double in [0.0, 1.0)
        const double s = perm_[knot] / 256.0;

        // Compute rescaled gradient between [-1.0, 1.0)
        const double grad = 2.0 * s - 1.0;

        // Return scalar product between distance and gradient
        return 2.0 * grad * delta;
    }

    AbstractPerlinProcess::AbstractPerlinProcess(
        double scale, std::vector<std::unique_ptr<AbstractPerlinNoiseOctave>> && octaves) noexcept
    :
    scale_{scale},
    octaves_(std::move(octaves))
    {
    }

    void AbstractPerlinProcess::reset()
    {
        // Reset every octave successively
        for (auto & octave : octaves_)
        {
            octave->reset();
        }

        // Compute scaling factor to get values in range [-1.0, 1.0]
        double amplitudeSquared = 0.0;
        for (const auto & octave : octaves_)
        {
            amplitudeSquared += std::pow(octave->getScale(), 2);
        }
        amplitude_ = scale_ * std::sqrt(amplitudeSquared);
    }

    double AbstractPerlinProcess::operator()(const float & t)
    {
        // Compute sum of octaves' values
        double value = 0.0;
        for (const auto & octave : octaves_)
        {
            value += (*octave)(t);
        }

        // Return scaled value by total amplitude
        return value / amplitude_;
    }

    double AbstractPerlinProcess::getWavelength() const noexcept
    {
        return std::transform_reduce(
            octaves_.cbegin(),
            octaves_.cend(),
            INF,
            std::less<double>(),
            [](const std::unique_ptr<AbstractPerlinNoiseOctave> & octave) -> double
            { return octave->getWavelength(); });
    }

    std::size_t AbstractPerlinProcess::getNumOctaves() const noexcept
    {
        return octaves_.size();
    }

    double AbstractPerlinProcess::getScale() const noexcept
    {
        return scale_;
    }

    std::vector<std::unique_ptr<AbstractPerlinNoiseOctave>> buildPerlinNoiseOctaves(
        double wavelength,
        std::size_t numOctaves,
        std::function<std::unique_ptr<AbstractPerlinNoiseOctave>(double, double)> factory)
    {
        std::vector<std::unique_ptr<AbstractPerlinNoiseOctave>> octaves_;
        octaves_.reserve(numOctaves);
        double scale = 1.0;
        for (std::size_t i = 0; i < numOctaves; ++i)
        {
            octaves_.push_back(factory(wavelength, scale));
            wavelength *= PERLIN_NOISE_LACUNARITY;
            scale *= PERLIN_NOISE_PERSISTENCE;
        }
        return octaves_;
    }

    RandomPerlinProcess::RandomPerlinProcess(
        double wavelength, double scale, std::size_t numOctaves) :
    AbstractPerlinProcess(
        scale,
        buildPerlinNoiseOctaves(
            wavelength,
            numOctaves,
            [](double wavelengthIn, double scaleIn) -> std::unique_ptr<AbstractPerlinNoiseOctave>
            { return std::make_unique<RandomPerlinNoiseOctave>(wavelengthIn, scaleIn); }))
    {
    }

    PeriodicPerlinProcess::PeriodicPerlinProcess(
        double wavelength, double period, double scale, std::size_t numOctaves) :
    AbstractPerlinProcess(
        scale,
        buildPerlinNoiseOctaves(
            wavelength,
            numOctaves,
            [period](double wavelengthIn,
                     double scaleIn) -> std::unique_ptr<AbstractPerlinNoiseOctave> {
                return std::make_unique<PeriodicPerlinNoiseOctave>(wavelengthIn, period, scaleIn);
            })),
    period_{period}
    {
        // Make sure the period is larger than the wavelength
        assert(period_ >= wavelength && "Period must be larger than wavelength.");
    }

    double PeriodicPerlinProcess::getPeriod() const noexcept
    {
        return period_;
    }

    template<typename VectorLike>
    std::enable_if_t<is_eigen_vector_v<VectorLike>, double> randomDouble(
        const Eigen::MatrixBase<VectorLike> & key, int64_t sparsity, double scale, uint32_t seed)
    {
        const int32_t keyLen = static_cast<int32_t>(sizeof(typename VectorLike::Scalar)) *
                               static_cast<int32_t>(key.size());
        const uint32_t hash = MurmurHash3(key.derived().data(), keyLen, seed);
        if (hash % sparsity == 0)
        {
            double encoding(hash);
            encoding /= std::numeric_limits<uint32_t>::max();
            return scale * encoding;
        }
        return 0.0;
    }

    std::pair<double, double> tile2dInterp1d(Eigen::Vector2i & posIdx,
                                             const Eigen::Vector2d & posRel,
                                             uint32_t dim,
                                             const Eigen::Vector2d & size,
                                             int64_t sparsity,
                                             double heightMax,
                                             const Eigen::Vector2d & interpThreshold,
                                             uint32_t seed)
    {
        const double z = randomDouble(posIdx, sparsity, heightMax, seed);
        double height, dheight;
        if (posRel[dim] < interpThreshold[dim])
        {
            posIdx[dim] -= 1;
            const double z_m = randomDouble(posIdx, sparsity, heightMax, seed);
            posIdx[dim] += 1;

            const double ratio = (1.0 - posRel[dim] / interpThreshold[dim]) / 2.0;
            height = z + (z_m - z) * ratio;
            dheight = (z - z_m) / (2.0 * size[dim] * interpThreshold[dim]);
        }
        else if (1.0 - posRel[dim] < interpThreshold[dim])
        {
            posIdx[dim] += 1;
            const double z_p = randomDouble(posIdx, sparsity, heightMax, seed);
            posIdx[dim] -= 1;

            const double ratio = (1.0 + (posRel[dim] - 1.0) / interpThreshold[dim]) / 2.0;
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

    HeightmapFunctor randomTileGround(const Eigen::Vector2d & size,
                                      double heightMax,
                                      const Eigen::Vector2d & interpDelta,
                                      uint32_t sparsity,
                                      double orientation,
                                      uint32_t seed)
    {
        if ((0.01 <= interpDelta.array()).all() &&
            (interpDelta.array() <= size.array() / 2.0).all())
        {
            PRINT_WARNING("'interpDelta' must be in range [0.01, 'size'/2.0].");
        }

        Eigen::Vector2d interpThreshold = interpDelta.cwiseMax(0.01).cwiseMin(size / 2.0);
        interpThreshold.array() /= size.array();

        const Eigen::Vector2d offset = Eigen::Vector2d::NullaryExpr(
            [&size, seed](Eigen::Index i) -> double
            {
                Eigen::Matrix<Eigen::Index, 1, 1> key;
                key[0] = i;
                return randomDouble(key, 1, size[i], seed);
            });

        Eigen::Rotation2D<double> rotationMat(orientation);

        return
            [size, heightMax, interpDelta, rotationMat, sparsity, interpThreshold, offset, seed](
                const Eigen::Vector3d & pos3) -> std::pair<double, Eigen::Vector3d>
        {
            // Compute the tile index and relative coordinate
            Eigen::Vector2d pos = rotationMat * (pos3.head<2>() + offset);
            Eigen::Vector2d posRel = pos.array() / size.array();
            Eigen::Vector2i posIdx = posRel.array().floor().cast<int32_t>();
            posRel -= posIdx.cast<double>();

            // Interpolate height based on nearby tiles if necessary
            double height;
            Eigen::Vector3d normal;
            Eigen::Matrix<bool, 2, 1> isEdge = (posRel.array() < interpThreshold.array()) ||
                                               (1.0 - posRel.array() < interpThreshold.array());
            if (isEdge[0] && !isEdge[1])
            {
                double dheight_x;
                std::tie(height, dheight_x) = tile2dInterp1d(
                    posIdx, posRel, 0, size, sparsity, heightMax, interpThreshold, seed);
                const double normInv = std::sqrt(dheight_x * dheight_x + 1.0);
                normal << -dheight_x * normInv, 0.0, normInv;
            }
            else if (!isEdge[0] && isEdge[1])
            {
                double dheight_y;
                std::tie(height, dheight_y) = tile2dInterp1d(
                    posIdx, posRel, 1, size, sparsity, heightMax, interpThreshold, seed);
                const double normInv = std::sqrt(dheight_y * dheight_y + 1.0);
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

                    const double ratio = (1.0 - posRel[1] / interpThreshold[1]) / 2.0;
                    height = height_0 + (height_m - height_0) * ratio;
                    const double dheight_x = dheight_x_0 + (dheight_x_m - dheight_x_0) * ratio;
                    const double dheight_y =
                        (height_0 - height_m) / (2.0 * size[1] * interpThreshold[1]);
                    normal << -dheight_x, -dheight_y, 1.0;
                    normal.normalize();
                }
                else
                {
                    posIdx[1] += 1;
                    const auto [height_p, dheight_x_p] = tile2dInterp1d(
                        posIdx, posRel, 0, size, sparsity, heightMax, interpThreshold, seed);

                    const double ratio = (1.0 + (posRel[1] - 1.0) / interpThreshold[1]) / 2.0;
                    height = height_0 + (height_p - height_0) * ratio;
                    const double dheight_x = dheight_x_0 + (dheight_x_p - dheight_x_0) * ratio;
                    const double dheight_y =
                        (height_p - height_0) / (2.0 * size[1] * interpThreshold[1]);
                    normal << -dheight_x, -dheight_y, 1.0;
                    normal.normalize();
                }
            }
            else
            {
                height = randomDouble(posIdx, sparsity, heightMax, seed);
                normal = Eigen::Vector3d::UnitZ();
            }

            return std::make_pair(height, std::move(normal));
        };
    }

    HeightmapFunctor sumHeightmap(const std::vector<HeightmapFunctor> & heightmaps)
    {
        if (heightmaps.size() == 1)
        {
            return heightmaps[0];
        }
        return [heightmaps](const Eigen::Vector3d & pos3) -> std::pair<double, Eigen::Vector3d>
        {
            double height = 0.0;
            Eigen::Vector3d normal = Eigen::Vector3d::Zero();
            for (const HeightmapFunctor & heightmap : heightmaps)
            {
                const auto [height_i, normal_i] = heightmap(pos3);
                height += height_i;
                normal += normal_i;
            }
            normal.normalize();
            return std::make_pair(height, std::move(normal));
        };
    }

    HeightmapFunctor mergeHeightmap(const std::vector<HeightmapFunctor> & heightmaps)
    {
        if (heightmaps.size() == 1)
        {
            return heightmaps[0];
        }
        return [heightmaps](const Eigen::Vector3d & pos3) -> std::pair<double, Eigen::Vector3d>
        {
            double heightmax = -INF;
            Eigen::Vector3d normal;  // It will be initialized to `normal_i`
            bool isDirty = false;
            for (const HeightmapFunctor & heightmap : heightmaps)
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

    Eigen::MatrixXd discretizeHeightmap(
        const HeightmapFunctor & heightmap, double gridSize, double gridUnit)
    {
        // Allocate empty discrete grid
        uint32_t gridDim = static_cast<int32_t>(std::ceil(gridSize / gridUnit)) + 1U;
        Eigen::MatrixXd heightGrid(gridDim * gridDim, 6);

        // Fill x and y discrete grid coordinates
        const Eigen::VectorXd values =
            (Eigen::VectorXd::LinSpaced(gridDim, 0, gridDim - 1) * gridUnit).array() -
            (gridDim - 1) * (gridUnit / 2.0);
        Eigen::Map<Eigen::MatrixXd>(heightGrid.col(0).data(), gridDim, gridDim).colwise() = values;
        Eigen::Map<Eigen::MatrixXd>(heightGrid.col(1).data(), gridDim, gridDim).rowwise() =
            values.transpose();

        // Fill discrete grid
        for (uint32_t i = 0; i < heightGrid.rows(); ++i)
        {
            auto result = heightmap(heightGrid.block<1, 3>(i, 0));
            heightGrid(i, 2) = std::get<double>(result);
            heightGrid.block<1, 3>(i, 3) = std::get<Eigen::Vector3d>(result);
        }

        return heightGrid;
    }
}
