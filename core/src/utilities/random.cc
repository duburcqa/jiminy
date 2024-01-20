#include <numeric>

#include "jiminy/core/utilities/random.h"


namespace jiminy
{
    static inline constexpr double PERLIN_NOISE_PERSISTENCE{1.50};
    static inline constexpr double PERLIN_NOISE_LACUNARITY{1.15};

    // ***************************** Uniform random bit generators ***************************** //

    PCG32::PCG32(uint64_t state) noexcept :
    state_{state | 3ULL}
    {
    }

    uint32_t PCG32::operator()() noexcept
    {
        constexpr uint8_t bits = static_cast<uint8_t>(std::numeric_limits<uint64_t>::digits);
        constexpr uint8_t uint32Bits = static_cast<uint8_t>(std::numeric_limits<uint32_t>::digits);
        constexpr uint8_t spareBits = bits - uint32Bits;
        constexpr uint8_t opBits = spareBits - 5 >= 64 ? 5 :
                                   spareBits - 4 >= 32 ? 4 :
                                   spareBits - 3 >= 16 ? 3 :
                                   spareBits - 2 >= 4  ? 2 :
                                   spareBits - 1 >= 1  ? 1 :
                                                         0;
        constexpr uint8_t mask = (1 << opBits) - 1;
        constexpr uint8_t maxRandShift = mask;
        constexpr uint8_t topSpare = opBits;
        constexpr uint8_t bottomSpare = spareBits - topSpare;
        constexpr uint8_t xShift = topSpare + (uint32Bits + maxRandShift) / 2;

        state_ *= 6364136223846793005ULL;
        uint64_t state = state_;
        uint8_t rshift = opBits ? static_cast<uint8_t>(state >> (bits - opBits)) & mask : 0U;
        state ^= state >> xShift;
        return static_cast<uint32_t>(state >> (bottomSpare - maxRandShift + rshift));
    }

    // ****************************** Random number distributions ****************************** //

    float uniform(const uniform_random_bit_generator_ref<uint32_t> & g)
    {
        return std::generate_canonical<float, std::numeric_limits<float>::digits>(g);
    }

    float uniform(const uniform_random_bit_generator_ref<uint32_t> & g, float lo, float hi)
    {
        return std::uniform_real_distribution<float>(lo, hi)(g);
    }

    namespace internal::ziggurat
    {
        /// \details "Advanced" cmath functions are required by the standard to be constexpr
        ///          starting from C++26 instead of C++23 as initially expected (see P1383R2).
        ///          Generic compile-time computation supported by all compilers is not possible
        ///          without it, and must be postponed to import-time, which is still much better
        ///          than run-time.
        ///
        ///  \sa For reference, see:
        ///      https://stackoverflow.com/a/34465458/4820605
        ///      https://stackoverflow.com/q/36497632/4820605
        struct ZigguratNormalData
        {
            ZigguratNormalData()
            {
                constexpr double m1 = 2147483648.0;
                constexpr double vn = 9.91256303526217e-03;
                double dn = 3.442619855899;
                double tn = dn;

                const double q = vn / std::exp(-0.5 * dn * dn);

                kn[0] = static_cast<uint32_t>((dn / q) * m1);
                kn[1] = 0;

                wn[0] = static_cast<float>(q / m1);
                wn[127] = static_cast<float>(dn / m1);

                fn[0] = 1.0F;
                fn[127] = static_cast<float>(std::exp(-0.5 * dn * dn));

                for (uint8_t i = 126; 1 <= i; i--)
                {
                    dn = std::sqrt(-2.0 * std::log(vn / dn + std::exp(-0.5 * dn * dn)));
                    kn[i + 1] = static_cast<uint32_t>((dn / tn) * m1);
                    tn = dn;
                    fn[i] = static_cast<float>(std::exp(-0.5 * dn * dn));
                    wn[i] = static_cast<float>(dn / m1);
                }
            }

            std::array<uint32_t, 128> kn{};
            std::array<float, 128> fn{};
            std::array<float, 128> wn{};
        };

        static const ZigguratNormalData ZIGGURAT_NORMAL_DATA{};
        static const std::array<uint32_t, 128> kn = ZIGGURAT_NORMAL_DATA.kn;
        static const std::array<float, 128> fn = ZIGGURAT_NORMAL_DATA.fn;
        static const std::array<float, 128> wn = ZIGGURAT_NORMAL_DATA.wn;
    }

    namespace internal
    {

        float normal(const uniform_random_bit_generator_ref<uint32_t> & g)
        {
            using namespace internal::ziggurat;

            constexpr float r = 3.442620F;

            int32_t hz;
            uint32_t iz;
            float x;
            float y;

            hz = static_cast<int32_t>(g());
            iz = (static_cast<uint32_t>(hz) & 127UL);

            if (std::fabs(hz) < kn[iz])
            {
                return static_cast<float>(hz) * wn[iz];
            }

            while (true)
            {
                if (iz == 0)
                {
                    while (true)
                    {
                        x = -0.2904764F * std::log(uniform(g));
                        y = -std::log(uniform(g));
                        if (x * x <= y + y)
                        {
                            break;
                        }
                    }

                    if (hz <= 0)
                    {
                        return -r - x;
                    }
                    return r + x;
                }

                x = static_cast<float>(hz) * wn[iz];

                if (fn[iz] + uniform(g) * (fn[iz - 1] - fn[iz]) < std::exp(-0.5F * x * x))
                {
                    return x;
                }

                hz = static_cast<int32_t>(g());
                iz = (hz & 127);

                if (std::fabs(hz) < kn[iz])
                {
                    return static_cast<float>(hz) * wn[iz];
                }
            }
        }
    }

    float normal(const uniform_random_bit_generator_ref<uint32_t> & g, float mean, float stddev)
    {
        return internal::normal(g) * stddev + mean;
    }

    // **************************** Non-cryptographic hash function **************************** //

    static uint32_t rotl32(uint32_t x, int8_t r) noexcept
    {
        return (x << r) | (x >> (32 - r));
    }

    /// @brief MurmurHash3 is a non-cryptographic hash function initially designed
    ///        for hash-based lookup.
    ///
    /// @sa It was written by Austin Appleby, and is placed in the public domain.
    ///     The author hereby disclaims copyright to this source code:
    ///     https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
    static uint32_t MurmurHash3(const void * key, int32_t len, uint32_t seed) noexcept
    {
        // Define some internal constants
        constexpr uint32_t c1 = 0xcc9e2d51;
        constexpr uint32_t c2 = 0x1b873593;
        constexpr uint32_t c3 = 0xe6546b64;

        // Initialize has to seed value
        uint32_t h1 = seed;

        // Extract bytes from key
        const auto * data = static_cast<const uint8_t *>(key);
        const int32_t num_blocks = len / 4;  // len in bytes, so 32-bits blocks

        // Body
        const auto * blocks =
            reinterpret_cast<const uint32_t *>(data + static_cast<std::ptrdiff_t>(num_blocks * 4));
        for (int32_t i = -num_blocks; i; ++i)
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
        const auto * tail =
            reinterpret_cast<const uint8_t *>(data + static_cast<std::ptrdiff_t>(num_blocks * 4));
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

    // **************************** Continuous 1D Gaussian processes *************************** //

    PeriodicGaussianProcess::PeriodicGaussianProcess(double wavelength, double period) noexcept :
    wavelength_{wavelength},
    period_{period}
    {
        reset(std::random_device{});
    }

    void PeriodicGaussianProcess::reset(
        const uniform_random_bit_generator_ref<uint32_t> & g) noexcept
    {
        // Sample normal vector
        auto normalVec = normal(num_times_, 1, g);

        // Compute discrete periodic gaussian process values
        values_.noalias() =
            cov_sqrt_root_.triangularView<Eigen::Lower>() * normalVec.cast<double>();
    }

    double PeriodicGaussianProcess::operator()(float t)
    {
        // Wrap requested time in gaussian process period
        double tWrap = std::fmod(t, period_);
        if (tWrap < 0)
        {
            tWrap += period_;
        }

        // Compute closest left and right indices
        const int32_t tLeftIdx = static_cast<int32_t>(std::floor(tWrap / dt_));
        const int32_t tRightIdx = (tLeftIdx + 1) % num_times_;

        // Perform First order interpolation
        const double ratio = tWrap / dt_ - tLeftIdx;
        return values_[tLeftIdx] + ratio * (values_[tRightIdx] - values_[tLeftIdx]);
    }

    double PeriodicGaussianProcess::wavelength() const noexcept
    {
        return wavelength_;
    }

    double PeriodicGaussianProcess::period() const noexcept
    {
        return period_;
    }

    // **************************** Continuous 1D Fourier processes **************************** //

    PeriodicFourierProcess::PeriodicFourierProcess(double wavelength, double period) noexcept :
    wavelength_{wavelength},
    period_{period}
    {
        reset(std::random_device{});
    }

    void PeriodicFourierProcess::reset(
        const uniform_random_bit_generator_ref<uint32_t> & g) noexcept
    {
        // Sample normal vectors
        auto normalVec1 = normal(num_harmonics_, 1, g);
        auto normalVec2 = normal(num_harmonics_, 1, g);

        // Compute discrete periodic gaussian process values
        const double scale = M_SQRT2 / std::sqrt(2 * num_harmonics_ + 1);
        values_ = scale * cos_mat_ * normalVec1.cast<double>();
        values_.noalias() += scale * sin_mat_ * normalVec2.cast<double>();
    }

    double PeriodicFourierProcess::operator()(float t)
    {
        // Wrap requested time in guassian process period
        double tWrap = std::fmod(t, period_);
        if (tWrap < 0)
        {
            tWrap += period_;
        }

        // Compute closest left and right indices
        const int32_t tLeftIdx = static_cast<int32_t>(std::floor(tWrap / dt_));
        const int32_t tRightIdx = (tLeftIdx + 1) % num_times_;

        // Perform First order interpolation
        const double ratio = tWrap / dt_ - tLeftIdx;
        return values_[tLeftIdx] + ratio * (values_[tRightIdx] - values_[tLeftIdx]);
    }

    double PeriodicFourierProcess::wavelength() const noexcept
    {
        return wavelength_;
    }

    double PeriodicFourierProcess::period() const noexcept
    {
        return period_;
    }

    // ***************************** Continuous 1D Perlin processes **************************** //

    AbstractPerlinNoiseOctave::AbstractPerlinNoiseOctave(double wavelength) :
    wavelength_{wavelength}
    {
        assert(wavelength_ > 0 && "wavelength must be strictly larger than 0.0.");
        shift_ = uniform(std::random_device{});
    }

    void AbstractPerlinNoiseOctave::reset(
        const uniform_random_bit_generator_ref<uint32_t> & g) noexcept
    {
        // Sample random phase shift
        shift_ = uniform(g);
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
        return lerp(ratio, yLeft, yRight);
    }

    double AbstractPerlinNoiseOctave::wavelength() const noexcept
    {
        return wavelength_;
    }

    double AbstractPerlinNoiseOctave::fade(double delta) noexcept
    {
        /* Improved Smoothstep function by Ken Perlin (aka Smootherstep).
           It has zero 1st and 2nd-order derivatives at dt = 0.0, and 1.0:
           https://en.wikipedia.org/wiki/Smoothstep#Variations */
        return std::pow(delta, 3) * (delta * (delta * 6.0 - 15.0) + 10.0);
    }

    double AbstractPerlinNoiseOctave::lerp(double ratio, double yLeft, double yRight) noexcept
    {
        return yLeft + ratio * (yRight - yLeft);
    }

    RandomPerlinNoiseOctave::RandomPerlinNoiseOctave(double wavelength) :
    AbstractPerlinNoiseOctave(wavelength)
    {
        seed_ = std::random_device{}();
    }

    void RandomPerlinNoiseOctave::reset(
        const uniform_random_bit_generator_ref<uint32_t> & g) noexcept
    {
        // Call base implementation
        AbstractPerlinNoiseOctave::reset(g);

        // Sample new random seed for MurmurHash
        seed_ = g();
    }

    double RandomPerlinNoiseOctave::grad(int32_t knot, double delta) const noexcept
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

    namespace internal
    {
        template<typename Generator, typename T>
        void randomizePermutationVector(Generator && g, T & vec)
        {
            // Re-Initialize the permutation vector with values from 0 to size
            std::iota(vec.begin(), vec.end(), 0);

            // Shuffle the permutation vector
            std::shuffle(vec.begin(), vec.end(), g);
        }
    }

    PeriodicPerlinNoiseOctave::PeriodicPerlinNoiseOctave(double wavelength, double period) :
    AbstractPerlinNoiseOctave(wavelength),
    period_{period}
    {
        // Make sure the wavelength is multiple of the period
        assert(std::abs(std::round(period / wavelength) * wavelength - period) < 1e-6 &&
               "wavelength must be multiple of period.");

        // Initialize the permutation vector with values from 0 to 255 and shuffle it
        internal::randomizePermutationVector(std::random_device{}, perm_);
    }

    void PeriodicPerlinNoiseOctave::reset(
        const uniform_random_bit_generator_ref<uint32_t> & g) noexcept
    {
        // Call base implementation
        AbstractPerlinNoiseOctave::reset(g);

        // Re-Initialize the permutation vector with values from 0 to 255
        internal::randomizePermutationVector(g, perm_);
    }

    double PeriodicPerlinNoiseOctave::grad(int32_t knot, double delta) const noexcept
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
        std::vector<OctaveScalePair> && octave_scale_pairs) noexcept :
    octave_scale_pairs_(std::move(octave_scale_pairs))
    {
        // Compute the scaling factor to keep values within range [-1.0, 1.0]
        double amplitude_squared = 0.0;
        for (const OctaveScalePair & octave_scale : octave_scale_pairs_)
        {
            const double scale = std::get<1>(octave_scale);
            amplitude_squared += std::pow(scale, 2);
        }
        amplitude_ = std::sqrt(amplitude_squared);
    }

    void AbstractPerlinProcess::reset(
        const uniform_random_bit_generator_ref<uint32_t> & g) noexcept
    {
        // Reset octaves
        for (OctaveScalePair & octave_scale : octave_scale_pairs_)
        {
            std::unique_ptr<AbstractPerlinNoiseOctave> & octave = std::get<0>(octave_scale);
            octave->reset(g);
        }
    }

    double AbstractPerlinProcess::operator()(float t)
    {
        // Compute sum of octaves' values
        double value = 0.0;
        for (const auto & [octave, scale] : octave_scale_pairs_)
        {
            value += scale * (*octave)(t);
        }

        // Scale sum by maximum amplitude
        return value / amplitude_;
    }

    double AbstractPerlinProcess::wavelength() const noexcept
    {
        double wavelength = INF;
        for (const OctaveScalePair & octave_scale : octave_scale_pairs_)
        {
            const std::unique_ptr<AbstractPerlinNoiseOctave> & octave = std::get<0>(octave_scale);
            wavelength = std::min(wavelength, octave->wavelength());
        }
        return wavelength;
    }

    std::size_t AbstractPerlinProcess::num_octaves() const noexcept
    {
        return octave_scale_pairs_.size();
    }

    std::vector<AbstractPerlinProcess::OctaveScalePair> buildPerlinNoiseOctaves(
        double wavelength,
        std::size_t num_octaves,
        std::function<std::unique_ptr<AbstractPerlinNoiseOctave>(double)> factory)
    {
        std::vector<AbstractPerlinProcess::OctaveScalePair> octave_scale_pairs;
        octave_scale_pairs.reserve(num_octaves);
        double scale = 1.0;
        for (std::size_t i = 0; i < num_octaves; ++i)
        {
            octave_scale_pairs.emplace_back(factory(wavelength), scale);
            wavelength *= PERLIN_NOISE_LACUNARITY;
            scale *= PERLIN_NOISE_PERSISTENCE;
        }
        return octave_scale_pairs;
    }

    RandomPerlinProcess::RandomPerlinProcess(double wavelength, std::size_t num_octaves) :
    AbstractPerlinProcess(buildPerlinNoiseOctaves(
        wavelength,
        num_octaves,
        [](double wavelengthIn) -> std::unique_ptr<AbstractPerlinNoiseOctave>
        { return std::make_unique<RandomPerlinNoiseOctave>(wavelengthIn); }))
    {
    }

    PeriodicPerlinProcess::PeriodicPerlinProcess(
        double wavelength, double period, std::size_t num_octaves) :
    AbstractPerlinProcess(buildPerlinNoiseOctaves(
        wavelength,
        num_octaves,
        [period](double wavelengthIn) -> std::unique_ptr<AbstractPerlinNoiseOctave>
        { return std::make_unique<PeriodicPerlinNoiseOctave>(wavelengthIn, period); })),
    period_{period}
    {
        // Make sure the period is larger than the wavelength
        assert(period_ >= wavelength && "Period must be larger than wavelength.");
    }

    double PeriodicPerlinProcess::period() const noexcept
    {
        return period_;
    }

    // ******************************* Random terrain generators ******************************* //

    template<typename Scalar>
    static float uniformSparseFromStateImpl(
        const MatrixX<Scalar> & state, int64_t sparsity, uint32_t seed) noexcept
    {
        const auto num_bytes = static_cast<int32_t>(sizeof(Scalar) * state.size());
        const uint32_t hash = MurmurHash3(state.data(), num_bytes, seed);
        if (hash % sparsity == 0)
        {
            return static_cast<float>(hash) /
                   static_cast<float>(std::numeric_limits<uint32_t>::max());
        }
        return 0.0F;
    }

    template<typename Derived>
    static float uniformSparseFromState(
        const Eigen::MatrixBase<Derived> & state, int64_t sparsity, uint32_t seed) noexcept
    {
        return uniformSparseFromStateImpl<typename Derived::Scalar>(
            state.derived(), sparsity, seed);
    }

    std::pair<double, double> tile2dInterp1d(Eigen::Vector2i & pos_indices,
                                             const Eigen::Vector2d & pos_rel,
                                             uint32_t dim,
                                             const Eigen::Vector2d & size,
                                             int64_t sparsity,
                                             double height_max,
                                             const Eigen::Vector2d & interp_thr,
                                             uint32_t seed)
    {
        double height, dheight;

        const double z = height_max * uniformSparseFromState(pos_indices, sparsity, seed);
        if (pos_rel[dim] < interp_thr[dim])
        {
            pos_indices[dim] -= 1;
            const double z_m = height_max * uniformSparseFromState(pos_indices, sparsity, seed);
            pos_indices[dim] += 1;

            const double ratio = (1.0 - pos_rel[dim] / interp_thr[dim]) / 2.0;
            height = z + (z_m - z) * ratio;
            dheight = (z - z_m) / (2.0 * size[dim] * interp_thr[dim]);
        }
        else if (1.0 - pos_rel[dim] < interp_thr[dim])
        {
            pos_indices[dim] += 1;
            const double z_p = height_max * uniformSparseFromState(pos_indices, sparsity, seed);
            pos_indices[dim] -= 1;

            const double ratio = (1.0 + (pos_rel[dim] - 1.0) / interp_thr[dim]) / 2.0;
            height = z + (z_p - z) * ratio;
            dheight = (z_p - z) / (2.0 * size[dim] * interp_thr[dim]);
        }
        else
        {
            height = z;
            dheight = 0.0;
        }

        return {height, dheight};
    }

    HeightmapFunctor tiles(const Eigen::Vector2d & size,
                           double height_max,
                           const Eigen::Vector2d & interp_delta,
                           uint32_t sparsity,
                           double orientation,
                           uint32_t seed)
    {
        if ((0.01 <= interp_delta.array()).all() &&
            (interp_delta.array() <= size.array() / 2.0).all())
        {
            PRINT_WARNING("'interp_delta' must be in range [0.01, 'size'/2.0].");
        }

        Eigen::Vector2d interp_thr = interp_delta.cwiseMax(0.01).cwiseMin(size / 2.0);
        interp_thr.array() /= size.array();

        const Eigen::Vector2d offset = Eigen::Vector2d::NullaryExpr(
            [&size, seed](Eigen::Index i) -> double {
                return size[i] *
                       uniformSparseFromState(Vector1<Eigen::Index>::Constant(i), 1, seed);
            });

        const Eigen::Rotation2D<double> rot_mat(orientation);

        return [size, height_max, interp_delta, rot_mat, sparsity, interp_thr, offset, seed](
                   const Eigen::Vector2d & pos, double & height, Eigen::Vector3d & normal) -> void
        {
            // Compute the tile index and relative coordinate
            Eigen::Vector2d pos_rel = (rot_mat * (pos + offset)).array() / size.array();
            Vector2<int32_t> pos_indices = pos_rel.array().floor().cast<int32_t>();
            pos_rel -= pos_indices.cast<double>();

            // Interpolate height based on nearby tiles if necessary
            Vector2<bool> is_edge = (pos_rel.array() < interp_thr.array()) ||
                                    (1.0 - pos_rel.array() < interp_thr.array());
            if (is_edge[0] && !is_edge[1])
            {
                double dheight_x;
                std::tie(height, dheight_x) = tile2dInterp1d(
                    pos_indices, pos_rel, 0, size, sparsity, height_max, interp_thr, seed);
                double const norm_inv = 1.0 / std::sqrt(dheight_x * dheight_x + 1.0);
                normal << -dheight_x * norm_inv, 0.0, norm_inv;
            }
            else if (!is_edge[0] && is_edge[1])
            {
                double dheight_y;
                std::tie(height, dheight_y) = tile2dInterp1d(
                    pos_indices, pos_rel, 1, size, sparsity, height_max, interp_thr, seed);
                double const norm_inv = 1.0 / std::sqrt(dheight_y * dheight_y + 1.0);
                normal << 0.0, -dheight_y * norm_inv, norm_inv;
            }
            else if (is_edge[0] && is_edge[1])
            {
                auto const [height_0, dheight_x_0] = tile2dInterp1d(
                    pos_indices, pos_rel, 0, size, sparsity, height_max, interp_thr, seed);
                if (pos_rel[1] < interp_thr[1])
                {
                    pos_indices[1] -= 1;
                    auto const [height_m, dheight_x_m] = tile2dInterp1d(
                        pos_indices, pos_rel, 0, size, sparsity, height_max, interp_thr, seed);

                    double const ratio = (1.0 - pos_rel[1] / interp_thr[1]) / 2.0;
                    height = height_0 + (height_m - height_0) * ratio;
                    double const dheight_x = dheight_x_0 + (dheight_x_m - dheight_x_0) * ratio;
                    double const dheight_y =
                        (height_0 - height_m) / (2.0 * size[1] * interp_thr[1]);
                    normal << -dheight_x, -dheight_y, 1.0;
                    normal.normalize();
                }
                else
                {
                    pos_indices[1] += 1;
                    auto const [height_p, dheight_x_p] = tile2dInterp1d(
                        pos_indices, pos_rel, 0, size, sparsity, height_max, interp_thr, seed);

                    double const ratio = (1.0 + (pos_rel[1] - 1.0) / interp_thr[1]) / 2.0;
                    height = height_0 + (height_p - height_0) * ratio;
                    double const dheight_x = dheight_x_0 + (dheight_x_p - dheight_x_0) * ratio;
                    double const dheight_y =
                        (height_p - height_0) / (2.0 * size[1] * interp_thr[1]);
                    normal << -dheight_x, -dheight_y, 1.0;
                    normal.normalize();
                }
            }
            else
            {
                height = height_max * uniformSparseFromState(pos_indices, sparsity, seed);
                normal = Eigen::Vector3d::UnitZ();
            }
        };
    }
}