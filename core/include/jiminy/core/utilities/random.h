#ifndef JIMINY_RANDOM_H
#define JIMINY_RANDOM_H

#include <optional>

#include "jiminy/core/fwd.h"


namespace jiminy
{
    // ************ Random number generator utilities ***************

    void JIMINY_DLLAPI resetRandomGenerators(const std::optional<uint32_t> & seed = std::nullopt);

    hresult_t JIMINY_DLLAPI getRandomSeed(uint32_t & seed);

    float64_t JIMINY_DLLAPI randUniform(const float64_t & lo = 0.0, const float64_t & hi = 1.0);

    float64_t JIMINY_DLLAPI randNormal(const float64_t & mean = 0.0, const float64_t & std = 1.0);

    Eigen::VectorXd JIMINY_DLLAPI randVectorNormal(
        const uint32_t & size, const float64_t & mean, const float64_t & std);

    Eigen::VectorXd JIMINY_DLLAPI randVectorNormal(const uint32_t & size, const float64_t & std);

    Eigen::VectorXd JIMINY_DLLAPI randVectorNormal(const Eigen::VectorXd & std);

    Eigen::VectorXd JIMINY_DLLAPI randVectorNormal(const Eigen::VectorXd & mean,
                                                   const Eigen::VectorXd & std);

    void JIMINY_DLLAPI shuffleIndices(std::vector<uint32_t> & vector);

    // ************ Continuous 1D Perlin processes ***************

    class JIMINY_DLLAPI PeriodicGaussianProcess
    {
    public:
        DISABLE_COPY(PeriodicGaussianProcess)

    public:
        PeriodicGaussianProcess(
            const float64_t & wavelength, const float64_t & period, const float64_t & scale = 1.0);

        ~PeriodicGaussianProcess() = default;

        void reset();

        float64_t operator()(const float & t);

        const float64_t & getWavelength() const;
        const float64_t & getPeriod() const;
        const float64_t & getDt() const;

    protected:
        void initialize();

    private:
        const float64_t wavelength_;
        const float64_t period_;
        const float64_t scale_;
        const float64_t dt_;
        const int32_t numTimes_;

        bool_t isInitialized_;
        Eigen::VectorXd values_;
        Eigen::MatrixXd covSqrtRoot_;
    };


    /// \see Based on "Smooth random functions, random ODEs, and Gaussianprocesses":
    ///      https://hal.inria.fr/hal-01944992/file/random_revision2.pdf */
    class JIMINY_DLLAPI PeriodicFourierProcess
    {
    public:
        DISABLE_COPY(PeriodicFourierProcess)

    public:
        PeriodicFourierProcess(
            const float64_t & wavelength, const float64_t & period, const float64_t & scale = 1.0);

        ~PeriodicFourierProcess() = default;

        void reset();

        float64_t operator()(const float & t);

        const float64_t & getWavelength() const;
        const float64_t & getPeriod() const;
        const int32_t & getNumHarmonics() const;
        const float64_t & getDt() const;

    protected:
        void initialize();

    private:
        const float64_t wavelength_;
        const float64_t period_;
        const float64_t scale_;
        const float64_t dt_;
        const int32_t numTimes_;
        const int32_t numHarmonics_;

        bool_t isInitialized_;
        Eigen::VectorXd values_;
        Eigen::MatrixXd cosMat_;
        Eigen::MatrixXd sinMat_;
    };

    class JIMINY_DLLAPI AbstractPerlinNoiseOctave
    {
    public:
        AbstractPerlinNoiseOctave(const float64_t & wavelength, const float64_t & scale);
        virtual ~AbstractPerlinNoiseOctave() = default;

        virtual void reset();

        float64_t operator()(const float64_t & t) const;

        const float64_t & getWavelength() const;
        const float64_t & getScale() const;

    protected:
        // Copy on purpose
        virtual float64_t grad(int32_t knot, const float64_t & delta) const = 0;

        float64_t fade(const float64_t & delta) const;

        float64_t lerp(
            const float64_t & ratio, const float64_t & yLeft, const float64_t & yRight) const;

    protected:
        const float64_t wavelength_;
        const float64_t scale_;

        float64_t shift_;
    };

    class JIMINY_DLLAPI RandomPerlinNoiseOctave : public AbstractPerlinNoiseOctave
    {
    public:
        RandomPerlinNoiseOctave(const float64_t & wavelength, const float64_t & scale);

        virtual ~RandomPerlinNoiseOctave() = default;

        virtual void reset() override final;

    protected:
        virtual float64_t grad(int32_t knot, const float64_t & delta) const override final;

    private:
        uint32_t seed_;
    };

    class JIMINY_DLLAPI PeriodicPerlinNoiseOctave : public AbstractPerlinNoiseOctave
    {
    public:
        PeriodicPerlinNoiseOctave(
            const float64_t & wavelength, const float64_t & period, const float64_t & scale);

        virtual ~PeriodicPerlinNoiseOctave() = default;

        virtual void reset() override final;

    protected:
        virtual float64_t grad(int32_t knot, const float64_t & delta) const override final;

    private:
        const float64_t period_;

        std::vector<uint8_t> perm_;
    };

    /// \brief  Sum of Perlin noise octaves.
    ///
    /// \details The original implementation uses fixed size permutation table to generate random
    ///          gradient directions. As a result, the generated process is inherently periodic,
    ///          which must be avoided. To circumvent this limitation, MurmurHash3 algorithm is
    ///          used to get random gradients at every point in time, without any periodicity, but
    ///          deterministically for a given seed. It is computationally more depending but not
    ///          critically slower.
    ///
    /// \sa  For technical references:
    ///      https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/perlin-noise-part-2
    ///      https://adrianb.io/2014/08/09/perlinnoise.html
    ///      https://gamedev.stackexchange.com/a/23705/148509
    ///      https://gamedev.stackexchange.com/q/161923/148509
    ///      https://gamedev.stackexchange.com/q/134561/148509
    ///
    /// \sa  For reference about the implementation:
    ///      https://github.com/bradykieffer/SimplexNoise/blob/master/simplexnoise/noise.py
    ///      https://github.com/sol-prog/Perlin_Noise/blob/master/PerlinNoise.cpp
    ///      https://github.com/ashima/webgl-noise/blob/master/src/classicnoise2D.glsl
    class JIMINY_DLLAPI AbstractPerlinProcess
    {
    public:
        DISABLE_COPY(AbstractPerlinProcess)

    public:
        AbstractPerlinProcess(const float64_t & wavelength,
                              const float64_t & scale = 1.0,
                              const uint32_t & numOctaves = 8U);
        virtual ~AbstractPerlinProcess() = default;

        void reset();

        float64_t operator()(const float & t);

        const float64_t & getWavelength() const;
        const uint32_t & getNumOctaves() const;
        const float64_t & getScale() const;

    protected:
        virtual void initialize() = 0;

    protected:
        const float64_t wavelength_;
        const uint32_t numOctaves_;
        const float64_t scale_;

        bool_t isInitialized_;
        std::vector<std::unique_ptr<AbstractPerlinNoiseOctave>> octaves_;
        float64_t amplitude_;

        float64_t grad_;
    };

    class JIMINY_DLLAPI RandomPerlinProcess : public AbstractPerlinProcess
    {
    public:
        RandomPerlinProcess(const float64_t & wavelength,
                            const float64_t & scale = 1.0,
                            const uint32_t & numOctaves = 6U);

        virtual ~RandomPerlinProcess() = default;

    protected:
        virtual void initialize() override final;
    };

    class PeriodicPerlinProcess : public AbstractPerlinProcess
    {
    public:
        PeriodicPerlinProcess(const float64_t & wavelength,
                              const float64_t & period,
                              const float64_t & scale = 1.0,
                              const uint32_t & numOctaves = 6U);

        virtual ~PeriodicPerlinProcess() = default;

        const float64_t & getPeriod() const;

    protected:
        virtual void initialize() override final;

    private:
        const float64_t period_;
    };

    // ************ Random terrain generators ***************

    HeightmapFunctor JIMINY_DLLAPI randomTileGround(const Eigen::Vector2d & size,
                                                    const float64_t & heightMax,
                                                    const Eigen::Vector2d & interpDelta,
                                                    const uint32_t & sparsity,
                                                    const float64_t & orientation,
                                                    const uint32_t & seed);

    HeightmapFunctor JIMINY_DLLAPI sumHeightmap(const std::vector<HeightmapFunctor> & heightmaps);
    HeightmapFunctor JIMINY_DLLAPI mergeHeightmap(
        const std::vector<HeightmapFunctor> & heightmaps);

    Eigen::MatrixXd JIMINY_DLLAPI discretizeHeightmap(const HeightmapFunctor & heightmap,
                                                      const float64_t & gridSize,
                                                      const float64_t & gridUnit);
}

#endif  // JIMINY_RANDOM_H
