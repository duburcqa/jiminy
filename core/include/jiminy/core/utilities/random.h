#ifndef JIMINY_RANDOM_H
#define JIMINY_RANDOM_H

#include "jiminy/core/macros.h"
#include "jiminy/core/types.h"


namespace jiminy
{
    // ************ Random number generator utilities ***************

    void resetRandomGenerators(const std::optional<uint32_t> & seed = std::nullopt);

    hresult_t getRandomSeed(uint32_t & seed);

    float64_t randUniform(const float64_t & lo = 0.0, const float64_t & hi = 1.0);

    float64_t randNormal(const float64_t & mean = 0.0, const float64_t & std = 1.0);

    vectorN_t randVectorNormal(
        const uint32_t & size, const float64_t & mean, const float64_t & std);

    vectorN_t randVectorNormal(const uint32_t & size, const float64_t & std);

    vectorN_t randVectorNormal(const vectorN_t & std);

    vectorN_t randVectorNormal(const vectorN_t & mean, const vectorN_t & std);

    void shuffleIndices(std::vector<uint32_t> & vector);

    // ************ Continuous 1D Perlin processes ***************

    class PeriodicGaussianProcess
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
        vectorN_t values_;
        matrixN_t covSqrtRoot_;
    };


    /// \see Based on "Smooth random functions, random ODEs, and Gaussianprocesses":
    ///      https://hal.inria.fr/hal-01944992/file/random_revision2.pdf */
    class PeriodicFourierProcess
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
        vectorN_t values_;
        matrixN_t cosMat_;
        matrixN_t sinMat_;
    };

    class AbstractPerlinNoiseOctave
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

    class RandomPerlinNoiseOctave : public AbstractPerlinNoiseOctave
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

    class PeriodicPerlinNoiseOctave : public AbstractPerlinNoiseOctave
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
    class AbstractPerlinProcess
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

    class RandomPerlinProcess : public AbstractPerlinProcess
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

    heightmapFunctor_t randomTileGround(const vector2_t & size,
                                        const float64_t & heightMax,
                                        const vector2_t & interpDelta,
                                        const uint32_t & sparsity,
                                        const float64_t & orientation,
                                        const uint32_t & seed);

    heightmapFunctor_t sumHeightmap(const std::vector<heightmapFunctor_t> & heightmaps);
    heightmapFunctor_t mergeHeightmap(const std::vector<heightmapFunctor_t> & heightmaps);

    matrixN_t discretizeHeightmap(const heightmapFunctor_t & heightmap,
                                  const float64_t & gridSize,
                                  const float64_t & gridUnit);
}

#endif  // JIMINY_RANDOM_H
