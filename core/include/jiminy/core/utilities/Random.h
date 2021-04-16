#ifndef JIMINY_RANDOM_H
#define JIMINY_RANDOM_H

#include "jiminy/core/Macros.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    // ************ Random number generator utilities ***************

    void resetRandomGenerators(boost::optional<uint32_t> const & seed = boost::none);

    hresult_t getRandomSeed(uint32_t & seed);

    float64_t randUniform(float64_t const & lo = 0.0,
                          float64_t const & hi = 1.0);

    float64_t randNormal(float64_t const & mean = 0.0,
                         float64_t const & std = 1.0);

    vectorN_t randVectorNormal(uint32_t  const & size,
                                float64_t const & mean,
                                float64_t const & std);

    vectorN_t randVectorNormal(uint32_t  const & size,
                                float64_t const & std);

    vectorN_t randVectorNormal(vectorN_t const & std);

    vectorN_t randVectorNormal(vectorN_t const & mean,
                                vectorN_t const & std);

    void shuffleIndices(std::vector<uint32_t> & vector);

    class PeriodicGaussianProcess
    {
    public:
        // Forbid the copy of the class
        PeriodicGaussianProcess(PeriodicGaussianProcess const & process) = delete;
        PeriodicGaussianProcess & operator = (PeriodicGaussianProcess const & process) = delete;

    public:
        PeriodicGaussianProcess(float64_t const & wavelength,
                                float64_t const & period);

        ~PeriodicGaussianProcess(void) = default;

        void reset(void);

        float64_t operator()(float const & t);

        float64_t const & getWavelength(void) const;
        float64_t const & getPeriod(void) const;
        float64_t const & getDt(void) const;

    protected:
        void initialize(void);

    private:
        float64_t const wavelength_;
        float64_t const period_;
        float64_t const dt_;
        uint32_t const numTimes_;

        bool_t isInitialized_;
        vectorN_t values_;
        matrixN_t covSqrtRoot_;
    };


    class PeriodicFourierProcess
    {
    /* Based on "Smooth random functions, random ODEs, and Gaussianprocesses":
        - https://hal.inria.fr/hal-01944992/file/random_revision2.pdf */
    public:
        // Forbid the copy of the class
        PeriodicFourierProcess(PeriodicFourierProcess const & process) = delete;
        PeriodicFourierProcess & operator = (PeriodicFourierProcess const & process) = delete;

    public:
        PeriodicFourierProcess(float64_t const & wavelength,
                               float64_t const & period);

        ~PeriodicFourierProcess(void) = default;

        void reset(void);

        float64_t operator()(float const & t);

        float64_t const & getWavelength(void) const;
        float64_t const & getPeriod(void) const;
        uint32_t const & getNumHarmonics(void) const;
        float64_t const & getDt(void) const;

    protected:
        void initialize(void);

    private:
        float64_t const wavelength_;
        float64_t const period_;
        float64_t const dt_;
        uint32_t const numTimes_;
        uint32_t const numHarmonics_;

        bool_t isInitialized_;
        vectorN_t values_;
        matrixN_t cosMat_;
        matrixN_t sinMat_;
    };

    class AbstractPerlinNoiseOctave
    {
    public:
        AbstractPerlinNoiseOctave(float64_t const & wavelength,
                                  float64_t const & scale);

        virtual ~AbstractPerlinNoiseOctave(void) = default;

        virtual void reset(void);

        float64_t operator()(float64_t const & t) const;

        float64_t const & getWavelength(void) const;
        float64_t const & getScale(void) const;

    protected:
        virtual float64_t grad(int32_t knot,
                               float64_t const & delta) const = 0;  // Copy on purpose

        float64_t fade(float64_t const & delta) const;

        float64_t lerp(float64_t const & ratio,
                       float64_t const & yLeft,
                       float64_t const & yRight) const;

    protected:
        float64_t const wavelength_;
        float64_t const scale_;

        float64_t shift_;
    };

    class RandomPerlinNoiseOctave : public AbstractPerlinNoiseOctave
    {
    public:
        RandomPerlinNoiseOctave(float64_t const & wavelength,
                                float64_t const & scale);

        virtual ~RandomPerlinNoiseOctave(void) = default;

        virtual void reset(void) override final;

    protected:
        virtual float64_t grad(int32_t knot,
                               float64_t const & delta) const override final;

    private:
        uint32_t seed_;
    };

    class PeriodicPerlinNoiseOctave : public AbstractPerlinNoiseOctave
    {
    public:
        PeriodicPerlinNoiseOctave(float64_t const & wavelength,
                                  float64_t const & period,
                                  float64_t const & scale);

        virtual ~PeriodicPerlinNoiseOctave(void) = default;

        virtual void reset(void) override final;

    protected:
        virtual float64_t grad(int32_t knot,
                               float64_t const & delta) const override final;

    private:
        float64_t const period_;

        std::vector<uint8_t> perm_;
    };

    class AbstractPerlinProcess
    {
    /* \brief  Sum of Perlin noise octaves.

        \details  The original implementation uses fixed size permutation table to generate
                    random gradient directions. As a result, the generated process is inherently
                    periodic, which must be avoided. To circumvent this limitation, MurmurHash3
                    algorithm is used to get random gradients at every point in time, without any
                    periodicity, but deterministically for a given seed. It is computationally more
                    depending but not critically slower.

        /sa  For technical references:
            - https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/perlin-noise-part-2
            - https://adrianb.io/2014/08/09/perlinnoise.html
            - https://gamedev.stackexchange.com/a/23705/148509
            - https://gamedev.stackexchange.com/q/161923/148509
            - https://gamedev.stackexchange.com/q/134561/148509

        /sa  For reference about the implementation:
            - https://github.com/bradykieffer/SimplexNoise/blob/master/simplexnoise/noise.py
            - https://github.com/sol-prog/Perlin_Noise/blob/master/PerlinNoise.cpp
            - https://github.com/ashima/webgl-noise/blob/master/src/classicnoise2D.glsl
    */
    public:
        // Forbid the copy of the class
        AbstractPerlinProcess(AbstractPerlinProcess const & process) = delete;
        AbstractPerlinProcess & operator = (AbstractPerlinProcess const & process) = delete;

    public:
        AbstractPerlinProcess(float64_t const & wavelength,
                              uint32_t  const & numOctaves = 8U);

        virtual ~AbstractPerlinProcess(void) = default;

        void reset(void);

        float64_t operator()(float const & t);

        float64_t const & getWavelength(void) const;
        uint32_t const & getNumOctaves(void) const;

    protected:
        virtual void initialize(void) = 0;

    protected:
        float64_t const wavelength_;
        uint32_t const numOctaves_;

        bool_t isInitialized_;
        std::vector<std::unique_ptr<AbstractPerlinNoiseOctave> > octaves_;
        float64_t amplitude_;
    };

    class RandomPerlinProcess : public AbstractPerlinProcess
    {
    public:
        RandomPerlinProcess(float64_t const & wavelength,
                            uint32_t  const & numOctaves = 6U);

        virtual ~RandomPerlinProcess(void) = default;

    protected:
        virtual void initialize(void) override final;
    };

    class PeriodicPerlinProcess : public AbstractPerlinProcess
    {
    public:
        PeriodicPerlinProcess(float64_t const & wavelength,
                              float64_t const & period,
                              uint32_t  const & numOctaves = 6U);

        virtual ~PeriodicPerlinProcess(void) = default;

        float64_t const & getPeriod(void) const;

    protected:
        virtual void initialize(void) override final;

    private:
        float64_t const period_;
    };
}

#endif  // JIMINY_RANDOM_H
