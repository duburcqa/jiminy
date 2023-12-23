#ifndef JIMINY_RANDOM_H
#define JIMINY_RANDOM_H

#include <optional>

#include "jiminy/core/fwd.h"


namespace jiminy
{
    // ************ Random number generator utilities ***************

    void JIMINY_DLLAPI resetRandomGenerators(
        const std::optional<uint32_t> & seed = std::nullopt) noexcept;

    hresult_t JIMINY_DLLAPI getRandomSeed(uint32_t & seed);

    double JIMINY_DLLAPI randUniform(double lo = 0.0, double hi = 1.0);

    double JIMINY_DLLAPI randNormal(double mean = 0.0, double std = 1.0);

    Eigen::VectorXd JIMINY_DLLAPI randVectorNormal(uint32_t size, double mean, double std);

    Eigen::VectorXd JIMINY_DLLAPI randVectorNormal(uint32_t size, double std);

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
        PeriodicGaussianProcess(double wavelength, double period, double scale = 1.0) noexcept;

        ~PeriodicGaussianProcess() = default;

        void reset();

        double operator()(const float & t);

        double getWavelength() const;
        double getPeriod() const;
        double getDt() const;

    protected:
        void initialize();

    private:
        const double wavelength_;
        const double period_;
        const double scale_;
        const double dt_;
        const int32_t numTimes_;

        bool isInitialized_{false};
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
        PeriodicFourierProcess(double wavelength, double period, double scale = 1.0);

        ~PeriodicFourierProcess() = default;

        void reset();

        double operator()(const float & t);

        double getWavelength() const;
        double getPeriod() const;
        int32_t getNumHarmonics() const;
        double getDt() const;

    protected:
        void initialize();

    private:
        const double wavelength_;
        const double period_;
        const double scale_;
        const double dt_;
        const int32_t numTimes_;
        const int32_t numHarmonics_;

        bool isInitialized_;
        Eigen::VectorXd values_;
        Eigen::MatrixXd cosMat_;
        Eigen::MatrixXd sinMat_;
    };

    class JIMINY_DLLAPI AbstractPerlinNoiseOctave
    {
    public:
        AbstractPerlinNoiseOctave(double wavelength, double scale);
        virtual ~AbstractPerlinNoiseOctave() = default;

        virtual void reset();

        double operator()(double t) const;

        double getWavelength() const;
        double getScale() const;

    protected:
        // Copy on purpose
        virtual double grad(int32_t knot, double delta) const = 0;

        double fade(double delta) const;

        double lerp(double ratio, double yLeft, double yRight) const;

    protected:
        const double wavelength_;
        const double scale_;

        double shift_;
    };

    class JIMINY_DLLAPI RandomPerlinNoiseOctave : public AbstractPerlinNoiseOctave
    {
    public:
        RandomPerlinNoiseOctave(double wavelength, double scale);

        virtual ~RandomPerlinNoiseOctave() = default;

        virtual void reset() override final;

    protected:
        virtual double grad(int32_t knot, double delta) const override final;

    private:
        uint32_t seed_;
    };

    class JIMINY_DLLAPI PeriodicPerlinNoiseOctave : public AbstractPerlinNoiseOctave
    {
    public:
        PeriodicPerlinNoiseOctave(double wavelength, double period, double scale);

        virtual ~PeriodicPerlinNoiseOctave() = default;

        virtual void reset() override final;

    protected:
        virtual double grad(int32_t knot, double delta) const override final;

    private:
        const double period_;

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
        AbstractPerlinProcess(double wavelength, double scale = 1.0, uint32_t numOctaves = 8U);
        virtual ~AbstractPerlinProcess() = default;

        void reset();

        double operator()(const float & t);

        double getWavelength() const;
        uint32_t getNumOctaves() const;
        double getScale() const;

    protected:
        virtual void initialize() = 0;

    protected:
        const double wavelength_;
        const uint32_t numOctaves_;
        const double scale_;

        bool isInitialized_;
        std::vector<std::unique_ptr<AbstractPerlinNoiseOctave>> octaves_;
        double amplitude_;

        double grad_;
    };

    class JIMINY_DLLAPI RandomPerlinProcess : public AbstractPerlinProcess
    {
    public:
        RandomPerlinProcess(double wavelength, double scale = 1.0, uint32_t numOctaves = 6U);

        virtual ~RandomPerlinProcess() = default;

    protected:
        virtual void initialize() override final;
    };

    class PeriodicPerlinProcess : public AbstractPerlinProcess
    {
    public:
        PeriodicPerlinProcess(
            double wavelength, double period, double scale = 1.0, uint32_t numOctaves = 6U);

        virtual ~PeriodicPerlinProcess() = default;

        double getPeriod() const;

    protected:
        virtual void initialize() override final;

    private:
        const double period_;
    };

    // ************ Random terrain generators ***************

    HeightmapFunctor JIMINY_DLLAPI randomTileGround(const Eigen::Vector2d & size,
                                                    double heightMax,
                                                    const Eigen::Vector2d & interpDelta,
                                                    uint32_t sparsity,
                                                    double orientation,
                                                    uint32_t seed);

    HeightmapFunctor JIMINY_DLLAPI sumHeightmap(const std::vector<HeightmapFunctor> & heightmaps);
    HeightmapFunctor JIMINY_DLLAPI mergeHeightmap(
        const std::vector<HeightmapFunctor> & heightmaps);

    Eigen::MatrixXd JIMINY_DLLAPI discretizeHeightmap(
        const HeightmapFunctor & heightmap, double gridSize, double gridUnit);
}

#endif  // JIMINY_RANDOM_H
