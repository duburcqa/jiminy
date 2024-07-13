#ifndef JIMINY_GEOMETRY_H
#define JIMINY_GEOMETRY_H

#include <vector>  // `std::vector`

#include "jiminy/core/fwd.h"  // `HeightmapFunction`

#include "hpp/fcl/fwd.hh"  // `hpp::fcl::BVHModelPtr_t`


namespace jiminy
{
    HeightmapFunction JIMINY_DLLAPI sumHeightmaps(
        const std::vector<HeightmapFunction> & heightmaps);

    HeightmapFunction JIMINY_DLLAPI mergeHeightmaps(
        const std::vector<HeightmapFunction> & heightmaps);

    hpp::fcl::CollisionGeometryPtr_t JIMINY_DLLAPI discretizeHeightmap(
        const HeightmapFunction & heightmap,
        double xMin,
        double xMax,
        double xUnit,
        double yMin,
        double yMax,
        double yUnit,
        bool mustSimplify = false);

    /// \brief Unidirectional periodic stairs ground of even parity, consisting of alternating
    ///        ascending and descending staircases.
    ///                               _______                         _______
    ///     __                _______|H      |_______         _______|H      |_______
    ///       |______________|H                     H|_______|H                     H|_ . . .
    ///       .       .      .       .       .       .       .       .       .       .
    ///       .   W   .  W   .   W   .   W   .   W   .   W   .   W   .   W   .   W   .
    ///       .  i=0  . i=0  .  i=1  .  i=N  . i=N+1 .  i=0  .  i=1  .  i=N  . i=N+1 .
    ///               |------>
    ///             x = 0
    ///
    /// \details The stairs have identical height and width, and each staircase has an identical
    ///          step number. This number corresponds to the amount of steps to climb in order to
    ///          reach the highest steps from the lowest ones. The above ASCII art shows staircases
    ///          with a step number of two.
    ///
    /// \param[in] stepWidth   Width of the steps.
    /// \param[in] stepHeight  Heigh of the steps.
    /// \param[in] stepNumber  Number of steps in the ascending or descending direction.
    /// \param[in] orientation Orientation of the staircases in the XY plane.
    HeightmapFunction JIMINY_DLLAPI periodicStairs(
        double stepWidth, double stepHeight, uint32_t stepNumber, double orientation);

    HeightmapFunction JIMINY_DLLAPI unidirectionalRandomPerlinGround(
        double wavelength, std::size_t numOctaves, double orientation, uint32_t seed);

    HeightmapFunction JIMINY_DLLAPI unidirectionalPeriodicPerlinGround(double wavelength,
                                                                       double period,
                                                                       std::size_t numOctaves,
                                                                       double orientation,
                                                                       uint32_t seed);

    HeightmapFunction JIMINY_DLLAPI randomPerlinGround(
        double wavelength, std::size_t numOctaves, uint32_t seed);

    HeightmapFunction JIMINY_DLLAPI periodicPerlinGround(
        double wavelength, double period, std::size_t numOctaves, uint32_t seed);
}

#endif  // JIMINY_GEOMETRY_H