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
}

#endif  // JIMINY_GEOMETRY_H