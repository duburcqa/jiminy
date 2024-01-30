#ifndef JIMINY_GEOMETRY_H
#define JIMINY_GEOMETRY_H

#include <vector>  // `std::vector`

#include "jiminy/core/fwd.h"  // `HeightmapFunctor`

#include "hpp/fcl/fwd.hh"  // `hpp::fcl::BVHModelPtr_t`


namespace jiminy
{
    HeightmapFunctor JIMINY_DLLAPI sumHeightmaps(std::vector<HeightmapFunctor> heightmaps);

    HeightmapFunctor JIMINY_DLLAPI mergeHeightmaps(std::vector<HeightmapFunctor> heightmaps);

    hpp::fcl::CollisionGeometryPtr_t JIMINY_DLLAPI discretizeHeightmap(
        const HeightmapFunctor & heightmap,
        double xMin,
        double xMax,
        double xUnit,
        double yMin,
        double yMax,
        double yUnit,
        bool mustSimplify = false);
}

#endif  // JIMINY_GEOMETRY_H