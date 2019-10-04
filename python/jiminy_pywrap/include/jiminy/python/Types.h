///////////////////////////////////////////////////////////////////////////////
/// \brief   Types used for python bindings.
///
/// \copyright Wandercraft
///////////////////////////////////////////////////////////////////////////////

#ifndef WDC_OPTIMAL_PYTHON_TYPES_H
#define WDC_OPTIMAL_PYTHON_TYPES_H

#include <boost/python.hpp>
#include <eigenpy/eigenpy.hpp>

#include "wdc/optimal/Types.h"

namespace jiminy
{
    namespace python
    {
        namespace bp = boost::python;

        // Typedef for Numpy scalar type.
        typedef float64_t npScalar_t;

        // Typedef for Eigen unaligned equivalents.
        typedef VectorN vectorN_fx_t;
        typedef MatrixN matrixN_fx_t;
    }
}

#endif  // WDC_OPTIMAL_PYTHON_TYPES_H
