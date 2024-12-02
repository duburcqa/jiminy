#ifndef FWD_PYTHON_H
#define FWD_PYTHON_H

/* Specify a unique symbol for Numpy C API that is jiminy-specific to avoid conflict
   with other pre-compiled shared libraries.
   See: https://github.com/numpy/numpy/issues/26091
        https://github.com/numpy/numpy/issues/9309 */
#define PY_ARRAY_UNIQUE_SYMBOL JIMINY_ARRAY_API

/* Numpy headers drags Python with them. As a result, it is necessary to include the desired Python
   library before Numpy picks the default one, as it would be impossible to to change it afterward.
   Boost::Python provides a helper specifically dedicated to selecting the right Python library
   depending on build type, so let's make use of it. */
#if !defined(NO_IMPORT_ARRAY) && !defined(NO_IMPORT)
#    include <boost/python/detail/wrap_python.hpp>
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#    include "numpy/ndarrayobject.h"
#endif

// Eigenpy must be imported before Boost::Python as it sets pre-processor definitions
#include "pinocchio/bindings/python/fwd.hpp"

#include <boost/python/numpy.hpp>

#endif  // FWD_PYTHON_H
