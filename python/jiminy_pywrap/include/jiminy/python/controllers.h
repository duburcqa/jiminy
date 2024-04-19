#ifndef CONTROLLERS_PYTHON_H
#define CONTROLLERS_PYTHON_H

#include "jiminy/core/fwd.h"


namespace jiminy::python
{
    void exposeAbstractController();
    void exposeFunctionalController();
}

#endif  // CONTROLLERS_PYTHON_H
