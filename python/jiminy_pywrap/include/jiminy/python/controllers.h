#ifndef CONTROLLERS_PYTHON_H
#define CONTROLLERS_PYTHON_H

#include "jiminy/core/fwd.h"


namespace jiminy::python
{
    void JIMINY_DLLAPI exposeAbstractController();
    void JIMINY_DLLAPI exposeControllerFunctor();
}

#endif  // CONTROLLERS_PYTHON_H
