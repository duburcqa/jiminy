#ifndef CONSTRAINTS_PYTHON_H
#define CONSTRAINTS_PYTHON_H

#include "jiminy/core/fwd.h"


namespace jiminy::python
{
    void JIMINY_DLLAPI exposeConstraint();
    void JIMINY_DLLAPI exposeConstraintTree();
}

#endif  // CONSTRAINTS_PYTHON_H
