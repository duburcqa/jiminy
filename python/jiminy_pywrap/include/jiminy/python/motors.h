#ifndef MOTORS_PYTHON_H
#define MOTORS_PYTHON_H

#include "jiminy/core/fwd.h"


namespace jiminy::python
{
    void JIMINY_DLLAPI exposeAbstractMotor();
    void JIMINY_DLLAPI exposeSimpleMotor();
}

#endif  // MOTORS_PYTHON_H
