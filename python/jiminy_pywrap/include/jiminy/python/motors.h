#ifndef MOTORS_PYTHON_H
#define MOTORS_PYTHON_H

#include "jiminy/core/fwd.h"


namespace jiminy::python
{
    void exposeAbstractMotor();
    void exposeBasicMotors();
}

#endif  // MOTORS_PYTHON_H
