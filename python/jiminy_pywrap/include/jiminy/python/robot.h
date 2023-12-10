#ifndef ROBOT_PYTHON_H
#define ROBOT_PYTHON_H

#include "jiminy/core/fwd.h"


namespace jiminy::python
{
    void JIMINY_DLLAPI exposeModel();
    void JIMINY_DLLAPI exposeRobot();
}

#endif  // ROBOT_PYTHON_H
