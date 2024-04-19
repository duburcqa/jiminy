#ifndef SIMULATOR_PYTHON_H
#define SIMULATOR_PYTHON_H

#include "jiminy/core/fwd.h"


namespace jiminy::python
{
    void exposeForces();
    void exposeStepperState();
    void exposeRobotState();
    void exposeEngine();
}

#endif  // SIMULATOR_PYTHON_H
