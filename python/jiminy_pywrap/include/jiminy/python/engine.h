#ifndef SIMULATOR_PYTHON_H
#define SIMULATOR_PYTHON_H

#include "jiminy/core/fwd.h"


namespace jiminy::python
{
    void JIMINY_DLLAPI exposeForces();
    void JIMINY_DLLAPI exposeStepperState();
    void JIMINY_DLLAPI exposeRobotState();
    void JIMINY_DLLAPI exposeEngine();
}

#endif  // SIMULATOR_PYTHON_H
