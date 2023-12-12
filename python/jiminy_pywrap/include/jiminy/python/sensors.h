#ifndef SENSORS_PYTHON_H
#define SENSORS_PYTHON_H

#include "jiminy/core/fwd.h"


namespace jiminy::python
{
    void JIMINY_DLLAPI exposeSensorsDataMap();
    void JIMINY_DLLAPI exposeAbstractSensor();
    void JIMINY_DLLAPI exposeBasicSensors();
}

#endif  // SENSORS_PYTHON_H
