#ifndef SENSORS_PYTHON_H
#define SENSORS_PYTHON_H

#include "jiminy/core/fwd.h"


namespace jiminy::python
{
    void exposeSensorMeasurementTree();
    void exposeAbstractSensor();
    void exposeBasicSensors();
}

#endif  // SENSORS_PYTHON_H
