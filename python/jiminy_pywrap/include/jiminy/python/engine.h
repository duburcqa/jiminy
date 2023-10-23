#ifndef SIMULATOR_PYTHON_H
#define SIMULATOR_PYTHON_H


namespace jiminy::python
{
    void exposeForces();
    void exposeStepperState();
    void exposeSystemState();
    void exposeSystem();
    void exposeEngineMultiRobot();
    void exposeEngine();
}

#endif  // SIMULATOR_PYTHON_H
