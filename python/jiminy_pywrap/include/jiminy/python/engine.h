#ifndef SIMULATOR_PYTHON_H
#define SIMULATOR_PYTHON_H


namespace jiminy::python
{
    void exposeForces(void);
    void exposeStepperState(void);
    void exposeSystemState(void);
    void exposeSystem(void);
    void exposeEngineMultiRobot(void);
    void exposeEngine(void);
}

#endif  // SIMULATOR_PYTHON_H
