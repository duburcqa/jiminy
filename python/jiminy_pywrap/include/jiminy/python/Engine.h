#ifndef SIMULATOR_PYTHON_H
#define SIMULATOR_PYTHON_H


namespace jiminy
{
namespace python
{
    void exposeForces(void);
    void exposeStepperState(void);
    void exposeSystemState(void);
    void exposeSystem(void);
    void exposeEngineMultiRobot(void);
    void exposeEngine(void);
}  // End of namespace python.
}  // End of namespace jiminy.

#endif  // SIMULATOR_PYTHON_H
