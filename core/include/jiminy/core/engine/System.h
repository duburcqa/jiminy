


#ifndef JIMINY_SYSTEM_H
#define JIMINY_SYSTEM_H

#include <set>

#include "jiminy/core/robot/Model.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    class Robot;
    class AbstractConstraintSolver;
    class AbstractConstraintBase;
    class AbstractController;
    class LockGuardLocal;

    struct forceProfile_t
    {
    public:
        forceProfile_t(void) = default;
        forceProfile_t(std::string           const & frameNameIn,
                       frameIndex_t          const & frameIdxIn,
                       float64_t             const & updatePeriodIn,
                       forceProfileFunctor_t const & forceFctIn);

    public:
        std::string frameName;
        frameIndex_t frameIdx;
        float64_t updatePeriod;
        pinocchio::Force forcePrev;
        forceProfileFunctor_t forceFct;
    };

    struct forceImpulse_t
    {
    public:
        forceImpulse_t(void) = default;
        forceImpulse_t(std::string      const & frameNameIn,
                       frameIndex_t     const & frameIdxIn,
                       float64_t        const & tIn,
                       float64_t        const & dtIn,
                       pinocchio::Force const & FIn);

    public:
        std::string frameName;
        frameIndex_t frameIdx;
        float64_t t;
        float64_t dt;
        pinocchio::Force F;
    };

    struct forceCoupling_t
    {
    public:
        forceCoupling_t(void) = default;
        forceCoupling_t(std::string            const & systemName1In,
                        int32_t                const & systemIdx1In,
                        std::string            const & systemName2In,
                        int32_t                const & systemIdx2In,
                        std::string            const & frameName1In,
                        frameIndex_t           const & frameIdx1In,
                        std::string            const & frameName2In,
                        frameIndex_t           const & frameIdx2In,
                        forceCouplingFunctor_t const & forceFctIn);

    public:
        std::string systemName1;
        int32_t systemIdx1;
        std::string systemName2;
        int32_t systemIdx2;
        std::string frameName1;
        frameIndex_t frameIdx1;
        std::string frameName2;
        frameIndex_t frameIdx2;
        forceCouplingFunctor_t forceFct;
    };

    using forceProfileRegister_t = std::vector<forceProfile_t>;
    using forceImpulseRegister_t = std::vector<forceImpulse_t>;

    struct systemHolder_t
    {
    public:
        systemHolder_t(void);
        systemHolder_t(std::string const & systemNameIn,
                       std::shared_ptr<Robot> robotIn,
                       std::shared_ptr<AbstractController> controllerIn,
                       callbackFunctor_t callbackFctIn);
        systemHolder_t(systemHolder_t const & other) = default;
        systemHolder_t(systemHolder_t && other) = default;
        systemHolder_t & operator = (systemHolder_t const & other) = default;
        systemHolder_t & operator = (systemHolder_t && other) = default;
        ~systemHolder_t(void) = default;

    public:
        std::string name;
        std::shared_ptr<Robot> robot;
        std::shared_ptr<AbstractController> controller;
        callbackFunctor_t callbackFct;
    };

    struct systemState_t
    {
    public:
        // Non-default constructor to be considered initialized even if not
        systemState_t(void);

        hresult_t initialize(Robot const & robot);
        bool_t const & getIsInitialized(void) const;

        void clear(void);

    public:
        vectorN_t q;
        vectorN_t v;
        vectorN_t a;
        vectorN_t command;
        vectorN_t u;
        vectorN_t uMotor;
        vectorN_t uInternal;
        vectorN_t uCustom;
        forceVector_t fExternal;

    private:
        bool_t isInitialized_;
    };

    struct systemDataHolder_t
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
        std::unique_ptr<LockGuardLocal> robotLock;

        forceProfileRegister_t forcesProfile;
        forceImpulseRegister_t forcesImpulse;
        std::set<float64_t> forcesImpulseBreaks;                       ///< Ordered list (without repetitions) of the start and end time associated with the forces
        std::set<float64_t>::const_iterator forcesImpulseBreakNextIt;  ///< Iterator related to the time of the next breakpoint associated with the impulse forces
        std::vector<bool_t> forcesImpulseActive;                       ///< Flag to active the forces. This is used to handle t-, t+ properly. Otherwise, it is impossible to determine at time t if the force is active or not.

        std::unique_ptr<AbstractConstraintSolver> constraintSolver;
        constraintsHolder_t constraintsHolder;                         ///< Store copy of constraints register for fast access.
        forceVector_t contactFramesForces;                             ///< Contact forces for each contact frames in local frame
        vector_aligned_t<forceVector_t> collisionBodiesForces;         ///< Contact forces for each geometries of each collision bodies in local frame
        matrix6N_t jointJacobian;                                      ///< Buffer used for intermediary computation of `data.u`

        std::vector<std::string> logFieldnamesPosition;
        std::vector<std::string> logFieldnamesVelocity;
        std::vector<std::string> logFieldnamesAcceleration;
        std::vector<std::string> logFieldnamesForceExternal;
        std::vector<std::string> logFieldnamesCommand;
        std::vector<std::string> logFieldnamesMotorEffort;
        std::string logFieldnameEnergy;

        systemState_t state;       ///< Internal buffer with the state for the integration loop
        systemState_t statePrev;   ///< Internal state for the integration loop at the end of the previous iteration
    };
}

#endif //end of JIMINY_STEPPERS_H
