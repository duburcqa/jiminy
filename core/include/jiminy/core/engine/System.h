


#ifndef JIMINY_SYSTEM_H
#define JIMINY_SYSTEM_H

#include <set>

#include "jiminy/core/robot/Model.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    class Robot;
    class AbstractConstraintBase;
    class AbstractController;
    class LockGuardLocal;

    struct forceProfile_t
    {
    public:
        forceProfile_t(void) = default;
        forceProfile_t(std::string           const & frameNameIn,
                       int32_t               const & frameIdxIn,
                       float64_t             const & updatePeriodIn,
                       forceProfileFunctor_t const & forceFctIn);
        ~forceProfile_t(void) = default;

    public:
        std::string frameName;
        int32_t frameIdx;
        float64_t updatePeriod;
        pinocchio::Force forcePrev;
        forceProfileFunctor_t forceFct;
    };

    struct forceImpulse_t
    {
    public:
        forceImpulse_t(void) = default;
        forceImpulse_t(std::string      const & frameNameIn,
                       int32_t          const & frameIdxIn,
                       float64_t        const & tIn,
                       float64_t        const & dtIn,
                       pinocchio::Force const & FIn);
        ~forceImpulse_t(void) = default;

    public:
        std::string frameName;
        int32_t frameIdx;
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
                        int32_t                const & frameIdx1In,
                        std::string            const & frameName2In,
                        int32_t                const & frameIdx2In,
                        forceCouplingFunctor_t const & forceFctIn);
        ~forceCoupling_t(void) = default;

    public:
        std::string systemName1;
        int32_t systemIdx1;
        std::string systemName2;
        int32_t systemIdx2;
        std::string frameName1;
        int32_t frameIdx1;
        std::string frameName2;
        int32_t frameIdx2;
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
        systemState_t(void);
        ~systemState_t(void) = default;

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
        systemDataHolder_t(void);
        systemDataHolder_t(systemDataHolder_t && other) = default;
        systemDataHolder_t & operator = (systemDataHolder_t && other) = default;
        ~systemDataHolder_t(void) = default;

    public:
        std::unique_ptr<LockGuardLocal> robotLock;

        forceProfileRegister_t forcesProfile;
        forceImpulseRegister_t forcesImpulse;
        std::set<float64_t> forcesImpulseBreaks;                          ///< Ordered list (without repetitions) of the start and end time associated with the forces
        std::set<float64_t>::const_iterator forcesImpulseBreakNextIt;     ///< Iterator related to the time of the next breakpoint associated with the impulse forces
        std::vector<bool_t> forcesImpulseActive;                          ///< Flag to active the forces. This is used to handle t-, t+ properly. Otherwise, it is impossible to determine at time t if the force is active or not.

        constraintsHolder_t constraintsHolder;                            ///< Store copy of constraints register for fast access.
        std::vector<int32_t> boundJointsActiveDir;                        ///< Store the active "direction" of the bound (0 for lower, 1 for higher)
        forceVector_t contactFramesForces;                                ///< Contact forces for each contact frames in local frame
        std::vector<forceVector_t> collisionBodiesForces;                 ///< Contact forces for each geometries of each collision bodies in local frame
        matrixN_t jointJacobian;                                          ///< Buffer used for intermediary computation of `uAugmented`
        vectorN_t uAugmented;                                             ///< Used to store the input effort plus the effect of external forces
        vectorN_t lo;                                                     ///< Lower bound of LCP problem
        vectorN_t hi;                                                     ///< Higher bound of LCP problem
        std::vector<int32_t> fIdx;                                        ///< Used to indicate linear coupling between bounds of LCP and the solution (i.e. friction pyramid: - mu * F_z < F_x/F_y < mu * F_z)

        std::vector<std::string> positionFieldnames;
        std::vector<std::string> velocityFieldnames;
        std::vector<std::string> accelerationFieldnames;
        std::vector<std::string> commandFieldnames;
        std::vector<std::string> motorEffortFieldnames;
        std::string energyFieldname;

        systemState_t state;       ///< Internal buffer with the state for the integration loop
        systemState_t statePrev;   ///< Internal state for the integration loop at the end of the previous iteration
    };
}

#endif //end of JIMINY_STEPPERS_H
