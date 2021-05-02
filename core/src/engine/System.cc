#include "pinocchio/spatial/force.hpp"                  // `pinocchio::Force`
#include "pinocchio/algorithm/joint-configuration.hpp"  // `pinocchio::neutral`

#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/constraints/AbstractConstraint.h"
#include "jiminy/core/control/AbstractController.h"
#include "jiminy/core/utilities/Helpers.h"
#include "jiminy/core/engine/System.h"


namespace jiminy
{
    // ====================================================
    // ================== forceProfile_t ==================
    // ====================================================

    forceProfile_t::forceProfile_t(std::string           const & frameNameIn,
                                   frameIndex_t          const & frameIdxIn,
                                   float64_t             const & updatePeriodIn,
                                   forceProfileFunctor_t const & forceFctIn) :
    frameName(frameNameIn),
    frameIdx(frameIdxIn),
    updatePeriod(updatePeriodIn),
    forcePrev(pinocchio::Force::Zero()),
    forceFct(forceFctIn)
    {
        // Empty on purpose
    }

    // ====================================================
    // ================== forceImpulse_t ==================
    // ====================================================

    forceImpulse_t::forceImpulse_t(std::string      const & frameNameIn,
                                   frameIndex_t     const & frameIdxIn,
                                   float64_t        const & tIn,
                                   float64_t        const & dtIn,
                                   pinocchio::Force const & FIn) :
    frameName(frameNameIn),
    frameIdx(frameIdxIn),
    t(tIn),
    dt(dtIn),
    F(FIn)
    {
        // Empty on purpose
    }

    // ====================================================
    // ================== forceCoupling_t =================
    // ====================================================

    forceCoupling_t::forceCoupling_t(std::string            const & systemName1In,
                                     int32_t                const & systemIdx1In,
                                     std::string            const & systemName2In,
                                     int32_t                const & systemIdx2In,
                                     std::string            const & frameName1In,
                                     frameIndex_t           const & frameIdx1In,
                                     std::string            const & frameName2In,
                                     frameIndex_t           const & frameIdx2In,
                                     forceCouplingFunctor_t const & forceFctIn) :
    systemName1(systemName1In),
    systemIdx1(systemIdx1In),
    systemName2(systemName2In),
    systemIdx2(systemIdx2In),
    frameName1(frameName1In),
    frameIdx1(frameIdx1In),
    frameName2(frameName2In),
    frameIdx2(frameIdx2In),
    forceFct(forceFctIn)
    {
        // Empty on purpose.
    }

    // ====================================================
    // ================== systemHolder_t ==================
    // ====================================================

    systemHolder_t::systemHolder_t(std::string const & systemNameIn,
                                   std::shared_ptr<Robot> robotIn,
                                   std::shared_ptr<AbstractController> controllerIn,
                                   callbackFunctor_t callbackFctIn) :
    name(systemNameIn),
    robot(std::move(robotIn)),
    controller(std::move(controllerIn)),
    callbackFct(std::move(callbackFctIn))
    {
        // Empty on purpose
    }

    systemHolder_t::systemHolder_t(void) :
    systemHolder_t("", nullptr, nullptr,
    [](float64_t const & /* t */,
       vectorN_t const & /* q */,
       vectorN_t const & /* v */) -> bool_t
    {
        return false;
    })
    {
        // Empty on purpose.
    }

    // ===============================================
    // ================ systemState_t ================
    // ===============================================

    systemState_t::systemState_t(void) :
    q(),
    v(),
    a(),
    command(),
    u(),
    uMotor(),
    uInternal(),
    uCustom(),
    fExternal(),
    isInitialized_(false)
    {
        // Empty on purpose.
    }

    hresult_t systemState_t::initialize(Robot const & robot)
    {
        if (!robot.getIsInitialized())
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        q = pinocchio::neutral(robot.pncModel_);
        v = vectorN_t::Zero(robot.nv());
        a = vectorN_t::Zero(robot.nv());
        command = vectorN_t::Zero(robot.getMotorsNames().size());
        u = vectorN_t::Zero(robot.nv());
        uMotor = vectorN_t::Zero(robot.getMotorsNames().size());
        uInternal = vectorN_t::Zero(robot.nv());
        uCustom = vectorN_t::Zero(robot.nv());
        fExternal = forceVector_t(robot.pncModel_.joints.size(),
                                  pinocchio::Force::Zero());
        isInitialized_ = true;

        return hresult_t::SUCCESS;
    }

    bool_t const & systemState_t::getIsInitialized(void) const
    {
        return isInitialized_;
    }

    void systemState_t::clear(void)
    {
        q = vectorN_t();
        v = vectorN_t();
        a = vectorN_t();
        command = vectorN_t();
        u = vectorN_t();
        uMotor = vectorN_t();
        uInternal = vectorN_t();
        uCustom = vectorN_t();
        fExternal.clear();
    }

    // ===============================================
    // ============== systemDataHolder_t =============
    // ===============================================

    systemDataHolder_t::systemDataHolder_t(void) :
    robotLock(nullptr),
    forcesProfile(),
    forcesImpulse(),
    forcesImpulseBreaks(),
    forcesImpulseBreakNextIt(),
    forcesImpulseActive(),
    constraintsHolder(),
    boundJointsActiveDir(),
    contactFramesForces(),
    collisionBodiesForces(),
    jointJacobian(),
    uAugmented(),
    lo(),
    hi(),
    fIdx(),
    positionFieldnames(),
    velocityFieldnames(),
    accelerationFieldnames(),
    forceExternalFieldnames(),
    commandFieldnames(),
    motorEffortFieldnames(),
    energyFieldname(),
    state(),
    statePrev()
    {
        // Empty on purpose
    }
}
