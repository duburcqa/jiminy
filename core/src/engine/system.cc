#include "pinocchio/spatial/force.hpp"                  // `pinocchio::Force`
#include "pinocchio/algorithm/joint-configuration.hpp"  // `pinocchio::neutral`

#include "jiminy/core/robot/robot.h"
#include "jiminy/core/solver/constraint_solvers.h"
#include "jiminy/core/constraints/abstract_constraint.h"
#include "jiminy/core/control/abstract_controller.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/engine/system.h"


namespace jiminy
{
    // ====================================================
    // ================== forceProfile_t ==================
    // ====================================================

    forceProfile_t::forceProfile_t(const std::string & frameNameIn,
                                   const frameIndex_t & frameIdxIn,
                                   const float64_t & updatePeriodIn,
                                   const forceProfileFunctor_t & forceFctIn) :
    frameName(frameNameIn),
    frameIdx(frameIdxIn),
    updatePeriod(updatePeriodIn),
    forcePrev(pinocchio::Force::Zero()),
    forceFct(forceFctIn)
    {
    }

    // ====================================================
    // ================== forceImpulse_t ==================
    // ====================================================

    forceImpulse_t::forceImpulse_t(const std::string & frameNameIn,
                                   const frameIndex_t & frameIdxIn,
                                   const float64_t & tIn,
                                   const float64_t & dtIn,
                                   const pinocchio::Force & FIn) :
    frameName(frameNameIn),
    frameIdx(frameIdxIn),
    t(tIn),
    dt(dtIn),
    F(FIn)
    {
    }

    // ====================================================
    // ================== forceCoupling_t =================
    // ====================================================

    forceCoupling_t::forceCoupling_t(const std::string & systemName1In,
                                     const int32_t & systemIdx1In,
                                     const std::string & systemName2In,
                                     const int32_t & systemIdx2In,
                                     const std::string & frameName1In,
                                     const frameIndex_t & frameIdx1In,
                                     const std::string & frameName2In,
                                     const frameIndex_t & frameIdx2In,
                                     const forceCouplingFunctor_t & forceFctIn) :
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
    }

    // ====================================================
    // ================== systemHolder_t ==================
    // ====================================================

    systemHolder_t::systemHolder_t(const std::string & systemNameIn,
                                   std::shared_ptr<Robot> robotIn,
                                   std::shared_ptr<AbstractController> controllerIn,
                                   callbackFunctor_t callbackFctIn) :
    name(systemNameIn),
    robot(robotIn),
    controller(controllerIn),
    callbackFct(std::move(callbackFctIn))
    {
    }

    systemHolder_t::systemHolder_t() :
    systemHolder_t("",
                   nullptr,
                   nullptr,
                   [](const float64_t & /* t */,
                      const Eigen::VectorXd & /* q */,
                      const Eigen::VectorXd & /* v */) -> bool_t { return false; })
    {
    }

    // ===============================================
    // ================ systemState_t ================
    // ===============================================

    systemState_t::systemState_t() :
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
    }

    hresult_t systemState_t::initialize(const Robot & robot)
    {
        if (!robot.getIsInitialized())
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        q = pinocchio::neutral(robot.pncModel_);
        v.setZero(robot.nv());
        a.setZero(robot.nv());
        command.setZero(robot.getMotorsNames().size());
        u.setZero(robot.nv());
        uMotor.setZero(robot.getMotorsNames().size());
        uInternal.setZero(robot.nv());
        uCustom.setZero(robot.nv());
        fExternal = forceVector_t(robot.pncModel_.joints.size(), pinocchio::Force::Zero());
        isInitialized_ = true;

        return hresult_t::SUCCESS;
    }

    const bool_t & systemState_t::getIsInitialized() const
    {
        return isInitialized_;
    }

    void systemState_t::clear()
    {
        q.resize(0);
        v.resize(0);
        a.resize(0);
        command.resize(0);
        u.resize(0);
        uMotor.resize(0);
        uInternal.resize(0);
        uCustom.resize(0);
        fExternal.clear();
    }

    systemDataHolder_t::systemDataHolder_t() = default;
    systemDataHolder_t::systemDataHolder_t(systemDataHolder_t &&) = default;
    systemDataHolder_t & systemDataHolder_t::operator=(systemDataHolder_t &&) = default;
    systemDataHolder_t::~systemDataHolder_t() = default;
}
