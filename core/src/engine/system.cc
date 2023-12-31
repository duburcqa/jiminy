#include "pinocchio/spatial/force.hpp"                  // `pinocchio::Force`
#include "pinocchio/algorithm/joint-configuration.hpp"  // `pinocchio::neutral`

#include "jiminy/core/robot/robot.h"
#include "jiminy/core/solver/constraint_solvers.h"
#include "jiminy/core/constraints/abstract_constraint.h"
#include "jiminy/core/control/abstract_controller.h"
#include "jiminy/core/engine/system.h"
#include "jiminy/core/utilities/helpers.h"


namespace jiminy
{
    // ====================================================
    // ================== ForceProfile ==================
    // ====================================================

    ForceProfile::ForceProfile(const std::string & frameNameIn,
                               pinocchio::FrameIndex frameIdxIn,
                               double updatePeriodIn,
                               const ForceProfileFunctor & forceFctIn) noexcept :
    frameName{frameNameIn},
    frameIdx{frameIdxIn},
    updatePeriod{updatePeriodIn},
    forceFct{forceFctIn}
    {
    }

    // ====================================================
    // ================== ForceImpulse ==================
    // ====================================================

    ForceImpulse::ForceImpulse(const std::string & frameNameIn,
                               pinocchio::FrameIndex frameIdxIn,
                               double tIn,
                               double dtIn,
                               const pinocchio::Force & FIn) noexcept :
    frameName{frameNameIn},
    frameIdx{frameIdxIn},
    t{tIn},
    dt{dtIn},
    F{FIn}
    {
    }

    // ====================================================
    // ================== ForceCoupling =================
    // ====================================================

    ForceCoupling::ForceCoupling(const std::string & systemName1In,
                                 std::ptrdiff_t systemIdx1In,
                                 const std::string & systemName2In,
                                 std::ptrdiff_t systemIdx2In,
                                 const std::string & frameName1In,
                                 pinocchio::FrameIndex frameIdx1In,
                                 const std::string & frameName2In,
                                 pinocchio::FrameIndex frameIdx2In,
                                 const ForceCouplingFunctor & forceFctIn) noexcept :
    systemName1{systemName1In},
    systemIdx1{systemIdx1In},
    systemName2{systemName2In},
    systemIdx2{systemIdx2In},
    frameName1{frameName1In},
    frameIdx1{frameIdx1In},
    frameName2{frameName2In},
    frameIdx2{frameIdx2In},
    forceFct{forceFctIn}
    {
    }

    // ===============================================
    // ================ systemState_t ================
    // ===============================================

    hresult_t systemState_t::initialize(const Robot & robot)
    {
        if (!robot.getIsInitialized())
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        Eigen::Index nv = robot.nv();
        std::size_t nMotors = robot.nmotors();
        std::size_t nJoints = robot.pncModel_.njoints;
        q = pinocchio::neutral(robot.pncModel_);
        v.setZero(nv);
        a.setZero(nv);
        command.setZero(nMotors);
        u.setZero(nv);
        uMotor.setZero(nMotors);
        uInternal.setZero(nv);
        uCustom.setZero(nv);
        fExternal = ForceVector(nJoints, pinocchio::Force::Zero());
        isInitialized_ = true;

        return hresult_t::SUCCESS;
    }

    bool systemState_t::getIsInitialized() const
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
