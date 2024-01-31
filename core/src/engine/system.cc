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
    // ******************************** External force functors ******************************** //

    ProfileForce::ProfileForce(const std::string & frameNameIn,
                               pinocchio::FrameIndex frameIndexIn,
                               double updatePeriodIn,
                               const ProfileForceFunction & forceFuncIn) noexcept :
    frameName{frameNameIn},
    frameIndex{frameIndexIn},
    updatePeriod{updatePeriodIn},
    func{forceFuncIn}
    {
    }

    ImpulseForce::ImpulseForce(const std::string & frameNameIn,
                               pinocchio::FrameIndex frameIndexIn,
                               double tIn,
                               double dtIn,
                               const pinocchio::Force & forceIn) noexcept :
    frameName{frameNameIn},
    frameIndex{frameIndexIn},
    t{tIn},
    dt{dtIn},
    force{forceIn}
    {
    }

    CouplingForce::CouplingForce(const std::string & systemName1In,
                                 std::ptrdiff_t systemIndex1In,
                                 const std::string & systemName2In,
                                 std::ptrdiff_t systemIndex2In,
                                 const std::string & frameName1In,
                                 pinocchio::FrameIndex frameIndex1In,
                                 const std::string & frameName2In,
                                 pinocchio::FrameIndex frameIndex2In,
                                 const CouplingForceFunction & forceFuncIn) noexcept :
    systemName1{systemName1In},
    systemIndex1{systemIndex1In},
    systemName2{systemName2In},
    systemIndex2{systemIndex2In},
    frameName1{frameName1In},
    frameIndex1{frameIndex1In},
    frameName2{frameName2In},
    frameIndex2{frameIndex2In},
    func{forceFuncIn}
    {
    }

    // ************************************** System state ************************************* //

    hresult_t SystemState::initialize(const Robot & robot)
    {
        if (!robot.getIsInitialized())
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        Eigen::Index nv = robot.nv();
        std::size_t nMotors = robot.nmotors();
        std::size_t nJoints = robot.pinocchioModel_.njoints;
        q = pinocchio::neutral(robot.pinocchioModel_);
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

    bool SystemState::getIsInitialized() const
    {
        return isInitialized_;
    }

    void SystemState::clear()
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

    SystemData::SystemData() = default;
    SystemData::SystemData(SystemData &&) = default;
    SystemData & SystemData::operator=(SystemData &&) = default;
    SystemData::~SystemData() = default;
}
